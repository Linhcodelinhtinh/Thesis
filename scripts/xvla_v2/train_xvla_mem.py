#!/usr/bin/env python3
"""X-VLA LoRA training for F0/F1 on the V2 shards.

Same structure as upstream peft_train.py (X-VLA 6c4ecfb): LoRA r=8, alpha=16, all-linear,
modules_to_save = soft_prompt_hub + action_encoder + action_decoder, upstream
build_optimizer / update_group_lrs, AdamW betas (0.9, 0.95), grad clip 1.0.
Defaults follow configs/training/xvla_mem_adapter_v2.json (lr 5e-5, coef 0.5, freeze 0,
warmup 250, cosine to 0.1x); upstream values are freeze 1000 / warmup 2000 / lr 1e-4 / constant. Added, because upstream lacks them:
  * real gradient accumulation (--grad-accum); one "step" = one optimizer update
  * resumable checkpoints (adapter + optimizer + step) and --max-hours stop for Kaggle
  * per-update jsonl log: losses, LR, share of Mem samples and memory-bearing prompts
  * registers the V2 shard handler at runtime (no edit of upstream files)

Launch (2x T4):
  accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 train_xvla_mem.py \
     --xvla-root /kaggle/working/X-VLA --models /kaggle/working/x_vla_libero \
     --metas /kaggle/working/xvla_v2/metas_f1 --output-dir /kaggle/working/runs/f1 \
     --batch-size 4 --grad-accum 4 --iters 5000 --save-interval 500
Effective batch = batch_size * grad_accum * num_processes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--xvla-root", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--metas", required=True, help="dir with base.json + mem.json (or one json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4, help="per GPU micro-batch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--iters", type=int, default=5000, help="optimizer updates")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=5e-5, help="transformer LoRA + action heads")
    p.add_argument("--learning-coef", type=float, default=0.5, help="multiplier for VLM LoRA + soft prompts")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--freeze-steps", type=int, default=0, help="upstream 1000; LoRA LR=0 before this")
    p.add_argument("--warmup-steps", type=int, default=250, help="upstream 2000")
    p.add_argument("--use-cosine-decay", action="store_true", default=True)
    p.add_argument("--no-cosine-decay", dest="use_cosine_decay", action="store_false")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", default="auto", help="auto | none | path to ckpt-N")
    p.add_argument("--max-hours", type=float, default=0, help="save + exit cleanly after this wall time (0=off)")
    p.add_argument("--find-unused-parameters", action="store_true", default=True)
    p.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    return p.parse_args()


def latest_ckpt(out: Path):
    best = None
    for c in out.glob("ckpt-*"):
        s = c / "state.json"
        if s.is_file() and (c / "optimizer.pt").is_file():
            step = json.loads(s.read_text())["global_step"]
            if best is None or step > best[0]:
                best = (step, c)
    return best[1] if best else None


def main():
    args = parse()
    sys.path.insert(0, args.xvla_root)
    sys.path.insert(0, str(HERE))
    from shard_handler import register
    register(args.xvla_root)

    from accelerate import Accelerator, DistributedDataParallelKwargs
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from datasets.dataset import InfiniteDataReader
    from datasets import worker_init_fn
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    import peft_train as up  # upstream helpers: build_optimizer, update_group_lrs, get_logger

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    logger = up.get_logger("xvla_v2", output_dir=out, accelerator=accelerator)

    resume = None
    if args.resume == "auto":
        resume = latest_ckpt(out)
    elif args.resume != "none":
        resume = Path(args.resume)
    start_step = json.loads((resume / "state.json").read_text())["global_step"] if resume else 0
    if start_step >= args.iters:
        logger.info(f"already finished: {resume}")
        return

    seed = args.seed + 1000 * start_step + accelerator.process_index
    torch.manual_seed(seed); np.random.seed(seed % 2**32); random.seed(seed)

    model = XVLA.from_pretrained(args.models)
    if resume:
        model = PeftModel.from_pretrained(model, str(resume), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, bias="none", target_modules="all-linear",
            modules_to_save=["transformer.soft_prompt_hub", "transformer.action_encoder",
                             "transformer.action_decoder"]))
    trainable = {}
    for name, prm in model.named_parameters():
        if prm.requires_grad:
            key = "vlm" if ".vlm." in name else ("modules_to_save" if "modules_to_save" in name else "transformer_lora")
            trainable[key] = trainable.get(key, 0) + prm.numel()
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        logger.info(f"trainable by part: {trainable}")
    processor = XVLAProcessor.from_pretrained(args.models)

    reader = InfiniteDataReader(args.metas, num_actions=model.num_actions, training=True,
                                action_mode=model.action_mode)
    loader = DataLoader(reader, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=worker_init_fn, persistent_workers=args.num_workers > 0)

    ns = argparse.Namespace(learning_rate=args.learning_rate, learning_coef=args.learning_coef,
                            freeze_steps=args.freeze_steps, warmup_steps=args.warmup_steps, iters=args.iters,
                            use_cosine_decay=args.use_cosine_decay, min_lr_ratio=args.min_lr_ratio)
    optim = up.build_optimizer(model, args.learning_rate, args.weight_decay, tuple(args.betas), args.learning_coef)
    if resume:
        optim.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu"))
        logger.info(f"resumed from {resume} at step {start_step}")
    model, optim = accelerator.prepare(model, optim)

    def save(step):
        if not accelerator.is_main_process:
            return
        d = out / f"ckpt-{step}"
        d.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(str(d), safe_serialization=True)
        torch.save(optim.state_dict(), d / "optimizer.pt")
        (d / "state.json").write_text(json.dumps(dict(global_step=step, args=vars(args), seed=args.seed,
                                                    trainable_parameters=trainable)))
        logger.info(f"saved {d}")

    model.train()
    step, t0, t_start = start_step, time.time(), time.time()
    acc = dict(n=0, mem=0, hasmem=0)
    bad = 0
    logf = open(out / "train_log.jsonl", "a") if accelerator.is_main_process else None
    for batch in loader:
        is_mem = batch.pop("is_mem")
        has_mem = batch.pop("has_memory")
        acc["n"] += len(is_mem); acc["mem"] += int(is_mem.sum()); acc["hasmem"] += int(has_mem.sum())
        lang = processor.encode_language(batch.pop("language_instruction"))
        inputs = {k: v.to(accelerator.device, non_blocking=True) for k, v in {**batch, **lang}.items()}
        up.update_group_lrs(optim, step, ns)
        with accelerator.accumulate(model):
            loss_dict = model(**inputs)
            loss = sum(loss_dict.values())
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()
        if not accelerator.sync_gradients:
            continue
        step += 1
        bad = bad + 1 if not torch.isfinite(loss.detach()).all() else 0
        if bad >= 20:  # fp16 on T4: persistent NaN/inf is not recoverable by the grad scaler
            raise RuntimeError(f"loss non-finite for {bad} consecutive updates at step {step}; "
                               "rerun with --mixed_precision no or a smaller LR, resume from the last ckpt")
        if step % args.log_interval == 0 and accelerator.is_main_process:
            dt = (time.time() - t0) / args.log_interval; t0 = time.time()
            rec = {k: float(v.detach().float().item()) for k, v in loss_dict.items()}
            rec.update(step=step, loss_total=float(loss.detach().item()), sec_per_update=dt,
                       frac_mem=acc["mem"] / max(1, acc["n"]), frac_has_memory=acc["hasmem"] / max(1, acc["n"]),
                       lr={g["name"]: g["lr"] for g in optim.param_groups},
                       peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            logger.info(f"[{step}/{args.iters}] loss={rec['loss_total']:.4f} mem={rec['frac_mem']:.2f} "
                        f"hasmem={rec['frac_has_memory']:.2f} {dt:.2f}s/upd vram={rec['peak_vram_gib']:.1f}G")
            acc = dict(n=0, mem=0, hasmem=0)
        if step % args.save_interval == 0 or step >= args.iters:
            save(step)
        if step >= args.iters:
            break
        if args.max_hours and step % args.log_interval == 0:
            over = torch.tensor(float((time.time() - t_start) / 3600 > args.max_hours), device=accelerator.device)
            over = accelerator.reduce(over, reduction="sum")  # same decision on every rank
        else:
            over = None
        if over is not None and over.item() > 0:
            if step % args.save_interval:
                save(step)
            logger.info(f"max-hours reached at step {step}; rerun with --resume auto")
            break
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
