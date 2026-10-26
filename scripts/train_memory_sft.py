#!/usr/bin/env python
"""V2 supervised SmolVLA adaptation with official loss/processors and Accelerate.

No simulator or RL. Run with accelerate launch (two T4s: fp16, NOT bf16).
"""
import argparse
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.memory_data import sha256, write_json, code_hashes, MODEL_REVISION, BACKBONE_REVISION
from src.training.export_utils import preserve_policy_type


def validate_inputs(args):
    assets = Path(args.assets_dir)
    lock = json.loads((assets / "assets.json").read_text())
    if lock["model_revision"] != MODEL_REVISION or lock["backbone_revision"] != BACKBONE_REVISION:
        raise ValueError("Asset revisions differ from audited contract")
    for name, digest in lock["files"].items():
        if sha256(assets / name) != digest:
            raise ValueError(f"Asset corrupted: {name}")
    root = Path(args.data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["memory_code_sha256"] != code_hashes(ROOT):
        raise ValueError("Memory pipeline changed since preparation; regenerate dataset")
    for name, digest in manifest["shards"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Dataset shard corrupted: {name}")
    for name, digest in manifest["tokenizer_files_sha256"].items():
        if sha256(assets / "backbone" / name) != digest:
            raise ValueError(f"Preparation/training tokenizer mismatch: {name}")
    is_smoke = getattr(args, "steps", 0) <= 5 or getattr(args, "validation_max_batches", 0) > 0
    if args.condition == "memory" and not any(e["memory_frames"] for e in manifest["episodes"] if e["split"] == "train"):
        if not is_smoke:
            raise ValueError("Zero memory-bearing training samples. Fix/annotate memory or run explicit no_memory control.")
    return manifest


def export_model(policy, assets, destination, run_info):
    """Merge PEFT for the existing evaluation adapter; keep normalizers byte-identical."""
    destination.mkdir(parents=True, exist_ok=False)
    if hasattr(policy, "merge_and_unload"):
        policy = policy.merge_and_unload(safe_merge=True)
    policy.config.use_peft = False
    policy.config.push_to_hub = False
    policy.config.pretrained_path = None
    policy.config.vlm_model_name = str((destination / "backbone").resolve())
    policy.config.load_vlm_weights = False
    policy.save_pretrained(destination)
    preserve_policy_type(destination, policy.config.type)
    shutil.copytree(assets / "backbone", destination / "backbone", ignore=shutil.ignore_patterns(".cache"))
    for path in (assets / "policy").glob("policy_*processor*"):
        shutil.copy2(path, destination / path.name)
    pre = json.loads((destination / "policy_preprocessor.json").read_text())
    for step in pre["steps"]:
        if step["registry_name"] == "tokenizer_processor":
            step["config"]["tokenizer_name"] = str((destination / "backbone").resolve())
    write_json(destination / "policy_preprocessor.json", pre)
    immutable = {str(p.relative_to(destination)): sha256(p) for p in destination.rglob("*")
                 if p.is_file() and str(p.relative_to(destination)) not in {"config.json", "policy_preprocessor.json"}}
    write_json(destination / "export_manifest.json", {"run": run_info, "immutable_files": immutable,
               "relocation": "Run scripts/memory_sft_assets.py relocate PATH after moving this directory."})


def train(args):
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    # The adapter applies the repository's existing PyAV compatibility setup.
    from src.models.smolvla import adapter as _adapter
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file
    from src.training.memory_dataset import MemoryDataset, collate_memory

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/fallback training")
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.accumulation,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.num_processes != args.expected_gpus:
        raise ValueError(f"Expected {args.expected_gpus} GPU processes, got {accelerator.num_processes}")
    set_seed(args.seed)
    # Validate expensive hashes once; propagate failure via Accelerate's process termination.
    with accelerator.main_process_first():
        manifest = validate_inputs(args) if accelerator.is_main_process else json.loads((Path(args.data_dir) / "manifest.json").read_text())
    out, assets = Path(args.output_dir).resolve(), Path(args.assets_dir).resolve()
    if accelerator.is_main_process:
        if (out / "pretrained_model").exists():
            raise FileExistsError("This run already has a final export; use a new output directory for a resumed branch")
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise FileExistsError(f"Use a fresh output directory or --resume: {out}")
        out.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    cfg = PreTrainedConfig.from_pretrained(assets / "policy")
    if not isinstance(cfg, SmolVLAConfig):
        raise TypeError("Expected the audited SmolVLA checkpoint")
    cfg.device = "cpu"
    cfg.vlm_model_name = str(assets / "backbone")
    cfg.load_vlm_weights = False  # architecture only, then STRICT load of complete policy weights below
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True
    cfg.push_to_hub = False
    cfg.pretrained_path = str(assets / "policy")
    policy = SmolVLAPolicy.from_pretrained(assets / "policy", config=cfg, strict=True).float()
    # The complete policy is now loaded; tell PEFT validation this is adaptation,
    # not a randomly initialized action expert. No additional weights are loaded.
    cfg.load_vlm_weights = True
    if cfg.chunk_size != 50 or cfg.tokenizer_max_length != manifest["max_tokens"]:
        raise ValueError("Unexpected checkpoint chunk/token contract")
    # Verify statistics from both processors; never refit on this small dataset.
    for filename in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        stats = load_file(str(assets / "policy" / filename))
        if stats["observation.state.mean"].shape != (8,) or stats["action.mean"].shape != (7,):
            raise ValueError("Unknown state/action normalization contract")
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=assets / "policy",
        preprocessor_overrides={"device_processor": {"device": str(accelerator.device)},
                                "tokenizer_processor": {"tokenizer_name": str(assets / "backbone")}},
        postprocessor_overrides={"device_processor": {"device": str(accelerator.device)}})
    if args.adaptation == "expert_lora":
        policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": args.rank,
                                                          "lora_alpha": args.rank, "lora_dropout": 0.0})
    trainable = {n: p.numel() for n, p in policy.named_parameters() if p.requires_grad}
    if not trainable or any("vlm_with_expert.vlm." in name for name in trainable):
        raise RuntimeError("Unexpected trainable parameter set: backbone must stay frozen")
    try:
        forward_src = inspect.getsource(SmolVLAPolicy.forward)
        has_pad = any(k in forward_src for k in ("action_is_pad", "actions_is_pad", "actions_id_pad"))
    except Exception:
        has_pad = True
    if not has_pad:
        raise RuntimeError("Installed SmolVLA padding API changed; audit loss mask before training")
    train_data = MemoryDataset(args.data_dir, "train", cfg.chunk_size, args.condition,
                               args.memory_dropout if args.condition == "memory" else 0.0, args.seed)
    val_data = MemoryDataset(args.data_dir, "val", cfg.chunk_size, args.condition)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_memory,
                       pin_memory=True, persistent_workers=False)
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)

    def lr_scale(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        fraction = min(1, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    policy, optimizer, train_loader, val_loader = accelerator.prepare(policy, optimizer, train_loader, val_loader)
    # One scheduler update per optimizer update (not per GPU or microbatch).
    accelerator.register_for_checkpointing(scheduler)
    run_info = {"arguments": vars(args), "data_manifest_sha256": sha256(Path(args.data_dir) / "manifest.json"),
                "assets_lock_sha256": sha256(assets / "assets.json"),
                "effective_batch_size": args.batch_size * args.accumulation * accelerator.num_processes,
                "trainable_parameters": sum(trainable.values()), "trainable_modules": trainable,
                "versions": {p: importlib.metadata.version(p) for p in
                             ["torch", "transformers", "lerobot", "peft", "accelerate", "h5py"]},
                "hardware": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "implementation_sha256": {name: sha256(ROOT / name) for name in
                    ["scripts/train_memory_sft.py", "scripts/prepare_memory_sft.py",
                     "src/training/memory_dataset.py", "src/models/smolvla/adapter.py"]},
                "official_policy_source_sha256": sha256(inspect.getfile(SmolVLAPolicy)),
                "normalization": "unchanged original checkpoint", "precision": "FP32 weights, FP16 AMP + GradScaler"}
    if accelerator.is_main_process:
        write_json(out / "run_config.json", run_info)
        (out / "environment_packages.txt").write_text("\n".join(sorted(
            f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())) + "\n")
    epoch, step, skip = 0, 0, 0
    if args.resume:
        saved = json.loads((Path(args.resume) / "progress.json").read_text())
        for key in ("data_manifest_sha256", "assets_lock_sha256"):
            if saved[key] != run_info[key]:
                raise ValueError(f"Resume mismatch: {key}")
        expected = {k: v for k, v in vars(args).items() if k not in {"resume", "output_dir", "data_dir", "assets_dir"}}
        if saved["training_arguments"] != expected:
            raise ValueError("Resume training arguments differ; use the original schedule and batch configuration")
        accelerator.load_state(args.resume)
        epoch, step, skip = saved["epoch"], saved["step"], saved["next_batch"]

    def checkpoint(next_batch):
        directory = out / f"checkpoint-{step:06d}"
        accelerator.save_state(str(directory))
        if accelerator.is_main_process:
            write_json(directory / "progress.json", {"epoch": epoch, "step": step, "next_batch": next_batch,
                "data_manifest_sha256": run_info["data_manifest_sha256"], "assets_lock_sha256": run_info["assets_lock_sha256"],
                "training_arguments": {k: v for k, v in vars(args).items()
                                       if k not in {"resume", "output_dir", "data_dir", "assets_dir"}}})
        accelerator.wait_for_everyone()

    def validate():
        policy.eval()
        totals = {}
        # Same validation frames and flow noise for memory vs empty-memory conditions.
        for condition in ("memory", "no_memory"):
            val_data.condition = condition
            total, count = 0.0, 0
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                torch.manual_seed(args.seed + 123)
                torch.cuda.manual_seed(args.seed + 123)
                for validation_index, batch in enumerate(val_loader):
                    if args.validation_max_batches and validation_index >= args.validation_max_batches:
                        break
                    padding = batch.get("actions_id_pad", batch.get("action_is_pad"))
                    batch = preprocessor(batch)
                    pad_dev = padding.to(accelerator.device)
                    batch["actions_id_pad"] = pad_dev
                    batch["action_is_pad"] = pad_dev
                    with accelerator.autocast():
                        losses, _ = policy(batch, reduction="none")
                    values = accelerator.gather_for_metrics(losses.detach().float())
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Non-finite validation loss")
                    total += values.sum().item()
                    count += values.numel()
            totals[condition] = total / count
        policy.train()
        if accelerator.is_main_process:
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "validation_flow_loss": totals,
                                    "validation_max_batches": args.validation_max_batches}) + "\n")
        accelerator.print({"step": step, "validation_flow_loss": totals})

    policy.train()
    optimizer.zero_grad()
    consecutive_overflows = 0
    while step < args.steps:
        train_data.epoch = epoch
        train_loader.set_epoch(epoch)
        active_loader = accelerator.skip_first_batches(train_loader, skip) if skip else train_loader
        for batch_index, batch in enumerate(active_loader, start=skip):
            with accelerator.accumulate(policy):
                padding = batch.get("actions_id_pad", batch.get("action_is_pad"))
                batch = preprocessor(batch)
                pad_dev = padding.to(accelerator.device)
                batch["actions_id_pad"] = pad_dev
                batch["action_is_pad"] = pad_dev
                with accelerator.autocast():
                    loss, _ = policy(batch)
                finite = accelerator.reduce(torch.isfinite(loss).int(), reduction="sum")
                if finite.item() != accelerator.num_processes:
                    raise FloatingPointError("Non-finite flow loss; stopping instead of accepting corrupt gradients")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients and accelerator.optimizer_step_was_skipped:
                consecutive_overflows += 1
                accelerator.print(f"AMP skipped an overflowing update ({consecutive_overflows}/20)")
                if consecutive_overflows >= 20:
                    raise FloatingPointError("20 consecutive AMP overflows; stop and inspect precision/data instead of looping forever")
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                consecutive_overflows = 0
                scheduler.step()
                step += 1
                mean_loss = accelerator.reduce(loss.detach().float(), reduction="mean").item()
                if step == 1 or step % args.log_every == 0:
                    row = {"step": step, "epoch": epoch, "last_microbatch_flow_loss": mean_loss,
                           "lr": scheduler.get_last_lr()[0],
                           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3}
                    accelerator.print(row)
                    if accelerator.is_main_process:
                        with (out / "metrics.jsonl").open("a") as f:
                            f.write(json.dumps(row) + "\n")
                if step % args.eval_every == 0 or step == args.steps:
                    validate()
                if step % args.save_every == 0 or step == args.steps:
                    checkpoint(batch_index + 1)
                if step >= args.steps:
                    break
        epoch += 1
        skip = 0
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        export_model(accelerator.unwrap_model(policy), assets, out / "pretrained_model", run_info)
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "assets-dir", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--condition", choices=["memory", "no_memory"], default="memory")
    p.add_argument("--adaptation", choices=["expert_lora", "expert_full"], default="expert_lora")
    p.add_argument("--expected-gpus", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2, help="Per GPU")
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="Start at zero to avoid duplicating processor imports in Kaggle workers")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--memory-dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--validation-max-batches", type=int, default=0, help="0=full validation; >0 is a labeled smoke-test subset")
    p.add_argument("--resume")
    args = p.parse_args()
    if any(getattr(args, n) < 1 for n in ["expected_gpus", "batch_size", "accumulation", "steps", "rank", "log_every", "save_every", "eval_every"]):
        p.error("GPU/batch/step/interval values must be positive")
    if not 0 <= args.memory_dropout < 1 or args.lr <= 0 or args.warmup_steps < 0 or args.validation_max_batches < 0:
        p.error("Invalid learning rate, dropout or warmup")
    train(args)


if __name__ == "__main__":
    main()
