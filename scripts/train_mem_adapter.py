#!/usr/bin/env python
"""Launch the expanded, memory-conditioned isolated adapter recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.mixed_sft import PREDICTED_MEMORY_SOURCES, validate_mixed_dataset
from src.training.mem_adapter_artifact import verify_memory_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=[1, 2], default=2)
    p.add_argument("--resume", type=Path, help="Resume training from an existing checkpoint directory")
    p.add_argument("--condition", choices=("memory", "no_memory"), default="memory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, help="Override number of training steps")
    p.add_argument("--save-every", type=int, help="Override checkpoint saving frequency")
    p.add_argument("--export-only", action="store_true",
                   help="Skip resumable training snapshots and keep the final adapter export")
    p.add_argument("--validation-max-batches", type=int, default=0,
                   help="Limit validation only for a smoke of at most five optimizer steps")
    a = p.parse_args()
    if a.export_only and (a.resume or a.save_every is not None):
        p.error("--export-only cannot be combined with --resume or --save-every")
    manifest = validate_mixed_dataset(a.data_dir)
    recipe = json.loads((ROOT / "configs/training/mem_adapter_v1.json").read_text())
    recipe["condition"] = a.condition
    if a.steps:
        recipe["steps"] = a.steps
    if a.save_every:
        recipe["save_every"] = a.save_every
    if a.validation_max_batches < 0 or (a.validation_max_batches and recipe["steps"] > 5):
        p.error("Truncated validation is only supported for training smoke")
    if (len(manifest["retention_replay_tasks"]) != recipe["required_base_tasks"] or
            manifest["memory_source"] not in PREDICTED_MEMORY_SOURCES.values()):
        raise ValueError("Requires expanded ten-task replay with predicted causal memory")
    if a.condition == "memory":
        verify_memory_code(manifest["memory_provenance"])
    args = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(a.gpus),
            "--num_machines", "1", "--mixed_precision", "fp16", "--dynamo_backend", "no"]
    if a.gpus > 1:
        args.append("--multi_gpu")
    args += [str(ROOT / "scripts/train_memory_sft.py"), "--data-dir", str(a.data_dir.resolve()),
             "--assets-dir", str(a.assets_dir.resolve()), "--output-dir", str(a.output_dir.resolve()),
             "--expected-gpus", str(a.gpus), "--batch-size", "2", "--accumulation", str(16 // a.gpus),
             "--seed", str(a.seed), "--validation-max-batches", str(a.validation_max_batches)]
    if a.resume:
        args += ["--resume", str(a.resume.resolve())]
    if a.export_only:
        args.append("--export-only")
    for key in ("condition", "adaptation", "export_format", "lora_scope", "rank", "lr", "steps",
                "base_fraction", "memory_dropout", "save_every", "eval_every"):
        args += ["--" + key.replace("_", "-"), str(recipe[key])]
    subprocess.run(args, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
