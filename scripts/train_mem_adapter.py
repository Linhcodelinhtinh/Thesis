#!/usr/bin/env python
"""Launch the expanded, memory-conditioned isolated adapter recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.mixed_sft import validate_mixed_dataset
from src.training.mem_adapter_artifact import verify_memory_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    manifest = validate_mixed_dataset(a.data_dir)
    recipe = json.loads((ROOT / "configs/training/mem_adapter_v1.json").read_text())
    if len(manifest["retention_replay_tasks"]) != recipe["required_base_tasks"] or manifest["memory_source"] != recipe["memory_source"]:
        raise ValueError("Requires expanded ten-task replay with predicted causal memory")
    verify_memory_code(manifest["memory_provenance"])
    args = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(a.gpus),
            "--num_machines", "1", "--mixed_precision", "fp16", "--dynamo_backend", "no"]
    if a.gpus > 1:
        args.append("--multi_gpu")
    args += [str(ROOT / "scripts/train_memory_sft.py"), "--data-dir", str(a.data_dir.resolve()),
             "--assets-dir", str(a.assets_dir.resolve()), "--output-dir", str(a.output_dir.resolve()),
             "--expected-gpus", str(a.gpus), "--batch-size", "2", "--accumulation", str(16 // a.gpus)]
    for key in ("condition", "adaptation", "export_format", "lora_scope", "rank", "lr", "steps",
                "base_fraction", "memory_dropout", "save_every", "eval_every"):
        args += ["--" + key.replace("_", "-"), str(recipe[key])]
    subprocess.run(args, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
