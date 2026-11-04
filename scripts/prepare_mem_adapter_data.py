#!/usr/bin/env python
"""Build expanded full-LIBERO-10 + T1/T3 causal-memory data, never a pilot fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.lock_memory_sft_split import build_lock
from src.training.split_lock import load_split_lock, seal_lock, validate_lock
from src.training.mixed_sft import validate_mixed_dataset
from src.evaluation.libero_mem_protocol import load_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--perception-python", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--previous-lock", type=Path, default=ROOT / "resources/manifests/libero_mem_t1_t3_base7_split_v1.json")
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    out = a.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(python, script, *args):
        subprocess.run([str(python), str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, check=True)

    if a.download:
        run(sys.executable, "download_memory_training_sources.py")
    lock_path = out / "split.lock.json"
    if not lock_path.exists():
        lock = build_lock(ROOT / "data/libero_mem", ROOT / "resources/demonstrations",
                          ROOT / "resources/manifests/libero_10_sft_sources.json")
        previous = load_split_lock(a.previous_lock)
        assignments = {e["initial_state_sha256"]: e["split"] for e in previous["episodes"]}
        for e in lock["episodes"]:
            if e["initial_state_sha256"] in assignments:
                e["split"] = assignments[e["initial_state_sha256"]]
        lock.pop("lock_sha256")
        lock["previous_lock_sha256"] = previous["lock_sha256"]
        lock = validate_lock(seal_lock(lock))
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    lock = load_split_lock(lock_path)
    base_tasks = {e["task"] for e in lock["episodes"] if e["domain"] == "base"}
    if len(base_tasks) != 10:
        raise ValueError("Expanded run requires all 10 official base tasks")
    for task in {e["task"] for e in lock["episodes"]}:
        if {e["split"] for e in lock["episodes"] if e["task"] == task} != {"train", "val", "test"}:
            raise ValueError("Expanded task has an empty split: " + task)
    indices = []
    for task in load_protocol()["tasks"]:
        index = out / ("memory_" + task["alias"])
        if not (index / "manifest.json").exists():
            run(a.perception_python, "prepare_libero_mem_sft.py", "--output-dir", index,
                "--task-name", task["name"], "--stream", "predicted_memory",
                "--tokenizer-path", a.tokenizer_path, "--split-lock", lock_path)
        indices.append(index)
    dataset = out / "dataset"
    if not (dataset / "manifest.json").exists():
        run(sys.executable, "prepare_mixed_memory_sft.py", "--split-lock", lock_path,
            "--tokenizer-path", a.tokenizer_path, "--output-dir", dataset, "--memory-index", *indices)
    manifest = validate_mixed_dataset(dataset)
    if manifest["memory_source"] != "predicted_rgb_v3" or len(manifest["retention_replay_tasks"]) != 10:
        raise ValueError("Dataset does not satisfy expanded Mem adapter contract")
    if manifest["split_lock_sha256"] != lock["lock_sha256"]:
        raise ValueError("Dataset split differs from expanded lock")
    print(json.dumps(dict(status="complete", dataset=str(dataset), counts=manifest["counts"])))


if __name__ == "__main__":
    main()
