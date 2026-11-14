#!/usr/bin/env python
"""Build expanded full-LIBERO-10 + T1/T3 causal-memory data, never a pilot fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.lock_memory_sft_split import build_lock, PREVIOUS_LOCK, RAW_ROOT
from src.training.split_lock import AUTHOR_MEM_SPLIT, load_split_lock
from src.training.mixed_sft import PREDICTED_MEMORY_SOURCES, validate_mixed_dataset
from src.evaluation.libero_mem_protocol import load_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, help="Optional separate scratch location for large training shards")
    p.add_argument("--perception-python", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--previous-lock", type=Path, default=PREVIOUS_LOCK, help="Preserve base replay assignments")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--writer", choices=("rgb_cycle_v3", "rgb_relations_v1"), default="rgb_relations_v1")
    p.add_argument("--rgb-compression", choices=("gzip", "lzf"), default="gzip")
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    out = a.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(python, script, *args):
        subprocess.run([str(python), str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, check=True)

    if a.download:
        run(sys.executable, "download_memory_training_sources.py", "--raw-root", a.raw_root)
    lock_path = out / "split.lock.json"
    if not lock_path.exists():
        lock = build_lock(ROOT / "data/libero_mem", ROOT / "resources/demonstrations",
                          ROOT / "resources/manifests/libero_10_sft_sources.json",
                          raw_root=a.raw_root, previous_lock=a.previous_lock)
        with lock_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(lock, indent=2) + "\n")
    lock = load_split_lock(lock_path)
    if lock.get("mem_split_policy") != AUTHOR_MEM_SPLIT or "parent_lock_sha256" in lock:
        raise ValueError("Use a fresh output root for the author-state split")
    base_tasks = {e["task"] for e in lock["episodes"] if e["domain"] == "base"}
    if len(base_tasks) != 10:
        raise ValueError("Expanded run requires all 10 official base tasks")
    indices = []
    for task in load_protocol()["tasks"]:
        index = out / ("memory_" + task["alias"])
        if not (index / "manifest.json").exists():
            run(a.perception_python, "prepare_libero_mem_sft.py", "--output-dir", index,
                "--task-name", task["name"], "--stream", "predicted_memory",
                "--tokenizer-path", a.tokenizer_path, "--split-lock", lock_path, "--writer", a.writer,
                "--resume")
        index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
        if (index_manifest.get("split_lock_sha256") != lock["lock_sha256"]
                or index_manifest.get("writer") != a.writer):
            raise ValueError("Existing index uses another split or writer: " + str(index))
        indices.append(index)
    dataset = a.dataset_dir.resolve() if a.dataset_dir else out / "dataset"
    if not (dataset / "manifest.json").exists():
        run(sys.executable, "prepare_mixed_memory_sft.py", "--split-lock", lock_path,
            "--tokenizer-path", a.tokenizer_path, "--output-dir", dataset,
            "--rgb-compression", a.rgb_compression, "--memory-index", *indices)
    manifest = validate_mixed_dataset(dataset)
    if manifest["memory_source"] != PREDICTED_MEMORY_SOURCES[a.writer] or len(manifest["retention_replay_tasks"]) != 10:
        raise ValueError("Dataset does not satisfy expanded Mem adapter contract")
    if manifest["split_lock_sha256"] != lock["lock_sha256"]:
        raise ValueError("Dataset split differs from expanded lock")
    print(json.dumps(dict(status="complete", dataset=str(dataset), counts=manifest["counts"])))


if __name__ == "__main__":
    main()
