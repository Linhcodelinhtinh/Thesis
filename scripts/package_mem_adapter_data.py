#!/usr/bin/env python
"""Atomically package completed, verified expanded data for Kaggle."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import math
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.mixed_sft import PREDICTED_MEMORY_SOURCES, validate_mixed_dataset
from src.training.mem_adapter_artifact import verify_memory_code
from src.utils.file_hash import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--memory-index", type=Path, nargs="+",
                   help="Include each source index manifest and check it against the dataset receipt")
    p.add_argument("--min-free-gib", type=float, default=0,
                   help="Require space for an uncompressed archive plus this reserve")
    a = p.parse_args()
    m = validate_mixed_dataset(a.data_dir)
    if m["memory_source"] not in PREDICTED_MEMORY_SOURCES.values() or len(m["retention_replay_tasks"]) != 10:
        raise ValueError("Refusing to package reduced or no-memory data")
    verify_memory_code(m["memory_provenance"])
    out = a.output.resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError("Use a new immutable package path")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = ["manifest.json", "split.lock.json", "protocol.json", "samples.jsonl", *m["shards"]]
    indices = sorted(a.data_dir.parent.glob("memory_*/manifest.json"))
    if a.memory_index is not None:
        indices = [index / "manifest.json" for index in a.memory_index]
        expected = sorted(item["manifest_sha256"] for item in m["memory_provenance"])
        if sorted(sha256_file(index) for index in indices) != expected:
            raise ValueError("Index manifest inventory differs from the dataset receipt")
    if len({index.parent.name for index in indices}) != len(indices):
        raise ValueError("Index names must be distinct in the archive")
    if not math.isfinite(a.min_free_gib) or a.min_free_gib < 0:
        p.error("--min-free-gib must be finite and nonnegative")
    if a.min_free_gib:
        size = sum((a.data_dir / name).stat().st_size for name in names)
        size += sum(index.stat().st_size for index in indices)
        if shutil.disk_usage(out.parent).free < size + a.min_free_gib * 1024**3:
            raise OSError("Insufficient free space for a second dataset copy. Use a separate output volume.")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for name in names:
            z.write(a.data_dir / name, "dataset/" + name)
        for index in indices:
            z.write(index, "memory_provenance/" + index.parent.name + ".json")
    partial.rename(out)
    summary = dict(status="complete", archive=out.name, sha256=sha256_file(out),
                   bytes=out.stat().st_size, counts=m["counts"],
                   dataset_manifest_sha256=sha256_file(a.data_dir / "manifest.json"),
                   split_lock_sha256=m["split_lock_sha256"],
                   memory_source=m["memory_source"], base_tasks=len(m["retention_replay_tasks"]),
                   note="Memory predictions are not oracle labels; policy success has not been measured")
    if "mem_split_policy" in m:
        summary.update(mem_split_policy=m["mem_split_policy"],
                       evaluation_split_lock_sha256=m["evaluation_split_lock_sha256"])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
