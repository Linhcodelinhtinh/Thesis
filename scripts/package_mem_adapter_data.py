#!/usr/bin/env python
"""Atomically package completed, verified expanded data for Kaggle."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.mixed_sft import validate_mixed_dataset
from src.training.mem_adapter_artifact import verify_memory_code
from src.utils.file_hash import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = validate_mixed_dataset(a.data_dir)
    if m["memory_source"] != "predicted_rgb_v3" or len(m["retention_replay_tasks"]) != 10:
        raise ValueError("Refusing to package reduced or no-memory data")
    verify_memory_code(m["memory_provenance"])
    out = a.output.resolve()
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError("Use a new immutable package path")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = ["manifest.json", "split.lock.json", "protocol.json", "samples.jsonl", *m["shards"]]
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for name in names:
            z.write(a.data_dir / name, "dataset/" + name)
        for index in sorted(a.data_dir.parent.glob("memory_*/manifest.json")):
            z.write(index, "memory_provenance/" + index.parent.name + ".json")
    partial.rename(out)
    summary = dict(status="complete", archive=out.name, sha256=sha256_file(out),
                   bytes=out.stat().st_size, counts=m["counts"],
                   dataset_manifest_sha256=sha256_file(a.data_dir / "manifest.json"),
                   memory_source=m["memory_source"], base_tasks=len(m["retention_replay_tasks"]),
                   note="Memory predictions are not oracle labels; policy success has not been measured")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
