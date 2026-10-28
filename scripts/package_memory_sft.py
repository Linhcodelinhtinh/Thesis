#!/usr/bin/env python
"""Create a small, source-only bundle for a private Kaggle Code Dataset."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="experiments/training/memory_sft_code.zip")
    args = p.parse_args()
    files = list((ROOT / "src").rglob("*.py"))
    files += [ROOT / "scripts" / name for name in (
        "prepare_memory_sft.py", "prepare_libero_mem_sft.py", "train_memory_sft.py", "memory_sft_assets.py",
        "replay_libero_mem_writer.py", "download_memory_detector.py", "audit_libero_mem_writer.py",
        "memory_detector_worker.py",
        "download_memory_training_sources.py", "lock_memory_sft_split.py", "prepare_mixed_memory_sft.py",
        "audit_libero_mem.py", "validate_mixed_memory_sft.py", "audit_mixed_memory_sft.py",
        "kaggle_memory_sft.sh", "package_memory_sft.py", "validate_memory_sft_export.py")]
    files += [ROOT / name for name in (
        "envs/kaggle_memory_sft.txt", "resources/manifests/models/smolvla_libero.yaml",
        "envs/memory_perception.txt", "resources/manifests/memory_perception_source.json",
        "resources/manifests/libero_mem_planning_sources.json",
        "resources/manifests/libero_10_sft_sources.json", "resources/manifests/libero_10_sft_pilot7_sources.json",
        "resources/manifests/libero_mem_t1_t3_base7_split_v1.json", "docs/V2_MEMORY_SFT_KAGGLE.md",
        "docs/MEMORY_WRITER_V3_T1_T3.md",
        "docs/LIBERO_MEM_FINETUNE_DATA_PROTOCOL.md", "configs/benchmarks/libero_mem_t1_t3_protocol.json",
        "notebooks/kaggle_libero_mem_finetune.ipynb")]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(ROOT))
    print(out.resolve())


if __name__ == "__main__":
    main()
