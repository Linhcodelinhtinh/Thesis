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
        "prepare_memory_sft.py", "train_memory_sft.py", "memory_sft_assets.py",
        "kaggle_memory_sft.sh", "package_memory_sft.py", "validate_memory_sft_export.py")]
    files += [ROOT / name for name in (
        "envs/kaggle_memory_sft.txt", "resources/manifests/models/smolvla_libero.yaml",
        "resources/manifests/libero_10_sft_sources.json", "docs/V2_MEMORY_SFT_KAGGLE.md",
        "notebooks/kaggle_memory_sft.ipynb")]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(ROOT))
    print(out.resolve())


if __name__ == "__main__":
    main()
