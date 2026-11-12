#!/usr/bin/env python
"""Bundle the locked raw LIBERO-Mem-derived sources for Kaggle preparation."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import content_digest, load_split_lock
from src.utils.file_hash import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = load_split_lock(args.split_lock)
    manifest_path = args.source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (lock.get("classification") != "CUSTOM_V2_LIBERO_MEM_DERIVED_TRAINING_DATA"
            or lock.get("mem_image_convention") != "raw_opengl"
            or manifest.get("status") != "complete"
            or manifest.get("manifest_sha256") != content_digest(payload)
            or lock.get("derived_manifest_sha256") != manifest["manifest_sha256"]
            or lock.get("derived_manifest_file_sha256") != sha256_file(manifest_path)):
        raise ValueError("Derived source manifest does not match the training split lock")
    inventory = {row["path"]: row["sha256"] for row in lock["sources"] if row["domain"] == "mem"}
    if manifest["artifacts"] != inventory:
        raise ValueError("Derived source inventory differs from the split lock")
    for name, expected in inventory.items():
        if Path(name).name != name or sha256_file(args.source_dir / name) != expected:
            raise ValueError("Derived source hash/path mismatch: " + name)

    output = args.output.resolve()
    sidecar = output.with_suffix(".json")
    partial = output.with_suffix(".zip.partial")
    if any(path.exists() for path in (output, sidecar, partial)):
        raise FileExistsError("Use a new archive path for the immutable source bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(inventory):
            archive.write(args.source_dir / name, "derived_raw/" + name)
        archive.write(manifest_path, "derived_raw/manifest.json")
        archive.write(args.split_lock, "split.lock.json")
    receipt = dict(status="complete", archive=output.name, sha256=sha256_file(partial),
                   bytes=partial.stat().st_size, split_lock_sha256=lock["lock_sha256"],
                   derived_manifest_sha256=manifest["manifest_sha256"], source_hashes=inventory)
    partial.rename(output)
    sidecar.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
