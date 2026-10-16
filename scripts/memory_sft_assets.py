#!/usr/bin/env python
"""Fetch pinned training assets, or relocate a self-contained exported policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.memory_data import (
    MODEL_REPO, MODEL_REVISION, BACKBONE_REPO, BACKBONE_REVISION, sha256, write_json,
)


def fetch_assets(directory):
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(directory).resolve()
    contract = yaml.safe_load((ROOT / "resources/manifests/models/smolvla_libero.yaml").read_text())
    for filename, item in contract["files"].items():
        path = hf_hub_download(MODEL_REPO, filename, revision=MODEL_REVISION, local_dir=root / "policy")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {filename}")
    snapshot_download(BACKBONE_REPO, revision=BACKBONE_REVISION, local_dir=root / "backbone",
                      allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    inventory = {str(p.relative_to(root)): sha256(p) for folder in ("policy", "backbone")
                 for p in (root / folder).iterdir() if p.is_file()}
    write_json(root / "assets.json", {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                                     "backbone_repo": BACKBONE_REPO, "backbone_revision": BACKBONE_REVISION,
                                     "files": inventory})
    print(f"Verified pinned assets: {root}")


def relocate(directory):
    root = Path(directory).resolve()
    provenance = json.loads((root / "export_manifest.json").read_text())
    for name, digest in provenance["immutable_files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Export integrity mismatch: {name}")
    config = json.loads((root / "config.json").read_text())
    config["vlm_model_name"] = str(root / "backbone")
    config["load_vlm_weights"] = False  # full merged model.safetensors already includes the VLM
    write_json(root / "config.json", config)
    processor = json.loads((root / "policy_preprocessor.json").read_text())
    for step in processor["steps"]:
        if step["registry_name"] == "tokenizer_processor":
            step["config"]["tokenizer_name"] = str(root / "backbone")
    write_json(root / "policy_preprocessor.json", processor)
    print(f"Relocated local backbone/tokenizer paths to {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["fetch", "relocate"])
    p.add_argument("directory")
    args = p.parse_args()
    (fetch_assets if args.command == "fetch" else relocate)(args.directory)
