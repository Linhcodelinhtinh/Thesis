#!/usr/bin/env python
"""Strict-load a merged policy through the evaluation adapter on a real sample.

This is an interface/inference check, not a simulated episode or benchmark score.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    import numpy as np
    import yaml
    from src.training.memory_data import sha256
    from src.training.memory_dataset import MemoryDataset
    from src.models.smolvla.adapter import SmolVLAAdapter
    root = Path(args.checkpoint)
    manifest = json.loads((root / "export_manifest.json").read_text())
    for name, digest in manifest["immutable_files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Corrupt export: {name}")
    base = yaml.safe_load((ROOT / "resources/manifests/models/smolvla_libero.yaml").read_text())
    for name, info in base["files"].items():
        if "normalizer_processor.safetensors" in name and sha256(root / name) != info["sha256"]:
            raise ValueError("Normalization differs from the original baseline")
    data = MemoryDataset(args.data_dir, "val")
    sample = data[0]
    robot = sample["robot"]
    observation = {"agentview_image": sample["agentview"], "robot0_eye_in_hand_image": sample["wrist"],
                   "robot0_gripper_qpos": robot[:2], "robot0_eef_pos": robot[2:5], "robot0_eef_quat": robot[5:9]}
    policy = SmolVLAAdapter()
    policy.load(str(root.resolve()), device=args.device, strict=True)
    chunk = policy.predict_action_chunk(observation, sample["task"])
    if chunk.shape != (50, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"Invalid exported action chunk: {chunk.shape}")
    print(json.dumps({"strict_reload": "passed", "original_normalization": "passed",
                      "action_chunk_shape": list(chunk.shape), "finite": True,
                      "benchmark_success": "NOT_MEASURED"}))


if __name__ == "__main__":
    main()
