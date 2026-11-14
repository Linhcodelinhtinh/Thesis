#!/usr/bin/env python
"""Strict-load a policy export through the evaluation adapter on a real sample.

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
    p.add_argument("--base-checkpoint", default=str(ROOT / "resources/checkpoints/smolvla_libero"))
    args = p.parse_args()
    import numpy as np
    import yaml
    from src.training.memory_data import sha256
    from src.training.memory_dataset import MemoryDataset
    from src.models.smolvla.adapter import SmolVLAAdapter
    from src.training.mem_adapter_artifact import adapter_training_condition, validate_adapter, verify_files
    root = Path(args.checkpoint)
    base = yaml.safe_load((ROOT / "resources/manifests/models/smolvla_libero.yaml").read_text())
    treatment = None
    kwargs = {}
    if (root / "mem_adapter_manifest.json").is_file():
        verify_files(args.base_checkpoint, {name: info["sha256"] for name, info in base["files"].items()})
        manifest = validate_adapter(root, args.base_checkpoint)
        from src.training.mixed_sft import validate_mixed_dataset
        validate_mixed_dataset(args.data_dir)
        if sha256(Path(args.data_dir) / "manifest.json") != manifest["run"]["data_manifest_sha256"]:
            raise ValueError("Dataset manifest differs from the adapter training input")
        treatment = "F0" if adapter_training_condition(manifest) == "no_memory" else "F1-off"
        kwargs["base_checkpoint"] = str(Path(args.base_checkpoint).resolve())
    else:
        manifest = json.loads((root / "export_manifest.json").read_text())
        verify_files(root, manifest["immutable_files"])
        for name, info in base["files"].items():
            if "normalizer_processor.safetensors" in name and sha256(root / name) != info["sha256"]:
                raise ValueError("Normalization differs from the original baseline")
    data = MemoryDataset(args.data_dir, "val", condition="no_memory" if treatment else "memory")
    try:
        sample = data[0]
    finally:
        data.close()
    robot = sample["robot"]
    observation = {"agentview_image": sample["agentview"], "robot0_eye_in_hand_image": sample["wrist"],
                   "robot0_gripper_qpos": robot[:2], "robot0_eef_pos": robot[2:5], "robot0_eef_quat": robot[5:9]}
    policy = SmolVLAAdapter()
    policy.load(str(root.resolve()), device=args.device, strict=True, **kwargs)
    policy.reset()
    if treatment:
        from src.interfaces.libero_mem_text import LiberoMemTextRenderer
        renderer = LiberoMemTextRenderer(tokenizer=policy.tokenizer, strict_tokenizer=True)
        if renderer.count_tokens(sample["task"]) > renderer.max_total_tokens:
            raise ValueError("Instruction exceeds the locked token budget")
        policy.bind_instruction_only_runtime(treatment, sample["task"], renderer)
    chunk = policy.predict_action_chunk(observation, sample["task"])
    if chunk.shape != (policy.chunk_size, policy.action_dim) or not np.isfinite(chunk).all():
        raise ValueError(f"Invalid exported action chunk: {chunk.shape}")
    print(json.dumps({"strict_reload": "passed", "original_normalization": "passed",
                      "action_chunk_shape": list(chunk.shape), "finite": True,
                      "sample_split": "val", "sample_domain": sample["domain"],
                      "inference_treatment": treatment or "merged_export",
                      "benchmark_success": "NOT_MEASURED"}))


if __name__ == "__main__":
    main()
