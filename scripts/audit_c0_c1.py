"""SmolVLA Parity Audit Runner (Phase C0 & C1).

Complies strictly with docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md:
- C0: Data & Runtime Freeze Manifest
- C1: Offline Layer-by-Layer Parity Probe (Frozen Observations + Fixed Noise Tensors)
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ROOT = REPO_ROOT / "experiments" / "audits" / "smolvla_parity_20261013"

MANIFEST_DIR = AUDIT_ROOT / "manifest"
SNAPSHOTS_DIR = AUDIT_ROOT / "snapshots"
OUTPUTS_DIR = AUDIT_ROOT / "outputs"
REPORTS_DIR = AUDIT_ROOT / "reports"


# =====================================================================
# Phase C0: Provenance & Manifest Generation
# =====================================================================

def compute_file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_c0_manifest(platform_tag: str, device: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"manifest_{platform_tag}.json"

    # 1. Git provenance
    git_hash = None
    git_dirty = None
    git_diff_hash = None
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip()
        git_dirty = len(status_out) > 0
        diff_out = subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO_ROOT)
        git_diff_hash = hashlib.sha256(diff_out).hexdigest()
    except Exception:
        pass

    # 2. Package inventory
    pip_freeze = {}
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            pip_freeze[dist.metadata["Name"]] = dist.version
    except Exception:
        pass

    # 3. Model checkpoint files
    ckpt_dir = REPO_ROOT / "resources" / "checkpoints" / "smolvla_libero"
    ckpt_hashes = {}
    if ckpt_dir.exists():
        for p in ckpt_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                rel = str(p.relative_to(ckpt_dir))
                ckpt_hashes[rel] = compute_file_sha256(p)

    # 4. GL context / rendering info
    gl_renderer = os.environ.get("MUJOCO_GL", "default")
    gl_info = {
        "MUJOCO_GL": gl_renderer,
        "GALLIUM_DRIVER": os.environ.get("GALLIUM_DRIVER"),
        "MESA_D3D12_DEFAULT_ADAPTER_NAME": os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME"),
    }

    manifest = {
        "run_id": "smolvla_parity_20261013",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform_tag": platform_tag,
        "device": device,
        "os": {
            "platform": sys.platform,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "sys_path": sys.path[:5],
        },
        "git": {
            "commit": git_hash,
            "is_dirty": git_dirty,
            "diff_sha256": git_diff_hash,
        },
        "gl_environment": gl_info,
        "key_packages": {
            "torch": pip_freeze.get("torch"),
            "torchvision": pip_freeze.get("torchvision"),
            "transformers": pip_freeze.get("transformers"),
            "diffusers": pip_freeze.get("diffusers"),
            "lerobot": pip_freeze.get("lerobot"),
            "robosuite": pip_freeze.get("robosuite"),
            "bddl": pip_freeze.get("bddl"),
            "mujoco": pip_freeze.get("mujoco"),
            "gymnasium": pip_freeze.get("gymnasium"),
            "gym": pip_freeze.get("gym"),
        },
        "checkpoint_hashes": ckpt_hashes,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[C0] Saved manifest for {platform_tag} to: {manifest_path}")
    return manifest_path


# =====================================================================
# Phase C1: Observation & Noise Snapshot Creation
# =====================================================================

def create_snapshots_and_noise() -> None:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    from src.simulator.libero_env import LiberoEnv

    print("\n[C1] Generating fixed observation snapshots and noise tensors...")

    # 1. Fixed Noise Tensors (shape: (1, 50, 32), dtype: float32)
    # Using fixed deterministic generator
    for seed_val, name in [(10420, "noise_0"), (20420, "noise_1")]:
        noise_file = SNAPSHOTS_DIR / f"{name}.pt"
        if not noise_file.exists():
            g = torch.Generator(device="cpu").manual_seed(seed_val)
            noise_tensor = torch.randn((1, 50, 32), generator=g, dtype=torch.float32)
            torch.save(noise_tensor, noise_file)
            print(f"  - Saved fixed noise: {noise_file} (shape={noise_tensor.shape}, mean={noise_tensor.mean():.4f}, std={noise_tensor.std():.4f})")
        else:
            print(f"  - Fixed noise already exists: {noise_file}")

    # 2. Observation Snapshot: snap_init_state0
    snap_init_file = SNAPSHOTS_DIR / "snap_init_state0.pt"
    if not snap_init_file.exists():
        print("  - Creating snap_init_state0 from LiberoEnv(libero_object, task_id=0)...")
        env = LiberoEnv(
            benchmark_name="libero_object",
            task_id=0,
            mode="LIBERO-DERIVED",
            camera_height=256,
            camera_width=256,
            num_steps_wait=10,
        )
        obs = env.reset(initial_state_id=0)

        # Snapshot dict with all numpy arrays preserved
        snap_data = {
            "agentview_image": obs["agentview_image"].copy(),
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"].copy(),
            "robot0_eef_pos": obs["robot0_eef_pos"].copy(),
            "robot0_eef_quat": obs["robot0_eef_quat"].copy(),
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"].copy(),
            "robot0_joint_pos": obs["robot0_joint_pos"].copy(),
            "instruction": env.language_instruction,
            "task_name": env.task_name,
        }
        torch.save(snap_data, snap_init_file)
        print(f"  - Saved snapshot to: {snap_init_file}")
        env.close()
    else:
        print(f"  - Snapshot already exists: {snap_init_file}")

    # 3. Observation Snapshot: snap_near_grasp (Step 38 from successful trajectory)
    snap_grasp_file = SNAPSHOTS_DIR / "snap_near_grasp.pt"
    if not snap_grasp_file.exists():
        print("  - Creating snap_near_grasp by stepping simulator with 38 demo actions...")
        traj_path = REPO_ROOT / "experiments/results/baseline_full_40/libero_object_task0/init_0/trajectory.npz"
        actions = np.load(traj_path)["actions"]
        env = LiberoEnv(
            benchmark_name="libero_object",
            task_id=0,
            mode="LIBERO-DERIVED",
            camera_height=256,
            camera_width=256,
            num_steps_wait=10,
        )
        obs = env.reset(initial_state_id=0)
        for i in range(38):
            obs, _, _, _ = env.step(actions[i])

        snap_grasp = {
            "agentview_image": obs["agentview_image"].copy(),
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"].copy(),
            "robot0_eef_pos": obs["robot0_eef_pos"].copy(),
            "robot0_eef_quat": obs["robot0_eef_quat"].copy(),
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"].copy(),
            "robot0_joint_pos": obs["robot0_joint_pos"].copy(),
            "instruction": env.language_instruction,
            "task_name": env.task_name,
            "step": 38,
        }
        torch.save(snap_grasp, snap_grasp_file)
        print(f"  - Saved near-grasp snapshot to: {snap_grasp_file}")
        env.close()
    else:
        print(f"  - Near-grasp snapshot already exists: {snap_grasp_file}")


# =====================================================================
# Phase C1: Layer-by-Layer Model Inference Probe
# =====================================================================

def run_layer_probe(
    platform_tag: str,
    device: str,
    snapshot_key: str,
    noise_key: str,
    adapter: Optional[Any] = None,
) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / f"{platform_tag}_{snapshot_key}_{noise_key}.pt"

    from src.models.registry import get_model_class

    print(f"\n[C1 Probe] Running: {platform_tag} | {snapshot_key} | {noise_key} on {device}...")

    # 1. Load Snapshot & Noise
    snap_data = torch.load(SNAPSHOTS_DIR / f"{snapshot_key}.pt", weights_only=False)
    noise_tensor = torch.load(SNAPSHOTS_DIR / f"{noise_key}.pt", weights_only=False)

    instruction = snap_data["instruction"]
    obs = {
        "agentview_image": snap_data["agentview_image"],
        "robot0_eye_in_hand_image": snap_data["robot0_eye_in_hand_image"],
        "robot0_eef_pos": snap_data["robot0_eef_pos"],
        "robot0_eef_quat": snap_data["robot0_eef_quat"],
        "robot0_gripper_qpos": snap_data["robot0_gripper_qpos"],
        "robot0_joint_pos": snap_data["robot0_joint_pos"],
    }

    # 2. Instantiate and Load Policy if not provided
    if adapter is None:
        policy_cls = get_model_class("smolvla_libero")
        adapter = policy_cls(
            chunk_size=50,
            action_dim=7,
            invert_gripper_action=False,
        )
        adapter.load(REPO_ROOT / "resources/checkpoints/smolvla_libero", device=device)

    # 3. Stage 1: Build Raw Features (Env Preprocessing)
    t0 = time.time()
    raw_features = adapter.build_raw_features(obs, instruction)
    stage1_time = time.time() - t0

    # 4. Stage 2: Policy Preprocessor (Normalization & Tokenization)
    t0 = time.time()
    processed_batch = adapter.preprocessor(raw_features)
    stage2_time = time.time() - t0

    # 5. Stage 3: Forward Pass with Fixed Noise
    t0 = time.time()
    with torch.no_grad():
        # Move noise to policy device
        noise_input = noise_tensor.to(device=device, dtype=torch.float32)

        # Batch preparation
        if hasattr(adapter.policy, "_prepare_batch"):
            batch = adapter.policy._prepare_batch(processed_batch)
        else:
            batch = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) and v.dim() < 4 else v.to(device)
                for k, v in processed_batch.items()
            }

        # Predict chunk with explicit noise argument
        raw_chunk = adapter.policy.predict_action_chunk(batch, noise=noise_input)
    stage3_time = time.time() - t0

    # 6. Stage 4: Postprocessor Unnormalization
    t0 = time.time()
    unnormalized_actions = adapter.postprocessor(raw_chunk)
    executed_chunk = adapter.postprocess(unnormalized_actions, apply_gripper_inversion=False)
    stage4_time = time.time() - t0

    # 7. Collect and save tensors on CPU
    probe_output = {
        "platform_tag": platform_tag,
        "device": device,
        "snapshot_key": snapshot_key,
        "noise_key": noise_key,
        "timing_ms": {
            "stage1_raw_features": stage1_time * 1000,
            "stage2_preprocessor": stage2_time * 1000,
            "stage3_model_forward": stage3_time * 1000,
            "stage4_postprocessor": stage4_time * 1000,
        },
        # Stage 1 representations
        "raw_image": raw_features.get("observation.images.image", raw_features.get("observation.images.camera1", torch.tensor([]))).cpu(),
        "raw_image2": raw_features.get("observation.images.image2", raw_features.get("observation.images.camera2", torch.tensor([]))).cpu(),
        "raw_state": raw_features["observation.state"].cpu(),
        # Stage 2 representations
        "preprocessed_image": processed_batch.get("observation.images.camera1", processed_batch.get("observation.images.image", torch.tensor([]))).cpu(),
        "preprocessed_image2": processed_batch.get("observation.images.camera2", processed_batch.get("observation.images.image2", torch.tensor([]))).cpu(),
        "preprocessed_state": processed_batch["observation.state"].cpu(),
        "lang_tokens": processed_batch.get("observation.language.tokens", torch.tensor([])).cpu(),
        # Stage 3 raw normalized model output
        "raw_chunk": raw_chunk.cpu(),
        # Stage 4 physical action chunk (50, 7)
        "executed_chunk": torch.from_numpy(executed_chunk).float(),
    }


    torch.save(probe_output, out_file)
    print(f"  -> Saved probe output to: {out_file}")
    return out_file


# =====================================================================
# Phase C1: Comparison & Differential Evaluation
# =====================================================================

def evaluate_parity_report() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "OFFLINE_PARITY_AUDIT_REPORT.md"

    print("\n[C1 Evaluation] Computing layer-by-layer parity comparisons...")

    # Find matching output files
    pairs_to_compare = [
        ("win_cpu", "wsl_cpu"),
        ("wsl_cpu", "wsl_cuda"),
        ("win_cpu", "wsl_cuda"),
    ]

    report_lines = [
        "# Offline Parity Audit Report (Phase C1)",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- Target Protocol: `docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md`",
        "",
        "## 1. Summary of Comparisons",
        "",
        "| Comparison Pair | Snapshot | Noise | Stage 1 (Raw State) L2 | Stage 2 (Prep State) L2 | Stage 3 (Raw Model Chunk) L2 | Stage 4 (Action Chunk) L2 | Max Action Diff |",
        "|---|---|---|---|---|---|---|---|",
    ]

    detailed_diffs = []

    for pA, pB in pairs_to_compare:
        for snap in ["snap_init_state0", "snap_near_grasp"]:
            for noise in ["noise_0", "noise_1"]:
                fA = OUTPUTS_DIR / f"{pA}_{snap}_{noise}.pt"
                fB = OUTPUTS_DIR / f"{pB}_{snap}_{noise}.pt"
                if not fA.exists() or not fB.exists():
                    continue

                dA = torch.load(fA, weights_only=False)
                dB = torch.load(fB, weights_only=False)

                # 1. Stage 1 State diff
                s1_l2 = float(torch.norm(dA["raw_state"] - dB["raw_state"]))

                # 2. Stage 2 Prep State diff
                s2_l2 = float(torch.norm(dA["preprocessed_state"] - dB["preprocessed_state"]))

                # 3. Stage 3 Raw Model Chunk diff
                rawA = dA["raw_chunk"]
                rawB = dB["raw_chunk"]
                if rawA.shape != rawB.shape:
                    rawA = rawA.squeeze()
                    rawB = rawB.squeeze()
                s3_l2 = float(torch.norm(rawA - rawB))

                # 4. Stage 4 Executed Chunk diff (50, 7)
                actA = dA["executed_chunk"]
                actB = dB["executed_chunk"]
                act_diff = torch.abs(actA - actB)
                s4_l2 = float(torch.norm(actA - actB))
                s4_max = float(torch.max(act_diff))
                s4_action0_l2 = float(torch.norm(actA[0] - actB[0]))

                report_lines.append(
                    f"| `{pA}` vs `{pB}` | `{snap}` | `{noise}` | {s1_l2:.6e} | {s2_l2:.6e} | {s3_l2:.6e} | {s4_l2:.6e} | {s4_max:.6e} |"
                )

                detailed_diffs.append({
                    "pair": f"{pA} vs {pB}",
                    "snapshot": snap,
                    "noise": noise,
                    "stage1_raw_state_l2": s1_l2,
                    "stage2_prep_state_l2": s2_l2,
                    "stage3_raw_chunk_l2": s3_l2,
                    "stage4_executed_chunk_l2": s4_l2,
                    "stage4_action0_l2": s4_action0_l2,
                    "stage4_max_abs": s4_max,
                    "per_dim_max_diff": torch.max(act_diff, dim=0)[0].tolist(),
                })

    report_lines.extend([
        "",
        "## 2. Detailed Dimension-wise Breakdown (Stage 4 Actions)",
        "",
        "Action semantics: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`",
        "",
    ])

    for d in detailed_diffs:
        report_lines.append(f"### `{d['pair']}` — `{d['snapshot']}` — `{d['noise']}`")
        report_lines.append(f"- **Full Chunk (50x7) L2**: `{d['stage4_executed_chunk_l2']:.6f}`")
        report_lines.append(f"- **Step 0 Action (1x7) L2**: `{d['stage4_action0_l2']:.6f}`")
        report_lines.append(f"- **Max Absolute Diff**: `{d['stage4_max_abs']:.6f}`")
        report_lines.append(f"- **Max Diff per dimension**: `{[round(x, 6) for x in d['per_dim_max_diff']]}`")
        report_lines.append("")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[C1] Offline Parity Report successfully generated at: {report_file}")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SmolVLA Parity Audit Runner")
    parser.add_argument("--mode", choices=["manifest", "snapshots", "probe", "evaluate", "all"], default="all")
    parser.add_argument("--platform-tag", type=str, default="local", help="e.g. win_cpu, wsl_cpu, wsl_cuda")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if args.mode in ["manifest", "all"]:
        generate_c0_manifest(args.platform_tag, args.device)

    if args.mode in ["snapshots", "all"]:
        create_snapshots_and_noise()

    if args.mode in ["probe", "all"]:
        from src.models.registry import get_model_class
        print(f"\n[C1] Pre-loading SmolVLA adapter for device={args.device}...")
        policy_cls = get_model_class("smolvla_libero")
        adapter = policy_cls(
            chunk_size=50,
            action_dim=7,
            invert_gripper_action=False,
        )
        adapter.load(REPO_ROOT / "resources/checkpoints/smolvla_libero", device=args.device)

        for snap in ["snap_init_state0", "snap_near_grasp"]:
            for noise in ["noise_0", "noise_1"]:
                run_layer_probe(args.platform_tag, args.device, snap, noise, adapter=adapter)

    if args.mode in ["evaluate", "all"]:
        evaluate_parity_report()


if __name__ == "__main__":
    main()
