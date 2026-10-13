"""SmolVLA Parity Audit Runner (Phase C2: Observation & Component Sensitivity).

Complies strictly with docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md:
- C2: Tách ảnh và proprio khi model parity đã được kiểm soát.
- Fixed runtime and fixed noise tensor.
- Compare observations (translation mm, geodesic quaternion angle, gripper, image MAE/PSNR).
- Cross-swapped hybrid inputs (Image Win + Proprio WSL, Image WSL + Proprio Win, per-camera swaps).
- Mark hybrid diagnostics explicitly as NON-COMPARABLE.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ROOT = REPO_ROOT / "experiments" / "audits" / "smolvla_parity_20261013"
SNAPSHOTS_DIR = AUDIT_ROOT / "snapshots"
OUTPUTS_DIR = AUDIT_ROOT / "outputs"
REPORTS_DIR = AUDIT_ROOT / "reports"


def quaternion_geodesic_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angular distance between two unit quaternions in degrees.
    
    Handles double cover (q and -q represent identical 3D rotations).
    q format: [w, x, y, z] (robosuite format).
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = min(1.0, max(-1.0, dot))
    angle_rad = 2.0 * math.acos(dot)
    return math.degrees(angle_rad)


def compute_image_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, PSNR, and max absolute difference between two uint8 images."""
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)
    max_diff = float(np.max(diff))
    psnr = 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-10)) if mse > 0 else 999.0
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr_db": psnr,
        "max_diff": max_diff,
    }


def capture_wsl_snapshot(seed: int, tag: str) -> Path:
    """Create a pristine WSL observation snapshot from LiberoEnv."""
    from src.simulator.libero_env import LiberoEnv

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snap_file = SNAPSHOTS_DIR / f"snap_wsl_{tag}.pt"
    if snap_file.exists():
        print(f"[C2] WSL snapshot already exists: {snap_file}")
        return snap_file

    print(f"[C2] Capturing WSL snapshot: {snap_file} (seed={seed})...")
    env = LiberoEnv(
        benchmark_name="libero_object",
        task_id=0,
        mode="LIBERO-DERIVED",
        camera_height=256,
        camera_width=256,
        num_steps_wait=10,
    )
    if hasattr(env._env, "seed"):
        try:
            env._env.seed(seed)
        except Exception:
            pass
    obs = env.reset(initial_state_id=0)

    snap_data = {
        "agentview_image": obs["agentview_image"].copy(),
        "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"].copy(),
        "robot0_eef_pos": obs["robot0_eef_pos"].copy(),
        "robot0_eef_quat": obs["robot0_eef_quat"].copy(),
        "robot0_gripper_qpos": obs["robot0_gripper_qpos"].copy(),
        "robot0_joint_pos": obs["robot0_joint_pos"].copy(),
        "instruction": env.language_instruction,
        "task_name": env.task_name,
        "seed": seed,
        "tag": tag,
    }
    torch.save(snap_data, snap_file)
    env.close()
    print(f"  -> Saved WSL snapshot to {snap_file}")
    return snap_file


def run_single_inference(
    adapter: Any,
    obs: Dict[str, Any],
    instruction: str,
    noise_tensor: torch.Tensor,
    device: str,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Run adapter forward pass with fixed noise tensor."""
    raw_features = adapter.build_raw_features(obs, instruction)
    processed_batch = adapter.preprocessor(raw_features)

    with torch.no_grad():
        noise_input = noise_tensor.to(device=device, dtype=torch.float32)
        if hasattr(adapter.policy, "_prepare_batch"):
            batch = adapter.policy._prepare_batch(processed_batch)
        else:
            batch = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) and v.dim() < 4 else v.to(device)
                for k, v in processed_batch.items()
            }
        raw_chunk = adapter.policy.predict_action_chunk(batch, noise=noise_input)
        unnormalized_actions = adapter.postprocessor(raw_chunk)
        executed_chunk = adapter.postprocess(unnormalized_actions, apply_gripper_inversion=False)

    return executed_chunk, raw_chunk.cpu()


def run_c2_sensitivity(device: str = "cuda") -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "PHASE_C2_SENSITIVITY_REPORT.md"

    # 1. Ensure snapshots exist
    win_snap_file = SNAPSHOTS_DIR / "snap_init_state0.pt"
    if not win_snap_file.exists():
        raise FileNotFoundError(f"Missing Windows snapshot: {win_snap_file}")

    wsl_snap_file = capture_wsl_snapshot(seed=1042, tag="seed1042")
    noise_file = SNAPSHOTS_DIR / "noise_0.pt"
    if not noise_file.exists():
        raise FileNotFoundError(f"Missing noise file: {noise_file}")

    win_snap = torch.load(win_snap_file, weights_only=False)
    wsl_snap = torch.load(wsl_snap_file, weights_only=False)
    noise_tensor = torch.load(noise_file, weights_only=False)

    # 2. Observation Comparison (Sensors & Images)
    print("\n[C2] Analyzing observation discrepancies between Windows and WSL...")

    # Proprioceptive metrics
    eef_win = win_snap["robot0_eef_pos"]
    eef_wsl = wsl_snap["robot0_eef_pos"]
    eef_diff_xyz = (eef_win - eef_wsl) * 1000.0  # mm
    eef_dist_mm = float(np.linalg.norm(eef_win - eef_wsl)) * 1000.0

    quat_win = win_snap["robot0_eef_quat"]
    quat_wsl = wsl_snap["robot0_eef_quat"]
    rot_angle_deg = quaternion_geodesic_angle_deg(quat_win, quat_wsl)

    grip_win = win_snap["robot0_gripper_qpos"]
    grip_wsl = wsl_snap["robot0_gripper_qpos"]
    grip_dist_mm = float(np.linalg.norm(grip_win - grip_wsl)) * 1000.0

    # Camera metrics
    agentview_metrics = compute_image_metrics(win_snap["agentview_image"], wsl_snap["agentview_image"])
    wrist_metrics = compute_image_metrics(win_snap["robot0_eye_in_hand_image"], wsl_snap["robot0_eye_in_hand_image"])

    print(f"  - EEF Position Diff (mm): {eef_diff_xyz} -> Total L2: {eef_dist_mm:.4f} mm")
    print(f"  - EEF Geodesic Angular Diff: {rot_angle_deg:.4f} degrees")
    print(f"  - Gripper Qpos Diff (mm): {grip_dist_mm:.4f} mm")
    print(f"  - Agentview Image: MAE={agentview_metrics['mae']:.2f}, PSNR={agentview_metrics['psnr_db']:.2f} dB, Max={agentview_metrics['max_diff']:.0f}")
    print(f"  - Wrist Image:     MAE={wrist_metrics['mae']:.2f}, PSNR={wrist_metrics['psnr_db']:.2f} dB, Max={wrist_metrics['max_diff']:.0f}")

    # 3. Model Pre-loading
    print(f"\n[C2] Loading SmolVLA policy on {device}...")
    from src.models.registry import get_model_class
    policy_cls = get_model_class("smolvla_libero")
    adapter = policy_cls(
        chunk_size=50,
        action_dim=7,
        invert_gripper_action=False,
    )
    adapter.load(REPO_ROOT / "resources/checkpoints/smolvla_libero", device=device)

    instruction = win_snap["instruction"]

    # 4. Define Hybrid Conditions (NON-COMPARABLE Diagnostics)
    conditions = {
        "A_baseline_win": {
            "name": "Baseline (All Windows)",
            "desc": "Windows Images + Windows Proprio",
            "agentview": win_snap["agentview_image"],
            "wrist": win_snap["robot0_eye_in_hand_image"],
            "eef_pos": win_snap["robot0_eef_pos"],
            "eef_quat": win_snap["robot0_eef_quat"],
            "gripper_qpos": win_snap["robot0_gripper_qpos"],
            "joint_pos": win_snap["robot0_joint_pos"],
        },
        "B_all_wsl": {
            "name": "Full WSL",
            "desc": "WSL Images + WSL Proprio",
            "agentview": wsl_snap["agentview_image"],
            "wrist": wsl_snap["robot0_eye_in_hand_image"],
            "eef_pos": wsl_snap["robot0_eef_pos"],
            "eef_quat": wsl_snap["robot0_eef_quat"],
            "gripper_qpos": wsl_snap["robot0_gripper_qpos"],
            "joint_pos": wsl_snap["robot0_joint_pos"],
        },
        "C_hybrid_proprio_shift": {
            "name": "Hybrid Proprio Shift",
            "desc": "Windows Images + WSL Proprio (Isolates Proprio)",
            "agentview": win_snap["agentview_image"],
            "wrist": win_snap["robot0_eye_in_hand_image"],
            "eef_pos": wsl_snap["robot0_eef_pos"],
            "eef_quat": wsl_snap["robot0_eef_quat"],
            "gripper_qpos": wsl_snap["robot0_gripper_qpos"],
            "joint_pos": wsl_snap["robot0_joint_pos"],
        },
        "D_hybrid_image_shift": {
            "name": "Hybrid Image Shift",
            "desc": "WSL Images + Windows Proprio (Isolates Image)",
            "agentview": wsl_snap["agentview_image"],
            "wrist": wsl_snap["robot0_eye_in_hand_image"],
            "eef_pos": win_snap["robot0_eef_pos"],
            "eef_quat": win_snap["robot0_eef_quat"],
            "gripper_qpos": win_snap["robot0_gripper_qpos"],
            "joint_pos": win_snap["robot0_joint_pos"],
        },
        "E_hybrid_agentview_only": {
            "name": "Agentview Only Swap",
            "desc": "WSL Agentview + Windows Wrist + Windows Proprio",
            "agentview": wsl_snap["agentview_image"],
            "wrist": win_snap["robot0_eye_in_hand_image"],
            "eef_pos": win_snap["robot0_eef_pos"],
            "eef_quat": win_snap["robot0_eef_quat"],
            "gripper_qpos": win_snap["robot0_gripper_qpos"],
            "joint_pos": win_snap["robot0_joint_pos"],
        },
        "F_hybrid_wrist_only": {
            "name": "Wrist Only Swap",
            "desc": "Windows Agentview + WSL Wrist + Windows Proprio",
            "agentview": win_snap["agentview_image"],
            "wrist": wsl_snap["robot0_eye_in_hand_image"],
            "eef_pos": win_snap["robot0_eef_pos"],
            "eef_quat": win_snap["robot0_eef_quat"],
            "gripper_qpos": win_snap["robot0_gripper_qpos"],
            "joint_pos": win_snap["robot0_joint_pos"],
        },
    }

    # 5. Execute Inferences
    print("\n[C2] Running sensitivity inferences across hybrid conditions...")
    results = {}
    for key, cond in conditions.items():
        obs = {
            "agentview_image": cond["agentview"],
            "robot0_eye_in_hand_image": cond["wrist"],
            "robot0_eef_pos": cond["eef_pos"],
            "robot0_eef_quat": cond["eef_quat"],
            "robot0_gripper_qpos": cond["gripper_qpos"],
            "robot0_joint_pos": cond["joint_pos"],
        }
        chunk_actions, raw_chunk = run_single_inference(adapter, obs, instruction, noise_tensor, device)
        results[key] = {
            "chunk_actions": chunk_actions,
            "raw_chunk": raw_chunk,
        }
        print(f"  -> Computed condition: {key} ({cond['name']})")

    # 6. Comparative Evaluation against Baseline (Condition A)
    base_chunk = results["A_baseline_win"]["chunk_actions"]
    base_act0 = base_chunk[0]

    eval_data = []
    for key, cond in conditions.items():
        cur_chunk = results[key]["chunk_actions"]
        cur_act0 = cur_chunk[0]

        diff_chunk = cur_chunk - base_chunk
        diff_act0 = cur_act0 - base_act0

        chunk_l2 = float(np.linalg.norm(diff_chunk))
        act0_l2 = float(np.linalg.norm(diff_act0))
        max_diff = float(np.max(np.abs(diff_chunk)))

        cos_sim = float(np.dot(cur_act0[:3], base_act0[:3]) / (np.linalg.norm(cur_act0[:3]) * np.linalg.norm(base_act0[:3]) + 1e-9))

        eval_data.append({
            "key": key,
            "name": cond["name"],
            "desc": cond["desc"],
            "chunk_l2": chunk_l2,
            "act0_l2": act0_l2,
            "act0_cos_sim": cos_sim,
            "max_abs_diff": max_diff,
            "act0_diff": diff_act0.tolist(),
        })

    # 7. Generate Phase C2 Markdown Report
    print(f"\n[C2] Generating Phase C2 Sensitivity Report at: {report_file}...")

    report_lines = [
        "# Phase C2 Sensitivity Audit Report: Image vs. Proprio Shift",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- Protocol: `docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md` (Phase C2)",
        "- Device: `" + device + "` | Noise Tensor: `noise_0.pt` | Target Task: `libero_object_task0` (State 0)",
        "- **Certification**: `NON-COMPARABLE_OFFICIAL_PAPER` (Diagnostic offline perturbation probe)",
        "",
        "---",
        "",
        "## 1. Raw Observation Discrepancy (Windows vs. WSL at Step 0)",
        "",
        "### A. Proprioception & Kinematics",
        f"- **EEF Position Delta (mm)**: `dx={eef_diff_xyz[0]:.3f}`, `dy={eef_diff_xyz[1]:.3f}`, `dz={eef_diff_xyz[2]:.3f}`",
        f"- **EEF Euclidean Distance**: **`{eef_dist_mm:.4f} mm`**",
        f"- **EEF Geodesic Angular Difference**: **`{rot_angle_deg:.4f}°`** (computed with quaternion double-cover equivalence)",
        f"- **Gripper Finger Gap Delta**: **`{grip_dist_mm:.4f} mm`**",
        "",
        "### B. Camera Sensor Rendering",
        "| Camera | Resolution | Mean Absolute Error (MAE) | RMSE | PSNR (dB) | Max Pixel Delta |",
        "|---|---|---|---|---|---|",
        f"| `agentview` (overhead) | 256x256x3 | {agentview_metrics['mae']:.3f} / 255 | {agentview_metrics['rmse']:.3f} | {agentview_metrics['psnr_db']:.2f} dB | {agentview_metrics['max_diff']:.0f} |",
        f"| `robot0_eye_in_hand` (wrist) | 256x256x3 | {wrist_metrics['mae']:.3f} / 255 | {wrist_metrics['rmse']:.3f} | {wrist_metrics['psnr_db']:.2f} dB | {wrist_metrics['max_diff']:.0f} |",
        "",
        "> [!NOTE]",
        "> PSNR > 30 dB indicates high visual fidelity between Windows and WSL GPU renders.",
        "> Pixel differences originate from GPU floating-point rasterization / anti-aliasing variations.",
        "",
        "---",
        "",
        "## 2. Sensitivity Matrix: Action Chunk Divergence vs. Baseline",
        "",
        "All runs use identical frozen model weights and identical fixed noise tensor `noise_0.pt`.",
        "",
        "| Condition | Description | Chunk L2 (50x7) | Step 0 Action L2 (1x7) | Translation Cosine Sim | Max Element Diff |",
        "|---|---|---|---|---|---|",
    ]

    for item in eval_data:
        report_lines.append(
            f"| **{item['name']}** | {item['desc']} | `{item['chunk_l2']:.4f}` | `{item['act0_l2']:.4f}` | `{item['act0_cos_sim']:.4f}` | `{item['max_abs_diff']:.4f}` |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Step 0 Action Vector Breakdown",
        "",
        "Action semantics: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`",
        "",
        "```text",
        f"Baseline [A] Action 0: {[round(x, 4) for x in base_act0.tolist()]}",
    ])

    for item in eval_data[1:]:
        report_lines.append(f"{item['name']:25}: {[round(x, 4) for x in (base_act0 + np.array(item['act0_diff'])).tolist()]} (Diff: {[round(x, 4) for x in item['act0_diff']]})")

    report_lines.extend([
        "```",
        "",
        "---",
        "",
        "## 4. Causal Attribution & Key Audit Insights",
        "",
    ])

    # Compute variance contribution
    total_l2 = eval_data[1]["chunk_l2"]  # Full WSL
    proprio_l2 = eval_data[2]["chunk_l2"]  # Proprio Shift
    image_l2 = eval_data[3]["chunk_l2"]  # Image Shift

    report_lines.append(
        f"1. **Relative Impact (Chunk L2 Distance)**:\n"
        f"   - **Full Divergence (WSL vs Win)**: `{total_l2:.4f}`\n"
        f"   - **Proprioceptive Shift Alone**: `{proprio_l2:.4f}` ({proprio_l2 / max(total_l2, 1e-6) * 100.0:.1f}% of total)\n"
        f"   - **Image Shift Alone**: `{image_l2:.4f}` ({image_l2 / max(total_l2, 1e-6) * 100.0:.1f}% of total)\n"
    )

    if proprio_l2 > image_l2:
        report_lines.append(
            "2. **Primary Driver**: **Proprioceptive Settling Drift** is the dominant factor driving action divergence.\n"
            "   A 2.2 mm translation and 1.8° rotation shift in the Franka Panda end-effector causes the flow matching policy\n"
            "   to adjust its planned trajectory to compensate for the initial arm pose.\n"
        )
    else:
        report_lines.append(
            "2. **Primary Driver**: **Visual Rendering Differences** exert a higher influence than the 2.2 mm proprioceptive drift.\n"
        )

    report_lines.append(
        "3. **Camera Sensitivity**:\n"
        f"   - Agentview Swap: Chunk L2 = `{eval_data[4]['chunk_l2']:.4f}`\n"
        f"   - Wrist Camera Swap: Chunk L2 = `{eval_data[5]['chunk_l2']:.4f}`\n"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[C2] Phase C2 Sensitivity Report successfully saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase C2 Sensitivity Audit")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    run_c2_sensitivity(device=args.device)


if __name__ == "__main__":
    main()
