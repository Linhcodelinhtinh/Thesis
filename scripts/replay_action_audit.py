"""Replay a saved policy action log and export per-step LIBERO physics telemetry.

This tool does not run the policy. It replays the already logged simulator actions
from an exact LIBERO initial-state index and records EEF/controller/object/gripper
state so a failure can be localized across the action-to-physics boundary.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulator.libero_env import BenchmarkMode, LiberoEnv


def _json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _pose(state: Any) -> Dict[str, np.ndarray]:
    pose = state.get_geom_state()
    return {
        "pos": np.asarray(pose["pos"], dtype=np.float64).copy(),
        "quat": np.asarray(pose["quat"], dtype=np.float64).copy(),
    }


def _find_state(states: Dict[str, Any], requested: str) -> str:
    if requested in states:
        return requested
    matches = [name for name in states if requested.lower() in name.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one object/site state matching {requested!r}; found {matches}. "
            "Pass the exact BDDL object or region name."
        )
    return matches[0]


def _world_goal(controller: Any) -> np.ndarray:
    goal = np.asarray(controller.goal_pos, dtype=np.float64).copy()
    if controller.input_ref_frame == "world":
        return goal
    if controller.input_ref_frame == "base":
        return np.asarray(controller.origin_pos) + np.asarray(controller.origin_ori) @ goal
    raise ValueError(f"Unsupported OSC input reference frame: {controller.input_ref_frame}")


def _contacts_for_target(domain: Any, target_name: str) -> List[Dict[str, Any]]:
    sim = domain.sim
    result = []
    for i in range(int(sim.data.ncon)):
        contact = sim.data.contact[i]
        geom1 = sim.model.geom_id2name(contact.geom1) or ""
        geom2 = sim.model.geom_id2name(contact.geom2) or ""
        if target_name.lower() in geom1.lower() or target_name.lower() in geom2.lower():
            result.append({"geom1": geom1, "geom2": geom2, "distance": float(contact.dist)})
    return result


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Action log is empty: {path}")
    for index, row in enumerate(rows):
        if "action" not in row or len(row["action"]) != 7:
            raise ValueError(f"Missing 7D executed action at log row {index}")
    return rows


def audit_saved_run(
    source_dir: Path,
    output_dir: Path,
    task_suite: str,
    task_id: int,
    initial_state_id: int,
    camera_resolution: int,
    target_object: str,
    target_region: Optional[str],
) -> Dict[str, Any]:
    source_log = source_dir / "model_output.jsonl"
    episode_path = source_dir / "episode.json"
    if not source_log.exists() or not episode_path.exists():
        raise FileNotFoundError(f"Expected model_output.jsonl and episode.json in {source_dir}")

    rows = _load_rows(source_log)
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    env = LiberoEnv(
        benchmark_name=task_suite,
        task_id=task_id,
        mode=BenchmarkMode.LIBERO_DERIVED,
        horizon=max(1000, len(rows)),
        camera_height=camera_resolution,
        camera_width=camera_resolution,
    )

    try:
        obs = env.reset(initial_state_id=initial_state_id)
        domain = env._env.env
        states = domain.object_states_dict
        object_key = _find_state(states, target_object)
        object_state = states[object_key]
        region_key = _find_state(states, target_region) if target_region else None
        region_state = states[region_key] if region_key else None

        robot = domain.robots[0]
        arm_name = next((name for name in robot.part_controllers if name != "right_gripper"), None)
        if arm_name is None:
            raise RuntimeError(f"No arm controller found: {list(robot.part_controllers)}")
        arm_controller = robot.part_controllers[arm_name]
        gripper_controller = robot.part_controllers.get(f"{arm_name}_gripper")

        initial_object = _pose(object_state)
        initial_eef = {
            "pos": np.asarray(obs.get("robot0_eef_pos"), dtype=np.float64).copy(),
            "quat": np.asarray(obs.get("robot0_eef_quat"), dtype=np.float64).copy(),
        }
        initial_region = _pose(region_state) if region_state else None
        region_size = None
        if region_key:
            site_id = domain.sim.model.site_name2id(region_key)
            region_size = np.asarray(domain.sim.model.site_size[site_id], dtype=np.float64).copy()
        initial_gripper = np.asarray(obs.get("robot0_gripper_qpos"), dtype=np.float64).copy()

        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / "trajectory_audit.csv"
        fieldnames = [
            "step", "chunk_step", "is_replanned", "inference_time_ms",
            "raw_normalized_action", "unnormalized_action", "executed_action",
            "controller_clipped_action", "controller_scaled_arm_delta",
            "eef_pos_pre_world", "eef_quat_pre_xyzw", "eef_goal_world", "eef_goal_base",
            "eef_goal_ori_world_matrix",
            "eef_pos_post_world", "eef_quat_post_xyzw", "eef_delta_actual",
            "target_pos_pre_world", "target_quat_pre_wxyz", "target_pos_post_world",
            "target_quat_post_wxyz", "target_velocity_post_world", "eef_to_target_pre_distance",
            "eef_goal_to_target_pre_distance", "container_center_world", "target_in_container_frame",
            "container_half_extents", "container_margin_xyz", "inside_container",
            "gripper_qpos_pre", "gripper_qpos_post", "gripper_aperture_pre_mean_abs",
            "gripper_aperture_post_mean_abs", "gripper_action", "target_contact_pairs",
            "target_finger_contact_pairs",
            "official_success", "replay_eef_error_vs_saved_log",
        ]
        initial_object_z = float(initial_object["pos"][2])
        max_replay_error = 0.0
        with audit_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for index, saved in enumerate(rows):
                action = np.asarray(saved["action"], dtype=np.float32)
                eef_pre = {
                    "pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
                    "quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
                }
                object_pre = _pose(object_state)
                gripper_pre = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy()
                saved_eef = saved.get("eef_pos_actual")
                replay_error = (
                    float(np.linalg.norm(eef_pre["pos"] - np.asarray(saved_eef, dtype=np.float64)))
                    if saved_eef is not None else float("nan")
                )
                max_replay_error = max(max_replay_error, replay_error) if np.isfinite(replay_error) else max_replay_error

                obs_post, _, _, _ = env.step(action)
                eef_post = {
                    "pos": np.asarray(obs_post["robot0_eef_pos"], dtype=np.float64).copy(),
                    "quat": np.asarray(obs_post["robot0_eef_quat"], dtype=np.float64).copy(),
                }
                obs = obs_post
                object_post = _pose(object_state)
                gripper_post = np.asarray(obs_post["robot0_gripper_qpos"], dtype=np.float64).copy()
                goal_base = np.asarray(arm_controller.goal_pos, dtype=np.float64).copy()
                goal_world = _world_goal(arm_controller)
                goal_ori_world = (
                    np.asarray(arm_controller.origin_ori) @ np.asarray(arm_controller.goal_ori)
                    if arm_controller.input_ref_frame == "base"
                    else np.asarray(arm_controller.goal_ori)
                )
                controller_clipped_action = np.clip(action, -1.0, 1.0)
                controller_scaled_arm_delta = arm_controller.scale_action(action[:6])
                object_body_id = domain.obj_body_id.get(object_key)
                object_body_name = (
                    domain.sim.model.body_id2name(object_body_id)
                    if object_body_id is not None else None
                )
                object_velocity = (
                    np.asarray(domain.sim.data.get_body_xvelp(object_body_name), dtype=np.float64).copy()
                    if object_body_name is not None else np.full(3, np.nan)
                )
                contacts = _contacts_for_target(domain, target_object)
                finger_contacts = [
                    pair for pair in contacts
                    if any(token in (pair[side] or "").lower()
                           for side in ("geom1", "geom2")
                           for token in ("gripper", "finger", "pad"))
                ]
                in_region = None
                local_object = None
                margins = None
                center = None
                if region_state is not None and initial_region is not None:
                    center = _pose(region_state)["pos"]
                    site_id = domain.sim.model.site_name2id(region_key)
                    site_rot = np.asarray(domain.sim.data.get_site_xmat(region_key), dtype=np.float64).reshape(3, 3)
                    local_object = site_rot.T @ (object_post["pos"] - center)
                    margins = region_size - np.abs(local_object)
                    in_region = bool(region_state.check_contain(object_state))

                writer.writerow({
                    "step": index,
                    "chunk_step": saved.get("chunk_step"),
                    "is_replanned": saved.get("is_replanned"),
                    "inference_time_ms": saved.get("inference_time_ms"),
                    "raw_normalized_action": _json(saved.get("raw_normalized_action")),
                    "unnormalized_action": _json(saved.get("unnormalized_action")),
                    "executed_action": _json(action),
                    "controller_clipped_action": _json(controller_clipped_action),
                    "controller_scaled_arm_delta": _json(controller_scaled_arm_delta),
                    "eef_pos_pre_world": _json(eef_pre["pos"]),
                    "eef_quat_pre_xyzw": _json(eef_pre["quat"]),
                    "eef_goal_world": _json(goal_world),
                    "eef_goal_base": _json(goal_base),
                    "eef_goal_ori_world_matrix": _json(goal_ori_world),
                    "eef_pos_post_world": _json(eef_post["pos"]),
                    "eef_quat_post_xyzw": _json(eef_post["quat"]),
                    "eef_delta_actual": _json(eef_post["pos"] - eef_pre["pos"]),
                    "target_pos_pre_world": _json(object_pre["pos"]),
                    "target_quat_pre_wxyz": _json(object_pre["quat"]),
                    "target_pos_post_world": _json(object_post["pos"]),
                    "target_quat_post_wxyz": _json(object_post["quat"]),
                    "target_velocity_post_world": _json(object_velocity),
                    "eef_to_target_pre_distance": float(np.linalg.norm(eef_pre["pos"] - object_pre["pos"])),
                    "eef_goal_to_target_pre_distance": float(np.linalg.norm(goal_world - object_pre["pos"])),
                    "container_center_world": _json(center) if center is not None else "",
                    "target_in_container_frame": _json(local_object) if local_object is not None else "",
                    "container_half_extents": _json(region_size) if region_size is not None else "",
                    "container_margin_xyz": _json(margins) if margins is not None else "",
                    "inside_container": in_region,
                    "gripper_qpos_pre": _json(gripper_pre),
                    "gripper_qpos_post": _json(gripper_post),
                    "gripper_aperture_pre_mean_abs": float(np.mean(np.abs(gripper_pre))),
                    "gripper_aperture_post_mean_abs": float(np.mean(np.abs(gripper_post))),
                    "gripper_action": float(action[-1]),
                    "target_contact_pairs": _json(contacts),
                    "target_finger_contact_pairs": _json(finger_contacts),
                    "official_success": bool(env.check_success()),
                    "replay_eef_error_vs_saved_log": replay_error,
                })

        runtime = env.get_provenance_metadata()
        osc = {
            "class": type(arm_controller).__name__,
            "input_type": arm_controller.input_type,
            "input_ref_frame": arm_controller.input_ref_frame,
            "goal_update_mode": arm_controller._goal_update_mode,
            "input_min": np.asarray(arm_controller.input_min),
            "input_max": np.asarray(arm_controller.input_max),
            "output_min": np.asarray(arm_controller.output_min),
            "output_max": np.asarray(arm_controller.output_max),
            "action_scale": np.asarray(arm_controller.action_scale),
            "action_output_transform": np.asarray(arm_controller.action_output_transform),
            "action_input_transform": np.asarray(arm_controller.action_input_transform),
            "origin_pos": np.asarray(arm_controller.origin_pos),
            "origin_ori": np.asarray(arm_controller.origin_ori),
            "control_dim": int(arm_controller.control_dim),
        }
        if gripper_controller is not None:
            gripper_config = {
                "class": type(gripper_controller).__name__,
                "input_min": np.asarray(gripper_controller.input_min),
                "input_max": np.asarray(gripper_controller.input_max),
                "output_min": np.asarray(gripper_controller.output_min),
                "output_max": np.asarray(gripper_controller.output_max),
            }
        else:
            gripper_config = None

        task_bddl_hash = hashlib.sha256(Path(env.bddl_file_path).read_bytes()).hexdigest()
        summary = {
            "source_run": str(source_dir),
            "source_log_sha256": hashlib.sha256(source_log.read_bytes()).hexdigest(),
            "replay": {
                "task_suite": task_suite,
                "task_id": task_id,
                "task_name": env.task_name,
                "instruction": env.language_instruction,
                "initial_state_id": initial_state_id,
                "steps_replayed": len(rows),
                "camera_resolution": camera_resolution,
                "control_freq_hz": env.control_freq,
                "horizon": env.horizon,
                "robot": "Panda",
                "gripper": env.get_robot_kinematics_info()["gripper_type"],
                "controller": env.controller_name,
                "sim_timestep_seconds": float(domain.sim.model.opt.timestep),
                "physics_steps_per_control_step": int(round(env.control_freq and (1.0 / env.control_freq) / domain.sim.model.opt.timestep)),
                "target_object_state": object_key,
                "target_region_state": region_key,
                "target_initial_pos_world": initial_object["pos"],
                "target_initial_quat_wxyz": initial_object["quat"],
                "eef_initial_pos_world": initial_eef["pos"],
                "eef_initial_quat_xyzw": initial_eef["quat"],
                "gripper_initial_qpos": initial_gripper,
                "container_initial_pos_world": initial_region["pos"] if initial_region else None,
                "container_initial_quat_xyzw": initial_region["quat"] if initial_region else None,
                "container_half_extents": region_size,
                "osc_controller": osc,
                "gripper_controller": gripper_config,
                "source_episode_metadata": episode,
                "replay_runtime_provenance": runtime,
                "bddl_path": env.bddl_file_path,
                "bddl_sha256": task_bddl_hash,
                "max_eef_replay_error_vs_saved_log_m": max_replay_error,
                "trajectory_csv": str(audit_path),
            },
        }
        summary_path = output_dir / "audit_metadata.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), encoding="utf-8")
        return summary
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Saved run folder with model_output.jsonl and episode.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output folder for audit metadata and CSV")
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-id", type=int, required=True)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--target-object", required=True, help="Exact BDDL target name or unique substring, e.g. ketchup_1")
    parser.add_argument("--target-region", default=None, help="Exact BDDL site-region name, e.g. basket_1_contain_region")
    args = parser.parse_args()
    summary = audit_saved_run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        task_suite=args.task_suite,
        task_id=args.task_id,
        initial_state_id=args.initial_state_id,
        camera_resolution=args.camera_resolution,
        target_object=args.target_object,
        target_region=args.target_region,
    )
    print(json.dumps(summary["replay"], indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
