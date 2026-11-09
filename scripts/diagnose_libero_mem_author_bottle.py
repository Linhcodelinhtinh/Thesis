#!/usr/bin/env python
"""Replay LIBERO-Mem's authored reset + 20-step warm-up in its vendored fork.

This diagnostic constructs the task using the same task mapping/controller
settings as the pinned author's libero_utils.py. That helper currently contains
an active `1/0`, so it cannot be called directly; the source is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MEM_ROOT = ROOT / "external/libero-mem"
EXPECTED_REV = "0eee0defa4024e51511b6e15a79f30e9620209de"
TASK = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default="demo_33")
    parser.add_argument("--split-lock", type=Path, default=ROOT / "experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json")
    parser.add_argument("--source-hdf5", type=Path, default=ROOT / "data/libero_mem/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/libero_mem_author_candidate/bottle_warmup")
    parser.add_argument("--on-screen", action="store_true", help="Match the source helper's onscreen renderer; requires a working display.")
    return parser.parse_args()


def digest_state(state):
    return hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()


def bottle_tilt_degrees(quat_wxyz):
    """Angle between the bottle's local vertical axis and world vertical."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    world_vertical_dot = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return float(np.degrees(np.arccos(world_vertical_dot)))


def main():
    args = parse_args()
    revision = subprocess.check_output(["git", "-C", str(MEM_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if revision != EXPECTED_REV:
        raise RuntimeError(f"Expected author source {EXPECTED_REV}, found {revision}")
    if importlib.metadata.version("robosuite") != "1.4.1":
        raise RuntimeError("This diagnostic requires the author's vendored robosuite 1.4.1 environment")
    if importlib.metadata.version("mujoco") != "3.2.2":
        raise RuntimeError("This replay requires the author's script-pinned MuJoCo 3.2.2")

    with args.split_lock.open(encoding="utf-8") as stream:
        lock = json.load(stream)
    row = next(item for item in lock["episodes"] if item["domain"] == "mem" and item["task"] == TASK and item["demo"] == args.demo)
    state = np.asarray(row["initial_state"], dtype=np.float64)
    actual_state_hash = digest_state(state)
    if actual_state_hash != row["initial_state_sha256"]:
        raise RuntimeError("Locked initial-state bytes do not match their recorded SHA-256")

    config_dir = args.output / "libero-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    package = MEM_ROOT / "libero/libero"
    config = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(ROOT / "data/libero_mem"),
        "assets": str(package / "assets"),
    }
    (config_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(MEM_ROOT))

    from libero.libero.envs import TASK_MAPPING
    import libero.libero.envs.bddl_utils as BDDLUtils
    from robosuite.controllers import load_controller_config

    bddl = package / "bddl_files/libero_mem" / f"{TASK}.bddl"
    problem = BDDLUtils.get_problem_info(str(bddl))["problem_name"]
    # The pinned author's libero_utils.get_libero_env uses this exact task class,
    # Panda/OSC_POSE/default gripper, and 20 Hz. Only onscreen rendering is
    # disabled by default for headless WSL operation.
    env = TASK_MAPPING[problem](
        bddl_file_name=str(bddl),
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=args.on_screen,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    bottle_ids = [i for i, name in enumerate(env.sim.model.body_names) if "wine_bottle" in str(name).lower()]
    if not bottle_ids:
        env.close()
        raise RuntimeError("No wine_bottle body was found in the compiled author task model")

    def pose_snapshot(step):
        bodies = []
        for body_id in bottle_ids:
            bodies.append({
                "body_id": int(body_id),
                "body_name": str(env.sim.model.body_names[body_id]),
                "xyz": np.asarray(env.sim.data.body_xpos[body_id], dtype=float).tolist(),
                "quat_wxyz": np.asarray(env.sim.data.body_xquat[body_id], dtype=float).tolist(),
                "tilt_degrees": bottle_tilt_degrees(env.sim.data.body_xquat[body_id]),
            })
        return {"step": step, "bodies": bodies}

    try:
        env.seed(0)
        env.reset()
        env.reset_subgoal_progress()
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        env._check_success()
        env._post_process()
        env._update_observables(force=True)
        obs = env._get_observations()
        restored = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        if restored.shape != state.shape or not np.allclose(restored, state, rtol=0, atol=1e-12):
            raise RuntimeError("Author-fork simulator failed exact initial-state restoration")
        env.reset_subgoal_progress()

        rows = [pose_snapshot(0)]
        Image.fromarray(np.asarray(obs["agentview_image"], dtype=np.uint8)).save(args.output / "author_step_00_raw.png")
        for step in range(1, 21):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            rows.append(pose_snapshot(step))
            Image.fromarray(np.asarray(obs["agentview_image"], dtype=np.uint8)).save(args.output / f"author_step_{step:02d}_raw.png")

        source_frame_note = None
        frame_comparison = None
        if args.source_hdf5.exists():
            with h5py.File(args.source_hdf5, "r") as source:
                frame = np.asarray(source[f"data/{args.demo}/obs/agentview_rgb"][0], dtype=np.uint8)
                Image.fromarray(frame).save(args.output / "train_demo_first_stored_frame.png")
                Image.fromarray(frame[:, ::-1]).save(args.output / "train_demo_first_stored_frame_source_orientation.png")

                candidate_raw = np.asarray(obs["agentview_image"], dtype=np.uint8)
                # mem_3_refine_demonstration.get_visual first flips horizontally
                # and then rotates 180 degrees before writing agentview_rgb.
                # Their composition is a vertical flip of the raw observation.
                candidate_refined = candidate_raw[::-1]
                Image.fromarray(candidate_refined).save(args.output / "author_step_20_refined_view.png")
                difference = np.abs(candidate_refined.astype(np.int16) - frame.astype(np.int16))
                frame_comparison = {
                    "source_transform_applied_to_replay": "horizontal_flip_then_rotate_180 (equivalent to vertical_flip), matching mem_3_refine_demonstration.get_visual",
                    "mean_absolute_rgb_error_0_255": float(np.mean(difference)),
                    "rmse_rgb_0_255": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
                    "fraction_exact_pixels": float(np.mean(np.all(difference == 0, axis=-1))),
                }
                side_by_side = Image.new("RGB", (frame.shape[1] * 2, frame.shape[0] + 24), "white")
                from PIL import ImageDraw
                draw = ImageDraw.Draw(side_by_side)
                draw.text((4, 4), "Replay after 20 warm-up steps", fill="black")
                draw.text((frame.shape[1] + 4, 4), "Released demo frame 0", fill="black")
                side_by_side.paste(Image.fromarray(candidate_refined), (0, 24))
                side_by_side.paste(Image.fromarray(frame), (frame.shape[1], 24))
                side_by_side.save(args.output / "post_warmup_vs_train_frame.png")
                source_frame_note = "released refined HDF5 frame; replay comparison applies the author's horizontal flip then 180-degree rotation"

        manifest = {
            "classification": "AUTHOR_SOURCE_WITH_AUTHOR_SIMULATOR_DEPENDENCY_SUBSET",
            "runtime_scope": "simulator-relevant pins copied from scripts/requirements.txt; policy/training packages omitted",
            "upstream_revision": revision,
            "python": sys.version.split()[0],
            "robosuite": importlib.metadata.version("robosuite"),
            "robosuite_import": str(Path(__import__("robosuite").__file__).resolve()),
            "mujoco": importlib.metadata.version("mujoco"),
            "task": TASK,
            "demo": args.demo,
            "split": row["split"],
            "initial_state_sha256": actual_state_hash,
            "state_restore_max_abs_error": float(np.max(np.abs(restored - state))),
            "seed": 0,
            "controller": "OSC_POSE from author's vendored robosuite",
            "robot": "Panda",
            "control_freq_hz": 20,
            "warmup_steps": 20,
            "warmup_action": [0, 0, 0, 0, 0, 0, -1],
            "onscreen_renderer": bool(args.on_screen),
            "frame_comparison": frame_comparison,
            "mujoco_version_note": "The author README instructs demo collection from scripts/requirements.txt, which pins MuJoCo 3.2.2. Python 3.10.22 remains a project choice because the author setup does not pin Python.",
            "source_frame_note": source_frame_note,
            "trajectory": rows,
        }
        (args.output / "diagnostic.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: manifest[key] for key in (
            "classification", "upstream_revision", "python", "robosuite", "mujoco", "task", "demo",
            "initial_state_sha256", "state_restore_max_abs_error", "warmup_steps", "warmup_action",
        )}, indent=2))
        print("output:", args.output)
        for row_data in rows:
            print(json.dumps(row_data))
    finally:
        env.close()


if __name__ == "__main__":
    main()
