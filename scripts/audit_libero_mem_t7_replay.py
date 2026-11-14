#!/usr/bin/env python
"""Audit both T7 evaluator branches with official raw actions, without a policy."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import load_protocol, load_suite_source, task_spec
from src.simulator.libero_mem_loader import LIBERO_MEM_ROOT, setup_libero_mem_paths
from src.training.split_lock import seal_lock, state_digest
from src.utils.file_hash import sha256_file


def replay(env, actions, state, protocol, branches):
    upstream = env.env
    env.seed(protocol["reset"]["seed"])
    env.reset()
    upstream.reset_subgoal_progress()
    upstream._overshot = False
    env.set_init_state(state)
    np.testing.assert_allclose(env.get_sim_state(), state, rtol=0, atol=1e-12)
    upstream.reset_subgoal_progress()
    upstream._overshot = False
    for _ in range(protocol["reset"]["warmup_steps"]):
        env.step(protocol["reset"]["warmup_action"])
    upstream.reset_subgoal_progress()
    upstream._overshot = False
    milestones = []
    previous_count = 0
    for step in range(len(actions) + 1):
        before_check = len(upstream._satisfied_subgoals)
        if before_check > previous_count:
            milestones.append(dict(step=step, count=before_check, phase="environment_step"))
        returned = bool(upstream._check_success(inc=True))
        previous_count = len(upstream._satisfied_subgoals)
        if previous_count > before_check:
            milestones.append(dict(step=step, count=previous_count, phase="observation_check"))
        if step < len(actions):
            env.step(actions[step])
    satisfied = upstream._satisfied_subgoals
    completed = [i for i, branch in enumerate(branches) if branch == satisfied]
    return dict(executed_actions=len(actions), milestones=milestones, completed_branches=completed,
                last_upstream_return=returned, overshot=bool(upstream._overshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/libero_mem_raw")
    parser.add_argument("--demos", nargs="+", required=True)
    parser.add_argument("--without-cameras", action="store_true",
                        help="Only inspect evaluator timing without rendering RGB")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.demos) != len(set(args.demos)):
        parser.error("Choose a new output and distinct development demos")
    lock_path = ROOT / "resources/manifests/libero_mem_t7_raw_states_v1.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if seal_lock({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock:
        raise ValueError("T7 state inventory is corrupt")
    states = {row["demo"]: row for row in lock["states"] if row["raw_index"] < 20}
    if set(args.demos) - states.keys():
        raise ValueError("Use only the first 20 raw states for this development diagnostic")
    path = args.raw_root / lock["source"]["path"]
    if sha256_file(path) != lock["source"]["sha256"]:
        raise ValueError("Raw source hash differs from the T7 inventory")
    suite = load_suite_source(LIBERO_MEM_ROOT)
    task = task_spec(suite, "T7")
    if lock["bddl_sha256"] != suite["source_text_sha256"][task["bddl"]]:
        raise ValueError("T7 task source differs from state inventory")
    revision = subprocess.check_output(
        ["git", "-C", str(LIBERO_MEM_ROOT), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    if revision != suite["upstream_revision"] or revision != lock["upstream_revision"]:
        raise ValueError("Runtime checkout differs from the pinned upstream")
    protocol = load_protocol(source_root=LIBERO_MEM_ROOT)
    versions = {name: importlib.metadata.version(name)
                for name in ("robosuite", "numpy", "bddl", "mujoco", "robomimic")}
    if any(protocol["runtime"][name] != version for name, version in versions.items()):
        raise ValueError("Use the pinned simulator runtime")
    setup_libero_mem_paths()
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.controllers import load_controller_config

    if load_controller_config(default_controller="OSC_POSE") != protocol["simulation"]["controller"]:
        raise ValueError("Controller differs from the pinned protocol")
    results = {}
    env = ControlEnv(bddl_file_name=str(LIBERO_MEM_ROOT / task["bddl"]), robots=["Panda"],
        controller="OSC_POSE", gripper_types="default", control_freq=20, horizon=1020,
        use_camera_obs=not args.without_cameras, has_offscreen_renderer=not args.without_cameras,
        camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
        camera_depths=False, camera_segmentations=None, ignore_done=True, reward_shaping=True)
    try:
        if (env.env.action_dim != 7 or env.env.control_freq != 20
                or type(env.env.robots[0].robot_model).__name__ != protocol["simulation"]["actual_robot_class"]):
            raise ValueError("Unexpected robot/action/frequency contract")
        goal, = env.env.parsed_problem["goal_state"]
        if goal[0] != "or" or len(goal) != 3 or any(
                branch[0] != "sequence" or len(branch) != 4 for branch in goal[1:]):
            raise ValueError("Unexpected upstream T7 branches")
        branches = [branch[1:] for branch in goal[1:]]
        with h5py.File(path, "r") as source:
            for demo in args.demos:
                group = source["data"][demo]
                actions, state = group["actions"][()], group["init_state"][()]
                if (state_digest(state) != states[demo]["initial_state_sha256"]
                        or actions.shape != (states[demo]["actions"], 7)
                        or not np.isfinite(actions).all()):
                    raise ValueError(f"Unexpected reset/action source at {demo}")
                results[demo] = replay(env, actions, state, protocol, branches)
                print(demo, results[demo], flush=True)
    finally:
        env.close()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(dict(classification="CUSTOM_T7_RAW_REPLAY_NOT_POLICY_SCORE", runtime=versions,
                       state_lock_sha256=lock["lock_sha256"], source_sha256=lock["source"]["sha256"],
                       source_code_sha256=sha256_file(Path(__file__)), upstream_revision=revision,
                       camera_observations=not args.without_cameras,
                       warmup=protocol["reset"], results=results), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
