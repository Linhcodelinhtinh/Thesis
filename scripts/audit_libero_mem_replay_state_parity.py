#!/usr/bin/env python
"""Compare pinned T8 replay robot states against the released refined trace.

This is a custom simulator diagnostic, not a policy score. It replays one
author demonstration through the pinned upstream helper and compares the
pre-action robot observations with the corresponding refined HDF5 rows.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_libero_mem_author_replay import (  # noqa: E402
    LIBERO_MEM_ROOT,
    UPSTREAM_EVALUATOR,
    UPSTREAM_LIBERO_UTILS,
    load_pinned_environment_helpers,
    load_pinned_state_initializer,
    official_action_transform,
    setup_libero_mem_paths,
    unwrap_to_attribute,
    verify_pinned_source_file,
)
from src.evaluation.libero_mem_protocol import load_protocol, load_suite_source, task_spec  # noqa: E402
from src.training.libero_mem_hdf5_loader import read_episode_metadata  # noqa: E402
from src.training.split_lock import state_digest  # noqa: E402
from src.utils.file_hash import sha256_file  # noqa: E402

TASK = "T8"
DEMO = "demo_1"
TOLERANCE = 1e-5


def receipt(kind):
    path = ROOT / "resources/manifests" / f"libero_mem_{TASK.lower()}_{kind}_receipt_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    local = value.get("local_verification", {})
    if (value.get("classification") != "AUTHOR_RELEASE_LOCAL_SHA256_RECEIPT"
            or value.get("task") != TASK or value.get("source_kind") != kind
            or local.get("task_file_sha256_verified") is not True):
        raise ValueError(f"Invalid {kind} source receipt")
    return path, value


def robot_state(observation):
    required = ("robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
    if not all(key in observation for key in required):
        raise ValueError("Replay observation lacks the recorded robot-state fields")
    return np.concatenate([observation[key] for key in required]).astype(np.float64, copy=False)


def main():
    raw_receipt_path, raw_receipt = receipt("raw")
    refined_receipt_path, refined_receipt = receipt("refined")
    raw_name, raw_hash = next(iter(raw_receipt["sha256"].items()))
    refined_name, refined_hash = next(iter(refined_receipt["sha256"].items()))
    if raw_name != refined_name:
        raise ValueError("Raw and refined receipts identify different T8 task files")
    raw_path = ROOT / "data/libero_mem_raw" / raw_name
    refined_path = ROOT / "data/libero_mem" / refined_name
    if (not raw_path.is_file() or not refined_path.is_file()
            or raw_path.stat().st_size != raw_receipt["local_verification"]["task_file_size_bytes"]
            or refined_path.stat().st_size != refined_receipt["local_verification"]["task_file_size_bytes"]):
        raise ValueError("T8 source file is missing or its size differs from the local receipt")

    with h5py.File(raw_path, "r") as source:
        original_actions = source[f"data/{DEMO}/actions"][()]
        initial_state = source[f"data/{DEMO}/init_state"][()]
    with h5py.File(refined_path, "r") as source:
        refined_actions = source[f"data/{DEMO}/actions"][()]
        expected_robot_states = source[f"data/{DEMO}/robot_states"][()]
    transformed_actions, skipped = official_action_transform(original_actions)
    if (not np.array_equal(transformed_actions, refined_actions)
            or len(refined_actions) != len(expected_robot_states)):
        raise ValueError("T8 raw-to-refined action or robot-state alignment check failed")

    suite = load_suite_source(LIBERO_MEM_ROOT)
    task = task_spec(suite, TASK)
    metadata = read_episode_metadata(
        ROOT / "data/libero_mem/metainfo.json", task["name"], [DEMO]
    )[DEMO]
    metadata_state = np.asarray(metadata["initial_state"], dtype=np.float64)
    if (metadata.get("success") is not True or not np.array_equal(initial_state, metadata_state)):
        raise ValueError("T8 refined metainfo success/state differs from the locked source")

    source_revision = subprocess.check_output(
        ["git", "-C", str(LIBERO_MEM_ROOT), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if source_revision != suite["upstream_revision"]:
        raise ValueError("LIBERO-Mem checkout differs from the pinned upstream revision")
    upstream_hashes = {
        UPSTREAM_EVALUATOR: verify_pinned_source_file(UPSTREAM_EVALUATOR, source_revision),
        UPSTREAM_LIBERO_UTILS: verify_pinned_source_file(UPSTREAM_LIBERO_UTILS, source_revision),
    }
    protocol = load_protocol(source_root=LIBERO_MEM_ROOT)
    versions = {name: importlib.metadata.version(name)
                for name in ("robosuite", "numpy", "bddl", "mujoco", "robomimic")}
    if any(protocol["runtime"][name] != version for name, version in versions.items()):
        raise ValueError("Use the pinned simulator runtime")

    setup_libero_mem_paths()
    sys.path.insert(0, str(LIBERO_MEM_ROOT / "scripts"))
    import os
    os.chdir(LIBERO_MEM_ROOT / "scripts")
    from libero.libero import benchmark

    matches = [candidate for candidate in benchmark.get_benchmark_dict()["libero_mem"]().tasks
               if candidate.name == task["name"]]
    if len(matches) != 1:
        raise ValueError("Pinned T8 task does not resolve exactly once")
    get_libero_env, get_dummy_action = load_pinned_environment_helpers(
        LIBERO_MEM_ROOT / UPSTREAM_LIBERO_UTILS
    )
    set_init_state = load_pinned_state_initializer(LIBERO_MEM_ROOT / UPSTREAM_EVALUATOR)
    env, _ = get_libero_env(matches[0], "llava", resolution=256)
    try:
        base = unwrap_to_attribute(env, "action_dim")
        if (base.action_dim != 7 or base.control_freq != 20
                or type(base.robots[0].robot_model).__name__
                != protocol["simulation"]["actual_robot_class"]):
            raise ValueError("Unexpected T8 robot/action/frequency contract")
        env.reset()
        set_init_state(env, initial_state)
        env.reset_subgoal_progress()
        dummy_action = get_dummy_action("llava")
        for _ in range(20):
            observation, _, _, _ = env.step(dummy_action)

        upstream = unwrap_to_attribute(env, "_satisfied_subgoals")
        max_abs = 0.0
        row_max = []
        first_divergence = None
        milestones = []
        last_return, done = False, False
        for i, action in enumerate(transformed_actions):
            observed = robot_state(observation)
            delta = np.abs(observed - expected_robot_states[i])
            error = float(delta.max())
            row_max.append(error)
            max_abs = max(max_abs, error)
            if first_divergence is None and error > TOLERANCE:
                first_divergence = dict(step=i, max_abs_error=error)
            if (i + 1) % 100 == 0:
                print(f"compared {i + 1}/{len(transformed_actions)} robot states", flush=True)

            before = len(upstream._satisfied_subgoals)
            last_return = bool(upstream._check_success(inc=True))
            after_check = len(upstream._satisfied_subgoals)
            if after_check > before:
                milestones.append(dict(step=i, count=after_check, phase="observation_check"))
            observation, _, done, _ = env.step(action)
            after_step = len(upstream._satisfied_subgoals)
            if after_step > after_check:
                milestones.append(dict(step=i + 1, count=after_step, phase="environment_step"))
        result = dict(
            classification="CUSTOM_AUTHOR_ACTION_STATE_PARITY_DIAGNOSTIC_NOT_POLICY_SCORE",
            task=TASK,
            demo=DEMO,
            raw_receipt_sha256=sha256_file(raw_receipt_path),
            refined_receipt_sha256=sha256_file(refined_receipt_path),
            raw_source_sha256=raw_hash,
            refined_source_sha256=refined_hash,
            raw_refined_initial_state_sha256=state_digest(initial_state),
            metadata_initial_state_sha256=state_digest(metadata_state),
            initial_state_exact_match=True,
            refined_metadata_success=True,
            raw_action_rows=int(len(original_actions)),
            leading_noops_removed=int(skipped),
            transformed_action_rows=int(len(transformed_actions)),
            transformed_action_sha256=hashlib.sha256(
                np.ascontiguousarray(transformed_actions).tobytes()).hexdigest(),
            refined_action_sha256=hashlib.sha256(
                np.ascontiguousarray(refined_actions).tobytes()).hexdigest(),
            actions_exact_match=True,
            refined_robot_state_rows=int(len(expected_robot_states)),
            robot_state_layout="gripper_qpos[2], eef_pos[3], eef_quat[4], captured before each recorded action",
            replay_robot_state_rows=int(len(row_max)),
            parity_tolerance=float(TOLERANCE),
            rows_within_tolerance=int(sum(error <= TOLERANCE for error in row_max)),
            first_state_divergence=first_divergence,
            max_abs_robot_state_error=max_abs,
            satisfied_subgoals=int(len(upstream._satisfied_subgoals)),
            milestones=milestones,
            last_upstream_return=last_return,
            final_step_done=bool(done),
            overshot=bool(upstream._overshot),
            upstream_revision=source_revision,
            upstream_source_sha256=upstream_hashes,
            runtime=versions,
            author_replay_code_sha256=sha256_file(ROOT / "scripts/audit_libero_mem_author_replay.py"),
            parity_script_sha256=sha256_file(SCRIPT_PATH),
            limitation="Exact action and initial-state source alignment does not identify the cause of any robot-state divergence or the terminal-success mismatch.",
        )
    finally:
        env.close()

    output = ROOT / "experiments/audits/t8-replay-state-parity-20261115.json"
    if output.exists():
        raise FileExistsError(f"Choose a new audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
