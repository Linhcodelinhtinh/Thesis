#!/usr/bin/env python
"""Replay pinned LIBERO-Mem raw author actions through the author's evaluator path.

This custom diagnostic audits simulator/evaluator wiring. It runs no policy and
does not produce a benchmark score. It is limited to the first 20 raw demos.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import load_protocol, load_suite_source, task_spec
from src.simulator.libero_mem_loader import LIBERO_MEM_ROOT, setup_libero_mem_paths
from src.training.split_lock import state_digest
from src.utils.file_hash import sha256_file

RAW_REPOSITORY = "libero-mem/LIBERO-Mem-Raw"
RAW_REVISION = "61917674055e8b1cbbfd22c1b54c28d20c1a1584"
UPSTREAM_EVALUATOR = "scripts/mem_5_run_evaluation_env_gt.py"
UPSTREAM_LIBERO_UTILS = "scripts/experiments/robot/libero/libero_utils.py"


def verify_pinned_source_file(path, revision):
    """Require local official source bytes to match the pinned Git revision."""
    local = Path(LIBERO_MEM_ROOT) / path
    expected = subprocess.check_output(
        ["git", "-C", str(LIBERO_MEM_ROOT), "show", f"{revision}:{path}"],
        timeout=30,
    )
    local_text = local.read_text(encoding="utf-8")
    expected_text = expected.decode("utf-8")
    normalize = lambda value: value.replace("\r\n", "\n").replace("\r", "\n")
    if normalize(local_text) != normalize(expected_text):
        raise ValueError(f"Official source file differs from pinned revision: {path}")
    return hashlib.sha256(expected).hexdigest()


def load_pinned_environment_helpers(helper_source):
    """Load exact upstream environment helper definitions without its TF import."""
    from libero.libero.envs import OffScreenRenderEnv, TASK_MAPPING
    import libero.libero.envs.bddl_utils as BDDLUtils
    from robosuite import load_controller_config
    from robosuite.wrappers import Wrapper

    module = ast.parse(Path(helper_source).read_text(encoding="utf-8"), filename=str(helper_source))
    wanted = {"DeterministicInitWrapper", "get_libero_env", "get_libero_dummy_action"}
    definitions = [node for node in module.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise ValueError("Pinned upstream helper is missing expected environment definitions")
    namespace = dict(
        os=os, np=np, BDDLUtils=BDDLUtils, TASK_MAPPING=TASK_MAPPING,
        load_controller_config=load_controller_config, OffScreenRenderEnv=OffScreenRenderEnv,
        Wrapper=Wrapper,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(helper_source), "exec"), namespace)
    return namespace["get_libero_env"], namespace["get_libero_dummy_action"]


def load_pinned_state_initializer(evaluator_source):
    """Load the exact state initialization helper used by upstream mem_5."""
    module = ast.parse(Path(evaluator_source).read_text(encoding="utf-8"),
                       filename=str(evaluator_source))
    definitions = [node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "set_init_state"]
    if len(definitions) != 1:
        raise ValueError("Pinned upstream evaluator lacks its unique set_init_state helper")
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(evaluator_source), "exec"), namespace)
    return namespace["set_init_state"]


def official_action_transform(actions):
    """Mirror the pinned mem_5 evaluator's leading-noop and gripper handling."""
    actions = np.asarray(actions)
    start_idx = 0
    for i, action in enumerate(actions):
        if np.linalg.norm(action[:-1]) >= 1e-4:
            start_idx = i
            break
    transformed = actions[start_idx:].tolist()
    for _ in range(20):
        transformed.append([0, 0, 0, 0, 0, 0, 0])
    for action in transformed:
        action[-1] = action[-1] if action[-1] != 0 else -1
    return np.asarray(transformed, dtype=np.float64), start_idx


def unwrap_to_attribute(env, attribute):
    current = env
    seen = set()
    while not hasattr(current, attribute):
        if not hasattr(current, "env") or id(current) in seen:
            raise AttributeError(f"Environment chain lacks {attribute}")
        seen.add(id(current))
        current = current.env
    return current


def replay(env, actions, state, dummy_action, set_init_state):
    """Follow the author's reset, warmup, evaluator-read, and step schedule."""
    env.reset()
    set_init_state(env, state)
    env.reset_subgoal_progress()
    for _ in range(20):
        env.step(dummy_action)

    upstream = unwrap_to_attribute(env, "_satisfied_subgoals")
    milestones = []
    last_return = False
    done = False
    for step, action in enumerate(actions):
        before = len(upstream._satisfied_subgoals)
        last_return = bool(upstream._check_success(inc=True))
        after_check = len(upstream._satisfied_subgoals)
        if after_check > before:
            milestones.append(dict(step=step, count=after_check, phase="observation_check"))
        _, _, done, _ = env.step(action)
        after_step = len(upstream._satisfied_subgoals)
        if after_step > after_check:
            milestones.append(dict(step=step + 1, count=after_step, phase="environment_step"))

    return dict(
        executed_actions=len(actions),
        milestones=milestones,
        satisfied_subgoals=len(upstream._satisfied_subgoals),
        last_upstream_return=last_return,
        final_step_done=bool(done),
        overshot=bool(upstream._overshot),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[f"T{i}" for i in range(7, 11)], required=True)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/libero_mem_raw")
    parser.add_argument("--demos", nargs="+", default=["demo_1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.demos) != len(set(args.demos)):
        parser.error("Choose a new output and distinct development demos")

    receipt_path = ROOT / "resources/manifests" / f"libero_mem_{args.task.lower()}_raw_receipt_v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("classification") != "AUTHOR_RELEASE_LOCAL_SHA256_RECEIPT"
            or receipt.get("task") != args.task
            or receipt.get("source_kind") != "raw"
            or receipt.get("repository") != RAW_REPOSITORY
            or receipt.get("revision") != RAW_REVISION
            or not receipt.get("local_verification", {}).get("task_file_sha256_verified")):
        raise ValueError("Raw source receipt does not match the pinned author release")
    if len(receipt.get("sha256", {})) != 1:
        raise ValueError("Expected exactly one task file in the raw source receipt")
    filename, expected_sha256 = next(iter(receipt["sha256"].items()))
    path = args.raw_root / filename
    if sha256_file(path) != expected_sha256:
        raise ValueError("Raw task file hash differs from its receipt")

    suite = load_suite_source(LIBERO_MEM_ROOT)
    task = task_spec(suite, args.task)
    revision = subprocess.check_output(
        ["git", "-C", str(LIBERO_MEM_ROOT), "rev-parse", "HEAD"],
        text=True, timeout=30,
    ).strip()
    if revision != suite["upstream_revision"]:
        raise ValueError("Runtime checkout differs from the pinned upstream")
    source_hashes = {
        UPSTREAM_EVALUATOR: verify_pinned_source_file(UPSTREAM_EVALUATOR, revision),
        UPSTREAM_LIBERO_UTILS: verify_pinned_source_file(UPSTREAM_LIBERO_UTILS, revision),
    }
    protocol = load_protocol(source_root=LIBERO_MEM_ROOT)
    versions = {name: importlib.metadata.version(name)
                for name in ("robosuite", "numpy", "bddl", "mujoco", "robomimic")}
    if any(protocol["runtime"][name] != version for name, version in versions.items()):
        raise ValueError("Use the pinned simulator runtime")

    selected = {}
    with h5py.File(path, "r") as source:
        all_demos = set(source["data"].keys())
        for demo in args.demos:
            try:
                raw_index = int(demo.rsplit("_", 1)[1]) - 1
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Unexpected raw demo name: {demo}") from exc
            if demo not in all_demos or not 0 <= raw_index < 20:
                raise ValueError("Use existing demonstrations from the first 20 development states")
            group = source["data"][demo]
            original_actions = group["actions"][()]
            state = group["init_state"][()]
            if (original_actions.ndim != 2 or original_actions.shape[1] != 7
                    or state.ndim != 1 or not np.isfinite(original_actions).all()
                    or not np.isfinite(state).all()):
                raise ValueError(f"Unexpected state/action contract in {demo}")
            actions, skipped_leading_noops = official_action_transform(original_actions)
            selected[demo] = dict(
                raw_index=raw_index,
                original_actions=original_actions,
                actions=actions,
                state=state,
                skipped_leading_noops=skipped_leading_noops,
            )

    setup_libero_mem_paths()
    sys.path.insert(0, str(LIBERO_MEM_ROOT / "scripts"))
    os.chdir(LIBERO_MEM_ROOT / "scripts")
    from libero.libero import benchmark
    get_libero_env, get_libero_dummy_action = load_pinned_environment_helpers(
        LIBERO_MEM_ROOT / UPSTREAM_LIBERO_UTILS)
    set_init_state = load_pinned_state_initializer(
        LIBERO_MEM_ROOT / UPSTREAM_EVALUATOR)
    task_suite = benchmark.get_benchmark_dict()["libero_mem"]()
    suite_tasks = [item for item in task_suite.tasks if item.name == task["name"]]
    if len(suite_tasks) != 1:
        raise ValueError("Pinned author task map does not resolve exactly once")
    env, _ = get_libero_env(suite_tasks[0], "llava", resolution=256)
    try:
        base = unwrap_to_attribute(env, "action_dim")
        if (base.action_dim != 7 or base.control_freq != 20
                or type(base.robots[0].robot_model).__name__
                != protocol["simulation"]["actual_robot_class"]):
            raise ValueError("Unexpected robot/action/frequency contract")
        goal, = base.parsed_problem["goal_state"]
        if goal[0] != task["goal_operator"]:
            raise ValueError("Parsed BDDL goal operator differs from the suite inventory")
        results = {}
        for demo, row in selected.items():
            transformed_sha = hashlib.sha256(
                np.ascontiguousarray(row["actions"]).tobytes()).hexdigest()
            results[demo] = dict(
                raw_index=row["raw_index"],
                initial_state_sha256=state_digest(row["state"]),
                original_action_sha256=hashlib.sha256(
                    np.ascontiguousarray(row["original_actions"]).tobytes()).hexdigest(),
                transformed_action_sha256=transformed_sha,
                skipped_leading_noops=row["skipped_leading_noops"],
                **replay(env, row["actions"], row["state"],
                         get_libero_dummy_action("llava"), set_init_state),
            )
            print(demo, results[demo], flush=True)
    finally:
        env.close()

    output_record = dict(
        classification="CUSTOM_AUTHOR_ACTION_REPLAY_NOT_POLICY_SCORE",
        task=args.task,
        source_repository=receipt["repository"],
        source_revision=receipt["revision"],
        source_filename=filename,
        source_sha256=expected_sha256,
        receipt_sha256=sha256_file(receipt_path),
        source_metadata_url=receipt["source_metadata_url"],
        upstream_repository=suite["upstream_repository"],
        upstream_revision=revision,
        upstream_source_sha256=source_hashes,
        bddl_path=task["bddl"],
        bddl_sha256=suite["source_text_sha256"][task["bddl"]],
        source_code_sha256=sha256_file(Path(__file__)),
        runtime=versions,
        goal_operator=task["goal_operator"],
        environment_helper="upstream scripts/experiments/robot/libero/libero_utils.py:get_libero_env",
        evaluator_source="upstream scripts/mem_5_run_evaluation_env_gt.py",
        action_transform="upstream mem_5 remove_initial_noops threshold=1e-4; append 20 zero vectors; map gripper 0 to -1",
        helper_loading="exact pinned helper AST definitions; TensorFlow-dependent unused imports omitted",
        camera_observations=True,
        warmup_steps=20,
        results=results,
    )
    output_path = ROOT / args.output if not args.output.is_absolute() else args.output
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(output_record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
