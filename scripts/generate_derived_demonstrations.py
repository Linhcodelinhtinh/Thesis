#!/usr/bin/env python
"""Replay pinned LIBERO-Mem demonstrations in the locked derived simulator.

Outputs raw OpenGL RGB (no author-refinement flip), exact measured state/action
fields, a strict terminal success record, and restartable source provenance.
This is data preparation; it never executes a VLA policy or produces a score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, load_protocol, source_digest, task_spec
from src.simulator.libero_mem_env import LiberoMemEnv
from src.training.libero_mem_hdf5_loader import read_episode_metadata
from src.training.memory_data import validate_episode
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK, content_digest, load_split_lock
from src.utils.file_hash import sha256_file

ARTIFACT = "libero_mem_derived_demonstrations"
DERIVED_PROTOCOL_ID = "libero_mem_derived_data_v1"


def atomic_json(path: Path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with open(temporary, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    os.replace(temporary, path)


def safe_copy(source: Path, destination: Path, buffer_size=16*1024*1024):
    """Copy file safely across filesystems (e.g. ext4 to 9P/drvfs) avoiding sendfile EIO."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as fsrc, open(destination, "wb") as fdst:
        while True:
            chunk = fsrc.read(buffer_size)
            if not chunk:
                break
            fdst.write(chunk)
        fdst.flush()
        os.fsync(fdst.fileno())


def atomic_copy(source: Path, destination: Path, buffer_size=16*1024*1024):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass
    safe_copy(source, temporary, buffer_size=buffer_size)
    if destination.exists():
        try:
            destination.unlink()
        except OSError:
            pass
    os.replace(temporary, destination)


def sealed_manifest(payload):
    result = dict(payload)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = content_digest(result)
    return result


def runtime_provenance():
    protocol = load_protocol()
    expected = protocol["runtime"]
    actual_packages = {
        name: importlib.metadata.version(name)
        for name in ("robosuite", "numpy", "bddl", "mujoco", "robomimic")
    }
    for name, version in actual_packages.items():
        if version != expected[name]:
            raise RuntimeError(f"Pinned LIBERO-Mem runtime mismatch for {name}: expected {expected[name]}, found {version}")
    python_version = platform.python_version()
    expected_python = expected["python"].split(".x", 1)[0]
    if not python_version.startswith(expected_python + "."):
        raise RuntimeError(f"Pinned Python runtime mismatch: expected {expected['python']}, found {python_version}")
    try:
        upstream_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "external/libero-mem"), "rev-parse", "HEAD"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        upstream_revision = None
    if upstream_revision is not None and upstream_revision != protocol["upstream_revision"]:
        raise RuntimeError("LIBERO-Mem source checkout differs from the pinned upstream revision")
    mujoco_gl = os.environ.get("MUJOCO_GL", "unset")
    if mujoco_gl.lower() != "osmesa":
        raise RuntimeError("Derived replay requires MUJOCO_GL=osmesa; use the supplied WSL runner")
    return dict(
        python=python_version,
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        packages=actual_packages,
        upstream_revision=upstream_revision,
        mujoco_gl=mujoco_gl,
    )


def code_provenance():
    files = (
        "scripts/generate_derived_demonstrations.py",
        "src/evaluation/libero_mem_protocol.py",
        "src/evaluation/libero_mem_metrics.py",
        "src/simulator/libero_mem_env.py",
        "src/simulator/libero_mem_loader.py",
        "src/training/split_lock.py",
        "src/training/memory_data.py",
    )
    return {name: sha256_file(ROOT / name) for name in files}


def output_artifacts(output_dir):
    files = sorted(output_dir.glob("*_demo.hdf5"))
    metainfo = output_dir / "metainfo.json"
    if metainfo.is_file():
        files.append(metainfo)
    return {path.name: sha256_file(path) for path in files}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("T1", "T3"), default=("T1", "T3"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/libero_mem_derived_v2")
    parser.add_argument("--staging-dir", type=Path, default=Path("/var/tmp/libero_mem_derived_staging_v2"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--require-success", action="store_true",
                        help="Store only successful complete replays; failures remain in run provenance")
    return parser.parse_args()


def task_description(metadata, task_name: str, demo_name: str):
    instruction = metadata[demo_name].get("task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Missing author task description: {task_name}/{demo_name}")
    return instruction.strip()


def locked_task_rows(parent_lock, task_alias, split):
    task_name = task_spec(load_protocol(), task_alias)["name"]
    rows = [row for row in parent_lock["episodes"]
            if row["domain"] == "mem" and row["task"] == task_name and row["split"] == split]
    return sorted(rows, key=lambda row: int(row["demo"][len("demo_"):]))


def make_run_config(task_alias, args, parent_lock, runtime, code_hashes):
    rows = locked_task_rows(parent_lock, task_alias, args.split)
    if not rows:
        raise ValueError(f"No parent-locked episodes for {task_alias}/{args.split}")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    candidates = rows[:args.max_episodes] if args.max_episodes is not None else rows
    source_name = candidates[0]["source"]
    if any(row["source"] != source_name for row in candidates):
        raise ValueError("Task episodes refer to inconsistent source files")
    source_lock = next((row for row in parent_lock["sources"]
                        if row["domain"] == "mem" and row["path"] == source_name), None)
    meta_source = next((row for row in parent_lock["sources"]
                        if row["domain"] == "mem" and row["path"] == "metainfo.json"), None)
    if source_lock is None or meta_source is None:
        raise ValueError("Parent split lock lacks pinned Mem source files")
    return dict(
        task=task_alias, task_name=task_spec(load_protocol(), task_alias)["name"], split=args.split,
        parent_split_lock_sha256=parent_lock["lock_sha256"],
        source_hdf5_sha256=source_lock["sha256"], source_metainfo_sha256=meta_source["sha256"],
        evaluator_protocol_sha256=source_digest(DEFAULT_PROTOCOL),
        generator_code_sha256=code_hashes["scripts/generate_derived_demonstrations.py"],
        code_sha256=code_hashes, runtime_sha256=content_digest(runtime),
        require_success=args.require_success, max_episodes=args.max_episodes,
        selected_demo_ids=[row["demo"] for row in candidates],
    )


def load_root_manifest(output_dir: Path, parent_lock, generator_hash, runtime, code_hashes):
    path = output_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("artifact") != ARTIFACT or manifest.get("schema_version") != 1:
            raise ValueError("Output directory contains another/legacy artifact; select a fresh --output-dir")
        if manifest.get("parent_split_lock_sha256") != parent_lock["lock_sha256"]:
            raise ValueError("Output directory is locked to a different parent split")
        if manifest.get("evaluator_protocol_sha256") != source_digest(DEFAULT_PROTOCOL):
            raise ValueError("Output directory uses another evaluator protocol")
        if manifest.get("generator_code_sha256") != generator_hash:
            raise ValueError("Generator code changed; select a fresh --output-dir to preserve provenance")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != content_digest(payload):
            raise ValueError("Derived manifest content/hash mismatch")
        if manifest.get("runtime") != runtime or manifest.get("code_sha256") != code_hashes:
            raise ValueError("Runtime or replay code provenance changed; select a fresh --output-dir")
        if manifest.get("status") == "complete":
            protocol = load_protocol()
            expected_artifacts = {task["name"] + "_demo.hdf5" for task in protocol["tasks"]}
            expected_artifacts.add("metainfo.json")
            if set(manifest.get("artifacts", {})) != expected_artifacts:
                raise ValueError("Completed manifest lacks the full derived artifact inventory")
            for name, digest in manifest.get("artifacts", {}).items():
                artifact = output_dir / name
                if not artifact.is_file() or sha256_file(artifact) != digest:
                    raise ValueError("Completed derived artifact hash mismatch: " + name)
        return manifest
    existing = [p.name for p in output_dir.iterdir() if p.is_file()] if output_dir.exists() else []
    if existing:
        raise ValueError("Output directory is nonempty but has no derived manifest; select a fresh directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return sealed_manifest(dict(
        artifact=ARTIFACT, schema_version=1, status="preparing",
        classification="CUSTOM_V2_LIBERO_MEM_DERIVED_TRAINING_DATA",
        derived_protocol_id=DERIVED_PROTOCOL_ID,
        parent_split_lock_sha256=parent_lock["lock_sha256"],
        parent_mem_sources=[row for row in parent_lock["sources"] if row["domain"] == "mem"],
        evaluator_protocol_sha256=source_digest(DEFAULT_PROTOCOL),
        generator_code_sha256=generator_hash,
        code_sha256=code_hashes,
        runtime=runtime,
        image_convention="raw_opengl",
        replay_contract="exact parent actions; 20 locked no-op warmup actions; one upstream policy-visible increment per observation",
        runs={}, artifacts={}, created_at=datetime.now(timezone.utc).isoformat(),
    ))


def run_task(task_alias: str, args, parent_lock, generator_hash, runtime, code_hashes, config):
    env = LiberoMemEnv(task_name=task_alias, split_lock=str(args.split_lock),
                       split=args.split, purpose="demo_replay")
    try:
        episode_indices = {row["demo"]: index for index, row in enumerate(env.episodes)}
        all_candidates = sorted(env.episodes, key=lambda row: int(row["demo"][len("demo_"):]))
        candidates = all_candidates[:args.max_episodes] if args.max_episodes is not None else all_candidates
        if not candidates:
            raise ValueError(f"No parent-locked episodes for {task_alias}/{args.split}")
        if [row["demo"] for row in candidates] != config["selected_demo_ids"]:
            raise ValueError("Environment episode roster differs from the recorded run configuration")

        task_name = env.task_name
        source_name = candidates[0]["source"]
        if any(row["source"] != source_name for row in candidates):
            raise ValueError("Task episodes refer to inconsistent source files")
        source_path = (args.source_dir / source_name).resolve()
        source_lock = next((row for row in parent_lock["sources"]
                            if row["domain"] == "mem" and row["path"] == source_name), None)
        if source_lock is None or sha256_file(source_path) != source_lock["sha256"]:
            raise ValueError("Source HDF5 is not the exact file pinned by the parent split lock")
        meta_source = next((row for row in parent_lock["sources"]
                            if row["domain"] == "mem" and row["path"] == "metainfo.json"), None)
        if meta_source is None or sha256_file(args.source_dir / "metainfo.json") != meta_source["sha256"]:
            raise ValueError("Source metainfo.json is not pinned by the parent split lock")

        run_key = f"{task_alias}/{args.split}"
        source_metadata = read_episode_metadata(
            args.source_dir / "metainfo.json", task_name, [row["demo"] for row in candidates])
        instructions = {row["demo"]: task_description(source_metadata, task_name, row["demo"])
                        for row in candidates}
        run_id = content_digest(config)
        stage_root = args.staging_dir / f"{task_alias}_{args.split}_{run_id[:16]}"
        stage_root.mkdir(parents=True, exist_ok=True)
        run_state_path = stage_root / "run-state.json"
        if run_state_path.exists():
            state = json.loads(run_state_path.read_text(encoding="utf-8"))
            if state.get("config_sha256") != run_id or state.get("config") != config:
                raise ValueError("Staging run configuration changed; select a fresh --staging-dir")
        else:
            state = dict(config=config, config_sha256=run_id, status="running", episodes={})
            atomic_json(run_state_path, state)

        target_h5 = args.output_dir / source_name
        work_h5 = stage_root / source_name
        if work_h5.exists() and not args.resume:
            work_h5.unlink()
            state["episodes"] = {}
            if (stage_root / "metainfo.json").exists():
                (stage_root / "metainfo.json").unlink()
        if not work_h5.exists():
            if target_h5.exists():
                safe_copy(target_h5, work_h5)
            else:
                with h5py.File(work_h5, "w"):
                    pass
        metainfo_out = args.output_dir / "metainfo.json"
        if metainfo_out.exists():
            metainfo = json.loads(metainfo_out.read_text(encoding="utf-8"))
        else:
            metainfo = {}
        staged_meta_path = stage_root / "metainfo.json"
        if staged_meta_path.exists():
            staged_meta = json.loads(staged_meta_path.read_text(encoding="utf-8"))
            for name, entries in staged_meta.items():
                metainfo.setdefault(name, {}).update(entries)
        metainfo.setdefault(task_name, {})

        with h5py.File(work_h5, "a") as dst:
            if (len(dst.keys()) > 0 or len(dst.attrs) > 0) and (
                    dst.attrs.get("derived_protocol_id") != DERIVED_PROTOCOL_ID or
                    dst.attrs.get("parent_split_lock_sha256") != parent_lock["lock_sha256"] or
                    dst.attrs.get("evaluator_protocol_sha256") != source_digest(DEFAULT_PROTOCOL) or
                    dst.attrs.get("source_sha256") != source_lock["sha256"]):
                raise ValueError("Staged/output HDF5 provenance differs; use a fresh output directory")
            if "data" in dst and (
                    dst["data"].attrs.get("derived_protocol_id") != DERIVED_PROTOCOL_ID or
                    dst["data"].attrs.get("macros_image_convention") != "raw_opengl" or
                    dst["data"].attrs.get("source_sha256") != source_lock["sha256"]):
                raise ValueError("Staged/output HDF5 data contract differs; use a fresh output directory")
            dst.attrs["derived_protocol_id"] = DERIVED_PROTOCOL_ID
            dst.attrs["parent_split_lock_sha256"] = parent_lock["lock_sha256"]
            dst.attrs["evaluator_protocol_sha256"] = source_digest(DEFAULT_PROTOCOL)
            dst.attrs["source_sha256"] = source_lock["sha256"]
            data = dst.require_group("data")
            data.attrs["problem_info"] = json.dumps(dict(problem_name=task_name,
                language_instruction=instructions[candidates[0]["demo"]]))
            data.attrs["env_args"] = json.dumps(dict(env_name=task_name, type=1, env_kwargs=dict(
                control_freq=20, controller_configs=dict(type="OSC_POSE", control_delta=True), robots=["Panda"])))
            data.attrs["macros_image_convention"] = "raw_opengl"
            data.attrs["derived_protocol_id"] = DERIVED_PROTOCOL_ID
            data.attrs["source_sha256"] = source_lock["sha256"]

            with h5py.File(source_path, "r") as source_h5:
                expected_source_demos = set(source_h5["data"].keys())
                for row in candidates:
                    demo = row["demo"]
                    if demo not in expected_source_demos:
                        raise KeyError(f"Parent-locked demo missing from source HDF5: {demo}")
                    source_actions = np.asarray(source_h5[f"data/{demo}/actions"][:], dtype=np.float32)
                    if len(source_actions) != row["frames"] or source_actions.shape != (row["frames"], 7):
                        raise ValueError(f"Source action length/shape differs from split lock: {demo}")
                    instruction = instructions[demo]
                    completed = state["episodes"].get(demo)
                    valid_saved = False
                    if args.resume and completed and completed.get("attempt_complete"):
                        if completed.get("saved") and demo in data:
                            try:
                                validate_episode(data[demo])
                                valid_saved = (
                                    completed.get("initial_state_sha256") == row["initial_state_sha256"] and
                                    completed.get("steps") == row["frames"] and
                                    completed.get("success") is True and
                                    bool(data[demo].attrs.get("success", False)) and
                                    data[demo].attrs.get("derived_protocol_id") == DERIVED_PROTOCOL_ID and
                                    hashlib.sha256(np.asarray(data[demo]["actions"][:], dtype=np.float32).tobytes()).hexdigest()
                                    == data[demo].attrs.get("source_actions_sha256") and
                                    np.array_equal(data[demo]["actions"][:], source_actions) and
                                    demo in metainfo[task_name] and
                                    metainfo[task_name][demo].get("task_description") == instruction and
                                    metainfo[task_name][demo].get("initial_state_sha256") == row["initial_state_sha256"]
                                )
                            except (KeyError, ValueError, AssertionError):
                                valid_saved = False
                    if valid_saved:
                        print(f"[{run_key}] {demo}: validated completed replay; resuming", flush=True)
                        continue
                    if args.resume and completed and completed.get("attempt_complete") and not completed.get("saved"):
                        if demo in data:
                            del data[demo]
                        metainfo[task_name].pop(demo, None)
                        print(f"[{run_key}] {demo}: preserving prior excluded replay outcome", flush=True)
                        continue
                    if demo in data:
                        del data[demo]
                    metainfo[task_name].pop(demo, None)
                    obs = env.reset(initial_state_id=episode_indices[demo])
                    captures = {name: [] for name in (
                        "agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "ee_states",
                        "joint_states", "gripper_states", "robot_states", "rewards", "dones")}
                    final_info, executed, terminated = {}, 0, False
                    started = time.time()
                    for step_index, action in enumerate(source_actions):
                        agent = obs["agentview_image"]
                        wrist = obs["robot0_eye_in_hand_image"]
                        pos = obs["robot0_eef_pos"]
                        quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
                        quat_norm = float(np.linalg.norm(quat))
                        if not np.isfinite(quat_norm) or quat_norm <= 0:
                            raise ValueError(f"Invalid EEF quaternion at {demo}/{step_index}")
                        quat /= quat_norm
                        grip = obs["robot0_gripper_qpos"]
                        joints = obs["robot0_joint_pos"]
                        robot_state = np.concatenate((grip[:2], pos, quat)).astype(np.float64)
                        captures["agentview_rgb"].append(agent)
                        captures["eye_in_hand_rgb"].append(wrist)
                        captures["ee_pos"].append(pos)
                        captures["ee_ori"].append(T.quat2axisangle(quat))
                        captures["ee_states"].append(np.concatenate((pos, T.quat2axisangle(quat))))
                        captures["joint_states"].append(joints)
                        captures["gripper_states"].append(grip[:2])
                        captures["robot_states"].append(robot_state)

                        obs, reward, done, info = env.step(action)
                        executed += 1
                        captures["rewards"].append(reward)
                        captures["dones"].append(done)
                        final_info = info
                        if done:
                            terminated = step_index + 1 < len(source_actions)
                            break

                    eval_result = final_info.get("eval_result")
                    completed_cycles = int(getattr(eval_result, "completed_subgoals", 0))
                    final_predicate = bool(final_info.get("final_predicate", False))
                    overshot = bool(getattr(eval_result, "overshot", False)) or bool(env.upstream._overshot)
                    success = completed_cycles == env.target_cycles and final_predicate and not overshot
                    full_replay = executed == len(source_actions) and not terminated
                    should_save = full_replay and (success or not args.require_success)
                    if success:
                        terminal_reason = "full_success"
                    elif terminated:
                        terminal_reason = "upstream_terminal_before_source_end"
                    elif overshot:
                        terminal_reason = "upstream_overshot"
                    elif completed_cycles < env.target_cycles:
                        terminal_reason = "required_milestones_not_completed"
                    elif not final_predicate:
                        terminal_reason = "final_predicate_false"
                    elif not full_replay:
                        terminal_reason = "replay_did_not_reach_source_end"
                    else:
                        terminal_reason = "unsuccessful_replay"
                    record = dict(
                        attempt_complete=True, terminal_reason=terminal_reason,
                        demo=demo, steps=executed, source_steps=len(source_actions),
                        completed_subgoals=completed_cycles, target_cycles=env.target_cycles,
                        final_predicate=final_predicate, overshot=overshot, success=success,
                        terminated_before_source_end=terminated, trajectory_complete=full_replay,
                        saved=should_save, initial_state_sha256=row["initial_state_sha256"],
                        instruction=instruction, duration_seconds=round(time.time() - started, 3),
                    )
                    if should_save:
                        group = data.create_group(demo)
                        group.create_dataset("actions", data=source_actions, compression="gzip")
                        group.create_dataset("rewards", data=np.asarray(captures["rewards"], dtype=np.float32), compression="gzip")
                        group.create_dataset("dones", data=np.asarray(captures["dones"], dtype=np.uint8), compression="gzip")
                        group.create_dataset("robot_states", data=np.asarray(captures["robot_states"], dtype=np.float64), compression="gzip")
                        observations = group.create_group("obs")
                        for name in ("agentview_rgb", "eye_in_hand_rgb"):
                            observations.create_dataset(name, data=np.asarray(captures[name], dtype=np.uint8), compression="gzip")
                        for name in ("ee_pos", "ee_ori", "ee_states", "joint_states", "gripper_states"):
                            observations.create_dataset(name, data=np.asarray(captures[name], dtype=np.float64), compression="gzip")
                        group.attrs["init_state"] = np.asarray(row["initial_state"], dtype=np.float64)
                        group.attrs["num_samples"] = executed
                        group.attrs["success"] = success
                        group.attrs["derived_protocol_id"] = DERIVED_PROTOCOL_ID
                        group.attrs["source_actions_sha256"] = hashlib.sha256(source_actions.tobytes()).hexdigest()
                        validate_episode(group)
                        metainfo[task_name][demo] = dict(
                            initial_state=[float(value) for value in row["initial_state"]],
                            task_description=instruction, success=success,
                            initial_state_sha256=row["initial_state_sha256"],
                            derived_protocol_id=DERIVED_PROTOCOL_ID,
                        )
                    state["episodes"][demo] = record
                    dst.flush()
                    atomic_json(staged_meta_path, {task_name: metainfo[task_name]})
                    atomic_json(run_state_path, state)
                    print(f"[{run_key}] {demo}: steps={executed}/{len(source_actions)} "
                          f"subgoals={completed_cycles}/{env.target_cycles} success={success} saved={should_save}", flush=True)
                    del captures
                    gc.collect()

                data.attrs["num_demos"] = len(data.keys())
                data.attrs["total"] = sum(len(data[name]["actions"]) for name in data.keys())
                dst.flush()

        # The companion metadata follows the exact finalized HDF5 inventory.
        saved_ids = {demo for demo, record in state["episodes"].items() if record.get("saved")}
        with h5py.File(work_h5, "r") as published_check:
            committed_demos = set(published_check["data"].keys())
        metainfo[task_name] = {demo: record for demo, record in metainfo[task_name].items()
                               if demo in committed_demos}
        if set(metainfo[task_name]) != committed_demos or not saved_ids <= committed_demos:
            raise ValueError("HDF5 and companion task/initial-state metadata inventories differ")
        atomic_json(staged_meta_path, {task_name: metainfo[task_name]})
        atomic_copy(work_h5, target_h5)
        # Merge committed metadata with other task entries from an existing output.
        if metainfo_out.exists():
            published_meta = json.loads(metainfo_out.read_text(encoding="utf-8"))
            published_meta[task_name] = metainfo[task_name]
        else:
            published_meta = metainfo
        atomic_json(metainfo_out, published_meta)

        state.update(status="complete", selection_complete=len(candidates) == len(all_candidates),
                     selected_episodes=len(candidates), expected_episodes=len(env.episodes),
                     saved_episodes=sum(bool(value.get("saved")) for value in state["episodes"].values()),
                     completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(run_state_path, state)
        return dict(
            status=state["status"], selection_complete=state["selection_complete"],
            config=config, config_sha256=run_id,
            output_hdf5=source_name, output_hdf5_sha256=sha256_file(target_h5),
            output_metainfo_sha256=sha256_file(metainfo_out),
            episodes=state["episodes"], selected_episodes=state["selected_episodes"],
            expected_episodes=state["expected_episodes"], saved_episodes=state["saved_episodes"],
            succeeded=sum(1 for value in state["episodes"].values() if value.get("success")),
            generator_code_sha256=generator_hash, code_sha256=code_hashes,
            runtime=runtime, runtime_sha256=content_digest(runtime),
        )
    except BaseException as exc:
        if "run_state_path" in locals() and run_state_path.exists():
            state["status"] = "failed"
            state["error"] = str(exc)
            atomic_json(run_state_path, state)
        raise
    finally:
        env.close()


def main():
    args = parse_args()
    parent_lock = load_split_lock(args.split_lock)
    aliases = {spec["name"]: spec["alias"] for spec in load_protocol()["tasks"]}
    required_runs = {f"{aliases[row['task']]}/{row['split']}" for row in parent_lock["episodes"]
                     if row["domain"] == "mem"}
    if any(f"{task}/{args.split}" not in required_runs for task in args.tasks):
        raise ValueError("Requested split is absent from the refined replay roster")
    runtime = runtime_provenance()
    code_hashes = code_provenance()
    generator_hash = code_hashes["scripts/generate_derived_demonstrations.py"]
    manifest = load_root_manifest(args.output_dir, parent_lock, generator_hash, runtime, code_hashes)
    for task in args.tasks:
        run_key = f"{task}/{args.split}"
        config = make_run_config(task, args, parent_lock, runtime, code_hashes)
        existing_run = manifest["runs"].get(run_key)
        if existing_run and existing_run.get("config") != config:
            raise ValueError("This task/split already has another replay config; choose a fresh --output-dir")
        if existing_run and existing_run.get("status") == "complete":
            print(f"[{run_key}] already complete with identical source/code/config; skipping")
            continue

        manifest["status"] = "preparing"
        manifest["runs"][run_key] = dict(status="running", config=config, started_at=datetime.now(timezone.utc).isoformat())
        manifest = sealed_manifest(manifest)
        atomic_json(args.output_dir / "manifest.json", manifest)
        try:
            result = run_task(task, args, parent_lock, generator_hash, runtime, code_hashes, config)
            result["config"] = config
            manifest["runs"][run_key] = result
        except BaseException as exc:
            manifest["runs"][run_key].update(status="failed", error=str(exc),
                failed_at=datetime.now(timezone.utc).isoformat())
            manifest = sealed_manifest(manifest)
            atomic_json(args.output_dir / "manifest.json", manifest)
            raise
        manifest["artifacts"] = output_artifacts(args.output_dir)
        manifest["status"] = "complete" if (
            set(manifest["runs"]) == required_runs and
            all(run.get("status") == "complete" and run.get("selection_complete")
                for run in manifest["runs"].values())) else "partial"
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest = sealed_manifest(manifest)
        atomic_json(args.output_dir / "manifest.json", manifest)
        task_name = task_spec(load_protocol(), task)["name"]
        summary = dict(task=task, task_name=task_name, split=args.split,
                       replayed=len(result["episodes"]), saved=result["saved_episodes"],
                       successful=result["succeeded"], output_sha256=result["output_hdf5_sha256"],
                       episodes=result["episodes"], timestamp=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output_dir / f"{task}_{args.split}_summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("task", "split", "replayed", "saved", "successful")}), flush=True)
    print(f"Derived replay inventory status: {manifest['status']}", flush=True)


if __name__ == "__main__":
    main()
