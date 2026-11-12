#!/usr/bin/env python
"""Seal a training-data lock from complete, provenance-checked replays.

Initial-state assignments are inherited from the parent lock. Only complete,
successful derived demonstrations enter the new lock. All failures remain in
the manifest as exclusions; the parent evaluation lock is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, load_protocol, source_digest
from src.training.memory_data import validate_episode
from src.training.split_lock import content_digest, load_split_lock, seal_lock, state_digest, validate_lock
from src.utils.file_hash import sha256_file


def demo_number(value):
    return int(value[len("demo_"):]) if value.startswith("demo_") else int(value)


def build_derived_lock(parent_lock_path, derived_dir, output, source_dir):
    parent = load_split_lock(parent_lock_path)
    derived_dir = Path(derived_dir).resolve()
    source_dir = Path(source_dir).resolve()
    manifest_path = derived_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact") != "libero_mem_derived_demonstrations" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a versioned derived demonstration manifest")
    if manifest.get("status") != "complete":
        raise ValueError("All task/split replays must be complete before locking derived data")
    if manifest.get("parent_split_lock_sha256") != parent["lock_sha256"]:
        raise ValueError("Derived replay was generated from a different parent split lock")
    if manifest.get("evaluator_protocol_sha256") != source_digest(DEFAULT_PROTOCOL):
        raise ValueError("Derived replay evaluator protocol differs from the current pinned protocol")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != content_digest(manifest_payload):
        raise ValueError("Derived manifest hash does not match the recorded replay inventory")
    if manifest.get("generator_code_sha256") != manifest.get("code_sha256", {}).get(
            "scripts/generate_derived_demonstrations.py"):
        raise ValueError("Manifest generator code identity is inconsistent")
    runtime = manifest.get("runtime", {})
    expected_runtime = load_protocol()["runtime"]
    if (runtime.get("packages") != {name: expected_runtime[name] for name in
            ("robosuite", "numpy", "bddl", "mujoco", "robomimic")} or
            not runtime.get("python", "").startswith(expected_runtime["python"].split(".x", 1)[0] + ".")):
        raise ValueError("Derived manifest does not record the pinned simulator runtime")

    protocol_tasks = {task["alias"]: task["name"] for task in load_protocol()["tasks"]}
    expected_runs = {f"{task}/{split}" for task in protocol_tasks for split in ("train", "val", "test")}
    runs = manifest.get("runs", {})
    if set(runs) != expected_runs:
        raise ValueError(f"Derived replay inventory is incomplete: missing={sorted(expected_runs-set(runs))}")

    expected_artifacts = {name + "_demo.hdf5" for name in protocol_tasks.values()}
    expected_artifacts.add("metainfo.json")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != expected_artifacts:
        raise ValueError("Manifest does not contain the exact final artifact inventory")
    for name, digest in artifacts.items():
        path = derived_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Derived artifact differs from its sealed manifest hash: " + name)

    parent_mem = [row for row in parent["episodes"] if row["domain"] == "mem"]
    parent_by_key = {(row["task"], row["demo"]): row for row in parent_mem}
    parent_source_hashes = {row["path"]: row["sha256"] for row in parent["sources"] if row["domain"] == "mem"}
    if manifest.get("parent_mem_sources") != [row for row in parent["sources"] if row["domain"] == "mem"]:
        raise ValueError("Derived manifest source roster differs from the parent lock")
    original_metainfo = source_dir / "metainfo.json"
    if parent_source_hashes.get("metainfo.json") is None or sha256_file(original_metainfo) != parent_source_hashes["metainfo.json"]:
        raise ValueError("Original Mem metainfo is not the file pinned by the parent lock")
    excluded, saved_by_task = [], {name: set() for name in protocol_tasks.values()}
    for run_key, run in runs.items():
        task_alias, split = run_key.split("/", 1)
        task_name = protocol_tasks[task_alias]
        expected_rows = sorted(
            (row for row in parent_mem if row["task"] == task_name and row["split"] == split),
            key=lambda row: demo_number(row["demo"]),
        )
        if not expected_rows:
            raise ValueError("Parent lock has no roster for " + run_key)
        expected_ids = [row["demo"] for row in expected_rows]
        if run.get("status") != "complete" or not run.get("selection_complete"):
            raise ValueError("Replay is incomplete or used a truncated selection: " + run_key)
        config = run.get("config", {})
        if (config.get("task") != task_alias or config.get("task_name") != task_name or
                config.get("split") != split or config.get("parent_split_lock_sha256") != parent["lock_sha256"] or
                config.get("selected_demo_ids") != expected_ids or config.get("max_episodes") is not None or
                config.get("require_success") is not True or
                config.get("evaluator_protocol_sha256") != source_digest(DEFAULT_PROTOCOL) or
                config.get("generator_code_sha256") != manifest["generator_code_sha256"] or
                config.get("code_sha256") != manifest["code_sha256"] or
                config.get("runtime_sha256") != content_digest(manifest["runtime"])):
            raise ValueError("Replay configuration does not cover the exact parent roster: " + run_key)
        expected_source = expected_rows[0]["source"]
        if (config.get("source_hdf5_sha256") != parent_source_hashes.get(expected_source) or
                config.get("source_metainfo_sha256") != parent_source_hashes["metainfo.json"]):
            raise ValueError("Replay input source hashes differ from the parent lock: " + run_key)
        if run.get("runtime") != manifest.get("runtime"):
            raise ValueError("Run runtime differs from the root manifest: " + run_key)
        replays = run.get("episodes", {})
        if set(replays) != set(expected_ids):
            raise ValueError("Per-episode replay records do not match the parent roster: " + run_key)
        for row in expected_rows:
            replay = replays[row["demo"]]
            if (not replay.get("attempt_complete") or replay.get("source_steps") != row["frames"] or
                    replay.get("initial_state_sha256") != row["initial_state_sha256"]):
                raise ValueError(f"Replay record is incomplete or mismatched: {run_key}/{row['demo']}")
            if replay.get("saved"):
                if (not replay.get("success") or not replay.get("trajectory_complete") or
                        replay.get("steps") != row["frames"]):
                    raise ValueError(f"Unsuccessful/incomplete replay was marked trainable: {run_key}/{row['demo']}")
                saved_by_task[task_name].add(row["demo"])
            else:
                excluded.append(f"{run_key}/{row['demo']}: {replay.get('terminal_reason', 'replay excluded')}")

    meta_path = derived_dir / "metainfo.json"
    metainfo = json.loads(meta_path.read_text(encoding="utf-8"))
    mem_sources, derived_rows = [], []
    for task_alias, task_name in protocol_tasks.items():
        task_rows = [row for row in parent_mem if row["task"] == task_name]
        if not task_rows:
            raise ValueError("Parent lock has no rows for " + task_alias)
        source = task_rows[0]["source"]
        if any(row["source"] != source for row in task_rows):
            raise ValueError("Parent lock maps one task to multiple source files: " + task_alias)
        original_path = source_dir / source
        if parent_source_hashes.get(source) is None or sha256_file(original_path) != parent_source_hashes[source]:
            raise ValueError("Original source HDF5 is not the file pinned by the parent lock: " + source)
        h5_path = derived_dir / source
        if not h5_path.is_file():
            raise FileNotFoundError("Missing derived HDF5: " + str(h5_path))
        if sha256_file(h5_path) != artifacts[source]:
            raise ValueError("Derived HDF5 does not match the sealed artifact inventory: " + source)
        mem_sources.append(dict(domain="mem", path=source, sha256=artifacts[source], image_convention="raw_opengl"))

        with h5py.File(original_path, "r") as source_h5, h5py.File(h5_path, "r") as h5:
            if h5.attrs.get("derived_protocol_id") != "libero_mem_derived_data_v1":
                raise ValueError("HDF5 is missing the derived data protocol marker: " + source)
            if h5.attrs.get("source_sha256") != parent_source_hashes[source]:
                raise ValueError("HDF5 parent source hash differs from the parent lock: " + source)
            if h5.attrs.get("parent_split_lock_sha256") != parent["lock_sha256"]:
                raise ValueError("HDF5 parent split lock marker differs: " + source)
            if h5.attrs.get("evaluator_protocol_sha256") != source_digest(DEFAULT_PROTOCOL):
                raise ValueError("HDF5 evaluator protocol marker differs: " + source)
            data = h5["data"]
            if data.attrs.get("macros_image_convention") != "raw_opengl":
                raise ValueError("HDF5 image convention is not raw_opengl: " + source)
            actual_demos = set(data.keys())
            if actual_demos != saved_by_task[task_name]:
                raise ValueError(f"Derived HDF5 inventory does not match successful replays: {task_alias}")
            task_metadata = metainfo.get(task_name, {})
            if set(task_metadata) != actual_demos:
                raise ValueError(f"Derived metainfo inventory differs from HDF5: {task_alias}")
            allowed = {row["demo"] for row in task_rows}
            if not actual_demos <= allowed:
                raise ValueError("Derived HDF5 contains demos outside parent split lock")

            for demo in sorted(actual_demos, key=demo_number):
                parent_row = parent_by_key[(task_name, demo)]
                group = data[demo]
                actions, robot, _ = validate_episode(group)
                source_actions = np.asarray(source_h5[f"data/{demo}/actions"][:], dtype=np.float32)
                record = task_metadata[demo]
                expected_state_hash = parent_row["initial_state_sha256"]
                if (not np.array_equal(actions, source_actions) or len(actions) != parent_row["frames"] or
                        len(group["rewards"]) != len(actions) or len(group["dones"]) != len(actions) or
                        group.attrs.get("source_actions_sha256") != hashlib.sha256(actions.tobytes()).hexdigest()):
                    raise ValueError(f"Derived action/reward inventory differs from its pinned source: {task_alias}/{demo}")
                if (state_digest(record["initial_state"]) != expected_state_hash or
                        state_digest(group.attrs["init_state"]) != expected_state_hash or
                        record.get("initial_state_sha256") != expected_state_hash):
                    raise ValueError(f"Derived initial state differs from parent lock: {task_alias}/{demo}")
                if not record.get("task_description") or not record.get("success") or not group.attrs.get("success", False):
                    raise ValueError(f"Derived training replay is not a successful labeled episode: {task_alias}/{demo}")
                if group.attrs.get("derived_protocol_id") != "libero_mem_derived_data_v1":
                    raise ValueError("Episode is missing its derived protocol marker: " + demo)
                derived_row = {key: parent_row[key] for key in (
                    "domain", "suite", "task", "demo", "source", "initial_state",
                    "initial_state_sha256", "split", "frames") if key in parent_row}
                derived_row.update(task_alias=task_alias, success=True, image_convention="raw_opengl")
                derived_rows.append(derived_row)

    mem_sources.append(dict(domain="mem", path="metainfo.json", sha256=artifacts["metainfo.json"],
                            image_convention="raw_opengl"))
    for task_alias in protocol_tasks:
        if not any(row.get("task_alias") == task_alias and row["split"] == "train" for row in derived_rows):
            raise ValueError(f"No successful derived training demonstrations remain for {task_alias}")
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in derived_rows):
            raise ValueError(f"No successful derived demonstrations remain in the Mem {split} split")

    parent_base = [row for row in parent["episodes"] if row["domain"] == "base"]
    base_sources = [row for row in parent["sources"] if row["domain"] == "base"]
    original_mem_sources = [row for row in parent["sources"] if row["domain"] == "mem"]
    payload = dict(
        schema_version=2,
        classification="CUSTOM_V2_LIBERO_MEM_DERIVED_TRAINING_DATA",
        parent_lock_sha256=parent["lock_sha256"],
        derived_manifest_sha256=manifest["manifest_sha256"],
        derived_manifest_file_sha256=sha256_file(manifest_path),
        generator_code_sha256=manifest["generator_code_sha256"],
        code_sha256=manifest["code_sha256"], runtime=manifest["runtime"],
        evaluator_protocol_sha256=source_digest(DEFAULT_PROTOCOL),
        mem_image_convention="raw_opengl", base_image_convention="raw_opengl",
        parent_mem_sources=original_mem_sources, excluded_replays=excluded,
        seed=parent["seed"], fractions=parent["fractions"], grouping=parent["grouping"],
        mem_revision=parent.get("mem_revision"), base_revision=parent.get("base_revision"),
        protocol_sha256=parent["protocol_sha256"],
        sources=base_sources + mem_sources,
        episodes=parent_base + derived_rows,
    )
    lock = validate_lock(seal_lock(payload))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Split locks are immutable; select a new output path")
    lock = build_derived_lock(args.parent_lock, args.derived_dir, args.output, args.source_dir)
    counts = {f"{domain}/{split}": sum(1 for item in lock["episodes"]
              if item["domain"] == domain and item["split"] == split)
              for domain in ("base", "mem") for split in ("train", "val", "test")}
    print(json.dumps(dict(lock_sha256=lock["lock_sha256"], episodes=len(lock["episodes"]),
                          mem_success_episodes=sum(1 for row in lock["episodes"] if row["domain"] == "mem"),
                          exclusions=len(lock["excluded_replays"]), counts=counts), indent=2))


if __name__ == "__main__":
    main()
