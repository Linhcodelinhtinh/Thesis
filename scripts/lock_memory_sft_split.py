#!/usr/bin/env python
"""Lock all released refined T1/T3 demos and the author's raw evaluation rosters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import AUTHOR_MEM_SPLIT, assign_locked_groups, load_split_lock, seal_lock, state_digest, validate_lock
from src.training.libero_mem_hdf5_loader import read_episode_metadata
from src.utils.file_hash import sha256_file
from src.evaluation.libero_mem_protocol import load_protocol, DEFAULT_PROTOCOL, source_digest


RAW_ROOT = ROOT / "data/libero_mem_raw"
PREVIOUS_LOCK = ROOT / "resources/manifests/libero_mem_t1_t3_base10_split_v1.json"


def build_lock(mem_root, base_root, base_inventory, base_tasks=None, seed=42, *,
               raw_root=RAW_ROOT, previous_lock=PREVIOUS_LOCK):
    import h5py
    protocol = load_protocol()
    source = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))
    hashes = {r["path"]: r["upstream_sha256"] for r in source["annotated_dataset"]["selected_pilot_files"]}
    raw_dataset = source["raw_dataset"]
    raw_sources, rosters = [], {"seen": [], "unseen": []}
    raw_by_task = {}
    for record in raw_dataset["selected_pilot_files"]:
        name = record["path"]
        task = name[:-len("_demo.hdf5")]
        path = raw_root / name
        if sha256_file(path) != record["upstream_sha256"]:
            raise ValueError("Raw Mem source bytes differ from author release: " + name)
        raw_sources.append(dict(path=name, sha256=record["upstream_sha256"]))
        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = sorted(data, key=lambda key: int(key.split("_")[-1]))
            if keys != [f"demo_{index + 1}" for index in range(120)]:
                raise ValueError("Unexpected raw Mem roster: " + task)
            rows = []
            for index, key in enumerate(keys):
                group = data[key]
                state = group["init_state"][()] if "init_state" in group else group["states"][0]
                rows.append(dict(domain="mem", suite="libero_mem", task=task, demo=key,
                    source=name, raw_index=index, initial_state=state.tolist(),
                    initial_state_sha256=state_digest(state)))
            if len({row["initial_state_sha256"] for row in rows}) != 120:
                raise ValueError("Duplicate raw initial state within task: " + task)
            raw_by_task[task] = rows
            rosters["seen"].extend(rows[:20])
            rosters["unseen"].extend(rows[100:])
    raw_train_states = {row["initial_state_sha256"] for rows in raw_by_task.values() for row in rows[:100]}
    if any(row["initial_state_sha256"] in raw_train_states for row in rosters["unseen"]):
        raise ValueError("Raw unseen state also appears in the author's raw training prefix")
    sources, mem_rows, base_rows = [], [], []
    for name, expected in hashes.items():
        if sha256_file(mem_root / name) != expected:
            raise ValueError("Mem source bytes differ from author release: " + name)
        sources.append(dict(domain="mem", path=name, sha256=expected))
    for spec in protocol["tasks"]:
        task = spec["name"]
        with h5py.File(mem_root / (task + "_demo.hdf5"), "r") as f:
            keys = sorted(f["data"], key=lambda s: int(s.split("_")[-1]))
            meta = read_episode_metadata(mem_root / "metainfo.json", task, keys)
            for key in keys:
                state = meta[key]["initial_state"]
                index = int(key.split("_")[-1]) - 1
                digest = state_digest(state)
                if (not 0 <= index < 100 or key != raw_by_task[task][index]["demo"]
                        or digest != raw_by_task[task][index]["initial_state_sha256"]):
                    raise ValueError(f"Refined state does not match its raw training index: {task}/{key}")
                mem_rows.append(dict(domain="mem", suite="libero_mem", task=task, demo=key,
                    source=task + "_demo.hdf5", initial_state=state, initial_state_sha256=digest,
                    raw_index=index, split="train", used_for_writer_development=index + 1 in (1, 5, 31),
                    frames=len(f["data"][key]["actions"])))
    inventory = json.loads(base_inventory.read_text(encoding="utf-8"))
    selected = inventory["demonstrations"]
    if base_tasks:
        selected = [r for r in selected if r["task_name"] in base_tasks]
        if {r["task_name"] for r in selected} != set(base_tasks):
            raise ValueError("Requested base task absent from source lock")
    if not selected:
        raise ValueError("Base replay inventory is empty")
    for record in selected:
        relative = record["task_suite"] + "/" + record["task_name"] + "_demo.hdf5"
        path = base_root / relative
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Base source bytes differ from author release: " + relative)
        sources.append(dict(domain="base", path=relative, sha256=record["sha256"]))
        with h5py.File(path, "r") as f:
            for key in sorted(f["data"]):
                state = f["data"][key].attrs["init_state"]
                base_rows.append(dict(domain="base", suite=record["task_suite"], task=record["task_name"], demo=key,
                    source=relative, initial_state_sha256=state_digest(state),
                    frames=len(f["data"][key]["actions"]) - 1))
    previous = load_split_lock(previous_lock) if previous_lock is not None else None
    base_revision = inventory["benchmark_provenance"]["pinned_commit"]
    if previous and previous["base_revision"] != base_revision:
        raise ValueError("Previous base assignments use another source revision")
    base_assignments = {
        row["initial_state_sha256"]: row["split"]
        for row in (previous["episodes"] if previous else ()) if row["domain"] == "base"
    }
    base_rows = assign_locked_groups(base_rows, seed)
    for row in base_rows:
        row["split"] = base_assignments.get(row["initial_state_sha256"], row["split"])
    counts = Counter((row["task"], row["split"]) for row in base_rows)
    for task in {row["task"] for row in base_rows}:
        if any(counts[task, split] == 0 for split in ("train", "val", "test")):
            raise ValueError("Base split missing a partition: " + task)
    result = seal_lock(dict(schema_version=2, classification="CUSTOM_V2_LIBERO_MEM_AUTHOR_STATE_ROSTERS",
        mem_split_policy=AUTHOR_MEM_SPLIT,
        seed=seed, fractions=dict(train=.8, val=.1, test=.1),
        grouping="Exact float64 initial-state hashes across tasks/domains; fractions apply to base replay only",
        previous_base_lock_sha256=previous["lock_sha256"] if previous else None,
        pretraining_note="Disjoint for this adaptation; cannot certify absence from original model pretraining",
        mem_revision=source["annotated_dataset"]["revision"],
        raw_revision=raw_dataset["revision"], raw_sources=raw_sources, evaluation_rosters=rosters,
        base_revision=base_revision,
        protocol_sha256=source_digest(DEFAULT_PROTOCOL), sources=sources, episodes=mem_rows + base_rows))
    return validate_lock(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mem-root", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--base-inventory", type=Path, default=ROOT / "resources/manifests/libero_10_sft_sources.json")
    p.add_argument("--base-tasks", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--previous-lock", type=Path, default=PREVIOUS_LOCK, help="Preserve previous base assignments only")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Split locks are immutable; choose a new version/path")
    lock = build_lock(a.mem_root, a.base_root, a.base_inventory, a.base_tasks, a.seed,
                      raw_root=a.raw_root, previous_lock=a.previous_lock)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(dict(lock_sha256=lock["lock_sha256"], episodes=len(lock["episodes"]),
        groups=len({r["initial_state_sha256"] for r in lock["episodes"]}),
        counts=dict(Counter(r["domain"] + "/" + r["split"] for r in lock["episodes"])),
        evaluation_counts={split: len(rows) for split, rows in lock["evaluation_rosters"].items()}), indent=2))


if __name__ == "__main__":
    main()
