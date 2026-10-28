#!/usr/bin/env python
"""Lock actual T1/T3 and selected base episodes before preparing or evaluating."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import assign_locked_groups, seal_lock, state_digest, validate_lock
from src.training.libero_mem_hdf5_loader import read_episode_metadata
from src.utils.file_hash import sha256_file
from src.evaluation.libero_mem_protocol import load_protocol, DEFAULT_PROTOCOL, source_digest


def build_lock(mem_root, base_root, base_inventory, base_tasks=None, seed=42):
    import h5py
    protocol = load_protocol()
    source = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))
    hashes = {r["path"]: r["upstream_sha256"] for r in source["annotated_dataset"]["selected_pilot_files"]}
    sources, episodes = [], []
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
                episodes.append(dict(domain="mem", suite="libero_mem", task=task, demo=key,
                    source=task + "_demo.hdf5", initial_state=state, initial_state_sha256=state_digest(state),
                    used_for_writer_development=int(key.split("_")[-1]) in (1, 5, 31),
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
                episodes.append(dict(domain="base", suite=record["task_suite"], task=record["task_name"], demo=key,
                    source=relative, initial_state_sha256=state_digest(state),
                    frames=len(f["data"][key]["actions"]) - 1))
    rows = assign_locked_groups(episodes, seed)
    counts = Counter((r["domain"], r["task"], r["split"]) for r in rows)
    for domain, task in {(r["domain"], r["task"]) for r in rows}:
        if any(counts[domain, task, s] == 0 for s in ("train", "val", "test")):
            raise ValueError("Split missing a task; inspect the source groups before proceeding")
    result = seal_lock(dict(schema_version=1, classification="CUSTOM_HELD_OUT_INITIAL_STATES",
        seed=seed, fractions=dict(train=.8, val=.1, test=.1),
        grouping="SHA256 canonical little-endian float64 complete initial-state vector, globally across tasks/domains",
        writer_development_note="T1/T3 demos 1,5,31 and every episode sharing those states forced to train",
        pretraining_note="Disjoint for this adaptation; cannot certify absence from original model pretraining",
        mem_revision=source["annotated_dataset"]["revision"],
        base_revision=inventory["benchmark_provenance"]["pinned_commit"],
        protocol_sha256=source_digest(DEFAULT_PROTOCOL), sources=sources, episodes=rows))
    return validate_lock(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mem-root", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--base-inventory", type=Path, default=ROOT / "resources/manifests/libero_10_sft_sources.json")
    p.add_argument("--base-tasks", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Split locks are immutable; choose a new version/path")
    lock = build_lock(a.mem_root, a.base_root, a.base_inventory, a.base_tasks, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(lock_sha256=lock["lock_sha256"], episodes=len(lock["episodes"]),
        groups=len({r["initial_state_sha256"] for r in lock["episodes"]}),
        counts=dict(Counter(r["domain"] + "/" + r["split"] for r in lock["episodes"]))), indent=2))


if __name__ == "__main__":
    main()
