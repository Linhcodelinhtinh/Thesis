"""Immutable episode/initial-state assignments shared by preparation and evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

SPLITS = ("train", "val", "test")
AUTHOR_MEM_SPLIT = "libero_mem_refined_train_raw_eval_v2"
AUTHOR_EVAL_SPLITS = ("seen", "unseen")
DEFAULT_AUTHOR_MEM_LOCK = Path(__file__).resolve().parents[2] / "resources/manifests/libero_mem_t1_t3_base10_split_v2.json"


def state_digest(state):
    state = np.asarray(state, dtype="<f8")
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
        raise ValueError("Initial state must be a nonempty finite vector")
    return hashlib.sha256(state.tobytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def seal_lock(payload):
    result = dict(payload)
    result["lock_sha256"] = content_digest(payload)
    return result


def validate_lock(lock):
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if content_digest(payload) != lock.get("lock_sha256"):
        raise ValueError("Split lock content/hash mismatch")
    seen, groups = set(), {}
    for row in lock["episodes"]:
        key = (row["domain"], row["task"], row["demo"])
        if key in seen or row["split"] not in SPLITS:
            raise ValueError("Duplicate episode or invalid split")
        seen.add(key)
        group = row["initial_state_sha256"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Initial-state leakage across splits")
        groups[group] = row["split"]
        if "initial_state" in row and state_digest(row["initial_state"]) != group:
            raise ValueError("Locked initial-state bytes/hash mismatch")
        if row.get("used_for_writer_development") and row["split"] != "train":
            raise ValueError("Writer development state cannot be held out")
    if not seen:
        raise ValueError("Empty split lock")
    if lock.get("mem_split_policy") == AUTHOR_MEM_SPLIT:
        _validate_author_mem_split(lock, groups)
    return lock


def _validate_author_mem_split(lock, groups):
    mem = [row for row in lock["episodes"] if row["domain"] == "mem"]
    tasks = {row["task"] for row in mem}
    raw_sources = {row["path"]: row["sha256"] for row in lock["raw_sources"]}
    if (not tasks or set(raw_sources) != {task + "_demo.hdf5" for task in tasks}
            or not lock["raw_revision"]):
        raise ValueError("Incomplete LIBERO-Mem raw source inventory")
    for row in mem:
        index = row["raw_index"]
        if (row["split"] != "train" or type(index) is not int or not 0 <= index < 100
                or row["demo"] != f"demo_{index + 1}"):
            raise ValueError("Author-refined Mem rows must be train-only and map to raw indices 0..99")
    rosters = lock["evaluation_rosters"]
    if set(rosters) != set(AUTHOR_EVAL_SPLITS):
        raise ValueError("Expected exact seen and unseen raw rosters")
    mem_states = {(row["task"], row["raw_index"]): row["initial_state_sha256"] for row in mem}
    seen_groups = set()
    for split, start in (("seen", 0), ("unseen", 100)):
        rows = rosters[split]
        if len(rows) != 20 * len(tasks) or {row["task"] for row in rows} != tasks:
            raise ValueError(f"Incomplete LIBERO-Mem raw {split} roster")
        for task in tasks:
            task_rows = [row for row in rows if row["task"] == task]
            if [row["raw_index"] for row in task_rows] != list(range(start, start + 20)):
                raise ValueError(f"Unexpected raw {split} indices for {task}")
            state_groups = set()
            for row in task_rows:
                index = row["raw_index"]
                group = row["initial_state_sha256"]
                if (row["domain"] != "mem" or row["source"] != task + "_demo.hdf5"
                        or row["demo"] != f"demo_{index + 1}"
                        or state_digest(row["initial_state"]) != group or group in state_groups):
                    raise ValueError(f"Invalid or duplicate raw {split} initial state")
                state_groups.add(group)
                if split == "seen":
                    if mem_states.get((task, index), group) != group:
                        raise ValueError("Seen raw state differs from refined training state")
                    seen_groups.add(group)
                elif group in groups or group in seen_groups:
                    raise ValueError("Raw unseen initial-state leakage")


def mem_episode_rows(lock, task, split):
    rows = lock["evaluation_rosters"][split] if split in AUTHOR_EVAL_SPLITS and (
        lock.get("mem_split_policy") == AUTHOR_MEM_SPLIT
    ) else [row for row in lock["episodes"] if row["split"] == split]
    return [row for row in rows if row["domain"] == "mem" and row["task"] == task]


def evaluation_lock_sha256(lock):
    return lock.get("parent_lock_sha256", lock["lock_sha256"])


def load_split_lock(path):
    return validate_lock(json.loads(Path(path).read_text(encoding="utf-8")))


def assign_locked_groups(episodes, seed=42, val_fraction=.1, test_fraction=.1):
    """Assign globally: identical states never split by task/domain or frame.

    Stratify connected task families. A task's states are unioned only for the
    purpose of choosing a balanced split, not merged into a single state group.
    Writer development states are forced to train before held-out allocation.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid held-out fractions")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    first = {}
    for e in episodes:
        task = e["domain"] + "/" + e["task"]
        state = e["initial_state_sha256"]
        if state in first:
            parent[find(task)] = find(first[state])
        else:
            first[state] = task
            find(task)
    families, forced = {}, {e["initial_state_sha256"] for e in episodes if e.get("used_for_writer_development")}
    for e in episodes:
        families.setdefault(find(e["domain"] + "/" + e["task"]), set()).add(e["initial_state_sha256"])
    assignments = {}
    for family in families.values():
        candidates = sorted(family - forced, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        nv, nt = max(1, round(len(family) * val_fraction)), max(1, round(len(family) * test_fraction))
        if len(candidates) < nv + nt or len(family) <= nv + nt:
            raise ValueError("Not enough unseen initial states for train/val/test")
        assignments.update({g: "train" for g in family})
        assignments.update({g: "val" for g in candidates[:nv]})
        assignments.update({g: "test" for g in candidates[nv:nv + nt]})
    return [dict(e, split=assignments[e["initial_state_sha256"]]) for e in episodes]


def locked_episode(lock, domain, task, demo, state=None):
    matches = [e for e in lock["episodes"] if (e["domain"], e["task"], e["demo"]) == (domain, task, demo)]
    if len(matches) != 1:
        raise ValueError(f"Episode is not uniquely present in split lock: {domain}/{task}/{demo}")
    row = matches[0]
    if state is not None and state_digest(state) != row["initial_state_sha256"]:
        raise ValueError("Source initial state changed since split was locked")
    return row
