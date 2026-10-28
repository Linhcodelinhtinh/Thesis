"""Immutable episode/initial-state assignments shared by preparation and evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

SPLITS = ("train", "val", "test")


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
    return lock


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
