"""Validate a raw B0 unseen run and pair it with the locked E2 F0/F1 cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.evaluation.mem_eval_io import evaluation_roster, write_json
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.mean(values) if values else None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def validate_artifacts(directory: Path, status: dict):
    artifacts = status.get("files_sha256", {})
    required = {"summary.json", "actions.jsonl", "trajectory.npz"}
    if not required.issubset(artifacts):
        raise ValueError(f"Required artifact hash missing: {directory}")
    for name, digest in artifacts.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Artifact hash mismatch: {path}")


def episode_result(directory: Path, row: dict, condition: str, seed: int):
    status = read_json(directory / "status.json")
    expected_identity = (row["task"], row["state_idx"], seed)
    actual_identity = (status.get("task"), status.get("state_idx"), status.get("policy_seed"))
    if status.get("status") != "complete" or actual_identity != expected_identity:
        raise ValueError(f"Incomplete or mismatched episode: {directory}")
    if (
        status.get("condition") != condition
        or status.get("split") != "unseen"
        or status.get("initial_state_sha256") != row["initial_state_sha256"]
        or status.get("source_demo") != row["source_demo"]
    ):
        raise ValueError(f"Episode metadata differs from locked roster: {directory}")
    validate_artifacts(directory, status)
    summary = read_json(directory / "summary.json")
    result = summary["result"]
    if (
        summary.get("task") != row["task"]
        or summary.get("state_idx") != row["state_idx"]
        or summary.get("policy_seed") != seed
        or summary.get("condition") != condition
        or summary.get("initial_state_sha256") != row["initial_state_sha256"]
        or type(result.get("success")) is not bool
    ):
        raise ValueError(f"Summary identity differs from locked roster: {directory}")
    return {
        "task": row["task"],
        "state_idx": int(row["state_idx"]),
        "initial_state_sha256": row["initial_state_sha256"],
        "condition": condition,
        "success": int(result["success"]),
        "scr": float(result["scr"]),
        "completed_subgoals": int(result["completed_subgoals"]),
        "total_subgoals": int(result["total_subgoals"]),
        "overshot": int(bool(result["overshot"])),
        "steps": int(result["total_steps"]),
        "wall_time_s": float(result["total_wall_time_s"]),
        "artifact_sha256": status["files_sha256"],
        "status_sha256": sha256_file(directory / "status.json"),
    }


def run_identity(manifest: dict):
    return {
        "base": manifest.get("base"),
        "local_assets_lock_sha256": manifest.get("local_assets_lock_sha256"),
        "environments": manifest.get("environments"),
        "cuda": manifest.get("cuda"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "split_lock_sha256": manifest.get("split_lock_sha256"),
    }


def load_b0(run_dir: Path, split_lock: Path, tasks: list[str], states: str, seed: int):
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    args = manifest["arguments"]
    _, lock, roster = evaluation_roster(split_lock, "unseen", tasks, states)
    if (
        manifest.get("status") != "complete"
        or manifest.get("classification") != "CUSTOM_V2_LIBERO_MEM"
        or args.get("split") != "unseen"
        or args.get("tasks") != tasks
        or args.get("conditions") != ["B0"]
        or str(args.get("states")) != states
        or args.get("policy_seed") != seed
        or args.get("preflight_only")
        or args.get("runtime_check")
        or manifest.get("expected_episodes") != len(roster)
        or manifest.get("completed_episodes") != len(roster)
        or manifest.get("split_lock_sha256") != lock["lock_sha256"]
        or manifest.get("protocol_sha256") != source_digest(DEFAULT_PROTOCOL)
        or manifest.get("roster") != roster
        or "B0" not in manifest.get("loaded_policies", {})
    ):
        raise ValueError(f"B0 run differs from locked unseen campaign: {run_dir}")
    episodes = {}
    for row in roster:
        key = (row["task"], int(row["state_idx"]))
        directory = run_dir / "episodes" / f"{row['task']}_B0_unseen_state{row['state_idx']}"
        episodes[key] = episode_result(directory, row, "B0", seed)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "identity": run_identity(manifest),
        "episodes": episodes,
    }


def load_e2(e2_root: Path, split_lock: Path, tasks: list[str], states: str, seed: int):
    _, lock, roster = evaluation_roster(split_lock, "unseen", tasks, states)
    planned = {(row["task"], int(row["state_idx"])): row for row in roster}
    episodes = {}
    manifests = []
    for manifest_path in sorted(e2_root.glob("batch-*/run_manifest.json")):
        manifest = read_json(manifest_path)
        args = manifest["arguments"]
        if (
            manifest.get("status") != "complete"
            or manifest.get("classification") != "CUSTOM_V2_LIBERO_MEM"
            or args.get("split") != "unseen"
            or args.get("tasks") != tasks
            or args.get("conditions") != ["F0", "F1"]
            or args.get("policy_seed") != seed
            or args.get("preflight_only")
            or args.get("runtime_check")
            or manifest.get("split_lock_sha256") != lock["lock_sha256"]
            or manifest.get("protocol_sha256") != source_digest(DEFAULT_PROTOCOL)
        ):
            raise ValueError(f"E2 run differs from the declared campaign: {manifest_path.parent}")
        expected_rows = evaluation_roster(split_lock, "unseen", tasks, str(args.get("states")))[2]
        if (
            manifest.get("roster") != expected_rows
            or manifest.get("expected_episodes") != len(expected_rows) * 2
            or manifest.get("completed_episodes") != manifest.get("expected_episodes")
        ):
            raise ValueError(f"E2 batch roster/count differs: {manifest_path.parent}")
        for row in expected_rows:
            key = (row["task"], int(row["state_idx"]))
            if key not in planned or planned[key] != row:
                raise ValueError(f"E2 batch row differs from locked roster: {key}")
            for condition in ("F0", "F1"):
                directory = manifest_path.parent / "episodes" / (
                    f"{row['task']}_{condition}_unseen_state{row['state_idx']}"
                )
                unit = (condition, *key)
                if unit in episodes:
                    raise ValueError(f"Duplicate E2 episode: {unit}")
                episodes[unit] = episode_result(directory, row, condition, seed)
        manifests.append({
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "source_sha256": manifest.get("source_sha256"),
            "identity": run_identity(manifest),
        })
    if len(manifests) != 10 or len(episodes) != len(roster) * 2:
        raise ValueError(f"Expected 10 complete E2 batches and 80 episodes, got {len(manifests)} and {len(episodes)}")
    return episodes, manifests


def summarize_conditions(episodes: dict, tasks: list[str]):
    out = {}
    conditions = sorted({key[0] for key in episodes})
    for condition in conditions:
        rows = [value for (arm, _task, _state), value in episodes.items() if arm == condition]
        out[condition] = {
            "n": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": mean([row["success"] for row in rows]),
            "mean_scr": mean([row["scr"] for row in rows]),
            "overshoots": sum(row["overshot"] for row in rows),
            "by_task": {},
        }
        for task in tasks:
            task_rows = [row for row in rows if row["task"] == task]
            out[condition]["by_task"][task] = {
                "n": len(task_rows),
                "successes": sum(row["success"] for row in task_rows),
                "success_rate": mean([row["success"] for row in task_rows]),
                "mean_scr": mean([row["scr"] for row in task_rows]),
                "overshoots": sum(row["overshot"] for row in task_rows),
            }
    return out


def compare_pair(left: dict, right: dict, condition_left: str, condition_right: str, tasks: list[str],
                 states: list[int], seed: int):
    metrics = ("success", "scr", "overshot")
    by_task = {}
    paired_rows = []
    for task in tasks:
        rows = []
        for state in states:
            a, b = left[(condition_left, task, state)], right[(condition_right, task, state)]
            if a["initial_state_sha256"] != b["initial_state_sha256"]:
                raise ValueError(f"Paired initial-state hash mismatch: {condition_left}/{condition_right}/{task}/{state}")
            row = {
                "task": task,
                "state_idx": state,
                "initial_state_sha256": a["initial_state_sha256"],
                "left": condition_left,
                "right": condition_right,
                "deltas_left_minus_right": {metric: a[metric] - b[metric] for metric in metrics},
                "left_success": a["success"],
                "right_success": b["success"],
            }
            rows.append(row)
            paired_rows.append(row)
        by_task[task] = {
            "n": len(rows),
            "left_only_success": sum(row["left_success"] > row["right_success"] for row in rows),
            "tie": sum(row["left_success"] == row["right_success"] for row in rows),
            "right_only_success": sum(row["left_success"] < row["right_success"] for row in rows),
            "mean_deltas": {
                metric: mean([row["deltas_left_minus_right"][metric] for row in rows])
                for metric in metrics
            },
        }
    rng = random.Random(42)
    draws = defaultdict(list)
    for _ in range(2000):
        sampled = [rng.choice(states) for _ in states]
        for metric in metrics:
            draws[metric].append(mean([
                mean([next(row for row in paired_rows
                           if row["task"] == task and row["state_idx"] == state)["deltas_left_minus_right"][metric]
                      for task in tasks])
                for state in sampled
            ]))
    intervals = {metric: [percentile(values, .025), percentile(values, .975)] for metric, values in draws.items()}
    return {
        "n_pairs": len(paired_rows),
        "grouping_unit": "locked state index; T1 and T3 stay together",
        "state_group_bootstrap_95": intervals,
        "by_task": by_task,
        "macro_mean_deltas_left_minus_right": {
            metric: mean([by_task[task]["mean_deltas"][metric] for task in tasks])
            for metric in metrics
        },
        "paired_rows": paired_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-run", type=Path, required=True)
    parser.add_argument("--e2-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--tasks", nargs="+", choices=("T1", "T3"), default=["T1", "T3"])
    parser.add_argument("--states", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    states = [int(value) for value in args.states.split(",")] if args.states != "all" else list(range(20))
    b0 = load_b0(args.b0_run, args.split_lock, args.tasks, args.states, args.seed)
    e2, e2_manifests = load_e2(args.e2_root, args.split_lock, args.tasks, args.states, args.seed)
    combined = {("B0", *key): value for key, value in b0["episodes"].items()}
    combined.update(e2)
    if set(b0["episodes"]) != {(task, state) for task in args.tasks for state in states}:
        raise ValueError("B0 run does not contain the exact requested roster")
    comparisons = {
        "B0_minus_F0": compare_pair(combined, combined, "B0", "F0", args.tasks, states, args.seed),
        "B0_minus_F1": compare_pair(combined, combined, "B0", "F1", args.tasks, states, args.seed),
        "F1_minus_F0": compare_pair(combined, combined, "F1", "F0", args.tasks, states, args.seed),
    }
    source_names = sorted(set(b0["manifest"].get("source_sha256", {})) & set(e2_manifests[0]["source_sha256"] or {}))
    source_parity = {
        name: all(
            b0["manifest"].get("source_sha256", {}).get(name) == item["source_sha256"].get(name)
            for item in e2_manifests
        )
        for name in source_names
    }
    result = {
        "classification": "CUSTOM_V2_LIBERO_MEM_UNSEEN_DESCRIPTIVE_NON_COMPARABLE_OFFICIAL_LIBERO",
        "design": {
            "conditions": ["B0", "F0", "F1"],
            "tasks": args.tasks,
            "state_indices_within_unseen_lock": states,
            "seed": args.seed,
            "planned_episodes_per_condition": len(args.tasks) * len(states),
            "causal_claim": False,
        },
        "provenance": {
            "split_lock_sha256": read_json(args.split_lock)["lock_sha256"],
            "protocol_sha256": source_digest(DEFAULT_PROTOCOL),
            "b0_run_manifest_sha256": b0["manifest_sha256"],
            "e2_batch_manifests": e2_manifests,
            "b0_run_identity": b0["identity"],
            "e2_run_identity": e2_manifests[0]["identity"],
            "source_code_hash_parity": source_parity,
        },
        "conditions": summarize_conditions(combined, args.tasks),
        "comparisons": comparisons,
        "episodes": {
            "B0": list(b0["episodes"].values()),
            "F0": [value for (condition, _task, _state), value in e2.items() if condition == "F0"],
            "F1": [value for (condition, _task, _state), value in e2.items() if condition == "F1"],
        },
        "interpretation_limit": "The report describes paired outcomes on the locked unseen roster and one seed. Any source/runtime mismatch is retained in provenance; custom results are not official LIBERO scores.",
    }
    write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "conditions": {key: value["successes"] for key, value in result["conditions"].items()},
        "macro_success_deltas": {
            key: value["macro_mean_deltas_left_minus_right"]["success"]
            for key, value in comparisons.items()
        },
        "source_hash_matches": sum(source_parity.values()),
        "source_hash_fields": len(source_parity),
    }, indent=2))


if __name__ == "__main__":
    main()
