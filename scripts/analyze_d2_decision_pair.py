"""Validate and summarize the paired D2 shadow/event development cohort."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.evaluation.memory_supervisor import (
    DEFAULT_DECISION_PROTOCOL,
    execution_treatment,
    load_decision_protocol,
)
from src.evaluation.mem_eval_io import evaluation_roster, write_json
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean(values):
    return statistics.mean(values) if values else None


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[index]


def validate_run(run_dir: Path, expected_arm: str, split_lock: Path, tasks: list[str], states: str, seed: int):
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    args = manifest["arguments"]
    mode = "shadow" if expected_arm == "shadow-projected" else "event"
    treatment = execution_treatment(
        load_decision_protocol(), args.get("decision_mode"), args.get("decision_exposure")
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("classification") != "CUSTOM_V2_LIBERO_MEM"
        or args.get("decision_mode") != mode
        or treatment.arm != expected_arm
        or args.get("decision_exposure") != "projected"
        or args.get("split") != "seen"
        or args.get("conditions") != ["F1"]
        or args.get("tasks") != tasks
        or str(args.get("states")) != states
        or args.get("policy_seed") != seed
        or args.get("preflight_only")
        or args.get("runtime_check")
    ):
        raise ValueError(f"Run metadata differs from D2 {expected_arm}: {run_dir}")
    if manifest.get("protocol_sha256") != source_digest(DEFAULT_PROTOCOL):
        raise ValueError(f"Evaluation protocol hash differs: {run_dir}")
    if manifest.get("decision_protocol_sha256") != source_digest(DEFAULT_DECISION_PROTOCOL):
        raise ValueError(f"Decision protocol hash differs: {run_dir}")

    _, lock, roster = evaluation_roster(split_lock, "seen", tasks, states)
    if manifest.get("split_lock_sha256") != lock["lock_sha256"]:
        raise ValueError(f"Split lock hash differs: {run_dir}")
    if manifest.get("roster") != roster or manifest.get("expected_episodes") != len(roster):
        raise ValueError(f"Run roster differs from the locked D2 roster: {run_dir}")
    if manifest.get("loaded_policies", {}).get("F1") is None:
        raise ValueError(f"F1 policy was not recorded as loaded: {run_dir}")

    episodes = {}
    for row in roster:
        key = (row["task"], int(row["state_idx"]), seed)
        episode_dir = run_dir / "episodes" / f"{row['task']}_F1_seen_state{row['state_idx']}"
        status = read_json(episode_dir / "status.json")
        if status.get("status") != "complete":
            raise ValueError(f"Incomplete episode {episode_dir}")
        identity = (status.get("task"), status.get("state_idx"), status.get("policy_seed"))
        if identity != key or status.get("decision_arm") != expected_arm:
            raise ValueError(f"Episode identity/treatment differs: {episode_dir}")
        if status.get("initial_state_sha256") != row["initial_state_sha256"]:
            raise ValueError(f"Initial state differs from locked roster: {episode_dir}")

        artifact_hashes = status.get("files_sha256", {})
        required = {"summary.json", "queries.jsonl", "actions.jsonl", "trajectory.npz", "memory_trace.jsonl"}
        if not required.issubset(artifact_hashes):
            raise ValueError(f"Required artifact hash missing: {episode_dir}")
        for name, expected_hash in artifact_hashes.items():
            artifact = episode_dir / name
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {artifact}")

        summary = read_json(episode_dir / "summary.json")
        result = summary["result"]
        if summary.get("initial_state_sha256") != row["initial_state_sha256"]:
            raise ValueError(f"Summary state hash differs from locked roster: {episode_dir}")
        queries = read_jsonl(episode_dir / "queries.jsonl")
        decisions = read_jsonl(episode_dir / "decisions.jsonl")
        requery = [d for d in decisions if d.get("decision") == "requery"]
        evidence_backed = [
            d for d in requery
            if d.get("changes") and all(
                change.get("after", {}).get("event_ids")
                and change.get("after", {}).get("evidence_ids")
                for change in d["changes"]
            )
        ]
        episode_metrics = {
            "task": row["task"],
            "state_idx": int(row["state_idx"]),
            "policy_seed": seed,
            "initial_state_sha256": row["initial_state_sha256"],
            "success": int(bool(result["success"])),
            "completed_subgoals": int(result["completed_subgoals"]),
            "total_subgoals": int(result["total_subgoals"]),
            "scr": float(result["scr"]),
            "overshot": int(bool(result["overshot"])),
            "steps": int(result["total_steps"]),
            "wall_time_s": float(result["total_wall_time_s"]),
            "queries": len(queries),
            "logged_policy_queries": result.get("decision_metrics", {}).get("queries"),
            "decision_records": len(decisions),
            "requeries": len(requery),
            "evidence_backed_requeries": len(evidence_backed),
            "requery_steps": [int(d["step"]) for d in requery],
            "requery_reasons": dict(Counter(d.get("reason", "unknown") for d in requery)),
            "artifact_sha256": artifact_hashes,
        }
        if episode_metrics["logged_policy_queries"] != episode_metrics["queries"]:
            raise ValueError(f"Query count differs between summary and log: {episode_dir}")
        episodes[key] = episode_metrics

    identity_fields = (
        "base", "loaded_policies", "local_assets_lock_sha256", "environments", "cuda",
        "source_sha256", "protocol_sha256", "decision_protocol_sha256", "split_lock_sha256",
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "identity": {key: manifest.get(key) for key in identity_fields},
        "episodes": episodes,
    }


def arm_summary(episodes: dict, tasks: list[str]):
    output = {}
    for task in tasks:
        rows = [value for key, value in episodes.items() if key[0] == task]
        output[task] = {
            "n": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": mean([row["success"] for row in rows]),
            "mean_scr": mean([row["scr"] for row in rows]),
            "overshoots": sum(row["overshot"] for row in rows),
            "mean_queries": mean([row["queries"] for row in rows]),
            "episodes_with_requery": sum(row["requeries"] > 0 for row in rows),
            "total_requeries": sum(row["requeries"] for row in rows),
            "evidence_backed_requeries": sum(row["evidence_backed_requeries"] for row in rows),
            "mean_wall_time_s": mean([row["wall_time_s"] for row in rows]),
        }
    all_rows = [value for value in episodes.values()]
    output["overall"] = {
        "n": len(all_rows),
        "successes": sum(row["success"] for row in all_rows),
        "success_rate": mean([row["success"] for row in all_rows]),
        "mean_scr": mean([row["scr"] for row in all_rows]),
        "overshoots": sum(row["overshot"] for row in all_rows),
        "mean_queries": mean([row["queries"] for row in all_rows]),
        "episodes_with_requery": sum(row["requeries"] > 0 for row in all_rows),
        "total_requeries": sum(row["requeries"] for row in all_rows),
        "evidence_backed_requeries": sum(row["evidence_backed_requeries"] for row in all_rows),
        "mean_wall_time_s": mean([row["wall_time_s"] for row in all_rows]),
    }
    return output


def paired_analysis(shadow: dict, event: dict, tasks: list[str], state_indices: list[int], bootstrap_seed: int):
    metrics = ("success", "scr", "overshot", "queries", "requeries", "wall_time_s")
    pairs = {}
    for task in tasks:
        rows = []
        for state_idx in state_indices:
            key = (task, state_idx, 42)
            a, b = event[key], shadow[key]
            if a["initial_state_sha256"] != b["initial_state_sha256"]:
                raise ValueError(f"Paired state hash mismatch: {key}")
            rows.append({
                "task": task,
                "state_idx": state_idx,
                "initial_state_sha256": a["initial_state_sha256"],
                "event_success": a["success"],
                "shadow_success": b["success"],
                "event_scr": a["scr"],
                "shadow_scr": b["scr"],
                "event_queries": a["queries"],
                "shadow_queries": b["queries"],
                "event_requeries": a["requeries"],
                "shadow_requeries": b["requeries"],
                "deltas": {name: a[name] - b[name] for name in metrics},
            })
        pairs[task] = rows

    rng = random.Random(bootstrap_seed)
    draws = defaultdict(list)
    for _ in range(2000):
        sampled_indices = [rng.choice(state_indices) for _ in state_indices]
        for metric in metrics:
            draws[metric].append(mean([
                mean([pairs[task][state_idx]["deltas"][metric] for task in tasks])
                for state_idx in sampled_indices
            ]))
    intervals = {}
    for metric, values in draws.items():
        values.sort()
        intervals[metric] = [percentile(values, 0.025), percentile(values, 0.975)]

    summary = {}
    for task in tasks:
        task_pairs = pairs[task]
        summary[task] = {
            "n": len(task_pairs),
            "event_successes": sum(row["event_success"] for row in task_pairs),
            "shadow_successes": sum(row["shadow_success"] for row in task_pairs),
            "success_win_tie_loss_event_vs_shadow": {
                "event_only_success": sum(row["event_success"] > row["shadow_success"] for row in task_pairs),
                "tie": sum(row["event_success"] == row["shadow_success"] for row in task_pairs),
                "shadow_only_success": sum(row["event_success"] < row["shadow_success"] for row in task_pairs),
            },
            "mean_event_minus_shadow": {
                metric: mean([row["deltas"][metric] for row in task_pairs]) for metric in metrics
            },
        }
    return {
        "pair_count": sum(len(rows) for rows in pairs.values()),
        "grouping_unit": "state_idx; both tasks kept together within each bootstrap draw",
        "bootstrap": {"resamples": 2000, "seed": bootstrap_seed, "macro_mean_95_percentile_interval": intervals},
        "by_task": summary,
        "macro_mean_event_minus_shadow": {
            metric: mean([summary[task]["mean_event_minus_shadow"][metric] for task in tasks])
            for metric in metrics
        },
        "paired_rows": [row for task in tasks for row in pairs[task]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-run", type=Path, required=True)
    parser.add_argument("--event-run", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--tasks", nargs="+", choices=("T1", "T3"), default=["T1", "T3"])
    parser.add_argument("--states", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    state_indices = [int(value) for value in args.states.split(",")]
    if len(set(state_indices)) != len(state_indices) or not state_indices:
        parser.error("States must be a nonempty comma-separated list without duplicates")

    shadow_run = validate_run(args.shadow_run, "shadow-projected", args.split_lock, args.tasks, args.states, args.seed)
    event_run = validate_run(args.event_run, "event-projected", args.split_lock, args.tasks, args.states, args.seed)
    if shadow_run["identity"] != event_run["identity"]:
        raise ValueError("The paired runs differ in checkpoint, runtime, source or locked assets")
    if set(shadow_run["episodes"]) != set(event_run["episodes"]):
        raise ValueError("Shadow and event arms do not contain the same completed units")
    protocol = load_decision_protocol()
    result = {
        "classification": "DECISION_DEVELOPMENT_CUSTOM_V2_NON_COMPARABLE_OFFICIAL_LIBERO",
        "design": {
            "comparison": "F1 projected-memory event-triggered retrieval vs shadow supervisor with the same periodic query cadence",
            "tasks": args.tasks,
            "state_indices_within_seen_lock": state_indices,
            "seed": args.seed,
            "planned_episodes_per_arm": len(args.tasks) * len(state_indices),
            "split": "seen/development",
            "causal_claim": False,
        },
        "provenance": {
            "split_lock_sha256": read_json(args.split_lock)["lock_sha256"],
            "protocol_sha256": source_digest(DEFAULT_PROTOCOL),
            "decision_protocol_sha256": source_digest(DEFAULT_DECISION_PROTOCOL),
            "shadow_run_manifest_sha256": shadow_run["manifest_sha256"],
            "event_run_manifest_sha256": event_run["manifest_sha256"],
            "bootstrap_seed": protocol["descriptive_bootstrap"]["seed"],
        },
        "arms": {
            "shadow-projected": arm_summary(shadow_run["episodes"], args.tasks),
            "event-projected": arm_summary(event_run["episodes"], args.tasks),
        },
        "paired_event_minus_shadow": paired_analysis(
            shadow_run["episodes"], event_run["episodes"], args.tasks, state_indices,
            protocol["descriptive_bootstrap"]["seed"],
        ),
        "requery_reason_counts": {
            "shadow-projected": dict(Counter(
                reason for row in shadow_run["episodes"].values() for reason, count in row["requery_reasons"].items()
                for _ in range(count)
            )),
            "event-projected": dict(Counter(
                reason for row in event_run["episodes"].values() for reason, count in row["requery_reasons"].items()
                for _ in range(count)
            )),
        },
        "episodes": {
            "shadow-projected": list(shadow_run["episodes"].values()),
            "event-projected": list(event_run["episodes"].values()),
        },
    }
    write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "shadow": result["arms"]["shadow-projected"]["overall"],
        "event": result["arms"]["event-projected"]["overall"],
        "paired": result["paired_event_minus_shadow"]["macro_mean_event_minus_shadow"],
    }, indent=2))


if __name__ == "__main__":
    main()
