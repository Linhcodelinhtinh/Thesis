#!/usr/bin/env python3
"""State-group bootstrap for the completed E2 F0/F1 matched rollout."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments/evaluation/e2_unseen_author_v2"
PAIRED_RECEIPT = RUN_ROOT / "paired_e2_final.json"
OUTPUT = ROOT / "docs/thesis/evidence/e2_paired_group_bootstrap_20261120.json"
TASKS = ("T1", "T3")
ARMS = ("F0", "F1")
STATE_GROUPS = 20
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main() -> None:
    paired_receipt = json.loads(PAIRED_RECEIPT.read_text(encoding="utf-8"))
    if (not paired_receipt.get("complete_paired_roster")
            or paired_receipt.get("state_groups") != STATE_GROUPS):
        raise ValueError("E2 aggregate does not certify the complete 20-group roster")

    episodes: dict[tuple[str, str, int, str, int], dict] = {}
    sources: list[Path] = [PAIRED_RECEIPT]
    manifests: dict[Path, dict] = {}
    for arm in ARMS:
        files = sorted(RUN_ROOT.glob(f"batch-*/episodes/*_{arm}_unseen_state*/summary.json"))
        if len(files) != STATE_GROUPS * len(TASKS):
            raise ValueError(f"Expected 40 {arm} summaries, found {len(files)}")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            status_path = path.parent / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (status.get("status") != "complete"
                    or data.get("condition") != arm
                    or data.get("split") != "unseen"
                    or data.get("task") not in TASKS
                    or data.get("policy_seed") != 42
                    or type(data.get("state_idx")) is not int
                    or type(data.get("result", {}).get("success")) is not bool):
                raise ValueError(f"Unexpected or incomplete E2 episode: {path}")
            if status.get("files_sha256", {}).get("summary.json") != sha256(path):
                raise ValueError(f"Episode summary hash differs from status receipt: {path}")
            key = (arm, data["task"], data["state_idx"],
                   data["initial_state_sha256"], data["policy_seed"])
            if key in episodes:
                raise ValueError(f"Duplicate matched episode key: {key}")
            episodes[key] = {
                "arm": arm,
                "success": data["result"]["success"],
                "source_demo": data["source_demo"],
                "path": path,
                "status_path": status_path,
            }
            sources.extend((path, status_path))
            batch_dir = path.parents[2]
            manifest_path = batch_dir / "run_manifest.json"
            if manifest_path not in manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                args = manifest.get("arguments", {})
                if (manifest.get("status") != "complete"
                        or args.get("conditions") != ["F0", "F1"]
                        or args.get("split") != "unseen"
                        or args.get("tasks") != list(TASKS)
                        or args.get("policy_seed") != 42):
                    raise ValueError(f"Batch manifest differs from E2 protocol: {manifest_path}")
                manifests[manifest_path] = manifest
                sources.append(manifest_path)
    if len(manifests) != 10:
        raise ValueError(f"Expected 10 complete E2 batch manifests, found {len(manifests)}")

    f0_keys = {key[1:] for key in episodes if key[0] == "F0"}
    f1_keys = {key[1:] for key in episodes if key[0] == "F1"}
    f0 = {key: episodes[("F0", *key)] for key in sorted(f0_keys)}
    f1 = {key: episodes[("F1", *key)] for key in sorted(f1_keys)}
    if set(f0) != set(f1) or len(f0) != STATE_GROUPS * len(TASKS):
        raise ValueError("F0 and F1 are not a complete matched 40-pair roster")

    pairs = []
    by_state: dict[int, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for task, state_idx, state_hash, seed in sorted(f0):
        left, right = f0[(task, state_idx, state_hash, seed)], f1[(task, state_idx, state_hash, seed)]
        if left["source_demo"] != right["source_demo"]:
            raise ValueError(f"Source demo differs between arms at {(task, state_idx)}")
        row = {
            "task": task,
            "state_idx": state_idx,
            "state_group_sha256": state_hash,
            "source_demo": left["source_demo"],
            "policy_seed": seed,
            "f0_success": left["success"],
            "f1_success": right["success"],
            "paired_difference_f1_minus_f0": int(right["success"]) - int(left["success"]),
            "f0_summary_sha256": sha256(left["path"]),
            "f1_summary_sha256": sha256(right["path"]),
        }
        pairs.append(row)
        by_state[state_idx].append(row)
        by_task[task].append(row)

    if set(by_state) != set(range(STATE_GROUPS)) or any(len(rows) != len(TASKS) for rows in by_state.values()):
        raise ValueError("Each locked state group must contain paired T1 and T3 results")
    for state_idx, rows in by_state.items():
        if len({row["state_group_sha256"] for row in rows}) != 1:
            raise ValueError(f"T1/T3 initial-state hashes do not match in group {state_idx}")

    outcome_counts = Counter(
        "both_success" if row["f0_success"] and row["f1_success"]
        else "f0_only" if row["f0_success"]
        else "f1_only" if row["f1_success"]
        else "both_failure"
        for row in pairs
    )
    f0_successes = sum(row["f0_success"] for row in pairs)
    f1_successes = sum(row["f1_success"] for row in pairs)
    overall = {
        "paired_task_state_units": len(pairs),
        "state_groups": len(by_state),
        "f0_successes": f0_successes,
        "f1_successes": f1_successes,
        "f0_success_rate": f0_successes / len(pairs),
        "f1_success_rate": f1_successes / len(pairs),
        "paired_difference_f1_minus_f0": (
            sum(row["paired_difference_f1_minus_f0"] for row in pairs) / len(pairs)
        ),
        "paired_outcomes": {name: outcome_counts[name] for name in
                            ("both_success", "f0_only", "f1_only", "both_failure")},
    }

    # Each resampled unit is a locked state group containing the matched T1/T3 pair.
    rng = random.Random(BOOTSTRAP_SEED)
    groups = [by_state[state_idx] for state_idx in range(STATE_GROUPS)]
    samples = []
    for _ in range(BOOTSTRAPS):
        drawn_groups = rng.choices(groups, k=STATE_GROUPS)
        differences = [
            row["paired_difference_f1_minus_f0"]
            for group in drawn_groups for row in group
        ]
        samples.append(sum(differences) / len(differences))
    samples.sort()

    per_task = []
    for task in TASKS:
        rows = by_task[task]
        per_task.append({
            "task": task,
            "paired_units": len(rows),
            "f0_successes": sum(row["f0_success"] for row in rows),
            "f1_successes": sum(row["f1_success"] for row in rows),
            "paired_difference_f1_minus_f0": sum(
                row["paired_difference_f1_minus_f0"] for row in rows
            ) / len(rows),
        })

    sources = sorted(set(sources))
    result = {
        "analysis_id": "e2-paired-state-group-bootstrap-2026-11-20",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "CUSTOM_V2_LIBERO_MEM_RAW_UNSEEN",
        "comparability": "NON-COMPARABLE_OFFICIAL_PAPER",
        "arms": {
            "F0": "V2 no-memory adapter",
            "F1": "V2 memory adapter with rgb_relations_v1",
        },
        "method": {
            "estimand": "paired success-rate difference (F1 minus F0)",
            "bootstrap": (
                "10,000 paired state-group resamples; each sampled group retains "
                "its matched T1/T3 outcomes"
            ),
            "interval": "two-sided 95% percentile interval",
            "seed": BOOTSTRAP_SEED,
            "scope_note": (
                "Descriptive uncertainty conditional on the 20 locked unseen state groups "
                "and one policy seed. It does not generalize across seeds or official LIBERO."
            ),
        },
        "overall": overall,
        "paired_difference_bootstrap_95": [
            percentile(samples, 0.025),
            percentile(samples, 0.975),
        ],
        "per_task": per_task,
        "pairs": pairs,
        "provenance": {
            "run_root": "experiments/evaluation/e2_unseen_author_v2",
            "source_files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for path in sources
            ],
            "split_lock_sha256": paired_receipt["split_lock_sha256"],
            "analysis_code_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "analysis_code_sha256": sha256(Path(__file__)),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "overall": overall,
        "paired_difference_bootstrap_95": result["paired_difference_bootstrap_95"],
        "complete_batch_manifests": len(manifests),
        "source_file_count": len(sources),
        "source_hashes_verified": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
