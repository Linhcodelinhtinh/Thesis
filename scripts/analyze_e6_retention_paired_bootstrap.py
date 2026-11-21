#!/usr/bin/env python3
"""Paired, task-stratified uncertainty analysis for the completed E6 retention run."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments/results/e6_retention_locked_50"
OUTPUT = ROOT / "docs/thesis/evidence/e6_retention_paired_bootstrap_20261120.json"
N_TASKS = 10
EPISODES_PER_TASK = 5
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


def load_arm(arm: str) -> tuple[dict[tuple[int, int, int], dict], list[Path]]:
    episodes: dict[tuple[int, int, int], dict] = {}
    sources: list[Path] = []
    for task_id in range(N_TASKS):
        task_dir = RUN_ROOT / f"{arm}_task_{task_id}" / f"libero_10_task{task_id}"
        files = sorted(task_dir.glob("init_*/episode.json"))
        if len(files) != EPISODES_PER_TASK:
            raise ValueError(
                f"{arm} task {task_id}: expected {EPISODES_PER_TASK} episode.json files, "
                f"found {len(files)} in {task_dir}"
            )
        summary = task_dir.parent / "benchmark_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing summary: {summary}")
        sources.append(summary)
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            state_id = data.get("initial_state_id")
            seed = data.get("seed")
            success = data.get("success")
            provenance = data.get("provenance") or {}
            if type(state_id) is not int or type(seed) is not int or type(success) is not bool:
                raise ValueError(f"Malformed outcome identity/status: {path}")
            if (provenance.get("certification") != "NON-COMPARABLE_OFFICIAL_PAPER"
                    or provenance.get("model_name") != "smolvla_libero"
                    or provenance.get("memory_condition") != "off"
                    or data.get("execution_horizon_s") != 10
                    or provenance.get("policy_chunk_size") != 50):
                raise ValueError(f"Unexpected evaluation/provenance contract: {path}")
            checkpoint = str(data.get("checkpoint", "")).replace("\\", "/")
            expected_checkpoint = (
                "resources/checkpoints/smolvla_libero"
                if arm == "B0"
                else "results_f1_author_split_v2/output_f1/pretrained_model"
            )
            if not checkpoint.endswith(expected_checkpoint):
                raise ValueError(f"{arm} episode loaded an unexpected checkpoint: {path}")
            key = (task_id, state_id, seed)
            if key in episodes:
                raise ValueError(f"Duplicate episode key {key} in {arm}")
            episodes[key] = {
                "success": success,
                "task_name": data.get("task_name"),
                "path": path,
            }
            sources.append(path)
    return episodes, sources


def main() -> None:
    b0, b0_sources = load_arm("B0")
    f1, f1_sources = load_arm("F1_actual")
    if set(b0) != set(f1) or len(b0) != N_TASKS * EPISODES_PER_TASK:
        raise ValueError("B0 and F1_actual do not form the complete 50-pair roster")

    pairs = []
    by_task: dict[int, list[dict]] = defaultdict(list)
    for task_id, state_id, seed in sorted(b0):
        left, right = b0[(task_id, state_id, seed)], f1[(task_id, state_id, seed)]
        if left["task_name"] != right["task_name"]:
            raise ValueError(f"Task name mismatch at task/state/seed {(task_id, state_id, seed)}")
        row = {
            "task_id": task_id,
            "task_name": left["task_name"],
            "initial_state_id": state_id,
            "seed": seed,
            "b0_success": left["success"],
            "f1_off_success": right["success"],
            "paired_difference_f1_off_minus_b0": int(right["success"]) - int(left["success"]),
            "b0_episode_sha256": sha256(left["path"]),
            "f1_off_episode_sha256": sha256(right["path"]),
        }
        pairs.append(row)
        by_task[task_id].append(row)

    outcomes = Counter(
        (
            "both_success" if row["b0_success"] and row["f1_off_success"]
            else "b0_only" if row["b0_success"]
            else "f1_off_only" if row["f1_off_success"]
            else "both_failure"
        )
        for row in pairs
    )
    overall = {
        "pairs": len(pairs),
        "b0_successes": sum(row["b0_success"] for row in pairs),
        "f1_off_successes": sum(row["f1_off_success"] for row in pairs),
        "b0_success_rate": sum(row["b0_success"] for row in pairs) / len(pairs),
        "f1_off_success_rate": sum(row["f1_off_success"] for row in pairs) / len(pairs),
        "paired_difference_f1_off_minus_b0": sum(
            row["paired_difference_f1_off_minus_b0"] for row in pairs
        ) / len(pairs),
        "paired_outcomes": {name: outcomes[name] for name in
                            ("both_success", "b0_only", "f1_off_only", "both_failure")},
    }

    # Resample paired states within each fixed task, so every task keeps five
    # observations and both arms remain paired throughout the bootstrap.
    rng = random.Random(BOOTSTRAP_SEED)
    task_rows = [by_task[task_id] for task_id in range(N_TASKS)]
    samples = []
    for _ in range(BOOTSTRAPS):
        differences = []
        for rows in task_rows:
            differences.extend(
                rng.choices(rows, k=len(rows))
            )
        samples.append(sum(
            row["paired_difference_f1_off_minus_b0"] for row in differences
        ) / len(differences))
    samples.sort()

    per_task = []
    for task_id in range(N_TASKS):
        rows = by_task[task_id]
        per_task.append({
            "task_id": task_id,
            "task_name": rows[0]["task_name"],
            "episodes_per_arm": len(rows),
            "b0_successes": sum(row["b0_success"] for row in rows),
            "f1_off_successes": sum(row["f1_off_success"] for row in rows),
            "paired_difference_f1_off_minus_b0": sum(
                row["paired_difference_f1_off_minus_b0"] for row in rows
            ) / len(rows),
        })

    sources = sorted(set(b0_sources + f1_sources))
    result = {
        "analysis_id": "e6-retention-paired-state-bootstrap-2026-11-20",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "CUSTOM_LIBERO_LOCAL",
        "comparability": "NON-COMPARABLE_OFFICIAL_PAPER",
        "arms": {
            "B0": "SmolVLA base checkpoint",
            "F1-off": "F1 V2 adapter with memory suppressed",
        },
        "method": {
            "estimand": "paired success-rate difference (F1-off minus B0)",
            "bootstrap": (
                "10,000 paired-state resamples, stratified within each of the 10 fixed tasks; "
                "five state pairs sampled with replacement per task"
            ),
            "interval": "two-sided 95% percentile interval",
            "seed": BOOTSTRAP_SEED,
            "scope_note": (
                "Descriptive uncertainty conditional on this fixed local task roster. "
                "It does not generalize to LIBERO-90 or isolate a causal memory effect."
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
            "run_root": "experiments/results/e6_retention_locked_50",
            "source_files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for path in sources
            ],
            "analysis_code_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "analysis_code_sha256": sha256(Path(__file__)),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "overall": overall,
        "paired_difference_bootstrap_95": result["paired_difference_bootstrap_95"],
        "source_file_count": len(sources),
        "source_hashes_verified": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
