#!/usr/bin/env python3
"""Audit paired E6 B0 reruns across the historical and WSL runtimes.

This is a descriptive reproducibility diagnostic. It does not estimate a
policy effect because the two runs use different execution environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SUITE_SEED_OFFSET = {"libero_10": 30}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_task_episodes(root: Path, task_id: int) -> dict[int, tuple[dict[str, Any], Path]]:
    task_root = root / f"libero_10_task{task_id}"
    files = sorted(task_root.rglob("episode.json"))
    if not files:
        raise RuntimeError(f"No episode.json files found under {task_root}")

    episodes: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        state_id = int(episode["initial_state_id"])
        if state_id in episodes:
            raise RuntimeError(f"Duplicate state {state_id} under {task_root}")
        episodes[state_id] = (episode, path)
    return episodes


def arm_environment(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    provenance_fields = (
        "execution_tier",
        "certification",
        "python_version",
        "robosuite_version",
        "bddl_version",
        "git_commit",
        "model_config_sha256",
        "base_seed",
        "num_steps_wait",
        "sample_level",
        "model_name",
        "checkpoint",
        "camera_resolution",
    )
    environments = []
    for episode in episodes:
        provenance = episode.get("provenance", {})
        environment = {key: provenance.get(key) for key in provenance_fields}
        environment["execution_horizon_s"] = provenance.get("execution_horizon_s")
        environment["policy_chunk_size"] = provenance.get("policy_chunk_size")
        environment["memory_condition"] = provenance.get("memory_condition")
        environment["non_comparable_reasons"] = provenance.get("non_comparable_reasons", [])
        environments.append(environment)

    first = environments[0]
    inconsistent = [item for item in environments[1:] if item != first]
    if inconsistent:
        raise RuntimeError("Runtime provenance varies within a compared arm")
    return first


def validate_episode(episode: dict[str, Any], task_id: int, state_id: int, arm: str) -> None:
    provenance = episode.get("provenance", {})
    if int(episode.get("initial_state_id", -1)) != state_id:
        raise RuntimeError(f"{arm}: state ID mismatch for task {task_id}, state {state_id}")
    if episode.get("success") not in (True, False):
        raise RuntimeError(f"{arm}: missing boolean success for task {task_id}, state {state_id}")
    if provenance.get("certification") != "NON-COMPARABLE_OFFICIAL_PAPER":
        raise RuntimeError(f"{arm}: expected NON-COMPARABLE_OFFICIAL_PAPER certification")
    if provenance.get("model_name") != "smolvla_libero":
        raise RuntimeError(f"{arm}: unexpected model for task {task_id}, state {state_id}")
    if provenance.get("checkpoint") != "resources/checkpoints/smolvla_libero":
        raise RuntimeError(f"{arm}: unexpected checkpoint for task {task_id}, state {state_id}")
    if provenance.get("model_config_sha256") != "f70fa95d16eaf98b362e60e2ab839f875427185c4acdc6f015f38f96227026eb":
        raise RuntimeError(f"{arm}: unexpected model config hash for task {task_id}, state {state_id}")
    if provenance.get("memory_condition") != "off":
        raise RuntimeError(f"{arm}: expected memory_condition=off for task {task_id}, state {state_id}")
    if int(provenance.get("execution_horizon_s", -1)) != 10:
        raise RuntimeError(f"{arm}: unexpected execution horizon for task {task_id}, state {state_id}")
    if int(provenance.get("policy_chunk_size", -1)) != 50:
        raise RuntimeError(f"{arm}: unexpected policy chunk size for task {task_id}, state {state_id}")
    expected_seed = 42 + (SUITE_SEED_OFFSET["libero_10"] + task_id) * 100 + state_id
    if int(episode.get("seed", -1)) != expected_seed:
        raise RuntimeError(
            f"{arm}: unexpected seed for task {task_id}, state {state_id}: "
            f"expected {expected_seed}, observed {episode.get('seed')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-root",
        default="experiments/results/e6_retention_locked_50",
        help="Directory containing B0_task_<id> historical outputs.",
    )
    parser.add_argument(
        "--reproduction-root",
        default="experiments/results/e6_reproduce_historical_50_20261120_101024",
        help="Directory containing task_<id> WSL rerun outputs.",
    )
    parser.add_argument(
        "--output",
        default="docs/thesis/evidence/e6_cross_environment_reproduction_20261120.json",
        help="Output JSON receipt path relative to the repository root.",
    )
    args = parser.parse_args()

    historical_root = (REPO / args.historical_root).resolve()
    reproduction_root = (REPO / args.reproduction_root).resolve()
    output_path = (REPO / args.output).resolve()
    script_path = Path(__file__).resolve()

    paired_rows: list[dict[str, Any]] = []
    historical_episodes: list[dict[str, Any]] = []
    reproduction_episodes: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    per_task: list[dict[str, Any]] = []

    for task_id in range(10):
        historical_dir = historical_root / f"B0_task_{task_id}"
        reproduction_dir = reproduction_root / f"task_{task_id}"
        historical = read_task_episodes(historical_dir, task_id)
        reproduction = read_task_episodes(reproduction_dir, task_id)
        historical_states = set(historical)
        reproduction_states = set(reproduction)
        if historical_states != reproduction_states:
            missing = sorted(historical_states - reproduction_states)
            extra = sorted(reproduction_states - historical_states)
            raise RuntimeError(
                f"Task {task_id} roster mismatch: missing={missing}, extra={extra}"
            )
        if len(historical_states) != 5:
            raise RuntimeError(f"Task {task_id} must contain exactly five matched states")

        task_rows: list[dict[str, Any]] = []
        for state_id in sorted(historical_states):
            historical_episode, historical_path = historical[state_id]
            reproduction_episode, reproduction_path = reproduction[state_id]
            validate_episode(historical_episode, task_id, state_id, "historical")
            validate_episode(reproduction_episode, task_id, state_id, "reproduction")

            historical_seed = int(historical_episode["seed"])
            reproduction_seed = int(reproduction_episode["seed"])
            if historical_seed != reproduction_seed:
                raise RuntimeError(
                    f"Seed mismatch for task {task_id}, state {state_id}: "
                    f"{historical_seed} vs {reproduction_seed}"
                )

            historical_success = bool(historical_episode["success"])
            reproduction_success = bool(reproduction_episode["success"])
            row = {
                "task_id": task_id,
                "state_id": state_id,
                "task_name": historical_episode.get("task_name"),
                "seed": historical_seed,
                "historical_success": historical_success,
                "reproduction_success": reproduction_success,
                "outcome_agreement": historical_success == reproduction_success,
                "historical_num_steps": historical_episode.get("num_steps"),
                "reproduction_num_steps": reproduction_episode.get("num_steps"),
                "historical_episode_path": historical_path.relative_to(REPO).as_posix(),
                "historical_episode_sha256": sha256_file(historical_path),
                "reproduction_episode_path": reproduction_path.relative_to(REPO).as_posix(),
                "reproduction_episode_sha256": sha256_file(reproduction_path),
            }
            paired_rows.append(row)
            task_rows.append(row)
            historical_episodes.append(historical_episode)
            reproduction_episodes.append(reproduction_episode)
            for path in (historical_path, reproduction_path):
                source_files.append(
                    {
                        "path": path.relative_to(REPO).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )

        per_task.append(
            {
                "task_id": task_id,
                "episodes": len(task_rows),
                "historical_successes": sum(row["historical_success"] for row in task_rows),
                "reproduction_successes": sum(row["reproduction_success"] for row in task_rows),
                "outcome_agreements": sum(row["outcome_agreement"] for row in task_rows),
            }
        )

    if len(paired_rows) != 50:
        raise RuntimeError(f"Expected exactly 50 episode pairs, found {len(paired_rows)}")

    outcomes = Counter(
        (row["historical_success"], row["reproduction_success"])
        for row in paired_rows
    )
    historical_environment = arm_environment(historical_episodes)
    reproduction_environment = arm_environment(reproduction_episodes)
    environment_differences = {
        field: {
            "historical": historical_environment.get(field),
            "reproduction": reproduction_environment.get(field),
        }
        for field in historical_environment
        if historical_environment.get(field) != reproduction_environment.get(field)
    }

    summary = {
        "paired_episodes": len(paired_rows),
        "rosters_match": True,
        "seeds_match": True,
        "historical_successes": sum(row["historical_success"] for row in paired_rows),
        "reproduction_successes": sum(row["reproduction_success"] for row in paired_rows),
        "historical_success_rate": sum(row["historical_success"] for row in paired_rows) / 50,
        "reproduction_success_rate": sum(row["reproduction_success"] for row in paired_rows) / 50,
        "descriptive_difference_percentage_points": (
            sum(row["reproduction_success"] for row in paired_rows)
            - sum(row["historical_success"] for row in paired_rows)
        ) * 2,
        "outcome_agreements": sum(row["outcome_agreement"] for row in paired_rows),
        "outcome_agreement_rate": sum(row["outcome_agreement"] for row in paired_rows) / 50,
        "both_success": outcomes[(True, True)],
        "historical_only_success": outcomes[(True, False)],
        "reproduction_only_success": outcomes[(False, True)],
        "both_fail": outcomes[(False, False)],
    }

    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_non_comparable",
        "purpose": (
            "Descriptive cross-environment reproducibility check of E6 B0 outcomes; "
            "not an official LIBERO score and not a causal policy comparison."
        ),
        "analysis_script": {
            "path": script_path.relative_to(REPO).as_posix(),
            "sha256": sha256_file(script_path),
        },
        "input_roots": {
            "historical": historical_root.relative_to(REPO).as_posix(),
            "reproduction": reproduction_root.relative_to(REPO).as_posix(),
        },
        "matched_protocol": {
            "suite": "libero_10",
            "model": "smolvla_libero",
            "checkpoint": "resources/checkpoints/smolvla_libero",
            "model_config_sha256": "f70fa95d16eaf98b362e60e2ab839f875427185c4acdc6f015f38f96227026eb",
            "base_seed": 42,
            "seed_formula": "42 + (30 + task_id) * 100 + state_id",
            "execution_horizon_s": 10,
            "policy_chunk_size": 50,
            "memory_condition": "off",
            "episodes_per_task": 5,
        },
        "historical_environment": historical_environment,
        "reproduction_environment": reproduction_environment,
        "environment_differences": environment_differences,
        "summary": summary,
        "per_task": per_task,
        "episode_pairs": paired_rows,
        "source_files": source_files,
        "limitations": [
            "The historical run used Windows Python 3.12.2, robosuite 1.5.2, BDDL 3.6.0, and source commit 657f325b; the reproduction used WSL/Linux Python 3.10.22, robosuite 1.4.0, BDDL 1.0.1, and source commit 02e05ffe.",
            "Platform, Python, simulator, BDDL, and source revision changed together; outcome flips cannot be attributed to the policy or to a single environment component.",
            "This diagnostic is not comparable with official LIBERO paper scores and does not generalize beyond the fixed 50 task/state/seed pairs.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": output_path.relative_to(REPO).as_posix(), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
