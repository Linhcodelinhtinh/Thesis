#!/usr/bin/env python
"""Compare D2 online memory return counts with the simulator evaluator trace.

The simulator evaluator is used only as an offline audit signal. This report
does not label identity, intent, or action correctness, and does not feed
ground-truth state back to the policy or memory writer.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import validate_lock

RUNS = {
    "shadow-projected": ROOT / "experiments/evaluation/decision_shadow_dev12_t1_t3_20261120_wsl",
    "event-projected": ROOT / "experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl",
}
OUTPUT = ROOT / "experiments/evaluation/d2_evaluator_cycle_alignment_20261121"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def episode_alignment(episode_dir: Path):
    status = read_json(episode_dir / "status.json")
    if status.get("status") != "complete":
        raise ValueError(f"Incomplete source episode: {episode_dir}")
    required = ("memory_trace.jsonl", "observations.jsonl", "summary.json")
    input_hashes = {}
    for name in required:
        expected = status.get("files_sha256", {}).get(name)
        path = episode_dir / name
        if not expected or sha256(path) != expected:
            raise ValueError(f"Source artifact hash mismatch: {episode_dir.name}/{name}")
        input_hashes[name] = expected

    summary = read_json(episode_dir / "summary.json")
    memory = read_jsonl(episode_dir / "memory_trace.jsonl")
    observations = read_jsonl(episode_dir / "observations.jsonl")
    if not memory or not observations:
        raise ValueError(f"Missing trace rows: {episode_dir}")
    mem_by_step = {int(row["step"]): row["snapshot"] for row in memory}
    obs_by_step = {int(row["step"]): row["info"]["eval_result"] for row in observations}
    common = sorted(mem_by_step.keys() & obs_by_step.keys())
    if not common or common[-1] != int(summary["result"]["total_steps"]):
        raise ValueError(f"Trace horizon differs from episode summary: {episode_dir}")

    writer_values, evaluator_values = [], []
    writer_relation_status, history_incomplete = [], []
    for step in common:
        relation = mem_by_step[step].get("relations", {}).get("bowl_1")
        writer_values.append(None if relation is None else int(relation["returns"]))
        writer_relation_status.append(None if relation is None else relation.get("status"))
        history_incomplete.append(None if relation is None else not bool(relation.get("history_complete")))
        evaluator_values.append(int(obs_by_step[step]["confirmed_cycles"]))

    if any(value is None for value in writer_values):
        raise ValueError(f"Writer relation is absent in a D2 trace: {episode_dir}")
    writer_increments = []
    for i in range(1, len(common)):
        if writer_values[i] > writer_values[i - 1]:
            for cycle in range(writer_values[i - 1] + 1, writer_values[i] + 1):
                writer_increments.append(dict(cycle=cycle, step=common[i]))
    evaluator_increments = []
    for i in range(1, len(common)):
        if evaluator_values[i] > evaluator_values[i - 1]:
            for cycle in range(evaluator_values[i - 1] + 1, evaluator_values[i] + 1):
                evaluator_increments.append(dict(cycle=cycle, step=common[i]))

    final_writer = writer_values[-1]
    final_evaluator = evaluator_values[-1]
    writer_by_cycle = {row["cycle"]: row["step"] for row in writer_increments}
    evaluator_by_cycle = {row["cycle"]: row["step"] for row in evaluator_increments}
    cycle_pairs = []
    for cycle in sorted(writer_by_cycle.keys() & evaluator_by_cycle.keys()):
        memory_step = writer_by_cycle[cycle]
        evaluator_step = evaluator_by_cycle[cycle]
        cycle_pairs.append(dict(
            episode_id=summary["episode_id"],
            task=summary["task"],
            decision_arm=summary["decision_arm"],
            cycle=cycle,
            memory_return_count_step=memory_step,
            evaluator_confirmed_cycle_step=evaluator_step,
            evaluator_minus_memory_step=evaluator_step - memory_step,
        ))

    result = summary["result"]
    return dict(
        episode_id=summary["episode_id"],
        task=summary["task"],
        decision_arm=summary["decision_arm"],
        state_idx=summary["state_idx"],
        initial_state_sha256=summary["initial_state_sha256"],
        policy_seed=summary["policy_seed"],
        success=bool(result["success"]),
        overshot=bool(result["overshot"]),
        total_steps=int(result["total_steps"]),
        final_writer_returns=final_writer,
        final_evaluator_confirmed_cycles=final_evaluator,
        final_count_exact_match=final_writer == final_evaluator,
        writer_return_increments=len(writer_increments),
        evaluator_cycle_increments=len(evaluator_increments),
        writer_evaluator_cycle_pairs=len(cycle_pairs),
        writer_relation_unknown_fraction=sum(x == "unknown" for x in writer_relation_status) / len(common),
        incomplete_history_fraction=sum(bool(x) for x in history_incomplete) / len(common),
        cycle_pairs=cycle_pairs,
        inputs_sha256=input_hashes,
        status_sha256=sha256(episode_dir / "status.json"),
    ), cycle_pairs


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing analysis output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    run_metadata = {}
    rows, pairs = [], []
    cohorts = {}
    for arm, run_dir in RUNS.items():
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("status") != "complete":
            raise ValueError(f"D2 source run is incomplete: {run_dir}")
        decision_protocol = manifest.get("decision_protocol", {})
        if decision_protocol.get("protocol_id") != "libero_mem_decision_dev_v1":
            raise ValueError(f"Unexpected decision protocol in {run_dir}")
        if manifest.get("arguments", {}).get("decision_mode") not in ("shadow", "event"):
            raise ValueError(f"Unexpected decision mode in {run_dir}")
        source_lock_path = ROOT / "resources/manifests/libero_mem_t1_t3_base10_split_v2.json"
        source_lock = read_json(source_lock_path)
        snapshot_lock = read_json(run_dir / "split.lock.json")
        validate_lock(source_lock)
        validate_lock(snapshot_lock)
        if (manifest.get("split_lock_sha256") != source_lock.get("lock_sha256")
                or manifest.get("split_lock_sha256") != snapshot_lock.get("lock_sha256")
                or manifest.get("split_lock_file_sha256") != sha256(source_lock_path)):
            raise ValueError(f"D2 split-lock provenance mismatch: {run_dir}")
        run_metadata[arm] = dict(
            path=run_dir.relative_to(ROOT).as_posix(),
            manifest_sha256=sha256(run_dir / "run_manifest.json"),
            decision_mode=manifest["arguments"]["decision_mode"],
            decision_protocol_sha256=manifest["decision_protocol_sha256"],
            protocol_sha256=manifest["protocol_sha256"],
            split_lock_sha256=manifest["split_lock_sha256"],
            source_split_lock_file_sha256=manifest["split_lock_file_sha256"],
            split_lock_snapshot_sha256=sha256(run_dir / "split.lock.json"),
            adapter_manifest_sha256=manifest["adapter_manifest_sha256"],
            runtime_writer_source_sha256=manifest["treatments"]["F1"]["runtime_writer_source_sha256"],
        )
        episode_dirs = sorted((run_dir / "episodes").iterdir())
        if len(episode_dirs) != 24 or not all(p.is_dir() for p in episode_dirs):
            raise ValueError(f"Unexpected D2 episode roster in {run_dir}: {len(episode_dirs)}")
        for episode_dir in episode_dirs:
            row, episode_pairs = episode_alignment(episode_dir)
            if row["decision_arm"] != arm:
                raise ValueError(f"Episode arm differs from containing run: {episode_dir}")
            rows.append(row)
            pairs.extend(episode_pairs)
        cohorts[arm] = {
            (row["task"], row["state_idx"], row["initial_state_sha256"], row["policy_seed"])
            for row in rows if row["decision_arm"] == arm
        }

    if len(cohorts) != 2 or len(set(map(frozenset, cohorts.values()))) != 1:
        raise ValueError("D2 shadow/event cohorts do not have identical task/state/policy-seed assignments")
    for cohort in cohorts.values():
        if len(cohort) != 24 or {task for task, *_ in cohort} != {"T1", "T3"}:
            raise ValueError("D2 cohort does not contain the expected 24 T1/T3 assignments")

    groups = []
    for arm in sorted(RUNS):
        for task in ("T1", "T3"):
            subset = [row for row in rows if row["decision_arm"] == arm and row["task"] == task]
            lags = [pair["evaluator_minus_memory_step"] for row in subset for pair in row["cycle_pairs"]]
            groups.append(dict(
                decision_arm=arm,
                task=task,
                episodes=len(subset),
                endpoint_count_exact_match=sum(row["final_count_exact_match"] for row in subset),
                writer_returns_total=sum(row["final_writer_returns"] for row in subset),
                evaluator_cycles_total=sum(row["final_evaluator_confirmed_cycles"] for row in subset),
                writer_cycle_increments=sum(row["writer_return_increments"] for row in subset),
                evaluator_cycle_increments=sum(row["evaluator_cycle_increments"] for row in subset),
                ordinal_cycle_pairs=sum(row["writer_evaluator_cycle_pairs"] for row in subset),
                evaluator_minus_memory_step_median=(statistics.median(lags) if lags else None),
                evaluator_minus_memory_step_min=(min(lags) if lags else None),
                evaluator_minus_memory_step_max=(max(lags) if lags else None),
            ))

    summary = dict(
        classification="D2_CLOSED_LOOP_SIMULATOR_EVALUATOR_ALIGNMENT_DESCRIPTIVE_NOT_BLIND_WRITER_ACCURACY",
        episodes=len(rows),
        arms=run_metadata,
        groups=groups,
        interpretation_limit=(
            "This compares online cumulative bowl-return counts with the simulator's independently computed confirmed-cycle count in existing D2 traces. "
            "Ordinal timing differences are descriptive and are not labeled false positives/false negatives. It does not measure identity, intent, target selection, or policy efficacy."
        ),
    )
    (OUTPUT / "episode_alignment.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (OUTPUT / "cycle_pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in pairs),
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# D2 closed-loop memory/evaluator cycle alignment",
        "",
        "Classification: `D2_CLOSED_LOOP_SIMULATOR_EVALUATOR_ALIGNMENT_DESCRIPTIVE_NOT_BLIND_WRITER_ACCURACY`.",
        "",
        "The analysis joins the actual online `bowl_1.returns` memory count with the simulator evaluator's `confirmed_cycles` at each logged observation.",
        "All 48 D2 event/shadow episodes and the memory, observation, and summary artifacts used here passed their recorded hashes.",
        "The evaluator is an offline oracle audit signal; it was not exposed to policy or memory during the rollouts.",
        "",
        "| Arm | Task | Episodes | Endpoint counts exactly equal | Memory returns | Evaluator cycles | Ordinal cycle pairs | Evaluator step minus memory step (median, range) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lag = ("n/a" if group["evaluator_minus_memory_step_median"] is None else
               f"{group['evaluator_minus_memory_step_median']:.1f} ({group['evaluator_minus_memory_step_min']}–{group['evaluator_minus_memory_step_max']})")
        lines.append(
            f"| {group['decision_arm']} | {group['task']} | {group['episodes']} | "
            f"{group['endpoint_count_exact_match']}/{group['episodes']} | {group['writer_returns_total']} | "
            f"{group['evaluator_cycles_total']} | {group['ordinal_cycle_pairs']} | {lag} |"
        )
    lines += [
        "",
        "For paired ordinal events, positive values mean the memory writer's return-count transition appeared before the simulator evaluator confirmed the corresponding cycle; negative values mean it appeared later.",
        "This is a timing description, not a correctness tolerance or an error rate. No independent identity, intent, or target labels are present in this analysis.",
        "",
        "See `episode_alignment.jsonl` for terminal per-episode counts and `cycle_pairs.jsonl` for per-cycle timing pairs.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    script_path = Path(__file__)
    receipt = dict(
        experiment_id="D2-evaluator-cycle-alignment-2026-11-21",
        classification=summary["classification"],
        status="complete",
        protocol_id="libero_mem_decision_dev_v1",
        source_run_metadata=run_metadata,
        episode_count=len(rows),
        episode_artifact_hashes_verified=[
            {"episode_id":row["episode_id"],"status_sha256":row["status_sha256"],"files_sha256":row["inputs_sha256"]}
            for row in rows
        ],
        analysis_script_path=script_path.relative_to(ROOT).as_posix(),
        analysis_script_sha256=sha256(script_path),
        outputs_sha256={
            name:sha256(OUTPUT/name)
            for name in ("summary.json","report.md","episode_alignment.jsonl","cycle_pairs.jsonl")
        },
        interpretation_limit=summary["interpretation_limit"],
    )
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
