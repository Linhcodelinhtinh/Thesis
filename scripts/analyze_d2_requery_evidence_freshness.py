#!/usr/bin/env python
"""Audit evidence freshness for actual D2 event-mode requery decisions.

This is an offline descriptive analysis of completed rollout traces. It does
not judge whether a requery was correct and does not feed evaluator state back
to the policy or memory writer.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import validate_lock  # noqa: E402

RUN = ROOT / "experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl"
SHADOW_RUN = ROOT / "experiments/evaluation/decision_shadow_dev12_t1_t3_20261120_wsl"
SOURCE_LOCK = ROOT / "resources/manifests/libero_mem_t1_t3_base10_split_v2.json"
DECISION_PROTOCOL = ROOT / "configs/benchmarks/libero_mem_decision_dev_v1.json"
OUTPUT = ROOT / "experiments/evaluation/d2_requery_evidence_freshness_20261121_v2"


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


def ids_from_change(change, field):
    before = change.get("before") or {}
    after = change.get("after") or {}
    before_ids = set(before.get(field, []))
    after_ids = set(after.get(field, []))
    return before_ids, after_ids


def observation_step(evidence_id):
    parts = evidence_id.split(":")
    if len(parts) > 1 and parts[0] in ("rgb", "gripper"):
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def summarize_decision(row, episode_id, task, state_idx, eval_by_step):
    changes = []
    before_events, after_events = set(), set()
    before_evidence, after_evidence = set(), set()
    text_only_history_incomplete = True
    for change in row.get("changes", []):
        before = change.get("before") or {}
        after = change.get("after") or {}
        b_events, a_events = ids_from_change(change, "event_ids")
        b_evidence, a_evidence = ids_from_change(change, "evidence_ids")
        before_events.update(b_events)
        after_events.update(a_events)
        before_evidence.update(b_evidence)
        after_evidence.update(a_evidence)
        before_value = before.get("value")
        after_value = after.get("value")
        if not (
            isinstance(before_value, list)
            and isinstance(after_value, list)
            and len(before_value) == 1
            and len(after_value) == 1
            and "History incomplete." not in before_value[0]
            and "History incomplete." in after_value[0]
            and before_value[0].replace(". History incomplete.", ".") == after_value[0].replace(" History incomplete.", "")
            and not (a_events - b_events)
            and not (a_evidence - b_evidence)
        ):
            text_only_history_incomplete = False
        changes.append(dict(
            key=change.get("key"),
            first_observation=change.get("first_observation"),
            before_value=before_value,
            after_value=after_value,
            new_event_ids=sorted(a_events - b_events),
            new_evidence_ids=sorted(a_evidence - b_evidence),
            after_evidence_ids=sorted(a_evidence),
        ))

    step = int(row["step"])
    evidence_steps = [observation_step(eid) for eid in after_evidence]
    evidence_steps = [value for value in evidence_steps if value is not None]
    latest_step = max(evidence_steps) if evidence_steps else None
    # At query step s, the latest completed observation normally has index s-1.
    evidence_age_actions = None if latest_step is None else step - 1 - latest_step
    if evidence_age_actions is not None and evidence_age_actions < 0:
        raise ValueError(f"Evidence ID points beyond the query observation: {episode_id} step {step}")
    eval_row = eval_by_step.get(step)
    return dict(
        episode_id=episode_id,
        task=task,
        state_idx=state_idx,
        step=step,
        query_id=int(row["query_id"]),
        reason=row.get("reason"),
        prefix_actions=int(row.get("prefix_actions", 0)),
        change_count=len(changes),
        changes=changes,
        after_event_id_count=len(after_events),
        new_event_id_count=len(after_events - before_events),
        after_evidence_id_count=len(after_evidence),
        new_evidence_id_count=len(after_evidence - before_evidence),
        latest_evidence_observation_step=latest_step,
        evidence_age_actions=evidence_age_actions,
        evaluator_confirmed_cycles_at_query=(
            None if eval_row is None else int(eval_row["confirmed_cycles"])
        ),
        history_incomplete_only_change=bool(changes) and text_only_history_incomplete,
        supervisor_ms=row.get("supervisor_ms"),
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    event_manifest = read_json(RUN / "run_manifest.json")
    shadow_manifest = read_json(SHADOW_RUN / "run_manifest.json")
    lock = read_json(SOURCE_LOCK)
    validate_lock(lock)
    if event_manifest.get("status") != "complete" or shadow_manifest.get("status") != "complete":
        raise ValueError("Both D2 source arms must be complete")
    if event_manifest.get("arguments", {}).get("decision_mode") != "event":
        raise ValueError("Source run is not event mode")
    if shadow_manifest.get("arguments", {}).get("decision_mode") != "shadow":
        raise ValueError("Paired source run is not shadow mode")
    if event_manifest.get("decision_protocol", {}).get("protocol_id") != "libero_mem_decision_dev_v1":
        raise ValueError("Unexpected D2 decision protocol")
    for manifest, run_dir in ((event_manifest, RUN), (shadow_manifest, SHADOW_RUN)):
        if manifest.get("decision_protocol_sha256") != sha256(DECISION_PROTOCOL):
            raise ValueError(f"Decision protocol hash mismatch: {run_dir}")
        if manifest.get("split_lock_sha256") != lock.get("lock_sha256"):
            raise ValueError(f"Semantic split-lock hash mismatch: {run_dir}")
        if manifest.get("split_lock_file_sha256") != sha256(SOURCE_LOCK):
            raise ValueError(f"Split-lock file hash mismatch: {run_dir}")

    event_episodes = sorted((RUN / "episodes").iterdir())
    shadow_episodes = sorted((SHADOW_RUN / "episodes").iterdir())
    if len(event_episodes) != 24 or len(shadow_episodes) != 24:
        raise ValueError("D2 source roster must contain 24 event and 24 shadow episodes")
    shadow_by_name = {path.name: path for path in shadow_episodes}
    if {path.name for path in event_episodes} != set(shadow_by_name):
        raise ValueError("D2 event/shadow episode names differ")

    rows, episode_rows = [], []
    cohort = set()
    for episode_dir in event_episodes:
        status = read_json(episode_dir / "status.json")
        if status.get("status") != "complete":
            raise ValueError(f"Incomplete D2 event episode: {episode_dir}")
        for name in ("summary.json", "observations.jsonl"):
            expected = status.get("files_sha256", {}).get(name)
            if not expected or sha256(episode_dir / name) != expected:
                raise ValueError(f"D2 event source hash mismatch: {episode_dir.name}/{name}")
        decisions_path = episode_dir / "decisions.jsonl"
        if decisions_path.exists():
            expected = status.get("files_sha256", {}).get("decisions.jsonl")
            if not expected or sha256(decisions_path) != expected:
                raise ValueError(f"D2 decision trace hash mismatch: {episode_dir.name}")
            decisions = read_jsonl(decisions_path)
        else:
            decisions = []

        summary = read_json(episode_dir / "summary.json")
        shadow_dir = shadow_by_name[episode_dir.name]
        shadow_status = read_json(shadow_dir / "status.json")
        shadow_summary_path = shadow_dir / "summary.json"
        shadow_summary = read_json(shadow_summary_path)
        if shadow_status.get("status") != "complete":
            raise ValueError(f"Incomplete paired D2 shadow episode: {shadow_dir}")
        shadow_summary_hash = shadow_status.get("files_sha256", {}).get("summary.json")
        if not shadow_summary_hash or sha256(shadow_summary_path) != shadow_summary_hash:
            raise ValueError(f"Paired shadow summary hash mismatch: {shadow_dir.name}")
        observation_rows = read_jsonl(episode_dir / "observations.jsonl")
        eval_by_step = {int(row["step"]): row["info"]["eval_result"] for row in observation_rows}
        identity = (
            summary["task"], int(summary["state_idx"]),
            summary["initial_state_sha256"], int(summary["policy_seed"]),
        )
        if summary["task"] not in ("T1", "T3") or not 0 <= int(summary["state_idx"]) <= 11:
            raise ValueError(f"Unexpected D2 development assignment: {episode_dir.name}")
        shadow_identity = (
            shadow_summary["task"], int(shadow_summary["state_idx"]),
            shadow_summary["initial_state_sha256"], int(shadow_summary["policy_seed"]),
        )
        if identity != shadow_identity:
            raise ValueError(f"Paired D2 event/shadow state or seed differs: {episode_dir.name}")
        cohort.add(identity)
        requery_rows = [row for row in decisions if row.get("decision") == "requery"]
        for row in requery_rows:
            if not row.get("changes"):
                raise ValueError(f"Requery without logged semantic changes: {episode_dir.name}")
            rows.append(summarize_decision(
                row, summary["episode_id"], summary["task"], int(summary["state_idx"]), eval_by_step
            ))
        episode_rows.append(dict(
            episode_id=summary["episode_id"],
            task=summary["task"],
            state_idx=int(summary["state_idx"]),
            initial_state_sha256=summary["initial_state_sha256"],
            policy_seed=int(summary["policy_seed"]),
            success=bool(summary["result"]["success"]),
            confirmed_cycles=int(summary["result"]["completed_subgoals"]),
            requery_count=len(requery_rows),
            decisions_sha256=(sha256(decisions_path) if decisions_path.exists() else None),
            status_sha256=sha256(episode_dir / "status.json"),
            paired_shadow_status_sha256=sha256(shadow_dir / "status.json"),
            paired_shadow_summary_sha256=sha256(shadow_summary_path),
        ))

    expected_assignments = {(task, state) for task in ("T1", "T3") for state in range(12)}
    if len(cohort) != 24 or {(task, state) for task, state, _, _ in cohort} != expected_assignments or len(rows) != 56:
        raise ValueError(f"Unexpected D2 event cohort/requery count: {len(cohort)} assignments, {len(rows)} requery decisions")
    if sum(row["requery_count"] > 0 for row in episode_rows) != 22:
        raise ValueError("Expected event requery in 22 of 24 episodes")
    if any(row["after_evidence_id_count"] == 0 for row in rows):
        raise ValueError("At least one event requery lacks carried evidence IDs")

    age_values = [row["evidence_age_actions"] for row in rows if row["evidence_age_actions"] is not None]
    no_new_evidence = [row for row in rows if row["new_evidence_id_count"] == 0]
    if len(no_new_evidence) != 15 or not all(row["history_incomplete_only_change"] for row in no_new_evidence):
        raise ValueError("The stale-evidence subset differs from the inspected history-incomplete pattern")
    if any(row["new_event_id_count"] != 0 for row in no_new_evidence):
        raise ValueError("No-new-evidence requery unexpectedly added an event ID")
    no_new_ages = [row["evidence_age_actions"] for row in no_new_evidence]
    if any(row["after_evidence_id_count"] != 9 for row in no_new_evidence):
        raise ValueError("Unexpected retained evidence-ID count in the history-incomplete subset")
    if any(row["evidence_age_actions"] != row["step"] - 3 for row in no_new_evidence):
        raise ValueError("Unexpected source observation range in the history-incomplete subset")
    if any(row["evidence_age_actions"] != 0 for row in rows if row["new_evidence_id_count"] > 0):
        raise ValueError("A requery with new evidence IDs did not cite the latest pre-query observation")

    groups = []
    for task in ("T1", "T3"):
        subset = [row for row in rows if row["task"] == task]
        groups.append(dict(
            task=task,
            episodes=sum(row["task"] == task for row in episode_rows),
            requery_count=len(subset),
            all_requeries_with_carried_evidence=sum(row["after_evidence_id_count"] > 0 for row in subset),
            requery_with_new_evidence_ids=sum(row["new_evidence_id_count"] > 0 for row in subset),
            requery_with_only_carried_evidence=sum(row["new_evidence_id_count"] == 0 for row in subset),
            history_incomplete_only_requeries=sum(row["history_incomplete_only_change"] for row in subset),
            latest_evidence_age_median_actions=(
                statistics.median(row["evidence_age_actions"] for row in subset
                                  if row["evidence_age_actions"] is not None)
                if any(row["evidence_age_actions"] is not None for row in subset) else None
            ),
            latest_evidence_age_max_actions=max(
                (row["evidence_age_actions"] for row in subset if row["evidence_age_actions"] is not None),
                default=None,
            ),
        ))

    summary = dict(
        classification="D2_EVENT_REQUERY_EVIDENCE_FRESHNESS_DESCRIPTIVE_NOT_DECISION_ACCURACY",
        event_episodes=len(episode_rows),
        episodes_with_requery=sum(row["requery_count"] > 0 for row in episode_rows),
        total_requeries=len(rows),
        requery_with_any_carried_evidence=sum(row["after_evidence_id_count"] > 0 for row in rows),
        requery_with_new_evidence_ids=sum(row["new_evidence_id_count"] > 0 for row in rows),
        requery_without_new_evidence_ids=len(no_new_evidence),
        history_incomplete_only_requeries=sum(row["history_incomplete_only_change"] for row in rows),
        no_new_evidence_retained_id_count=9,
        no_new_evidence_age_actions=dict(
            median=statistics.median(no_new_ages), minimum=min(no_new_ages), maximum=max(no_new_ages)
        ),
        latest_evidence_age_actions=dict(
            median=statistics.median(age_values), minimum=min(age_values), maximum=max(age_values)
        ),
        tasks=groups,
        episode_rows=episode_rows,
        interpretation_limit=(
            "Evidence-ID presence means a fact had recorded provenance; it does not mean the evidence was newly observed at the trigger. "
            "Evidence age is computed from the largest rgb/gripper observation index in the post-change fact IDs versus query step-1. "
            "No freshness tolerance, decision-correctness labels, or causal policy outcome test was specified."
        ),
        source_run_manifests=dict(
            event_sha256=sha256(RUN / "run_manifest.json"),
            shadow_sha256=sha256(SHADOW_RUN / "run_manifest.json"),
            split_lock_sha256=lock["lock_sha256"],
            split_lock_file_sha256=sha256(SOURCE_LOCK),
            decision_protocol_sha256=sha256(DECISION_PROTOCOL),
        ),
    )
    (OUTPUT / "per_requery.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (OUTPUT / "episode_summary.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# D2 event requery evidence freshness audit",
        "",
        "Classification: `D2_EVENT_REQUERY_EVIDENCE_FRESHNESS_DESCRIPTIVE_NOT_DECISION_ACCURACY`.",
        "",
        "The audit reads completed T1/T3 seen-development event traces (seed 42) and compares evidence IDs present after each requery with IDs present before that decision.",
        "All 24 event episodes and required decision, observation, and summary artifacts passed source hashes; the exact paired shadow task/state/initial-state/seed roster and split lock were also checked.",
        "",
        f"Across {len(rows)} requery decisions in {summary['episodes_with_requery']}/24 episodes, all had at least one evidence ID in the resulting fact. "
        f"{summary['requery_with_new_evidence_ids']} decisions added evidence IDs; {len(no_new_evidence)} reused only IDs already carried before the trigger.",
        f"All {summary['history_incomplete_only_requeries']} no-new-ID decisions only appended `History incomplete.` to the prior bowl fact, added no event/evidence IDs, and retained the same {summary['no_new_evidence_retained_id_count']} source IDs from observation steps 0–2.",
        f"For those {len(no_new_evidence)} decisions, evidence age had median {statistics.median(no_new_ages):.0f} actions and range {min(no_new_ages)}–{max(no_new_ages)} actions.",
        f"Latest recorded evidence age relative to query step minus one had median {statistics.median(age_values):.0f} actions and range {min(age_values)}–{max(age_values)}.",
        "",
        "| Task | Episodes | Requeries | With carried evidence IDs | Added new evidence IDs | Only carried IDs | History-incomplete-only | Evidence age median / max (actions) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['task']} | {group['episodes']} | {group['requery_count']} | "
            f"{group['all_requeries_with_carried_evidence']} | {group['requery_with_new_evidence_ids']} | "
            f"{group['requery_with_only_carried_evidence']} | {group['history_incomplete_only_requeries']} | "
            f"{group['latest_evidence_age_median_actions']:.0f} / {group['latest_evidence_age_max_actions']} |"
        )
    lines += [
        "",
        "This refines the statement that every requery carried evidence IDs: ID presence is not the same as fresh evidence. The 15 history-incomplete-only decisions had no newly added evidence or event IDs; they retained provenance from earlier observations. This is a trace description, not an FP/FN rate or an assessment of whether those decisions were correct.",
        "Evidence-age values are exact action-index differences under the stated query-step-1 convention; no threshold was predeclared, so the report does not classify them as fresh/stale accuracy errors.",
        "",
        "See `per_requery.jsonl` for decision-level changes and evidence ages, and `episode_summary.jsonl` for the complete event cohort.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    script_path = Path(__file__)
    receipt = dict(
        experiment_id="D2-event-requery-evidence-freshness-2026-11-21",
        classification=summary["classification"],
        status="complete",
        episode_count=len(episode_rows),
        requery_count=len(rows),
        source_hashes=summary["source_run_manifests"],
        episode_artifact_hashes=[
            dict(episode_id=row["episode_id"], status_sha256=row["status_sha256"], decisions_sha256=row["decisions_sha256"],
                 paired_shadow_status_sha256=row["paired_shadow_status_sha256"],
                 paired_shadow_summary_sha256=row["paired_shadow_summary_sha256"])
            for row in episode_rows
        ],
        analysis_script_path=script_path.relative_to(ROOT).as_posix(),
        analysis_script_sha256=sha256(script_path),
        output_sha256={
            name: sha256(OUTPUT / name)
            for name in ("summary.json", "report.md", "per_requery.jsonl", "episode_summary.jsonl")
        },
        interpretation_limit=summary["interpretation_limit"],
    )
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
