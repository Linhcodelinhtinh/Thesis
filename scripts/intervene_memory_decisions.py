#!/usr/bin/env python
"""Probe the offline count precondition with controlled fact interventions."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mem_eval_io import evaluation_roster, write_json
from src.evaluation.proposal_assessment import count_precondition, return_cycle_target
from src.memory.decision_context import DecisionFact
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file


def probe(fact, target):
    count, armed, identity = fact.value
    if count is None or fact.validity != "supported" or armed:
        return None
    versions = {
        "original": fact,
        "exact_repeat": fact,
        "rename_id": replace(fact, key="renamed_subject"),
        "refresh_evidence_id": replace(fact, evidence_ids=tuple("renamed:" + eid for eid in fact.evidence_ids)),
        "change_count": replace(fact, value=(target if count < target else target - 1, armed, identity)),
        "invalidate_evidence": replace(fact, value=(None, armed, identity), validity="unknown", evidence_ids=()),
    }
    return {name: count_precondition(value, target, "start_cycle").value for name, value in versions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--task", choices=("T1", "T3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new intervention output")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay["classification"] != "OPEN_LOOP_DEVELOPMENT_NOT_POLICY_SCORE" or replay["exposure"] != "snapshot":
        raise ValueError("Expected the offline snapshot replay artifact")
    _, _, roster = evaluation_roster(DEFAULT_AUTHOR_MEM_LOCK, "seen", [args.task], "0")
    target = return_cycle_target(roster[0]["instruction"])
    prefix = roster[0]["task_name"] + "_demo_"
    counts = Counter()
    for key, episode in replay["episodes"].items():
        if not key.startswith(prefix):
            raise ValueError("Replay task does not match the requested instruction")
        for request in episode["requests"]:
            for change in request["changes"]:
                data = change["after"]
                if data is None:
                    continue
                fact = DecisionFact(data["key"], tuple(data["value"]), data["validity"],
                                    tuple(data["event_ids"]), tuple(data["evidence_ids"]))
                if fact.version != data["version"]:
                    raise ValueError("Stored decision fact version differs")
                outcomes = probe(fact, target)
                if outcomes is None:
                    counts["excluded_unsupported_or_armed"] += 1
                    continue
                original = outcomes["original"]
                counts["evaluated"] += 1
                counts["exact_repeat_agrees"] += outcomes["exact_repeat"] == original
                counts["id_rename_agrees"] += outcomes["rename_id"] == original
                counts["evidence_refresh_agrees"] += outcomes["refresh_evidence_id"] == original
                counts["count_change_flips"] += outcomes["change_count"] != original
                counts["invalidation_yields_unknown"] += outcomes["invalidate_evidence"] == "unknown"
    write_json(args.output, dict(classification="OFFLINE_RULE_PROBE_NOT_POLICY_SCORE",
                                task=args.task, target=target, replay_sha256=sha256_file(args.replay),
                                rule_sha256=sha256_file(ROOT / "src/evaluation/proposal_assessment.py"),
                                counts=counts))
    print(dict(counts))


if __name__ == "__main__":
    main()
