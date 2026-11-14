#!/usr/bin/env python
"""Replay the decision scheduler on a hash-checked relation index."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mem_eval_io import write_json
from src.evaluation.libero_mem_protocol import source_digest
from src.evaluation.memory_supervisor import (
    DEFAULT_DECISION_PROTOCOL, MemorySupervisor, execution_treatment, load_decision_protocol,
)
from src.utils.file_hash import sha256_file


def replay(index, protocol, expected_episodes):
    treatment = execution_treatment(protocol, "event", "snapshot")
    with ((index / "memory_trace.jsonl").open(encoding="utf-8") as journal,
          (index / "samples.jsonl").open(encoding="utf-8") as samples):
        results = {}
        episode, step, query_at = None, -1, 0
        for trace_line, sample_line in zip(journal, samples, strict=True):
            row, sample = json.loads(trace_line), json.loads(sample_line)
            snapshot = row["snapshot"]
            key, new_step = row["episode_id"], row["step"]
            if ((key, new_step) != (sample["episode_id"], sample["step"])
                    or key != snapshot["episode_id"] or new_step != snapshot["as_of_observation"]):
                raise ValueError("Trace and sample chronology disagree")
            if key != episode:
                if key in results or key not in expected_episodes or new_step != 0:
                    raise ValueError("Unexpected or noncontiguous episode")
                episode, step, query_at = key, -1, 0
                supervisor = MemorySupervisor(treatment)
                result = results[key] = dict(frames=0, queries=0, requests=[], trigger_types=Counter())
            if new_step != step + 1:
                raise ValueError("Trace is missing an observation")
            step = new_step
            result["frames"] += 1
            if (step + 1 == expected_episodes[key] or step >= protocol["policy_actions"]
                    or "abstained_at" in result):
                continue
            early = False
            if step > query_at:
                changes = supervisor.evaluate(sample["instruction"], None, snapshot)
                for change in changes:
                    before, after = change["before"], change["after"]
                    change["observed_return_increase"] = bool(
                        snapshot["profile"]["count_cycles"] and after and after["value"][0] is not None
                        and after["value"][0] > (before["value"][0]
                            if before and before["value"][0] is not None else 0))
                    category = ("observed_return_increase" if change["observed_return_increase"]
                                else "other_semantic_change")
                    result["trigger_types"][category] += 1
                early = bool(changes) and step - query_at < protocol["parent_execution_horizon"]
                if early:
                    result["requests"].append(dict(step=step, changes=changes))
            if step == 0 or early or step - query_at >= protocol["parent_execution_horizon"]:
                if result["queries"] == treatment.max_queries:
                    result["abstained_at"] = step
                else:
                    supervisor.bind(result["queries"], sample["instruction"], None, snapshot)
                    result["queries"] += 1
                    query_at = step
        if {key: row["frames"] for key, row in results.items()} != expected_episodes:
            raise ValueError("Trace sizes do not match index manifest")
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--milestones", type=Path, help="Optional independently sourced author-milestone audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new audit output file")
    manifest_path = args.index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["writer"] != "rgb_relations_v1":
        raise ValueError("Expected a complete relation index")
    for name in ("memory_trace.jsonl", "samples.jsonl"):
        if sha256_file(args.index / name) != manifest["artifacts_sha256"][name]:
            raise ValueError("Index artifact hash differs: " + name)
    protocol = load_decision_protocol()
    episodes = replay(args.index, protocol, {row["id"]: row["frames"] for row in manifest["episodes"]})
    for episode in episodes.values():
        actions = min(episode["frames"] - 1, protocol["policy_actions"])
        for key, prefix in (("text_queries", protocol["parent_execution_horizon"]),
                            ("periodic_queries", protocol["periodic_prefix_actions"])):
            episode[key] = (actions + prefix - 1) // prefix
    report = dict(classification="OPEN_LOOP_DEVELOPMENT_NOT_POLICY_SCORE",
                  exposure="snapshot", index_sha256=sha256_file(manifest_path),
                  decision_protocol_sha256=source_digest(DEFAULT_DECISION_PROTOCOL), episodes=episodes,
                  source_sha256={name: source_digest(ROOT / name) for name in (
                      "scripts/audit_memory_decisions.py", "src/evaluation/memory_supervisor.py",
                      "src/memory/decision_context.py")})
    if args.milestones:
        labels = json.loads(args.milestones.read_text(encoding="utf-8"))
        if (labels["classification"] != "DEVELOPMENT_MILESTONES_NOT_POLICY_SCORE"
                or labels["index_sha256"] != report["index_sha256"]
                or set(episodes) != {row["episode"] for row in labels["episodes"]}):
            raise ValueError("Milestone audit does not identify this index")
        report["milestones_sha256"] = sha256_file(args.milestones)
        # Author milestone delays describe writer availability. They do not label
        # whether a policy would have selected the right next action.
        delay = [match["delay"] for row in labels["episodes"] for match in row["matches"]]
        linked = 0
        for row in labels["episodes"]:
            requests = episodes[row["episode"]]["requests"]
            for match in row["matches"]:
                linked += any(change["observed_return_increase"] and
                              match["predicted"] <= change["first_observation"] <= request["step"] <
                              match["predicted"] + protocol["parent_execution_horizon"]
                              for request in requests for change in request["changes"])
        report["writer_milestones"] = dict(
            matched=len(delay), median_delay_frames=median(delay) if delay else None,
            missing=sum(row["false_negative"] for row in labels["episodes"]),
            extra=sum(row["false_positive"] for row in labels["episodes"]),
            matched_writer_events_near_early_request=linked)
    write_json(args.output, report)
    print(f"{len(episodes)} stored episodes, {sum(r['frames'] for r in episodes.values())} frames, "
          f"{sum(len(r['requests']) for r in episodes.values())} early requests")


if __name__ == "__main__":
    main()
