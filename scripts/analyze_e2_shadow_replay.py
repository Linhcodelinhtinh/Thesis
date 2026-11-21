#!/usr/bin/env python
"""Replay projected-memory shadow decisions on the completed E2 F1 traces.

This is an offline observer analysis. It never reruns the policy or changes a
recorded action, so it is not a closed-loop event-mode efficacy experiment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.libero_mem_protocol import source_digest
from src.evaluation.memory_supervisor import (
    MemorySupervisor, execution_treatment, load_decision_protocol,
)
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.models import (
    EvidenceSource, IdentityStatus, ObjectIdentity, V2MemoryEvent,
    VisibilityStatus, _freeze_value,
)
from src.memory.relation_memory import RelationMemorySnapshot, RelationState
from src.memory.visual_pipeline import TOKENIZER_SHA256, load_prompt_tokenizer
from src.utils.file_hash import sha256_file


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def event_from_dict(row):
    if row is None:
        return None
    return V2MemoryEvent(
        id=row["id"], type=row["type"], occurred_at=row["occurred_at"],
        available_at=row["available_at"], source=EvidenceSource(row["source"]),
        probability=row.get("probability"), supersedes=row.get("supersedes"),
        evidence_ids=tuple(row.get("evidence_ids", ())), details=row.get("details"),
    )


def snapshot_from_dict(raw):
    profile = MemoryProfile(**raw["profile"])
    objects = {
        key: ObjectIdentity(
            object_id=key, origin_anchor=value["origin_anchor"],
            identity_status=IdentityStatus(value["identity_status"]),
            visibility=VisibilityStatus(value["visibility"]),
            current_location_desc=value.get("current_location_desc"),
            last_confirmed_in=value.get("last_confirmed_in"),
            metadata=value.get("metadata"),
        )
        for key, value in raw["objects"].items()
    }
    relations = {}
    for key, value in raw["relations"].items():
        relations[key] = RelationState(
            status=value["status"], reference=value.get("reference"),
            last_event=event_from_dict(value.get("last_event")),
            last_return=event_from_dict(value.get("last_return")),
            departure=event_from_dict(value.get("departure")),
            candidate_frames=value["candidate_frames"], returns=value["returns"],
            history_complete=value["history_complete"], armed=value["armed"],
            historical_relation_valid=value["historical_relation_valid"],
            current_evidence_ids=tuple(value.get("current_evidence_ids", ())),
            # This field was added after E2; renderer and decision logic do not read it.
            origin_verified=value.get("origin_verified", True),
        )
    return RelationMemorySnapshot(
        episode_id=raw["episode_id"], as_of_observation=raw["as_of_observation"],
        last_consumed_action=raw["last_consumed_action"], profile=profile,
        objects=MappingProxyType(objects), relations=MappingProxyType(relations),
        evidence=_freeze_value(raw["evidence"]),
        recent_events=tuple(event_from_dict(row) for row in raw["recent_events"]),
    )


def check_file_hash(directory, status, name):
    path = directory / name
    expected = status["files_sha256"].get(name)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"E2 artifact hash mismatch: {path}")
    return expected


def projected_return_count(fact):
    if not fact or not fact.get("value"):
        return None
    match = re.search(r": (\d+) observed returns\.", fact["value"][0])
    return int(match.group(1)) if match else None


def analyze_episode(directory, batch_manifest, renderer, protocol):
    status_path = directory / "status.json"
    summary_path = directory / "summary.json"
    status, summary = read_json(status_path), read_json(summary_path)
    if (status["status"] != "complete" or summary["condition"] != "F1"
            or summary["split"] != "unseen" or summary["task"] not in ("T1", "T3")
            or status["episode_id"] != directory.name):
        raise ValueError(f"Unexpected or incomplete E2 F1 episode: {directory}")
    if (summary["policy_seed"] != batch_manifest["arguments"]["policy_seed"]
            or status["state_idx"] != summary["state_idx"]
            or status["task"] != summary["task"]):
        raise ValueError(f"E2 identity mismatch: {directory}")
    artifact_hashes = {
        name: check_file_hash(directory, status, name)
        for name in ("memory_trace.jsonl", "queries.jsonl", "observations.jsonl", "summary.json")
    }
    recorded_actions = summary["result"]["total_steps"]
    if (not 1 <= recorded_actions <= protocol["policy_actions"]
            or summary["result"]["decision_metrics"] is not None):
        raise ValueError(f"Expected a completed fixed-horizon E2 rollout: {directory}")
    if (batch_manifest["arguments"].get("decision_mode") is not None
            or batch_manifest["protocol"]["rollout"]["execution_horizon"]
            != protocol["parent_execution_horizon"]):
        raise ValueError(f"E2 query schedule differs from the shadow replay protocol: {directory}")

    query_rows = list(jsonl(directory / "queries.jsonl"))
    if not query_rows or len({row["step"] for row in query_rows}) != len(query_rows):
        raise ValueError(f"Missing or duplicate query rows: {directory}")
    if query_rows[0]["step"] != 0:
        raise ValueError(f"E2 rollout does not begin with a query: {directory}")
    query_at = {row["step"]: row for row in query_rows}
    query_steps = sorted(query_at)
    horizon = protocol["parent_execution_horizon"]
    if (any(b - a != horizon for a, b in zip(query_steps, query_steps[1:]))
            or query_steps[-1] >= recorded_actions):
        raise ValueError(f"E2 query cadence is not the fixed horizon: {directory}")
    instruction = summary["instruction"]
    treatment = execution_treatment(protocol, "shadow", "projected")
    supervisor = MemorySupervisor(treatment)
    active_query_step = None
    last_step = -1
    frames = 0
    checked_prompts = 0
    changes = []

    for trace_row in jsonl(directory / "memory_trace.jsonl"):
        step = trace_row["step"]
        raw = trace_row["snapshot"]
        if step != last_step + 1 or step != raw["as_of_observation"]:
            raise ValueError(f"Noncontiguous E2 memory trace: {directory} step={step}")
        if raw["episode_id"] != directory.name:
            raise ValueError(f"Snapshot episode ID mismatch: {directory}")
        snapshot = snapshot_from_dict(raw)
        frames += 1

        if step > 0 and active_query_step is not None and step < recorded_actions:
            deltas = supervisor.evaluate(instruction, snapshot, raw, renderer)
            for delta in deltas:
                early = step - active_query_step < horizon
                before, after = delta["before"], delta["after"]
                before_returns = projected_return_count(before)
                after_returns = projected_return_count(after)
                is_return_increase = bool(
                    raw["profile"]["count_cycles"] and after_returns is not None
                    and after_returns > (before_returns if before_returns is not None else 0)
                )
                changes.append({
                    "step": step, "query_step": active_query_step,
                    "early_candidate": early,
                    "change_type": "observed_return_increase" if is_return_increase else "other_semantic_change",
                    "key": delta["key"], "before": before, "after": after,
                    "event_ids": list(after["event_ids"] if after else ()),
                    "evidence_ids": list(after["evidence_ids"] if after else ()),
                    "evidence_refs_present": bool(after) and all(
                        eid in raw["evidence"] for eid in after["evidence_ids"]
                    ),
                })

        if step in query_at:
            query = query_at[step]
            rendered = renderer.render(instruction, snapshot)
            if rendered.text != query["prompt"]:
                raise ValueError(f"Projected prompt replay differs at {directory.name}/{step}")
            if rendered.text != query["render"]["text"]:
                raise ValueError(f"Rendered metadata text differs at {directory.name}/{step}")
            if tuple(rendered.facts_included) != tuple(query["render"]["facts_included"]):
                raise ValueError(f"Rendered fact list differs at {directory.name}/{step}")
            supervisor.bind(checked_prompts, instruction, snapshot, raw, renderer, rendered)
            active_query_step = step
            checked_prompts += 1
        last_step = step

    if frames != recorded_actions + 1 or checked_prompts != len(query_rows):
        raise ValueError(f"Incomplete E2 memory/query replay: {directory}")
    return {
        "episode": directory.name, "task": summary["task"], "split": summary["split"],
        "state_idx": summary["state_idx"], "source_demo": summary["source_demo"],
        "initial_state_sha256": summary["initial_state_sha256"],
        "policy_seed": summary["policy_seed"], "frames": frames,
        "recorded_actions": recorded_actions,
        "recorded_queries": len(query_rows), "prompts_exactly_replayed": checked_prompts,
        "query_steps": query_steps, "semantic_changes": changes,
        "artifacts_sha256": artifact_hashes,
        "status_sha256": sha256_file(status_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "experiments/evaluation/e2_unseen_author_v2")
    parser.add_argument("--adapter-dir", type=Path,
                        default=ROOT / "experiments/training/results_f1_author_split_v2/output_f1/pretrained_model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new replay output file")

    protocol = load_decision_protocol()
    tokenizer_path = args.adapter_dir / "tokenizer.json"
    tokenizer = load_prompt_tokenizer(tokenizer_path)
    adapter_manifest = read_json(args.adapter_dir / "mem_adapter_manifest.json")
    provenance = adapter_manifest["memory_provenance"][0]
    writer_config = provenance["writer_config"]
    if (provenance["writer"] != "rgb_relations_v1"
            or writer_config != {"projection": "relation_text_v1", "strategy": "relevant",
                                 "max_total_tokens": 48}):
        raise ValueError("E2 F1 adapter is not the expected Relation V1 projection")
    renderer = RelationMemoryRenderer(tokenizer, max_total_tokens=writer_config["max_total_tokens"],
                                      strategy=writer_config["strategy"])
    batch_paths = sorted(args.run_root.glob("batch-*/run_manifest.json"))
    if len(batch_paths) != 10:
        raise ValueError(f"Expected ten completed E2 batch manifests, found {len(batch_paths)}")
    manifests, manifest_hashes, code_snapshots = {}, {}, set()
    for path in batch_paths:
        manifest = read_json(path)
        if (manifest["status"] != "complete" or manifest["arguments"].get("split") != "unseen"
                or manifest["arguments"].get("decision_mode") is not None
                or manifest["arguments"].get("conditions") != ["F0", "F1"]):
            raise ValueError(f"Unexpected E2 run manifest: {path}")
        if sha256_file(path.parent / "code_snapshot.zip") != manifest["code_snapshot_sha256"]:
            raise ValueError(f"E2 source snapshot hash mismatch: {path.parent}")
        if manifest["protocol"]["rollout"]["execution_horizon"] != protocol["parent_execution_horizon"]:
            raise ValueError(f"Unexpected E2 action query horizon: {path}")
        manifests[path.parent] = manifest
        manifest_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
        code_snapshots.add(manifest["code_snapshot_sha256"])
    if len(code_snapshots) != 1:
        raise ValueError("E2 batches do not share the same source snapshot")

    episode_dirs = sorted(args.run_root.glob("batch-*/episodes/T[13]_F1_unseen_state*"))
    if len(episode_dirs) != 40:
        raise ValueError(f"Expected 40 completed E2 F1 episodes, found {len(episode_dirs)}")
    episodes = []
    for directory in episode_dirs:
        batch_manifest = manifests[directory.parent.parent]
        episodes.append(analyze_episode(directory, batch_manifest, renderer, protocol))
    if {(row["task"], row["state_idx"]) for row in episodes} != {
            (task, index) for task in ("T1", "T3") for index in range(20)}:
        raise ValueError("E2 F1 episodes do not cover the 20 paired state groups")
    state_pairs = {}
    for row in episodes:
        state_pairs.setdefault(row["state_idx"], {})[row["task"]] = row["initial_state_sha256"]
    if any(set(pair) != {"T1", "T3"} or pair["T1"] != pair["T3"] for pair in state_pairs.values()):
        raise ValueError("E2 T1/T3 episodes do not share identical paired state hashes")

    all_changes = [change for episode in episodes for change in episode["semantic_changes"]]
    early = [change for change in all_changes if change["early_candidate"]]
    query_counts = Counter(row["recorded_queries"] for row in episodes)
    trigger_counts = Counter(change["change_type"] for change in all_changes)
    current_sources = (
        "scripts/analyze_e2_shadow_replay.py",
        "src/evaluation/memory_supervisor.py",
        "src/memory/decision_context.py",
        "src/interfaces/relation_memory_text.py",
        "src/memory/visual_pipeline.py",
    )
    historical_source_names = (
        "src/evaluation/memory_supervisor.py", "src/memory/decision_context.py",
        "src/interfaces/relation_memory_text.py", "src/memory/relation_memory.py",
        "src/evaluation/libero_mem_rollout.py",
    )
    with __import__("zipfile").ZipFile(batch_paths[0].parent / "code_snapshot.zip") as archive:
        historical_source_comparison = {
            name: {
                "e2_sha256": __import__("hashlib").sha256(archive.read(name)).hexdigest(),
                "current_sha256": sha256_file(ROOT / name),
                "identical": __import__("hashlib").sha256(archive.read(name)).hexdigest()
                == sha256_file(ROOT / name),
            }
            for name in historical_source_names
        }
    result = {
        "schema_version": 1,
        "analysis_id": "e2_projected_memory_shadow_replay_20261120",
        "classification": "OFFLINE_SHADOW_REPLAY_ON_E2_TRAJECTORIES_NOT_CLOSED_LOOP_EVENT_EFFICACY",
        "comparability": "NON_COMPARABLE_OFFICIAL_PAPER",
        "interpretation": (
            "Replays the projected-memory decision observer on completed E2 F1 traces while preserving "
            "their recorded policy actions and fixed 10-action query schedule. Early candidates are "
            "opportunities seen on historical trajectories; they do not establish what would happen "
            "after an event-triggered requery changed the policy trajectory."
        ),
        "inputs": {
            "run_root": str(args.run_root.relative_to(ROOT)).replace("\\", "/"),
            "batch_manifest_count": len(batch_paths), "batch_manifest_sha256": manifest_hashes,
            "code_snapshot_sha256": next(iter(code_snapshots)),
            "adapter_dir": str(args.adapter_dir.relative_to(ROOT)).replace("\\", "/"),
            "adapter_manifest_sha256": sha256_file(args.adapter_dir / "mem_adapter_manifest.json"),
            "writer": provenance["writer"], "writer_config": writer_config,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "expected_tokenizer_sha256": TOKENIZER_SHA256,
            "decision_protocol_sha256": source_digest(
                ROOT / "configs/benchmarks/libero_mem_decision_dev_v1.json"),
            "decision_protocol_id": protocol["protocol_id"],
            "decision_protocol_split": protocol["split"],
            "trace_split": "unseen",
            "query_horizon_actions": protocol["parent_execution_horizon"],
            "minimum_prefix_actions": protocol["minimum_prefix_actions"],
            "semantic_persistence_observations": protocol["semantic_persistence_observations"],
            "source_sha256": {name: source_digest(ROOT / name) for name in current_sources},
            "e2_source_comparison": historical_source_comparison,
        },
        "summary": {
            "episodes": len(episodes), "paired_state_groups": len(state_pairs),
            "tasks": {task: sum(row["task"] == task for row in episodes) for task in ("T1", "T3")},
            "recorded_policy_actions": sum(row["recorded_actions"] for row in episodes),
            "recorded_policy_queries": sum(row["recorded_queries"] for row in episodes),
            "query_count_distribution": {str(key): value for key, value in sorted(query_counts.items())},
            "projected_prompts_exactly_replayed": sum(row["prompts_exactly_replayed"] for row in episodes),
            "shadow_semantic_change_checks": len(all_changes),
            "early_requery_candidates_on_recorded_trajectories": len(early),
            "early_candidate_types": dict(Counter(change["change_type"] for change in early)),
            "all_semantic_change_types": dict(trigger_counts),
            "evidence_reference_complete_candidates": sum(change["evidence_refs_present"] for change in early),
            "actual_requeries_in_E2": 0,
            "policy_actions_changed_by_this_analysis": 0,
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
