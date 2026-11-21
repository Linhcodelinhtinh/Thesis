#!/usr/bin/env python
"""Audit paired D1 text/shadow runs and report observer/action parity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import source_digest
from src.evaluation.memory_supervisor import DEFAULT_DECISION_PROTOCOL
from src.evaluation.memory_supervisor import execution_treatment, load_decision_protocol
from src.utils.file_hash import sha256_file

RUNTIME_SOURCES = (
    "scripts/evaluate_mem_adapter.py",
    "scripts/libero_mem_sim_worker.py",
    "scripts/memory_detector_worker.py",
    "src/evaluation/libero_mem_rollout.py",
    "src/evaluation/memory_supervisor.py",
    "src/memory/decision_context.py",
    "src/memory/event_writer.py",
    "src/memory/relation_pipeline.py",
    "src/interfaces/relation_memory_text.py",
    "src/interfaces/libero_mem_text.py",
    "src/simulator/libero_mem_env.py",
    "src/simulator/libero_mem_loader.py",
    "src/simulator/process_libero_mem_env.py",
    "src/models/smolvla/adapter.py",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def query_rows(directory):
    with (directory / "queries.jsonl").open(encoding="utf-8") as handle:
        return [(row["step"], row["prompt"], row["chunk_sha256"])
                for row in map(json.loads, handle)]


def episode_parity(text_dir, shadow_dir):
    text_queries, shadow_queries = query_rows(text_dir), query_rows(shadow_dir)
    with (np.load(text_dir / "trajectory.npz", allow_pickle=False) as text,
          np.load(shadow_dir / "trajectory.npz", allow_pickle=False) as shadow):
        arrays = {key: bool(np.array_equal(text[key], shadow[key]))
                  for key in ("actions", "query_steps", "action_chunks")}
    return dict(query_rows_identical=text_queries == shadow_queries,
                trajectory_arrays_identical=all(arrays.values()),
                trajectory_arrays=arrays,
                text_queries=len(text_queries), shadow_queries=len(shadow_queries),
                text_last_step=text_queries[-1][0] if text_queries else None,
                shadow_last_step=shadow_queries[-1][0] if shadow_queries else None)


def decision_observations(directory):
    path = directory / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
    changes = []
    for row in rows:
        for change in row.get("changes", []):
            after = change.get("after") or {}
            changes.append(dict(step=row["step"], decision=row["decision"], reason=row["reason"],
                                key=change.get("key"), evidence_ids=after.get("evidence_ids", []),
                                event_ids=after.get("event_ids", [])))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-run", type=Path, required=True)
    parser.add_argument("--shadow-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report output")

    text_manifest = read_json(args.text_run / "run_manifest.json")
    shadow_manifest = read_json(args.shadow_run / "run_manifest.json")
    if any(row["status"] != "complete" for row in (text_manifest, shadow_manifest)):
        raise ValueError("Both D1 runs must complete before parity analysis")
    if text_manifest["classification"] != "CUSTOM_V2_LIBERO_MEM" or shadow_manifest["classification"] != text_manifest["classification"]:
        raise ValueError("Unexpected benchmark classification")

    protocol = load_decision_protocol()
    if (text_manifest["decision_protocol_sha256"] != shadow_manifest["decision_protocol_sha256"]
            or text_manifest["decision_protocol_sha256"] != source_digest(DEFAULT_DECISION_PROTOCOL)):
        raise ValueError("Decision protocol changed between arms")

    source_text = text_manifest["source_sha256"]
    source_shadow = shadow_manifest["source_sha256"]
    source_keys = set(source_text) | set(source_shadow)
    source_differences = sorted(key for key in source_keys if source_text.get(key) != source_shadow.get(key))
    # This CLI is not on the evaluate_mem_adapter -> rollout call path. Its only
    # difference is the unrelated memory-off provenance flag used by run_benchmark.
    if source_differences != ["scripts/run_benchmark.py"]:
        raise ValueError(f"Unexpected source differences: {source_differences}")
    if any(source_text.get(name) != source_shadow.get(name) for name in RUNTIME_SOURCES):
        raise ValueError("A decision or rollout source changed between arms")
    if "run_benchmark" in (ROOT / "scripts/evaluate_mem_adapter.py").read_text(encoding="utf-8"):
        raise ValueError("run_benchmark is referenced by the D1 entrypoint; source divergence needs review")

    identity_fields = ("base", "loaded_policies", "local_assets_lock_sha256", "environments", "cuda")
    for field in identity_fields:
        if text_manifest.get(field) != shadow_manifest.get(field):
            raise ValueError(f"Run identity differs: {field}")
    if text_manifest["treatments"]["F1"]["adapter_manifest_sha256"] != shadow_manifest["treatments"]["F1"]["adapter_manifest_sha256"]:
        raise ValueError("F1 adapter manifest differs")
    for name in ("save_video", "stream_memory_trace", "split", "tasks", "conditions", "states", "policy_seed"):
        if text_manifest["arguments"][name] != shadow_manifest["arguments"][name]:
            raise ValueError(f"Run argument differs: {name}")
    if text_manifest["arguments"]["decision_mode"] != "text" or shadow_manifest["arguments"]["decision_mode"] != "shadow":
        raise ValueError("Expected text and shadow arms")
    if text_manifest["roster"] != shadow_manifest["roster"]:
        raise ValueError("Paired roster differs")

    text_treatment = execution_treatment(protocol, "text", "projected")
    shadow_treatment = execution_treatment(protocol, "shadow", "projected")
    episodes = []
    for row in text_manifest["roster"]:
        stem = f"{row['task']}_F1_seen_state{row['state_idx']}"
        text_dir = args.text_run / "episodes" / stem
        shadow_dir = args.shadow_run / "episodes" / stem
        text_status, shadow_status = read_json(text_dir / "status.json"), read_json(shadow_dir / "status.json")
        key = (row["task"], row["initial_state_sha256"], text_manifest["arguments"]["policy_seed"])
        if (shadow_status["task"], shadow_status["initial_state_sha256"], shadow_status["policy_seed"]) != key:
            raise ValueError(f"Shadow episode identity differs: {stem}")
        for directory, status, treatment, arm in ((text_dir, text_status, text_treatment, "text"),
                                                  (shadow_dir, shadow_status, shadow_treatment, "shadow-projected")):
            if status["status"] != "complete" or status["decision_arm"] != arm:
                raise ValueError(f"Incomplete or mislabeled episode: {directory}")
            required = ("summary.json", "queries.jsonl", "actions.jsonl", "trajectory.npz", "memory_trace.jsonl")
            for name in required:
                if sha256_file(directory / name) != status["files_sha256"].get(name):
                    raise ValueError(f"Artifact hash mismatch: {directory / name}")
        text_result = read_json(text_dir / "summary.json")["result"]
        shadow_result = read_json(shadow_dir / "summary.json")["result"]
        if text_result["total_steps"] != 1000 or shadow_result["total_steps"] != 1000:
            raise ValueError(f"Expected 1000-step episodes: {stem}")
        for result, treatment in ((text_result, text_treatment), (shadow_result, shadow_treatment)):
            metrics = result["decision_metrics"]
            if not 1 <= metrics["queries"] <= treatment.max_queries:
                raise ValueError(f"Query count exceeds the protocol: {stem}")
            if metrics["abstained"] != (result["termination_reason"] == "decision_abstention"):
                raise ValueError(f"Abstention metadata is inconsistent: {stem}")
        episodes.append(dict(
            episode=stem,
            task=row["task"],
            state_idx=row["state_idx"],
            initial_state_sha256=row["initial_state_sha256"],
            seed=text_manifest["arguments"]["policy_seed"],
            artifacts_sha256={
                arm: dict(status_sha256=sha256_file(
                    (text_dir if arm == "text" else shadow_dir) / "status.json"),
                    artifacts=status["files_sha256"])
                for arm, status in (("text", text_status), ("shadow", shadow_status))
            },
            text=dict(success=text_result["success"], scr=text_result["scr"],
                      queries=text_result["decision_metrics"]["queries"],
                      early_requeries=text_result["decision_metrics"]["early_requeries"],
                      wall_time_s=text_result["total_wall_time_s"]),
            shadow=dict(success=shadow_result["success"], scr=shadow_result["scr"],
                        queries=shadow_result["decision_metrics"]["queries"],
                        early_requeries=shadow_result["decision_metrics"]["early_requeries"],
                        triggered_checks=shadow_result["decision_metrics"]["triggered_checks"],
                        wall_time_s=shadow_result["total_wall_time_s"]),
            shadow_observed_changes=decision_observations(shadow_dir),
            parity=episode_parity(text_dir, shadow_dir),
        ))

    output = dict(
        experiment_id="D1-text-shadow-parity-2026-11-20",
        status="complete",
        stage="development_only",
        classification="DECISION_DEVELOPMENT_CUSTOM_V2",
        comparability="NON-COMPARABLE_OFFICIAL_PAPER",
        question="Does projected shadow observation preserve text-arm prompts and actions while logging evidence-backed semantic changes?",
        source_differences=[dict(path=key, text_sha256=source_text.get(key), shadow_sha256=source_shadow.get(key),
                                 impact="Unrelated run_benchmark CLI change; D1 entrypoint is evaluate_mem_adapter.py.")
                            for key in source_differences],
        provenance=dict(adapter_manifest_sha256=text_manifest["treatments"]["F1"]["adapter_manifest_sha256"],
                        decision_protocol_sha256=text_manifest["decision_protocol_sha256"],
                        split_lock_sha256=text_manifest["split_lock_sha256"],
                        source_sha256={key: source_text[key] for key in RUNTIME_SOURCES}),
        episodes=episodes,
        parity=dict(pairs=len(episodes), identical=sum(
            item["parity"]["query_rows_identical"] and item["parity"]["trajectory_arrays_identical"]
            for item in episodes)),
        limitations=["Two seen state-0 episodes, one policy seed; development trace/parity only.",
                     "Shadow records a potential semantic change but does not issue an early query.",
                     "The separate D2 event pilot is the only completed functional early-requery closed-loop example and remains one unbalanced pair."],
        analysis_source_sha256={name: sha256_file(ROOT / name) for name in (
            "scripts/analyze_d1_shadow_parity.py", "scripts/aggregate_memory_decisions.py")},
    )
    output["text_run_manifest_sha256"] = sha256_file(args.text_run / "run_manifest.json")
    output["shadow_run_manifest_sha256"] = sha256_file(args.shadow_run / "run_manifest.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"D1: {len(episodes)} paired episodes; {output['parity']['identical']} exact prompt/action parity")


if __name__ == "__main__":
    main()
