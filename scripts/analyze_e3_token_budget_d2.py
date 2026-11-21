#!/usr/bin/env python
"""Measure relation-memory prompt delivery across the locked E3 token sweep.

This is a fixed-trace development analysis of the completed D2 event cohort.
It replays only the recorded memory snapshots through the exact tokenizer and
renderer, and never re-runs SmolVLA, changes an action, or claims policy efficacy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_e2_shadow_replay import jsonl, read_json, snapshot_from_dict
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.memory.visual_pipeline import (
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    load_prompt_tokenizer,
)
from src.utils.file_hash import sha256_file


DEFAULT_RUN = ROOT / "experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl"
DEFAULT_ADAPTER = ROOT / "experiments/training/results_f1_author_split_v2/output_f1/pretrained_model"
DEFAULT_PROTOCOL = ROOT / "docs/thesis/evidence/mem_analysis_protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "experiments/evaluation/e3_token_budget_d2_dev12_20261121"

RATIOS = ((0, "0"), (1, "1/3"), (2, "2/3"), (3, "full"))
RENDER_SOURCE_PATHS = (
    "src/interfaces/relation_memory_text.py",
    "src/interfaces/libero_mem_text.py",
    "src/memory/relation_memory.py",
    "src/memory/models.py",
    "src/memory/libero_mem_profiles.py",
    "src/memory/visual_pipeline.py",
    "src/evaluation/libero_mem_rollout.py",
)


def verify_episode(directory: Path, roster_row: dict, seed: int) -> tuple[list[dict], dict[str, dict]]:
    status_path = directory / "status.json"
    status = read_json(status_path)
    if status.get("status") != "complete" or status.get("logging_errors"):
        raise ValueError(f"Incomplete or logging-error episode: {directory.name}")
    if status.get("episode_id") != directory.name:
        raise ValueError(f"Episode ID mismatch: {directory.name}")
    if status.get("task") != roster_row["task"] or status.get("state_idx") != roster_row["state_idx"]:
        raise ValueError(f"Roster task/state mismatch: {directory.name}")
    if status.get("initial_state_sha256") != roster_row["initial_state_sha256"]:
        raise ValueError(f"Initial-state hash mismatch: {directory.name}")
    if status.get("policy_seed") != seed or status.get("instruction") != roster_row["instruction"]:
        raise ValueError(f"Seed or instruction mismatch: {directory.name}")
    for name, expected in status["files_sha256"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Episode artifact hash mismatch: {directory.name}/{name}")

    queries = list(jsonl(directory / "queries.jsonl"))
    memory_by_step = {}
    for row in jsonl(directory / "memory_trace.jsonl"):
        step = int(row["step"])
        if step in memory_by_step:
            raise ValueError(f"Duplicate memory snapshot at {directory.name}/{step}")
        memory_by_step[step] = row
    if not queries or not memory_by_step:
        raise ValueError(f"Missing query or memory trace: {directory.name}")
    if len({int(row["step"]) for row in queries}) != len(queries):
        raise ValueError(f"Duplicate query step: {directory.name}")
    return queries, memory_by_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--analysis-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preserving existing analysis output: {output_dir}")

    run_path = args.run_dir / "run_manifest.json"
    manifest = read_json(run_path)
    arguments = manifest["arguments"]
    if (manifest.get("status") != "complete" or manifest.get("completed_episodes") != 24
            or arguments.get("decision_mode") != "event"
            or arguments.get("decision_exposure") != "projected"
            or arguments.get("split") != "seen"
            or arguments.get("tasks") != ["T1", "T3"]
            or arguments.get("states") != ",".join(str(i) for i in range(12))
            or arguments.get("policy_seed") != 42):
        raise ValueError("Input run is not the locked D2 event development cohort")

    protocol = read_json(args.analysis_protocol)
    if protocol.get("protocol_id") != "thesis_mem_analysis_v2":
        raise ValueError("Unexpected E3 analysis protocol")
    protocol_budget = protocol["budget"]
    if (protocol_budget.get("pilot_text_total_tokens") != 48
            or protocol_budget.get("sweep") !=
            "0, 1/3, 2/3, full residual tokens after the instruction, selected using development data"):
        raise ValueError("E3 budget schedule differs from the locked analysis protocol")

    snapshot_path = args.run_dir / "code_snapshot.zip"
    if sha256_file(snapshot_path) != manifest["code_snapshot_sha256"]:
        raise ValueError("D2 code snapshot hash mismatch")
    adapter_manifest_path = args.adapter_dir / "mem_adapter_manifest.json"
    if sha256_file(adapter_manifest_path) != manifest["adapter_manifest_sha256"]:
        raise ValueError("F1 adapter manifest hash differs from the D2 run")
    adapter_manifest = read_json(adapter_manifest_path)
    provenance = next(item for item in adapter_manifest["memory_provenance"]
                      if item["writer"] == "rgb_relations_v1")
    writer_config = provenance["writer_config"]
    if writer_config != {"projection": "relation_text_v1", "strategy": "relevant",
                         "max_total_tokens": 48}:
        raise ValueError("F1 writer configuration differs from the D2 event run")
    if provenance.get("tokenizer_revision") != TOKENIZER_REVISION:
        raise ValueError("F1 tokenizer revision differs from the pinned renderer tokenizer")

    source_hashes = manifest["source_sha256"]
    checked_sources = {}
    for relative in RENDER_SOURCE_PATHS:
        expected = source_hashes.get(relative)
        actual = sha256_file(ROOT / relative)
        if expected != actual:
            raise ValueError(f"Renderer source differs from D2 run snapshot: {relative}")
        checked_sources[relative] = actual

    tokenizer_path = args.adapter_dir / "tokenizer.json"
    tokenizer = load_prompt_tokenizer(tokenizer_path)
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer artifact hash differs from the pinned source")

    roster = {}
    for item in manifest["roster"]:
        episode_id = f"{item['task']}_F1_seen_state{item['state_idx']}"
        if episode_id in roster:
            raise ValueError(f"Duplicate roster episode: {episode_id}")
        roster[episode_id] = item
    episode_dirs = sorted((args.run_dir / "episodes").glob("T[13]_F1_seen_state*"))
    if len(episode_dirs) != 24 or set(roster) != {path.name for path in episode_dirs}:
        raise ValueError("D2 event run does not contain exactly the locked 24-episode roster")

    base_renderer = RelationMemoryRenderer(tokenizer, max_total_tokens=48, strategy="relevant")
    renderer_cache = {48: base_renderer}
    summaries = defaultdict(lambda: {
        "queries": 0,
        "candidate_queries": 0,
        "queries_with_memory": 0,
        "candidate_facts": 0,
        "included_facts": 0,
        "dropped_facts": 0,
        "baseline_facts": 0,
        "baseline_facts_retained": 0,
        "total_tokens": [],
        "episodes_with_memory": set(),
        "state_groups_with_memory": set(),
    })
    detail_rows = []
    full_budget_replays = 0
    expected_queries = 0
    query_input_hashes = {}
    trace_input_hashes = {}
    episode_outcomes = []

    for episode_dir in episode_dirs:
        row = roster[episode_dir.name]
        queries, memory_by_step = verify_episode(episode_dir, row, arguments["policy_seed"])
        expected_queries += len(queries)
        query_input_hashes[str((episode_dir / "queries.jsonl").relative_to(ROOT)).replace("\\", "/")] = sha256_file(episode_dir / "queries.jsonl")
        trace_input_hashes[str((episode_dir / "memory_trace.jsonl").relative_to(ROOT)).replace("\\", "/")] = sha256_file(episode_dir / "memory_trace.jsonl")
        episode_query_count = 0
        episode_memory_counts = defaultdict(int)

        for query in queries:
            step = int(query["step"])
            trace_row = memory_by_step.get(step)
            if trace_row is None or trace_row["snapshot"]["as_of_observation"] != step:
                raise ValueError(f"No exact memory snapshot for query {episode_dir.name}/{step}")
            snapshot = snapshot_from_dict(trace_row["snapshot"])
            instruction = row["instruction"]
            original = base_renderer.render(instruction, snapshot, enabled=True)
            logged = query["render"]
            if (original.text != query["prompt"]
                    or list(original.facts_included) != logged["facts_included"]
                    or original.total_facts_considered != logged["total_facts_considered"]
                    or original.included_facts_count != logged["included_facts_count"]
                    or original.dropped_facts_count != logged["dropped_facts_count"]
                    or original.total_tokens_with_instruction != logged["total_tokens_with_instruction"]
                    or logged["budget_tokens"] != 48):
                raise ValueError(f"48-token renderer replay mismatch: {episode_dir.name}/{step}")
            full_budget_replays += 1
            episode_query_count += 1
            full_facts = tuple(original.facts_included)
            instruction_tokens = base_renderer.count_tokens(instruction)
            residual = max(0, 48 - instruction_tokens)

            for numerator, label in RATIOS:
                cap = instruction_tokens + (residual * numerator // 3)
                renderer = renderer_cache.get(cap)
                if renderer is None:
                    renderer = RelationMemoryRenderer(tokenizer, max_total_tokens=cap,
                                                      strategy="relevant")
                    renderer_cache[cap] = renderer
                result = renderer.render(instruction, snapshot, enabled=True)
                agg = summaries[(row["task"], label)]
                agg["queries"] += 1
                agg["candidate_queries"] += int(result.total_facts_considered > 0)
                agg["queries_with_memory"] += int(result.included_facts_count > 0)
                agg["candidate_facts"] += result.total_facts_considered
                agg["included_facts"] += result.included_facts_count
                agg["dropped_facts"] += result.dropped_facts_count
                agg["baseline_facts"] += len(full_facts)
                agg["baseline_facts_retained"] += sum(fact in result.facts_included for fact in full_facts)
                agg["total_tokens"].append(result.total_tokens_with_instruction)
                if result.included_facts_count:
                    agg["episodes_with_memory"].add(episode_dir.name)
                    agg["state_groups_with_memory"].add(row["state_idx"])
                    episode_memory_counts[label] += 1
                detail_rows.append({
                    "episode_id": episode_dir.name,
                    "task": row["task"],
                    "state_idx": row["state_idx"],
                    "initial_state_sha256": row["initial_state_sha256"],
                    "query_step": step,
                    "query_id": query["query_id"],
                    "instruction_tokens": instruction_tokens,
                    "residual_at_48": residual,
                    "budget_level": label,
                    "total_token_cap": cap,
                    "prompt_tokens": result.total_tokens_with_instruction,
                    "candidate_facts": result.total_facts_considered,
                    "facts_included": list(result.facts_included),
                    "facts_dropped": list(result.dropped_facts),
                    "baseline_facts": list(full_facts),
                    "rendered_prompt": result.text,
                })

        episode_outcomes.append({
            "episode_id": episode_dir.name,
            "task": row["task"],
            "state_idx": row["state_idx"],
            "initial_state_sha256": row["initial_state_sha256"],
            "recorded_queries": episode_query_count,
            "query_steps_with_memory_by_budget": dict(episode_memory_counts),
        })

    if full_budget_replays != expected_queries:
        raise ValueError("Not all recorded policy queries were replayed")

    aggregate = []
    for (task, label), row in sorted(summaries.items()):
        aggregate.append({
            "task": task,
            "budget_level": label,
            "queries": row["queries"],
            "candidate_queries": row["candidate_queries"],
            "queries_with_memory": row["queries_with_memory"],
            "memory_delivery_rate_all_queries": (row["queries_with_memory"] / row["queries"]
                                                  if row["queries"] else 0.0),
            "memory_delivery_rate_when_candidate_exists": (
                row["queries_with_memory"] / row["candidate_queries"] if row["candidate_queries"] else 0.0),
            "candidate_facts": row["candidate_facts"],
            "included_facts": row["included_facts"],
            "dropped_facts": row["dropped_facts"],
            "baseline_facts_retained_rate": (
                row["baseline_facts_retained"] / row["baseline_facts"] if row["baseline_facts"] else None),
            "mean_prompt_tokens": (sum(row["total_tokens"]) / len(row["total_tokens"])
                                   if row["total_tokens"] else 0.0),
            "episodes_with_memory": len(row["episodes_with_memory"]),
            "state_groups_with_memory": len(row["state_groups_with_memory"]),
        })

    output_dir.mkdir(parents=True)
    inputs = {
        "d2_run_manifest": {"path": str(run_path.relative_to(ROOT)).replace("\\", "/"),
                             "sha256": sha256_file(run_path)},
        "d2_code_snapshot": {"path": str(snapshot_path.relative_to(ROOT)).replace("\\", "/"),
                             "sha256": sha256_file(snapshot_path)},
        "f1_adapter_manifest": {"path": str(adapter_manifest_path.relative_to(ROOT)).replace("\\", "/"),
                                "sha256": sha256_file(adapter_manifest_path)},
        "analysis_protocol": {"path": str(args.analysis_protocol.relative_to(ROOT)).replace("\\", "/"),
                              "sha256": sha256_file(args.analysis_protocol)},
        "tokenizer": {"path": str(tokenizer_path.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": sha256_file(tokenizer_path), "repository": TOKENIZER_REPO,
                      "revision": TOKENIZER_REVISION},
        "query_logs_sha256": query_input_hashes,
        "memory_traces_sha256": trace_input_hashes,
    }
    receipt = {
        "experiment_id": "E3-token-budget-D2-fixed-trace-development-2026-11-21",
        "classification": "DEVELOPMENT_FIXED_TRACE_RENDER_BUDGET_SWEEP_NOT_POLICY_EFFICACY",
        "protocol_id": protocol["protocol_id"],
        "scope": {
            "source_run": str(args.run_dir.relative_to(ROOT)).replace("\\", "/"),
            "decision_arm": "event-projected",
            "split": "seen-development",
            "tasks": ["T1", "T3"],
            "state_indices_per_task": list(range(12)),
            "policy_seed": 42,
            "policy_episodes_reexecuted": 0,
            "policy_actions_or_prompts_changed": False,
            "interpretation_limit": "Renderer-level token delivery on recorded D2 snapshots only; no policy, closed-loop, success, or causal efficacy conclusion.",
        },
        "budget_schedule": {
            "max_total_tokens": 48,
            "levels": [
                {"label": label, "fraction_of_residual": label,
                 "definition": "instruction_tokens + floor((48 - instruction_tokens) * numerator / 3)",
                 "numerator": numerator}
                for numerator, label in RATIOS
            ],
            "token_counter": "pinned SmolVLM2 tokenizer; exact LiberoMemTextRenderer newline/special-token convention",
        },
        "verification": {
            "episodes": len(episode_dirs),
            "queries_replayed_at_full_48_token_budget": full_budget_replays,
            "full_budget_prompt_and_renderer_metadata_exact_match": True,
            "episode_artifact_hashes_verified": True,
            "d2_runtime_renderer_source_hashes_match": checked_sources,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "aggregate": aggregate,
        "episode_outcomes": episode_outcomes,
        "inputs": inputs,
        "analyzer": {
            "path": "scripts/analyze_e3_token_budget_d2.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    detail_path = output_dir / "per_query.jsonl"
    with detail_path.open("w", encoding="utf-8") as stream:
        for row in detail_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    lines = [
        "# E3 token-budget sweep on fixed D2 traces",
        "",
        "Classification: `DEVELOPMENT_FIXED_TRACE_RENDER_BUDGET_SWEEP_NOT_POLICY_EFFICACY`.",
        "",
        "The analysis replays the completed D2 event-projected development snapshots through the pinned relation renderer and tokenizer. It does not rerun SmolVLA, change actions, or measure closed-loop success.",
        "",
        f"- Roster: {len(episode_dirs)} episodes, T1/T3 seen states 0–11, policy seed 42.",
        f"- Queries replayed at the original 48-token budget with exact prompt and renderer metadata: {full_budget_replays}.",
        "- Budget levels allocate 0, 1/3, 2/3, and all residual tokens after the instruction.",
        "",
        "| Task | Residual budget | Queries | Candidate queries | Queries with memory | Delivery / candidates | Mean prompt tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['task']} | {row['budget_level']} | {row['queries']} | {row['candidate_queries']} | "
            f"{row['queries_with_memory']} | {row['memory_delivery_rate_when_candidate_exists']:.3f} | "
            f"{row['mean_prompt_tokens']:.2f} |"
        )
    lines.extend([
        "",
        "The sweep measures projection delivery only. Lower-budget outputs were not passed to the policy, so this report cannot establish whether token budget changes actions or task success. Query rows are repeated observations nested within 24 episodes and are not independent samples.",
        "",
        f"Receipt: `receipt.json` (SHA-256 `{sha256_file(receipt_path)}`).",
        f"Per-query audit rows: `per_query.jsonl` (SHA-256 `{sha256_file(detail_path)}`).",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "episodes": len(episode_dirs),
        "queries_replayed": full_budget_replays,
        "aggregate": aggregate,
        "output_dir": str(output_dir),
        "receipt_sha256": sha256_file(receipt_path),
    }, indent=2))


if __name__ == "__main__":
    main()
