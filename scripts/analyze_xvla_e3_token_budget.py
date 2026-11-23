#!/usr/bin/env python
"""Run the locked fixed-trace E3 renderer sweep on X-VLA E2 memory traces."""
import hashlib
import json
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_e2_shadow_replay import snapshot_from_dict
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.memory.visual_pipeline import TOKENIZER_SHA256, load_prompt_tokenizer

RUN = ROOT / "experiments/evaluation/xvla_e2_f1_mem_on_20261122"
TOKENIZER_DIR = ROOT / "experiments/training/assets_mem_adapter_v1/backbone"
OUTPUT = ROOT / "experiments/evaluation/xvla_e3_token_budget_20261123"
LEVELS = ((0, "0"), (1, "1/3"), (2, "2/3"), (3, "full"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_dir, output_dir = args.source_run.absolute(), args.output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError("Preserving existing output " + str(output_dir))
    manifest = read(run_dir / "run_manifest.json")
    decision_mode = manifest.get("arguments", {}).get("decision_mode")
    if decision_mode is None:
        expected_episodes, expected_split = 40, "unseen"
    elif decision_mode == "event":
        expected_episodes, expected_split = 24, "seen"
        if manifest.get("arguments", {}).get("decision_exposure") != "projected":
            raise ValueError("D2 E3 requires the projected event arm")
    else:
        raise ValueError("E3 source must be an instruction-fixed E2 trace or a projected D2 event trace")
    if (manifest.get("status") != "complete"
            or manifest.get("completed_episodes") != expected_episodes
            or manifest.get("arguments", {}).get("split") != expected_split):
        raise ValueError(f"Expected completed X-VLA cohort: {expected_episodes} {expected_split} episodes")
    manifest_roster = manifest.get("roster", [])
    manifest_assignments = {(r["task"], int(r["state_idx"]), r["initial_state_sha256"])
                            for r in manifest_roster}
    if len(manifest_assignments) != expected_episodes:
        raise ValueError("E3 source manifest roster is missing or contains duplicates")
    provenance = manifest["memory_contract"]
    if provenance["writer"] != "rgb_relations_v1" or provenance["writer_config"] != {
        "projection": "relation_text_v1", "strategy": "relevant", "max_total_tokens": 48
    }:
        raise ValueError("X-VLA memory contract does not match E3")
    tokenizer = load_prompt_tokenizer(TOKENIZER_DIR)
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    if sha(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Pinned projection tokenizer hash mismatch")
    renderer_cache = {}
    base_renderer = RelationMemoryRenderer(tokenizer, 48, "relevant")
    totals = defaultdict(lambda: {"queries": 0, "candidate_queries": 0, "delivered_queries": 0,
        "candidate_facts": 0, "included_facts": 0, "dropped_facts": 0,
        "baseline_facts": 0, "baseline_retained": 0, "tokens": [], "episodes": set()})
    details, input_hashes, query_count, replay_count = [], {}, 0, 0
    episode_dirs = sorted(p for p in (run_dir / "episodes").iterdir() if p.is_dir())
    if len(episode_dirs) != expected_episodes:
        raise ValueError(f"Expected exactly {expected_episodes} source episodes")
    if decision_mode == "event":
        roster = {(read(p / "status.json")["task"], read(p / "status.json")["state_idx"]) for p in episode_dirs}
        expected_roster = {(task, state) for task in ("T1", "T3") for state in range(12)}
        if roster != expected_roster:
            raise ValueError("D2 E3 source roster differs from seen T1/T3 states 0–11")
    for episode in episode_dirs:
        status = read(episode / "status.json")
        if status.get("status") != "complete" or status.get("logging_errors"):
            raise ValueError("Incomplete episode " + episode.name)
        expected_split = "seen" if decision_mode == "event" else "unseen"
        identity = (status.get("task"), int(status.get("state_idx", -1)), status.get("initial_state_sha256"))
        if (status.get("split") != expected_split
                or status.get("policy_seed") != manifest.get("arguments", {}).get("policy_seed", 42)
                or identity not in manifest_assignments):
            raise ValueError("E3 source episode identity differs from its run manifest: " + episode.name)
        for name in ("queries.jsonl", "memory_trace.jsonl"):
            expected = status.get("files_sha256", {}).get(name)
            if not expected or sha(episode / name) != expected:
                raise ValueError("Source hash mismatch " + episode.name + "/" + name)
            input_hashes[str((episode / name).relative_to(ROOT))] = expected
        queries = rows(episode / "queries.jsonl")
        trace = {int(r["step"]): r for r in rows(episode / "memory_trace.jsonl")}
        instruction = status["instruction"]
        query_count += len(queries)
        for query_id, query in enumerate(queries):
            step = int(query["step"])
            if step not in trace or int(trace[step]["snapshot"]["as_of_observation"]) != step:
                raise ValueError("No exact writer snapshot at " + episode.name + "/" + str(step))
            snapshot = snapshot_from_dict(trace[step]["snapshot"])
            original = base_renderer.render(instruction, snapshot, True)
            logged = query["render"]
            if (original.text != query["prompt"] or list(original.facts_included) != logged["facts_included"]
                    or original.total_tokens_with_instruction != logged["total_tokens_with_instruction"]
                    or int(logged["budget_tokens"]) != 48):
                raise ValueError("48-token exact replay mismatch at " + episode.name + "/" + str(step))
            replay_count += 1
            instruction_tokens = base_renderer.count_tokens(instruction)
            residual = max(0, 48 - instruction_tokens)
            base_facts = tuple(original.facts_included)
            for numerator, label in LEVELS:
                cap = instruction_tokens + residual * numerator // 3
                if cap not in renderer_cache:
                    renderer_cache[cap] = RelationMemoryRenderer(tokenizer, cap, "relevant")
                result = renderer_cache[cap].render(instruction, snapshot, True)
                agg = totals[(status["task"], label)]
                agg["queries"] += 1
                agg["candidate_queries"] += int(result.total_facts_considered > 0)
                agg["delivered_queries"] += int(result.included_facts_count > 0)
                agg["candidate_facts"] += result.total_facts_considered
                agg["included_facts"] += result.included_facts_count
                agg["dropped_facts"] += result.dropped_facts_count
                agg["baseline_facts"] += len(base_facts)
                agg["baseline_retained"] += sum(f in result.facts_included for f in base_facts)
                agg["tokens"].append(result.total_tokens_with_instruction)
                if result.included_facts_count:
                    agg["episodes"].add(episode.name)
                details.append({"episode_id": episode.name, "task": status["task"], "state_idx": status["state_idx"],
                    "query_step": step, "query_id": query_id, "budget_level": label, "token_cap": cap,
                    "prompt_tokens": result.total_tokens_with_instruction, "candidate_facts": result.total_facts_considered,
                    "facts_included": list(result.facts_included), "facts_dropped": list(result.dropped_facts),
                    "prompt": result.text})
    if query_count != replay_count:
        raise ValueError("Not every policy query was replayed")
    aggregate = []
    for (task, label), a in sorted(totals.items()):
        aggregate.append({"task": task, "budget_level": label, "queries": a["queries"],
            "candidate_queries": a["candidate_queries"], "delivered_queries": a["delivered_queries"],
            "delivery_rate_all": a["delivered_queries"] / a["queries"],
            "delivery_rate_candidates": a["delivered_queries"] / a["candidate_queries"] if a["candidate_queries"] else None,
            "candidate_facts": a["candidate_facts"], "included_facts": a["included_facts"],
            "dropped_facts": a["dropped_facts"],
            "baseline_retained_rate": a["baseline_retained"] / a["baseline_facts"] if a["baseline_facts"] else None,
            "mean_prompt_tokens": sum(a["tokens"]) / len(a["tokens"]), "episodes_with_memory": len(a["episodes"])})
    output_dir.mkdir(parents=True)
    (output_dir / "per_query.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in details), encoding="utf-8")
    result = {"status": "complete", "classification": "CUSTOM_V2_XVLA_E3_FIXED_TRACE_RENDERER_SWEEP",
        "source_manifest_sha256": sha(run_dir / "run_manifest.json"), "tokenizer_sha256": TOKENIZER_SHA256,
        "source_run": str(run_dir.relative_to(ROOT)), "decision_mode": decision_mode,
        "episodes": len(episode_dirs), "queries_replayed": replay_count, "aggregate": aggregate,
        "interpretation": "Fixed-trace prompt delivery analysis only. Reduced budgets were not sent to the policy and do not estimate action or task-success effects."}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# X-VLA E3 fixed-trace token budget sweep", "", "| Task | Budget | Queries | Candidate queries | Memory delivered | Delivery rate | Candidate facts | Included | Dropped | Mean prompt tokens | Episodes with memory |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in aggregate:
        lines.append(f"| {a['task']} | {a['budget_level']} | {a['queries']} | {a['candidate_queries']} | {a['delivered_queries']} | {a['delivery_rate_all']:.3f} | {a['candidate_facts']} | {a['included_facts']} | {a['dropped_facts']} | {a['mean_prompt_tokens']:.2f} | {a['episodes_with_memory']} |")
    lines += ["", result["interpretation"]]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"input_hashes": input_hashes, "source_manifest_sha256": result["source_manifest_sha256"],
        "outputs_sha256": {n: sha(output_dir / n) for n in ("per_query.jsonl", "summary.json", "report.md")}}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
