#!/usr/bin/env python
"""Offline D3/D5 evaluator-cycle alignment for the X-VLA D2 paired cohort."""
import hashlib
import json
import statistics
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "experiments/evaluation/xvla_d2_shadow_dev12_t1_t3_20261123_wsl"
EVENT = ROOT / "experiments/evaluation/xvla_d2_event_dev12_t1_t3_20261123_wsl"
OUTPUT = ROOT / "experiments/evaluation/xvla_d3_d5_d2_trace_analysis_20261123"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def run_rows(run_dir, expected_mode):
    manifest_path = run_dir / "run_manifest.json"
    manifest = read(manifest_path)
    args = manifest.get("arguments", {})
    if (manifest.get("status") != "complete" or manifest.get("completed_episodes") != 24
            or args.get("decision_mode") != expected_mode
            or args.get("decision_exposure") != "projected"
            or args.get("split") != "seen" or args.get("policy_seed") != 42
            or args.get("tasks") != ["T1", "T3"]
            or args.get("states") != "0,1,2,3,4,5,6,7,8,9,10,11"):
        raise ValueError(f"Unexpected/incomplete X-VLA D2 {expected_mode} manifest: {run_dir}")
    protocol = manifest.get("decision_protocol", {})
    if protocol.get("protocol_id") != "libero_mem_decision_dev_v1" or protocol.get("stage") != "development":
        raise ValueError("X-VLA D2 decision protocol differs from the locked development protocol")
    if manifest.get("decision_execution", {}).get("mode") != expected_mode or manifest.get("decision_execution", {}).get("exposure") != "projected":
        raise ValueError("Run manifest decision execution differs from the requested D2 arm")
    roster = manifest.get("roster", [])
    manifest_assignments = {(r["task"], int(r["state_idx"]), r["initial_state_sha256"]) for r in roster}
    if len(roster) != 24 or {(task, state) for task, state, _ in manifest_assignments} != {
            (task, state) for task in ("T1", "T3") for state in range(12)}:
        raise ValueError("D2 run manifest roster is not T1/T3 seen states 0–11")
    directories = sorted(p for p in (run_dir / "episodes").iterdir() if p.is_dir())
    if len(directories) != 24:
        raise ValueError(f"Expected 24 D2 episodes, found {len(directories)} in {run_dir}")
    rows = []
    for directory in directories:
        status = read(directory / "status.json")
        if status.get("status") != "complete" or status.get("logging_errors"):
            raise ValueError("Incomplete episode " + directory.name)
        if (status.get("split") != "seen" or status.get("policy_seed") != 42
                or (status.get("task"), int(status.get("state_idx", -1)), status.get("initial_state_sha256"))
                   not in manifest_assignments):
            raise ValueError("Episode identity differs from its locked D2 roster: " + directory.name)
        hashes = {}
        for name in ("memory_trace.jsonl", "observations.jsonl", "summary.json"):
            expected = status.get("files_sha256", {}).get(name)
            if not expected or digest(directory / name) != expected:
                raise ValueError("Artifact hash mismatch " + directory.name + "/" + name)
            hashes[name] = expected
        summary = read(directory / "summary.json")
        memory = {int(r["step"]): r["snapshot"] for r in jsonl(directory / "memory_trace.jsonl")}
        observations = {int(r["step"]): r["info"]["eval_result"] for r in jsonl(directory / "observations.jsonl")
                        if r.get("info", {}).get("eval_result") is not None}
        steps = sorted(memory.keys() & observations.keys())
        if not steps or steps[-1] != int(summary["result"]["total_steps"]):
            raise ValueError("Trace horizon mismatch " + directory.name)
        writer, evaluator, unknown, incomplete = [], [], [], []
        for step in steps:
            rel = memory[step].get("relations", {}).get("bowl_1")
            if rel is None:
                raise ValueError(f"Missing bowl_1 at {directory.name}/{step}")
            writer.append(int(rel["returns"]))
            evaluator.append(int(observations[step]["confirmed_cycles"]))
            unknown.append(rel.get("status") == "unknown")
            incomplete.append(not bool(rel.get("history_complete")))

        def events(values):
            output = []
            for i in range(1, len(values)):
                output.extend({"cycle": cycle, "step": steps[i]}
                              for cycle in range(values[i - 1] + 1, values[i] + 1))
            return output

        writer_events, evaluator_events = events(writer), events(evaluator)
        writer_steps = {x["cycle"]: x["step"] for x in writer_events}
        evaluator_steps = {x["cycle"]: x["step"] for x in evaluator_events}
        pairs = [{"cycle": cycle, "writer_step": writer_steps[cycle], "evaluator_step": evaluator_steps[cycle],
                  "evaluator_minus_writer_step": evaluator_steps[cycle] - writer_steps[cycle]}
                 for cycle in sorted(writer_steps.keys() & evaluator_steps.keys())]
        result = summary["result"]
        rows.append({"decision_mode": expected_mode, "episode_id": summary["episode_id"],
            "task": summary["task"], "state_idx": summary["state_idx"],
            "initial_state_sha256": summary["initial_state_sha256"], "policy_seed": summary["policy_seed"],
            "success": bool(result["success"]), "overshot": bool(result["overshot"]),
            "steps": steps[-1], "writer_returns": writer[-1], "evaluator_cycles": evaluator[-1],
            "endpoint_match": writer[-1] == evaluator[-1], "writer_events": len(writer_events),
            "evaluator_events": len(evaluator_events), "pairs": pairs,
            "unknown_fraction": sum(unknown) / len(steps), "incomplete_fraction": sum(incomplete) / len(steps),
            "input_hashes": hashes})
    return manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    output_dir = parser.parse_args(argv).output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError("Preserving existing output " + str(output_dir))
    shadow_manifest, shadow_rows = run_rows(SHADOW, "shadow")
    event_manifest, event_rows = run_rows(EVENT, "event")

    def identity(rows):
        return {(r["task"], r["state_idx"], r["initial_state_sha256"], r["policy_seed"]) for r in rows}
    if identity(shadow_rows) != identity(event_rows):
        raise ValueError("D2 shadow/event task-state-seed roster mismatch")
    if {(task, state, seed) for task, state, _, seed in identity(shadow_rows)} != {
            (task, state, 42) for task in ("T1", "T3") for state in range(12)}:
        raise ValueError("Unexpected D2 development assignment set")
    if (shadow_manifest["split_lock_sha256"] != event_manifest["split_lock_sha256"]
            or shadow_manifest["protocol_sha256"] != event_manifest["protocol_sha256"]
            or shadow_manifest["decision_protocol_sha256"] != event_manifest["decision_protocol_sha256"]
            or shadow_manifest["adapter_export"] != event_manifest["adapter_export"]
            or shadow_manifest["memory_contract"] != event_manifest["memory_contract"]
            or shadow_manifest["environments"] != event_manifest["environments"]
            or shadow_manifest["roster"] != event_manifest["roster"]):
        raise ValueError("D2 shadow/event protocol, adapter, memory contract, environments or roster differ")
    execution_files = ("scripts/run_x_vla_libero.py", "scripts/xvla_v2/evaluate_memory.py",
        "src/evaluation/libero_mem_rollout.py", "src/evaluation/memory_supervisor.py",
        "src/interfaces/relation_memory_text.py", "src/models/xvla_mem.py", "src/simulator/xvla_mem_env.py")
    for name in execution_files:
        if (shadow_manifest.get("source_sha256", {}).get(name)
                != event_manifest.get("source_sha256", {}).get(name)):
            raise ValueError("D2 execution source differs between arms: " + name)
    rows = shadow_rows + event_rows
    groups = []
    for mode in ("shadow", "event"):
        for task in ("T1", "T3"):
            subset = [r for r in rows if r["decision_mode"] == mode and r["task"] == task]
            lag = [pair["evaluator_minus_writer_step"] for r in subset for pair in r["pairs"]]
            groups.append({"decision_mode": mode, "task": task, "episodes": len(subset),
                "endpoint_matches": sum(r["endpoint_match"] for r in subset),
                "writer_returns": sum(r["writer_returns"] for r in subset),
                "evaluator_cycles": sum(r["evaluator_cycles"] for r in subset),
                "writer_events": sum(r["writer_events"] for r in subset),
                "evaluator_events": sum(r["evaluator_events"] for r in subset),
                "ordinal_pairs": sum(len(r["pairs"]) for r in subset),
                "lag_median": statistics.median(lag) if lag else None,
                "lag_min": min(lag) if lag else None, "lag_max": max(lag) if lag else None})
    output_dir.mkdir(parents=True)
    (output_dir / "episode_alignment.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output_dir / "cycle_pairs.jsonl").write_text("".join(json.dumps({"episode_id": r["episode_id"],
        "decision_mode": r["decision_mode"], "task": r["task"], **pair}, ensure_ascii=False) + "\n"
        for r in rows for pair in r["pairs"]), encoding="utf-8")
    summary = {"classification": "CUSTOM_V2_XVLA_D3_D5_D2_CYCLE_ALIGNMENT_DESCRIPTIVE", "status": "complete",
        "episodes": len(rows), "paired_assignments": len(shadow_rows), "groups": groups,
        "shadow_manifest_sha256": digest(SHADOW / "run_manifest.json"),
        "event_manifest_sha256": digest(EVENT / "run_manifest.json"),
        "interpretation": "Offline endpoint count agreement and ordinal timing only. Evaluator values were not exposed to policy or writer. This does not measure identity, intent, target accuracy, or policy efficacy."}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# X-VLA D3/D5 cycle alignment on D2 cohort", "",
        "Classification: `CUSTOM_V2_XVLA_D3_D5_D2_CYCLE_ALIGNMENT_DESCRIPTIVE`.", "",
        "All 48 D2 traces were hash-verified. Evaluator data is used only for this offline audit.", "",
        "| Arm | Task | N | Endpoint match | Writer returns | Evaluator cycles | Ordinal pairs | Evaluator minus writer step, median (range) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        lag = "n/a" if group["lag_median"] is None else f"{group['lag_median']:.1f} ({group['lag_min']}–{group['lag_max']})"
        lines.append(f"| {group['decision_mode']} | {group['task']} | {group['episodes']} | "
            f"{group['endpoint_matches']}/{group['episodes']} | {group['writer_returns']} | "
            f"{group['evaluator_cycles']} | {group['ordinal_pairs']} | {lag} |")
    lines += ["", summary["interpretation"], "", "See `episode_alignment.jsonl` and `cycle_pairs.jsonl` for details."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"episode_count": len(rows), "paired_assignment_count": len(shadow_rows),
        "shadow_manifest_sha256": summary["shadow_manifest_sha256"],
        "event_manifest_sha256": summary["event_manifest_sha256"],
        "episode_inputs": [{"episode_id": r["episode_id"], "decision_mode": r["decision_mode"],
            "sha256": r["input_hashes"]} for r in rows],
        "outputs_sha256": {name: digest(output_dir / name) for name in
            ("summary.json", "report.md", "episode_alignment.jsonl", "cycle_pairs.jsonl")}}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
