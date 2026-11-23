#!/usr/bin/env python
"""Offline D3/D5 alignment of X-VLA writer and evaluator cycle traces."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/evaluation/xvla_e2_f1_mem_on_20261122"
OUTPUT = ROOT / "experiments/evaluation/xvla_d3_d5_e2_trace_analysis_20261123"


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


def analyze(directory):
    status = read(directory / "status.json")
    if status.get("status") != "complete" or status.get("logging_errors"):
        raise ValueError("Incomplete episode " + directory.name)
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
            raise ValueError("Missing bowl_1 at step " + str(step))
        writer.append(int(rel["returns"]))
        evaluator.append(int(observations[step]["confirmed_cycles"]))
        unknown.append(rel.get("status") == "unknown")
        incomplete.append(not bool(rel.get("history_complete")))

    def events(values):
        result = []
        for i in range(1, len(values)):
            result.extend({"cycle": cycle, "step": steps[i]}
                          for cycle in range(values[i - 1] + 1, values[i] + 1))
        return result
    we, ee = events(writer), events(evaluator)
    wm = {x["cycle"]: x["step"] for x in we}
    em = {x["cycle"]: x["step"] for x in ee}
    pairs = [{"cycle": c, "writer_step": wm[c], "evaluator_step": em[c],
              "evaluator_minus_writer_step": em[c] - wm[c]} for c in sorted(wm.keys() & em.keys())]
    result = summary["result"]
    return {"episode_id": summary["episode_id"], "task": summary["task"], "state_idx": summary["state_idx"],
            "initial_state_sha256": summary["initial_state_sha256"], "policy_seed": summary["policy_seed"],
            "success": bool(result["success"]), "overshot": bool(result["overshot"]),
            "steps": steps[-1], "writer_returns": writer[-1], "evaluator_cycles": evaluator[-1],
            "endpoint_match": writer[-1] == evaluator[-1], "writer_events": len(we), "evaluator_events": len(ee),
            "pairs": pairs, "unknown_fraction": sum(unknown) / len(steps),
            "incomplete_fraction": sum(incomplete) / len(steps), "input_hashes": hashes}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserving existing output " + str(OUTPUT))
    manifest = read(RUN / "run_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("completed_episodes") != 40:
        raise ValueError("Expected completed 40-episode X-VLA F1-on E2 cohort")
    dirs = sorted(p for p in (RUN / "episodes").iterdir() if p.is_dir())
    if len(dirs) != 40:
        raise ValueError("Expected 40 episodes, found " + str(len(dirs)))
    rows = [analyze(p) for p in dirs]
    if len({(r["task"], r["initial_state_sha256"], r["policy_seed"]) for r in rows}) != 40:
        raise ValueError("Duplicate task/state/seed assignment")
    groups = []
    for task in ("T1", "T3"):
        subset = [r for r in rows if r["task"] == task]
        lag = [p["evaluator_minus_writer_step"] for r in subset for p in r["pairs"]]
        groups.append({"task": task, "episodes": len(subset),
                       "endpoint_matches": sum(r["endpoint_match"] for r in subset),
                       "writer_returns": sum(r["writer_returns"] for r in subset),
                       "evaluator_cycles": sum(r["evaluator_cycles"] for r in subset),
                       "writer_events": sum(r["writer_events"] for r in subset),
                       "evaluator_events": sum(r["evaluator_events"] for r in subset),
                       "ordinal_pairs": sum(len(r["pairs"]) for r in subset),
                       "lag_median": statistics.median(lag) if lag else None,
                       "lag_min": min(lag) if lag else None, "lag_max": max(lag) if lag else None})
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "episode_alignment.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (OUTPUT / "cycle_pairs.jsonl").write_text("".join(json.dumps({"episode_id": r["episode_id"], "task": r["task"], **p}) + "\n"
        for r in rows for p in r["pairs"]), encoding="utf-8")
    summary = {"classification": "CUSTOM_V2_XVLA_D3_D5_CYCLE_ALIGNMENT_DESCRIPTIVE", "status": "complete",
               "episodes": len(rows), "groups": groups, "source_manifest_sha256": digest(RUN / "run_manifest.json"),
               "interpretation": "Offline count agreement and timing only. Evaluator values were not exposed to policy or writer. This does not measure identity, intent, target accuracy, or policy efficacy."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# X-VLA D3/D5 cycle alignment", "", "Classification: CUSTOM_V2_XVLA_D3_D5_CYCLE_ALIGNMENT_DESCRIPTIVE.", "",
             "All 40 source episodes passed status and source artifact hash checks. Evaluator data is used only for offline audit.", "",
             "| Task | N | Endpoint match | Writer returns | Evaluator cycles | Event pairs | Evaluator minus writer step, median (range) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        lag = "n/a" if g["lag_median"] is None else f"{g['lag_median']:.1f} ({g['lag_min']}–{g['lag_max']})"
        lines.append(f"| {g['task']} | {g['episodes']} | {g['endpoint_matches']}/{g['episodes']} | {g['writer_returns']} | {g['evaluator_cycles']} | {g['ordinal_pairs']} | {lag} |")
    lines += ["", summary["interpretation"], "", "See episode_alignment.jsonl and cycle_pairs.jsonl for details."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"episode_count": len(rows), "source_manifest_sha256": summary["source_manifest_sha256"],
               "episode_inputs": [{"episode_id": r["episode_id"], "sha256": r["input_hashes"]} for r in rows],
               "outputs_sha256": {n: digest(OUTPUT / n) for n in ("summary.json", "report.md", "episode_alignment.jsonl", "cycle_pairs.jsonl")}}
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
