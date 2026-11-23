#!/usr/bin/env python
"""Verify X-VLA D1 text/shadow action and prompt parity."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "experiments/evaluation/xvla_d1_text_20261123_r3"
SHADOW = ROOT / "experiments/evaluation/xvla_d1_shadow_20261123"
OUTPUT = ROOT / "experiments/evaluation/xvla_d1_parity_20261123"


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


def verify_episode(directory, mode):
    status = read(directory / "status.json")
    if status.get("status") != "complete" or status.get("decision_arm") != mode:
        raise ValueError("Incomplete or mislabeled episode " + directory.name)
    files = ("summary.json", "queries.jsonl", "actions.jsonl", "trajectory.npz", "memory_trace.jsonl", "observations.jsonl")
    for name in files:
        expected = status.get("files_sha256", {}).get(name)
        if not expected or digest(directory / name) != expected:
            raise ValueError("Missing or changed artifact " + directory.name + "/" + name)
    return status


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserving existing output " + str(OUTPUT))
    text_manifest, shadow_manifest = read(TEXT / "run_manifest.json"), read(SHADOW / "run_manifest.json")
    if text_manifest.get("status") != "complete" or shadow_manifest.get("status") != "complete":
        raise ValueError("Both X-VLA D1 runs must be complete")
    for field in ("protocol_sha256", "split_lock_sha256", "decision_protocol_sha256", "memory_contract", "environments"):
        if text_manifest.get(field) != shadow_manifest.get(field):
            raise ValueError("D1 run identity differs in " + field)
    if text_manifest["arguments"]["decision_mode"] != "text" or shadow_manifest["arguments"]["decision_mode"] != "shadow":
        raise ValueError("Expected text and shadow decision modes")
    if text_manifest["roster"] != shadow_manifest["roster"] or len(text_manifest["roster"]) != 2:
        raise ValueError("D1 roster mismatch or unexpected size")
    episodes = []
    for row in text_manifest["roster"]:
        stem = f"{row['task']}_F1_mem-on_seen_state{row['state_idx']}"
        td, sd = TEXT / "episodes" / stem, SHADOW / "episodes" / stem
        ts, ss = verify_episode(td, "text"), verify_episode(sd, "shadow-projected")
        identity = (row["task"], row["initial_state_sha256"], text_manifest["arguments"]["policy_seed"])
        if (ts["task"], ts["initial_state_sha256"], ts["policy_seed"]) != identity:
            raise ValueError("Text episode roster identity mismatch " + stem)
        if (ss["task"], ss["initial_state_sha256"], ss["policy_seed"]) != identity:
            raise ValueError("Shadow episode roster identity mismatch " + stem)
        tq, sq = jsonl(td / "queries.jsonl"), jsonl(sd / "queries.jsonl")
        trows = [(r["step"], r["prompt"], r["chunk_sha256"]) for r in tq]
        srows = [(r["step"], r["prompt"], r["chunk_sha256"]) for r in sq]
        arrays = {}
        with np.load(td / "trajectory.npz", allow_pickle=False) as ta, np.load(sd / "trajectory.npz", allow_pickle=False) as sa:
            for key in ("actions", "query_steps", "action_chunks"):
                arrays[key] = bool(np.array_equal(ta[key], sa[key]))
        tr, sr = read(td / "summary.json")["result"], read(sd / "summary.json")["result"]
        episodes.append({"task": row["task"], "state_idx": row["state_idx"],
            "initial_state_sha256": row["initial_state_sha256"],
            "text": {"success": tr["success"], "scr": tr["scr"], "overshot": tr["overshot"], "steps": tr["total_steps"], "queries": len(tq)},
            "shadow": {"success": sr["success"], "scr": sr["scr"], "overshot": sr["overshot"], "steps": sr["total_steps"], "queries": len(sq)},
            "queries_identical": trows == srows, "trajectory_arrays_identical": arrays,
            "all_trajectory_arrays_identical": all(arrays.values()),
            "text_status_sha256": digest(td / "status.json"), "shadow_status_sha256": digest(sd / "status.json")})
    OUTPUT.mkdir(parents=True)
    result = {"status": "complete", "classification": "CUSTOM_V2_XVLA_D1_SHADOW_PARITY",
        "comparability": "NON-COMPARABLE_OFFICIAL_PAPER", "pairs": len(episodes),
        "exact_prompt_and_trajectory_pairs": sum(e["queries_identical"] and e["all_trajectory_arrays_identical"] for e in episodes),
        "episodes": episodes, "text_manifest_sha256": digest(TEXT / "run_manifest.json"),
        "shadow_manifest_sha256": digest(SHADOW / "run_manifest.json"),
        "interpretation": "D1 development-only parity check. Shadow observation must preserve text prompts, query timing, and actions. Two state-0 task episodes do not estimate efficacy."}
    (OUTPUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# X-VLA D1 text/shadow parity", "", "| Task | Text success / steps | Shadow success / steps | Query rows identical | Action, query-step and chunk arrays identical |", "|---|---:|---:|---|---|"]
    for e in episodes:
        lines.append(f"| {e['task']} | {e['text']['success']} / {e['text']['steps']} | {e['shadow']['success']} / {e['shadow']['steps']} | {e['queries_identical']} | {e['all_trajectory_arrays_identical']} |")
    lines += ["", result["interpretation"]]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"text_manifest_sha256": result["text_manifest_sha256"], "shadow_manifest_sha256": result["shadow_manifest_sha256"],
        "episode_status_hashes": [{"task": e["task"], "text": e["text_status_sha256"], "shadow": e["shadow_status_sha256"]} for e in episodes],
        "outputs_sha256": {n: digest(OUTPUT / n) for n in ("report.json", "report.md")}}
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
