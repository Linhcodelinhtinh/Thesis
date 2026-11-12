"""Read existing artifacts only; never launch policy, simulator, or training.

Run from anywhere with Python 3. No model dependencies are imported. Outputs
are a dated writing snapshot, not a new benchmark or a runtime certification.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/thesis/evidence"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = {}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_out(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


snapshot = {
    "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
    "scope": "Existing artifact inspection; no fresh benchmark/training/runtime tests",
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
    "limitations": [
        "Episode JSONs and selected small manifests were read and hashed.",
        "Only the four exported adapter files are independently rehashed here; large base weights, raw HDF5, shards and all videos are not rehashed.",
        "Historical runs are inventoried, not pooled or declared comparable.",
        "Stored test split has already been used for development; it is not a fresh confirmatory holdout.",
    ],
}

# Do not turn directories containing several runs into a pooled result.
groups = defaultdict(list)
for path in sorted((ROOT / "experiments/results").rglob("episode.json")):
    row = read(path)
    run = path.parent.parent.parent.relative_to(ROOT).as_posix()
    groups[run].append((path, row))
historical = []
for run, values in groups.items():
    historical.append({
        "run": run, "episodes_found": len(values),
        "successes_in_found_episodes": sum(r["success"] is True for _, r in values),
        "task_count": len({r["task_name"] for _, r in values}),
        "execution_horizons": json.dumps(sorted({r.get("execution_horizon_s") for _, r in values}, key=str)),
        "checkpoints_recorded": json.dumps(sorted({str(r.get("checkpoint")) for _, r in values})),
        "interpretation": "INVENTORY_ONLY; completeness and matched provenance not certified",
    })
csv_out("historical_run_inventory.csv", historical)
snapshot["historical_runs_found"] = len(historical)
snapshot["historical_episode_jsons_read"] = sum(len(v) for v in groups.values())
snapshot["frozen_v1"] = {
    "report": "experiments/baseline_v1/V1_BASELINE_REPORT.md",
    "signoff": read("experiments/baseline_v1/freeze_audit_signoff.json"),
    "referenced_raw_episode_count": len(list((ROOT / "experiments/results/raw_baseline").rglob("episode.json"))),
    "use": "Historical reported result only; referenced raw episodes absent at inspection",
}

run_root = Path("experiments/evaluation/results_mem_2/step5_test_20261109-093744")
episodes, manifests = [], {}
for condition in ["B0", "F1"]:
    folder = run_root / f"test_{condition}"
    manifest = read(folder / "run_manifest.json")
    manifests[condition] = manifest
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete selected run: {condition}")
    for p in sorted((ROOT / folder / "episodes").glob("*/summary.json")):
        s = read(p)
        status = read(p.with_name("status.json"))
        if status["status"] != "complete" or status.get("logging_errors"):
            raise ValueError(f"Invalid episode status: {p}")
        if digest(p) != status["files_sha256"]["summary.json"]:
            raise ValueError(f"Summary checksum mismatch: {p}")
        r = s["result"]
        trace = r.get("memory_trace", [])
        final_memory = trace[-1]["snapshot"] if trace else None
        episodes.append({
            "condition": condition, "task": s["task"], "state_idx": s["state_idx"],
            "source_demo": s["source_demo"], "initial_state_sha256": s["initial_state_sha256"],
            "policy_seed": s["policy_seed"], "success": r["success"], "scr": r["scr"],
            "overshot": r["overshot"], "steps": r["total_steps"],
            "evaluator_confirmed_cycles": r["confirmed_cycles"],
            "final_writer_confirmed_cycles": sum(c["confirmed"] for c in final_memory["cycles"].values()) if final_memory else "",
            "final_writer_history_complete": final_memory["history_complete"] if final_memory else "",
            "policy_queries": len(r["prompt_history"]),
            "queries_with_memory_suffix": sum(" Memory: " in t for t in r["prompt_history"]),
            "queries_with_incomplete_history_text": sum("history incomplete" in t for t in r["prompt_history"]),
            "avg_policy_ms": r["avg_latency_ms"], "wall_time_s": r["total_wall_time_s"],
            "completed_subgoals": r["completed_subgoals"],
            "termination_reason": r["termination_reason"],
            "summary_file": p.relative_to(ROOT).as_posix(),
        })
    found = [e for e in episodes if e["condition"] == condition]
    if len(found) != manifest["expected_episodes"]:
        raise ValueError(f"Roster count mismatch: {condition}")
    keys = {(e["task"], e["initial_state_sha256"], e["policy_seed"]) for e in found}
    expected = {(r["task"], r["initial_state_sha256"], manifest["arguments"]["policy_seed"]) for r in manifest["roster"]}
    if keys != expected or len(keys) != len(found):
        raise ValueError(f"Roster identity mismatch: {condition}")
csv_out("mem_test_episodes.csv", episodes)
counts = []
for condition in ["B0", "F1"]:
    for task in ["T1", "T3", "ALL"]:
        es = [e for e in episodes if e["condition"] == condition and (task == "ALL" or e["task"] == task)]
        counts.append({"condition": condition, "task": task, "n_task_episodes": len(es),
                       "unique_state_hashes": len({e["initial_state_sha256"] for e in es}),
                       "successes": sum(e["success"] for e in es),
                       "sr": sum(e["success"] for e in es) / len(es),
                       "mean_scr": sum(e["scr"] for e in es) / len(es),
                       "overshot_count": sum(e["overshot"] for e in es)})
csv_out("mem_test_counts.csv", counts)
pair_keys = lambda condition: {(e["task"], e["initial_state_sha256"], e["policy_seed"]): e for e in episodes if e["condition"] == condition}
b, f = pair_keys("B0"), pair_keys("F1")
if b.keys() != f.keys():
    raise ValueError("B0/F1 pairing mismatch")
shared = ["protocol_sha256", "split_lock_sha256", "split_lock_file_sha256", "base"]
snapshot["mem_test"] = {
    "run": run_root.as_posix(), "counts": counts, "paired_task_episodes": len(b),
    "unique_state_clusters": len({k[1] for k in b}),
    "outcome_pairs": dict(Counter(f"B0_{int(b[k]['success'])}_F1_{int(f[k]['success'])}" for k in b)),
    "matched_manifest_fields": {k: manifests["B0"][k] == manifests["F1"][k] for k in shared},
    "source_hash_differences": [k for k in sorted(set(manifests['B0']['source_sha256']) | set(manifests['F1']['source_sha256']))
                                if manifests['B0']['source_sha256'].get(k) != manifests['F1']['source_sha256'].get(k)],
    "run_provenance": {c: {k: m[k] for k in ["git_revision", "dirty_diff_sha256", "code_snapshot_sha256", "protocol_sha256", "split_lock_sha256", "base", "environments", "cuda", "loaded_policies"]} for c, m in manifests.items()},
    "primary_summary_hashes_verified": len(episodes),
}
if not all(snapshot["mem_test"]["matched_manifest_fields"].values()):
    raise ValueError("Selected run contracts do not match")

train = Path("experiments/training/results_mem_2/mem-adapter-20261105-083543")
cfg = read(train / "run_config.json")
progress = read(train / "checkpoint-003000/progress.json")
adapter = read(train / "pretrained_model/mem_adapter_manifest.json")
metrics_path = ROOT / train / "metrics.jsonl"
INPUTS[metrics_path.relative_to(ROOT).as_posix()] = digest(metrics_path)
metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
validation = []
for r in metrics:
    if "validation_flow_loss" in r:
        for mode, value in r["validation_flow_loss"].items():
            for domain, v in value["by_domain"].items():
                validation.append({"step": r["step"], "inference_memory": mode, "domain": domain, **v})
csv_out("validation_loss.csv", validation)
adapter_files = {}
for name, expected in adapter["adapter_files"].items():
    actual = digest(ROOT / train / "pretrained_model" / name)
    expected = expected["sha256"] if isinstance(expected, dict) else expected
    adapter_files[name] = {"sha256": actual, "matches_manifest": actual == expected}
    if actual != expected:
        raise ValueError(f"Adapter checksum mismatch: {name}")
dataset = read("experiments/training/results_mem_2/mem-adapter-v1/dataset-manifest.json")
dataset_path = "experiments/training/results_mem_2/mem-adapter-v1/dataset-manifest.json"
snapshot["training"] = {
    "run": train.as_posix(), "step": progress["step"],
    "arguments": cfg["arguments"], "effective_batch_size": cfg["effective_batch_size"],
    "trainable_parameters": cfg["trainable_parameters"], "versions": cfg["versions"],
    "hardware": cfg["hardware"], "adapter_files": adapter_files,
    "data_manifest_matches_training": INPUTS[dataset_path] == cfg["data_manifest_sha256"],
    "dataset_episode_counts": dict(Counter(f"{e['domain']}/{e['split']}" for e in dataset["episodes"])),
    "dataset_frame_counts": dataset["counts"],
    "dataset_image_contract": dataset["image_contract"],
    "dataset_sources": dataset["sources"],
    "validation_loss": validation,
}
snapshot["derived_data"] = {
    "versioned_v1_output_exists": (ROOT / "data/libero_mem_derived_v1").exists(),
    "legacy_replay_summaries": [],
    "note": "Summary metadata only; HDF5 contents, byte hashes and end-to-end readiness not certified",
}
for parent in ["data/libero_mem_derived", "data/libero_mem_derived_pilot"]:
    for p in (ROOT / parent).glob("*_summary.json"):
        r = read(p)
        snapshot["derived_data"]["legacy_replay_summaries"].append({
            "file": p.relative_to(ROOT).as_posix(),
            **{k: r.get(k) for k in ["task", "split", "total_replayed", "successful_episodes", "timestamp", "target_hdf5_sha256"]},
        })

# Preserve the earlier RGB event-writer audit as a separate evidence stream.
# Its demonstrations and metrics are not pooled with the later policy rollout.
writer_root = Path("experiments/audits/memory_writer_v3")
writer = read(writer_root / "verification_summary.json")
writer_evidence_hashes = {}
for relative, expected in writer["evidence_hashes"].items():
    artifact = writer_root / relative
    actual = digest(ROOT / artifact)
    if actual != expected:
        raise ValueError(f"Writer audit evidence checksum mismatch: {artifact}")
    writer_evidence_hashes[artifact.as_posix()] = actual
package = writer_root / "memory_writer_source.zip"
package_hash = digest(ROOT / package)
if package_hash != writer["source_package"]["sha256"]:
    raise ValueError("Writer source archive checksum mismatch")
writer_evidence_hashes[package.as_posix()] = package_hash
INPUTS.update(writer_evidence_hashes)
writer_core = ["src/perception/object_tracker.py", "src/memory/event_writer.py",
               "src/interfaces/libero_mem_text.py"]
writer_core_matches = {}
for relative in writer_core:
    actual = digest(ROOT / relative)
    expected = writer["code_hashes"][relative]
    writer_core_matches[relative] = {"sha256": actual, "matches_audit": actual == expected}
snapshot["writer_replay"] = {
    "summary": (writer_root / "verification_summary.json").as_posix(),
    "summary_status": writer["status"],
    "date": writer["date"],
    "real_demonstrations": writer["real_demonstrations"],
    "unique_initial_states": writer["unique_initial_states"],
    "frames": writer["frames"],
    "grounded_frames": writer["grounded_frames"],
    "memory_frames": writer["memory_frames"],
    "max_prompt_tokens": writer["max_prompt_tokens"],
    "annotated_returns": writer["annotated_returns"],
    "predicted_returns": writer["predicted_returns"],
    "matches": writer["matches"],
    "false_positive": writer["false_positive"],
    "false_negative": writer["false_negative"],
    "event_tolerance_frames": writer["event_tolerance_frames"],
    "confirmation_delay_frames": writer["confirmation_delay_frames"],
    "incomplete_histories": writer["incomplete_histories"],
    "accepted_bowl_detections": writer["accepted_bowl_detections"],
    "accepted_at_iou_0_5": writer["accepted_at_iou_0_5"],
    "predicted_index_live_detector_parity_frames": writer["predicted_index_live_detector_parity_frames"],
    "no_memory_index_parity_frames": writer["no_memory_index_parity_frames"],
    "worker_parity": writer["worker_parity"],
    "core_source_hashes": writer_core_matches,
    "limitations": writer["limitations"],
    "evidence_sha256": writer_evidence_hashes,
}

for folder in ["docs", "src", "configs", "envs", "resources/manifests"]:
    for p in (ROOT / folder).rglob("*"):
        if not p.is_file() or "thesis" in p.relative_to(ROOT).parts or "__pycache__" in p.parts:
            continue
        if p.suffix in [".md", ".py", ".json", ".yaml", ".yml", ".tex", ".bib", ".pdf"]:
            INPUTS[p.relative_to(ROOT).as_posix()] = digest(p)
save("snapshot.json", snapshot)
save("input_hashes.json", INPUTS)
print(json.dumps({"snapshot": str(OUT / "snapshot.json"), "historical_jsons_read": snapshot["historical_episode_jsons_read"],
                  "mem_test": counts, "training_step": progress["step"], "adapter_files_verified": len(adapter_files),
                  "dataset_episode_counts": snapshot["training"]["dataset_episode_counts"]}, ensure_ascii=False, indent=2))
