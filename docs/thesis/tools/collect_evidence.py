"""Read existing artifacts only; never launch policy, simulator, or training.

Run from anywhere with Python 3. No model dependencies are imported. Outputs
are a dated writing snapshot, not a new benchmark or a runtime certification.
Offline RGB writer indexes are recorded with their explicit scope and status.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    "scope": "Existing artifact inspection, one custom seen-state B0/F0 development pilot, offline T7-T10 writer diagnostics, and an offline MiniVLA source/checkpoint audit; no model training or official benchmark score",
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
    "limitations": [
        "Episode JSONs and selected small manifests were read and hashed.",
        "Only the four exported adapter files are independently rehashed here; large base weights, raw HDF5, shards and all videos are not rehashed.",
        "Historical runs are inventoried, not pooled or declared comparable.",
        "Stored test split has already been used for development; it is not a fresh confirmatory holdout.",
        "T7 RGB index evidence, when present, is an offline writer artifact and has no independent relation labels.",
        "T9 RGB index evidence is one author-refined training demonstration; it has no independent identity or containment labels.",
        "MiniVLA checkpoint/source audit is provenance and interface evidence only; it includes no model inference or policy rollout.",
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

# Version-2 data preparation and offline decision replay are separate from
# historical model-training/policy results above.
split_path = Path("resources/manifests/libero_mem_t1_t3_base10_split_v2.json")
split_lock = read(split_path)
alignment_path = Path("experiments/training/relations-author-split-v2/alignment-audit.json")
alignment = read(alignment_path)
if alignment.get("status") != "passed":
    raise ValueError("Selected V2 alignment audit did not pass")
snapshot["v2_data"] = {
    "classification": "DATA_ALIGNMENT_QA_NOT_POLICY_SCORE",
    "split_lock_path": split_path.as_posix(),
    "split_lock_file_sha256": INPUTS[split_path.as_posix()],
    "split_lock_content_sha256": split_lock.get("lock_sha256"),
    "alignment_path": alignment_path.as_posix(),
    "alignment_file_sha256": INPUTS[alignment_path.as_posix()],
    "alignment": {key: alignment.get(key) for key in [
        "status", "classification", "episodes", "all_state_action_rows_compared",
        "rgb_frames_compared_per_camera", "policy_adapter_parity_samples",
    ]},
}

decision_root = Path("experiments/audits/relation-e1-20261113-232401")
decision_protocol_path = Path("configs/benchmarks/libero_mem_decision_dev_v1.json")
decision_protocol = read(decision_protocol_path)
decision_reports = {}
for task in ["T1", "T3"]:
    report_path = decision_root / f"decision-dev-v1-final-{task}.json"
    report = read(report_path)
    rows = list(report.get("episodes", {}).values())
    decision_reports[task] = {
        "path": report_path.as_posix(),
        "sha256": INPUTS[report_path.as_posix()],
        "classification": report.get("classification"),
        "exposure": report.get("exposure"),
        "episodes": len(rows),
        "frames": sum(row.get("frames", 0) for row in rows),
        "scheduled_queries": sum(row.get("queries", 0) for row in rows),
        "trigger_requests": sum(len(row.get("requests", [])) for row in rows),
        "index_sha256": report.get("index_sha256"),
        "decision_protocol_sha256": report.get("decision_protocol_sha256"),
    }
snapshot["decision_replay"] = {
    "classification": "OPEN_LOOP_DEVELOPMENT_NOT_POLICY_SCORE",
    "protocol_path": decision_protocol_path.as_posix(),
    "protocol_file_sha256": INPUTS[decision_protocol_path.as_posix()],
    "protocol_id": decision_protocol.get("protocol_id"),
    "tasks": decision_reports,
}

receipt_path = Path("resources/manifests/libero_mem_t7_refined_receipt_v1.json")
receipt = read(receipt_path)
timing_path = Path("experiments/audits/t7-raw-branch-timing-20261114.json")
timing = read(timing_path)
t7_rgb_reports = {}
for name in ["t7-raw-rgb-branch1-20261114.json", "t7-raw-rgb-branch2-20261114.json"]:
    report = read(Path("experiments/audits") / name)
    rows = list(report.get("results", {}).values())
    t7_rgb_reports[name] = {
        "sha256": INPUTS[(Path("experiments/audits") / name).as_posix()],
        "classification": report.get("classification"),
        "camera_observations": report.get("camera_observations"),
        "demonstrations": len(rows),
        "completed_or_branches": sum(len(row.get("completed_branches", [])) for row in rows),
        "source_sha256": report.get("source_sha256"),
    }
t7_index_path = Path("experiments/audits/relation-e1-20261114-t7-demo1")
t7_manifest_path = t7_index_path / "manifest.json"
t7_index = {
    "classification": "T7_RGB_WRITER_OFFLINE_INDEX_NOT_POLICY_SCORE",
    "source_receipt_path": receipt_path.as_posix(),
    "source_receipt_file_sha256": INPUTS[receipt_path.as_posix()],
    "source_revision": receipt.get("revision"),
    "source_hdf5_sha256": receipt.get("sha256", {}).get(
        "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5"),
    "source_local_hash_verified": receipt.get("local_verification", {}).get("task_file_sha256_verified"),
    "independent_relation_labels": False,
}
if t7_manifest_path.is_file():
    manifest = read(t7_manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("T7 relation index manifest exists but is not complete")
    for name, expected in manifest.get("artifacts_sha256", {}).items():
        artifact = t7_index_path / name
        actual = digest(ROOT / artifact)
        if actual != expected:
            raise ValueError(f"T7 writer index checksum mismatch: {artifact}")
        INPUTS[artifact.as_posix()] = actual
    sample_path = t7_index_path / "samples.jsonl"
    samples = [json.loads(line) for line in (ROOT / sample_path).read_text(encoding="utf-8").splitlines()]
    trace_path = t7_index_path / "memory_trace.jsonl"
    traces = [json.loads(line) for line in (ROOT / trace_path).read_text(encoding="utf-8").splitlines()]
    detection_path = t7_index_path / "detections.jsonl"
    detections = [json.loads(line) for line in (ROOT / detection_path).read_text(encoding="utf-8").splitlines()]
    INPUTS[sample_path.as_posix()] = digest(ROOT / sample_path)
    INPUTS[trace_path.as_posix()] = digest(ROOT / trace_path)
    INPUTS[detection_path.as_posix()] = digest(ROOT / detection_path)
    identity_counts, relation_counts = Counter(), Counter()
    for row in traces:
        snapshot_row = row["snapshot"]
        identity_counts.update(obj.get("identity_status", "unknown")
                               for obj in snapshot_row.get("objects", {}).values())
        relation_counts.update(fact.get("status", "unknown")
                               for fact in snapshot_row.get("relations", {}).values())
    initial_detections = []
    for row in detections:
        evidence = row["evidence"]
        if evidence.get("step", 0) >= 20:
            continue
        boxes = evidence.get("views", {}).get("agentview_image", [])
        counts = Counter(box.get("label") for box in boxes)
        initial_detections.append({"step": evidence["step"],
                                   "bowl_detections": counts.get("bowl", 0),
                                   "plate_detections": counts.get("plate", 0)})
    snapshot_details = {
        "status": "complete",
        "manifest_path": t7_manifest_path.as_posix(),
        "manifest_sha256": INPUTS[t7_manifest_path.as_posix()],
        "episodes": manifest.get("episodes", []),
        "total_samples": manifest.get("total_samples"),
        "manifest_memory_frames": manifest.get("memory_frames"),
        "samples_with_memory": sum(bool(row.get("has_memory")) for row in samples),
        "samples_without_relevant_fact": sum(row.get("skipped_reason") == "no_relevant_fact" for row in samples),
        "identity_status_observations": dict(identity_counts),
        "relation_status_observations": dict(relation_counts),
        "first_20_detection_counts": initial_detections,
        "detector_revision": (manifest.get("detector") or {}).get("revision"),
    }
else:
    snapshot_details = {"status": "not_complete_or_not_available"}
t7_index["index"] = snapshot_details
snapshot["t7_writer_audit"] = t7_index
snapshot["t7_runtime_replay"] = {
    "timing_report_path": timing_path.as_posix(),
    "timing_report_sha256": INPUTS[timing_path.as_posix()],
    "classification": timing.get("classification"),
    "camera_observations": timing.get("camera_observations"),
    "source_sha256": timing.get("source_sha256"),
    "demonstrations": len(timing.get("results", {})),
    "rgb_branch_reports": t7_rgb_reports,
}

t7_box_audit_path = Path("experiments/audits/t7-detector-box-e1-20261115/report.json")
if t7_box_audit_path.is_file():
    t7_box_audit = read(t7_box_audit_path)
    if (t7_box_audit.get("classification") != "CUSTOM_OFFLINE_DETECTOR_AUDIT_AGAINST_AUTHOR_EXO_BOXES"
            or t7_box_audit.get("frames") != 425
            or t7_box_audit.get("source_frame_hashes_verified") != 425):
        raise ValueError("T7 detector-box audit is incomplete or has an unexpected scope")
    if (t7_box_audit.get("input_sha256", {}).get("manifest.json") != INPUTS[t7_manifest_path.as_posix()]
            or t7_box_audit.get("input_sha256", {}).get("detections.jsonl") != INPUTS[detection_path.as_posix()]):
        raise ValueError("T7 detector-box audit does not refer to the collected writer artifacts")
    INPUTS[t7_box_audit_path.as_posix()] = digest(ROOT / t7_box_audit_path)
    snapshot["t7_detector_box_audit"] = {
        "classification": t7_box_audit["classification"],
        "report_path": t7_box_audit_path.as_posix(),
        "report_sha256": INPUTS[t7_box_audit_path.as_posix()],
        "frames": t7_box_audit["frames"],
        "source_frame_hashes_verified": t7_box_audit["source_frame_hashes_verified"],
        "matching": t7_box_audit["matching"],
        "all_frames": t7_box_audit["all_frames"],
        "first_20_frames": t7_box_audit["first_20_frames"],
        "limitations": t7_box_audit["limitations"],
    }
else:
    snapshot["t7_detector_box_audit"] = {"status": "not_available"}

t9_receipt_path = Path("resources/manifests/libero_mem_t9_source_receipt_v1.json")
t9_receipt = read(t9_receipt_path)
t9_index_path = Path("experiments/audits/relation-e1-20261115-t9-demo1")
t9_manifest_path = t9_index_path / "manifest.json"
t9_manifest = read(t9_manifest_path)
t9_contract_path = t9_index_path / "index_contract.json"
t9_contract = read(t9_contract_path)
if (t9_manifest.get("status") != "complete"
        or t9_manifest.get("artifact") != "causal_prompt_index_not_training_shards"
        or t9_manifest.get("writer") != "rgb_relations_v1"
        or t9_manifest.get("stream") != "predicted_memory"):
    raise ValueError("T9 relation writer diagnostic is incomplete or has an unexpected scope")
if (t9_receipt.get("repository") != "libero-mem/LIBERO-Mem"
        or t9_receipt.get("revision") != t9_manifest.get("source_revision")
        or not t9_receipt.get("local_verification", {}).get("task_file_sha256_verified")
        or not t9_receipt.get("local_verification", {}).get("metainfo_sha256_verified")):
    raise ValueError("T9 source receipt does not match the locally verified refined source")
for name, expected in t9_manifest.get("artifacts_sha256", {}).items():
    artifact = t9_index_path / name
    actual = digest(ROOT / artifact)
    if actual != expected:
        raise ValueError(f"T9 writer index checksum mismatch: {artifact}")
    INPUTS[artifact.as_posix()] = actual
if (digest(ROOT / t9_contract_path) != t9_manifest.get("index_contract_sha256")
        or t9_contract.get("writer") != t9_manifest.get("writer")
        or t9_contract.get("stream") != t9_manifest.get("stream")
        or t9_contract.get("demo_ids") != [1]
        or t9_contract.get("source_hashes") != t9_manifest.get("sources")):
    raise ValueError("T9 relation index contract does not match the run manifest")
INPUTS[t9_contract_path.as_posix()] = digest(ROOT / t9_contract_path)
t9_samples_path = t9_index_path / "samples.jsonl"
t9_trace_path = t9_index_path / "memory_trace.jsonl"
t9_samples = [json.loads(line) for line in (ROOT / t9_samples_path).read_text(encoding="utf-8").splitlines() if line.strip()]
t9_traces = [json.loads(line) for line in (ROOT / t9_trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
t9_identity_counts, t9_relation_counts = Counter(), Counter()
t9_event_count = 0
for row in t9_traces:
    state = row["snapshot"]
    t9_identity_counts.update(obj.get("identity_status", "unknown") for obj in state.get("objects", {}).values())
    t9_relation_counts.update(fact.get("status", "unknown") for fact in state.get("relations", {}).values())
    t9_event_count += len(row.get("signals", {}).get("new_events", []))
if (len(t9_samples) != t9_manifest.get("total_samples")
        or len(t9_traces) != t9_manifest.get("total_samples")
        or t9_manifest.get("episodes", [{}])[0].get("split") != "train"
        or t9_manifest.get("episodes", [{}])[0].get("frames") != len(t9_samples)
        or sum(bool(row.get("has_memory")) for row in t9_samples) != t9_manifest.get("memory_frames")):
    raise ValueError("T9 relation index rows do not match its complete manifest")
t9_source_name = next((name for name in t9_receipt.get("sha256", {}) if name != "metainfo.json"), None)
if (not t9_source_name
        or t9_manifest.get("sources", {}).get(t9_source_name) != t9_receipt["sha256"].get(t9_source_name)
        or t9_manifest.get("sources", {}).get("metainfo.json") != t9_receipt.get("sha256", {}).get("metainfo.json")):
    raise ValueError("T9 writer index sources differ from its local source receipt")
snapshot["t9_writer_diagnostic"] = {
    "classification": "T9_SINGLE_TRAIN_DEMO_OFFLINE_WRITER_DIAGNOSTIC_NOT_ACCURACY_OR_POLICY_SCORE",
    "source_receipt_path": t9_receipt_path.as_posix(),
    "source_receipt_sha256": INPUTS[t9_receipt_path.as_posix()],
    "source_revision": t9_manifest.get("source_revision"),
    "source_hdf5_sha256": t9_manifest.get("sources", {}).get(t9_source_name),
    "manifest_path": t9_manifest_path.as_posix(),
    "manifest_sha256": INPUTS[t9_manifest_path.as_posix()],
    "index_contract_path": t9_contract_path.as_posix(),
    "index_contract_sha256": INPUTS[t9_contract_path.as_posix()],
    "episode": t9_manifest.get("episodes"),
    "total_samples": len(t9_samples),
    "samples_with_memory": sum(bool(row.get("has_memory")) for row in t9_samples),
    "samples_without_relevant_fact": sum(row.get("skipped_reason") == "no_relevant_fact" for row in t9_samples),
    "identity_status_observations": dict(t9_identity_counts),
    "relation_status_observations": dict(t9_relation_counts),
    "new_events": t9_event_count,
    "detector_revision": (t9_manifest.get("detector") or {}).get("revision"),
    "independent_identity_or_containment_labels": False,
}

# T8 and T10 are held to the same single-demo, offline-writer receipt checks
# as T9. Keep the released raw files and the refined RGB files separately
# identified; neither receipt is a policy or identity-accuracy claim.
PINNED_SUITE_WRITER_SOURCES = {
    "T8": {
        "refined_name": "KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate_demo.hdf5",
        "refined_sha256": "399061494f4a0b4ac8a5ce646aa5755a6698ec65fbd4e6843143de5e95c1c555",
        "refined_size": 28219269155,
        "raw_sha256": "77dcad3d4f04b36441a70dc95b4cd2131189c5ebfa105bf92ce5d2a8125a16af",
        "raw_size": 74236352,
    },
    "T10": {
        "refined_name": "KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center_demo.hdf5",
        "refined_sha256": "bcb9357581d17a6312a34c1f7bd855b9bb7357ca3d5a6aa1c5be2d2a58e1d85b",
        "refined_size": 19947570362,
        "raw_sha256": "53afeac724e9ead7eedf289e9fc985f87b071aa9146b3694d6c25e16760bc72c",
        "raw_size": 25280904,
    },
}
PINNED_REFINED_REVISION = "6df130c05b82e09294fa9534e84a069687b72d33"
PINNED_RAW_REVISION = "61917674055e8b1cbbfd22c1b54c28d20c1a1584"
PINNED_METAINFO_SHA256 = "2afe9bc01f9579d4adeec63a6cf012f8560de3b5a4b7d5a96c0643c94777ad0c"

for task, pins in PINNED_SUITE_WRITER_SOURCES.items():
    slug = task.lower()
    refined_receipt_path = Path(f"resources/manifests/libero_mem_{slug}_refined_receipt_v1.json")
    raw_receipt_path = Path(f"resources/manifests/libero_mem_{slug}_raw_receipt_v1.json")
    refined_receipt = read(refined_receipt_path)
    raw_receipt = read(raw_receipt_path)
    refined_filename = pins["refined_name"]
    raw_filename = refined_filename
    refined_local = refined_receipt.get("local_verification", {})
    raw_local = raw_receipt.get("local_verification", {})
    refined_local_path = Path(refined_local.get("task_file_path", ""))
    raw_local_path = Path(raw_local.get("task_file_path", ""))
    if (refined_receipt.get("repository") != "libero-mem/LIBERO-Mem"
            or refined_receipt.get("revision") != PINNED_REFINED_REVISION
            or refined_receipt.get("sha256", {}).get(refined_filename) != pins["refined_sha256"]
            or refined_receipt.get("sha256", {}).get("metainfo.json") != PINNED_METAINFO_SHA256
            or refined_local.get("task_file_size_bytes") != pins["refined_size"]
            or not refined_local.get("task_file_sha256_verified")
            or not refined_local.get("metainfo_sha256_verified")
            or not refined_local_path.is_file()
            or refined_local_path.stat().st_size != pins["refined_size"]):
        raise ValueError(f"{task} refined receipt or local source does not match the pinned release")
    if (raw_receipt.get("repository") != "libero-mem/LIBERO-Mem-Raw"
            or raw_receipt.get("revision") != PINNED_RAW_REVISION
            or raw_receipt.get("sha256", {}).get(raw_filename) != pins["raw_sha256"]
            or raw_local.get("task_file_size_bytes") != pins["raw_size"]
            or not raw_local.get("task_file_sha256_verified")
            or not raw_local_path.is_file()
            or raw_local_path.stat().st_size != pins["raw_size"]):
        raise ValueError(f"{task} raw receipt or local source does not match the pinned release")

    index_path = Path(f"experiments/audits/relation-e1-20261115-{slug}-demo1")
    manifest_path = index_path / "manifest.json"
    manifest = read(manifest_path)
    contract_path = index_path / "index_contract.json"
    contract = read(contract_path)
    if (manifest.get("status") != "complete"
            or manifest.get("artifact") != "causal_prompt_index_not_training_shards"
            or manifest.get("writer") != "rgb_relations_v1"
            or manifest.get("stream") != "predicted_memory"
            or manifest.get("source_kind") != "author_refined"
            or manifest.get("source_revision") != PINNED_REFINED_REVISION):
        raise ValueError(f"{task} relation writer diagnostic is incomplete or has an unexpected scope")
    for name, expected in manifest.get("artifacts_sha256", {}).items():
        artifact = index_path / name
        actual = digest(ROOT / artifact)
        if actual != expected:
            raise ValueError(f"{task} writer index checksum mismatch: {artifact}")
        INPUTS[artifact.as_posix()] = actual
    if (digest(ROOT / contract_path) != manifest.get("index_contract_sha256")
            or contract.get("writer") != manifest.get("writer")
            or contract.get("stream") != manifest.get("stream")
            or contract.get("demo_ids") != [1]
            or contract.get("source_hashes") != manifest.get("sources")):
        raise ValueError(f"{task} relation index contract does not match its run manifest")
    INPUTS[contract_path.as_posix()] = digest(ROOT / contract_path)

    samples_path = index_path / "samples.jsonl"
    trace_path = index_path / "memory_trace.jsonl"
    samples = [json.loads(line) for line in (ROOT / samples_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = [json.loads(line) for line in (ROOT / trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    identity_counts, relation_counts = Counter(), Counter()
    event_count = 0
    for row in traces:
        state = row["snapshot"]
        identity_counts.update(obj.get("identity_status", "unknown") for obj in state.get("objects", {}).values())
        relation_counts.update(fact.get("status", "unknown") for fact in state.get("relations", {}).values())
        event_count += len(row.get("signals", {}).get("new_events", []))
    if (len(samples) != manifest.get("total_samples")
            or len(traces) != manifest.get("total_samples")
            or manifest.get("episodes", [{}])[0].get("split") != "train"
            or manifest.get("episodes", [{}])[0].get("frames") != len(samples)
            or sum(bool(row.get("has_memory")) for row in samples) != manifest.get("memory_frames")
            or manifest.get("sources", {}).get(refined_filename) != pins["refined_sha256"]
            or manifest.get("sources", {}).get("metainfo.json") != PINNED_METAINFO_SHA256):
        raise ValueError(f"{task} relation index rows or source hashes differ from the pinned manifest")
    snapshot[f"{slug}_writer_diagnostic"] = {
        "classification": f"{task}_SINGLE_TRAIN_DEMO_OFFLINE_WRITER_DIAGNOSTIC_NOT_ACCURACY_OR_POLICY_SCORE",
        "refined_source_receipt_path": refined_receipt_path.as_posix(),
        "refined_source_receipt_sha256": INPUTS[refined_receipt_path.as_posix()],
        "raw_source_receipt_path": raw_receipt_path.as_posix(),
        "raw_source_receipt_sha256": INPUTS[raw_receipt_path.as_posix()],
        "source_revision": manifest.get("source_revision"),
        "source_hdf5_sha256": manifest.get("sources", {}).get(refined_filename),
        "raw_source_hdf5_sha256": raw_receipt.get("sha256", {}).get(raw_filename),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": INPUTS[manifest_path.as_posix()],
        "index_contract_path": contract_path.as_posix(),
        "index_contract_sha256": INPUTS[contract_path.as_posix()],
        "episode": manifest.get("episodes"),
        "total_samples": len(samples),
        "samples_with_memory": sum(bool(row.get("has_memory")) for row in samples),
        "samples_without_relevant_fact": sum(row.get("skipped_reason") == "no_relevant_fact" for row in samples),
        "identity_status_observations": dict(identity_counts),
        "relation_status_observations": dict(relation_counts),
        "new_events": event_count,
        "detector_revision": (manifest.get("detector") or {}).get("revision"),
        "independent_identity_or_containment_labels": False,
    }

# Pinned author-action replays exercise task-specific environment setup and
# evaluator semantics without a policy. Keep one development demo per task;
# T8's partial sequence result is retained as an unresolved source/runtime
# discrepancy rather than converted into a pass.
author_replay_script_path = Path("scripts/audit_libero_mem_author_replay.py")
author_replay_script_sha256 = digest(ROOT / author_replay_script_path)
INPUTS[author_replay_script_path.as_posix()] = author_replay_script_sha256
author_replay_tasks = {}
for task in ["T8", "T9", "T10"]:
    slug = task.lower()
    report_path = Path(f"experiments/audits/{slug}-author-replay-upstream-20261115.json")
    report = read(report_path)
    raw_receipt_path = Path(f"resources/manifests/libero_mem_{slug}_raw_receipt_v1.json")
    raw_receipt = read(raw_receipt_path)
    raw_name, raw_sha256 = next(iter(raw_receipt["sha256"].items()))
    if (report.get("classification") != "CUSTOM_AUTHOR_ACTION_REPLAY_NOT_POLICY_SCORE"
            or report.get("task") != task
            or report.get("source_revision") != raw_receipt.get("revision")
            or report.get("source_filename") != raw_name
            or report.get("source_sha256") != raw_sha256
            or report.get("receipt_sha256") != INPUTS[raw_receipt_path.as_posix()]
            or report.get("upstream_revision") != "0eee0defa4024e51511b6e15a79f30e9620209de"
            or report.get("environment_helper") != "upstream scripts/experiments/robot/libero/libero_utils.py:get_libero_env"
            or report.get("evaluator_source") != "upstream scripts/mem_5_run_evaluation_env_gt.py"):
        raise ValueError(f"{task} author replay does not match its source receipt or pinned evaluator")
    result_rows = report.get("results", {})
    if set(result_rows) != {"demo_1"}:
        raise ValueError(f"{task} author replay is not the one-demo development diagnostic")
    result = result_rows["demo_1"]
    if task in ("T9", "T10") and (
            result.get("last_upstream_return") is not True
            or result.get("final_step_done") is not True
            or result.get("overshot") is not False):
        raise ValueError(f"{task} source-action replay did not reach the recorded upstream terminal")
    if task == "T8" and (
            result.get("satisfied_subgoals") != 3
            or result.get("last_upstream_return") is not False
            or result.get("final_step_done") is not False):
        raise ValueError("T8 source-action result differs from the recorded partial sequence replay")
    author_replay_tasks[task] = {
        "classification": report["classification"],
        "report_path": report_path.as_posix(),
        "report_sha256": INPUTS[report_path.as_posix()],
        "raw_source_receipt_path": raw_receipt_path.as_posix(),
        "raw_source_receipt_sha256": INPUTS[raw_receipt_path.as_posix()],
        "raw_source_sha256": report["source_sha256"],
        "upstream_revision": report["upstream_revision"],
        "upstream_source_sha256": report["upstream_source_sha256"],
        "bddl_sha256": report["bddl_sha256"],
        "camera_observations": report["camera_observations"],
        "action_transform": report["action_transform"],
        "result": result,
    }
snapshot["author_action_replay_diagnostics"] = {
    "classification": "CUSTOM_SINGLE_DEMO_RUNTIME_DIAGNOSTIC_NOT_POLICY_SCORE",
    "script_path": author_replay_script_path.as_posix(),
    "script_sha256_at_collection": author_replay_script_sha256,
    "tasks": author_replay_tasks,
    "limitations": [
        "One author-source demonstration per task; no policy was run.",
        "T8 reached three recorded sequence subgoals but did not report final upstream success; investigate the discrepancy before using it as a gate pass.",
        "T9 and T10 each reported final upstream success on demo_1; this does not measure perception, memory, or policy competence.",
    ],
}

# T8 refined metadata and upstream-helper replay share the exact same initial
# state and transformed author actions. Compare the live robot-state trace so
# the retained terminal discrepancy is localized without claiming its cause.
t8_raw_receipt_path = Path("resources/manifests/libero_mem_t8_raw_receipt_v1.json")
t8_refined_receipt_path = Path("resources/manifests/libero_mem_t8_refined_receipt_v1.json")
t8_raw_receipt = read(t8_raw_receipt_path)
t8_refined_receipt = read(t8_refined_receipt_path)
t8_parity_script_path = Path("scripts/audit_libero_mem_replay_state_parity.py")
t8_parity_script_sha256 = digest(ROOT / t8_parity_script_path)
INPUTS[t8_parity_script_path.as_posix()] = t8_parity_script_sha256
t8_parity_path = Path("experiments/audits/t8-replay-state-parity-20261115.json")
t8_parity = read(t8_parity_path)
t8_first_divergence = t8_parity.get("first_state_divergence", {})
if (
    t8_parity.get("classification")
    != "CUSTOM_AUTHOR_ACTION_STATE_PARITY_DIAGNOSTIC_NOT_POLICY_SCORE"
    or t8_parity.get("task") != "T8"
    or t8_parity.get("demo") != "demo_1"
    or t8_parity.get("raw_receipt_sha256") != INPUTS[t8_raw_receipt_path.as_posix()]
    or t8_parity.get("refined_receipt_sha256") != INPUTS[t8_refined_receipt_path.as_posix()]
    or t8_parity.get("raw_source_sha256") != next(iter(t8_raw_receipt["sha256"].values()))
    or t8_parity.get("refined_source_sha256") != next(iter(t8_refined_receipt["sha256"].values()))
    or t8_parity.get("parity_script_sha256") != t8_parity_script_sha256
    or t8_parity.get("author_replay_code_sha256") != author_replay_script_sha256
    or t8_parity.get("actions_exact_match") is not True
    or t8_parity.get("initial_state_exact_match") is not True
    or t8_parity.get("refined_metadata_success") is not True
    or t8_parity.get("transformed_action_rows") != 613
    or t8_parity.get("replay_robot_state_rows") != 613
    or t8_first_divergence.get("step") != 219
    or t8_parity.get("rows_within_tolerance", 613) >= 613
    or len(t8_parity.get("milestones", [])) != 3
    or t8_parity.get("last_upstream_return") is not False
    or t8_parity.get("final_step_done") is not False
):
    raise ValueError("T8 state-parity report does not match the pinned author replay evidence")
snapshot["author_action_replay_diagnostics"]["tasks"]["T8"]["state_parity"] = {
    "classification": t8_parity["classification"],
    "report_path": t8_parity_path.as_posix(),
    "report_sha256": INPUTS[t8_parity_path.as_posix()],
    "script_path": t8_parity_script_path.as_posix(),
    "script_sha256": t8_parity_script_sha256,
    "raw_refined_actions_exact_match": t8_parity["actions_exact_match"],
    "raw_refined_initial_state_exact_match": t8_parity["initial_state_exact_match"],
    "rows_within_tolerance": t8_parity["rows_within_tolerance"],
    "state_row_count": t8_parity["replay_robot_state_rows"],
    "parity_tolerance": t8_parity["parity_tolerance"],
    "first_state_divergence": t8_first_divergence,
    "max_abs_robot_state_error": t8_parity["max_abs_robot_state_error"],
    "milestones": t8_parity["milestones"],
    "last_upstream_return": t8_parity["last_upstream_return"],
    "final_step_done": t8_parity["final_step_done"],
    "limitation": t8_parity["limitation"],
}
snapshot["author_action_replay_diagnostics"]["limitations"][1] = (
    "T8 raw/refined demo_1 actions and initial state match exactly; a pinned-runtime replay "
    "first exceeds the diagnostic 1e-5 robot-state tolerance at step 219 and still does not "
    "report terminal success. The cause remains unresolved; this is not a gate pass."
)

# The existing V2 F0 artifact and its latest B0/F0 policy runs are separate
# from the historical B0/F1 pilot above. Verify each run and retain its own
# hashes and scope; never pool the four development episodes with the historic
# 20-episode-per-condition result.
f0_adapter_path = Path("experiments/training/results_f0_author_split_v2/output_f0/pretrained_model/mem_adapter_manifest.json")
f0_adapter = read(f0_adapter_path)
image_parity_path = Path("experiments/audits/smolvla-image-preprocessing-parity-20261115/report.json")
image_parity = read(image_parity_path)
image_dataset_path = Path("experiments/training/results_f0_author_split_v2/dataset-manifest.json")
image_dataset = read(image_dataset_path)
if (image_parity.get("finding", {}).get("result") !=
        "NO_IMAGE_ORIENTATION_MISMATCH_FOUND_AT_THE_DECLARED_TRANSFORM_LEVEL"):
    raise ValueError("SmolVLA image preprocessing audit has an unexpected finding")
if image_parity["training"]["adapter_manifest_sha256"] != INPUTS[f0_adapter_path.as_posix()]:
    raise ValueError("Image preprocessing audit refers to a different F0 artifact")
if image_parity["training"]["dataset_manifest_sha256"] != INPUTS[image_dataset_path.as_posix()]:
    raise ValueError("Image preprocessing audit refers to a different training dataset manifest")
if image_dataset["image_contract"] != image_parity["training"]["image_contract"]:
    raise ValueError("Image preprocessing report disagrees with the recorded dataset contract")
if image_parity["training"]["lerobot_version_recorded"] != f0_adapter["run"]["versions"]["lerobot"]:
    raise ValueError("Image preprocessing report disagrees with the F0 LeRobot version")
if (image_parity["evaluation"]["lerobot_version_installed"] !=
        image_parity["training"]["lerobot_version_recorded"]):
    raise ValueError("Training/evaluation LeRobot versions differ in the image audit")
if not image_parity["training"]["training_collator_matches_f0_implementation_record"]:
    raise ValueError("F0 manifest does not attest the image-audit training collator")
if (image_parity["training"]["training_collator_sha256"] !=
        f0_adapter["run"]["implementation_sha256"][image_parity["training"]["training_collator_source"]]):
    raise ValueError("Image audit training collator differs from the F0 implementation record")
for relative, details in image_parity["training"]["data_preparation_sources"].items():
    if digest(ROOT / relative) != details["current_sha256"] or not details["matches"]:
        raise ValueError(f"Image preprocessing source changed since its audit: {relative}")
if digest(ROOT / image_parity["training"]["training_collator_source"]) != image_parity["training"]["training_collator_sha256"]:
    raise ValueError("Image preprocessing training collator source changed")
if digest(ROOT / image_parity["evaluation"]["simulator_source"]) != image_parity["evaluation"]["simulator_source_sha256"]:
    raise ValueError("Image preprocessing simulator source changed")
if digest(ROOT / image_parity["evaluation"]["adapter_source"]) != image_parity["evaluation"]["adapter_source_sha256"]:
    raise ValueError("Image preprocessing evaluation adapter source changed")
snapshot["smolvla_image_preprocessing_parity"] = {
    "classification": image_parity["classification"],
    "report_path": image_parity_path.as_posix(),
    "report_sha256": INPUTS[image_parity_path.as_posix()],
    "result": image_parity["finding"]["result"],
    "training_collator_matches_f0_implementation_record": image_parity["training"]["training_collator_matches_f0_implementation_record"],
    "data_preparation_sources_match": all(row["matches"] for row in image_parity["training"]["data_preparation_sources"].values()),
    "training_lerobot_version": image_parity["training"]["lerobot_version_recorded"],
    "evaluation_lerobot_version": image_parity["evaluation"]["lerobot_version_installed"],
    "evaluation_processor_class_source_sha256": image_parity["evaluation"]["processor_class_source_sha256"],
    "training_processor_source_sha256_recorded": False,
    "limitations": image_parity["finding"]["limits"],
}

minivla_path = Path("experiments/audits/minivla-contract-audit-20261115/report.json")
minivla_audit = read(minivla_path)
if minivla_audit.get("evaluation_status") != "BLOCKED_OFFICIAL_CONTRACT_AUDIT":
    raise ValueError("MiniVLA must remain blocked until official runtime parity is verified")
if minivla_audit.get("model_inference") is not False or minivla_audit.get("policy_rollouts") != 0:
    raise ValueError("MiniVLA contract audit must not contain model inference or policy rollouts")
snapshot["minivla_candidate_gate"] = {
    "classification": minivla_audit["classification"],
    "report_path": minivla_path.as_posix(),
    "report_sha256": INPUTS[minivla_path.as_posix()],
    "evaluation_status": minivla_audit["evaluation_status"],
    "checkpoint_candidates_verified": len(minivla_audit["checkpoint"]["candidates"]),
    "checkpoint_selection_status": minivla_audit["checkpoint"]["selection_status"],
    "official_source_revision": minivla_audit["official_source"]["revision"],
    "fail_closed_gate_present": minivla_audit["local_adapter"]["fail_closed_gate_present"],
    "model_inference": minivla_audit["model_inference"],
    "policy_rollouts": minivla_audit["policy_rollouts"],
    "finding": minivla_audit["finding"],
}
freeze_path = Path("resources/manifests/pip_freeze_smolvla.txt")
INPUTS[freeze_path.as_posix()] = digest(ROOT / freeze_path)
software_manifest_path = Path("resources/manifests/software_manifest.yaml")
INPUTS[software_manifest_path.as_posix()] = digest(ROOT / software_manifest_path)

runtime_root = Path("experiments/evaluation/e0-f0-runtime-ext4-20261115")
runtime_manifest_path = runtime_root / "run_manifest.json"
runtime_manifest = read(runtime_manifest_path)
if runtime_manifest.get("status") != "runtime_checked" or runtime_manifest.get("completed_runtime_checks") != 4:
    raise ValueError("Expected four completed B0/F0 V2 runtime checks")
runtime_checks = []
runtime_episode_root = ROOT / runtime_root / "episodes"
for folder in sorted(runtime_episode_root.iterdir()):
    if not folder.is_dir():
        continue
    relative_status = runtime_root / "episodes" / folder.name / "status.json"
    status = read(relative_status)
    if status.get("status") != "diagnostic_complete" or status.get("error"):
        raise ValueError(f"Runtime diagnostic did not complete cleanly: {folder.name}")
    for name, expected in status.get("files_sha256", {}).items():
        relative = runtime_root / "episodes" / folder.name / name
        actual = digest(ROOT / relative)
        INPUTS[relative.as_posix()] = actual
        if actual != expected:
            raise ValueError(f"Runtime diagnostic checksum mismatch: {relative}")
    action_path = folder / "actions.jsonl"
    action_count = sum(1 for line in action_path.open(encoding="utf-8") if line.strip())
    if action_count != 2:
        raise ValueError(f"Expected exactly two diagnostic actions: {folder.name}")
    runtime_checks.append({
        "episode_id": status.get("episode_id"), "task": status.get("task"),
        "condition": status.get("condition"), "state_idx": status.get("state_idx"),
        "status": status.get("status"), "actions": action_count,
        "initial_state_sha256": status.get("initial_state_sha256"),
    })
runtime_report = runtime_root / "report.md"
INPUTS[runtime_report.as_posix()] = digest(ROOT / runtime_report)
if len(runtime_checks) != 4:
    raise ValueError(f"Expected four runtime diagnostic episodes, got {len(runtime_checks)}")
snapshot["f0_v2_artifact"] = {
    "classification": "EXISTING_NO_MEMORY_ADAPTER_ARTIFACT",
    "manifest_path": f0_adapter_path.as_posix(),
    "manifest_sha256": INPUTS[f0_adapter_path.as_posix()],
    "condition": f0_adapter.get("run", {}).get("arguments", {}).get("condition"),
    "training_steps": f0_adapter.get("run", {}).get("arguments", {}).get("steps"),
    "data_manifest_sha256": f0_adapter.get("run", {}).get("data_manifest_sha256"),
    "weight_identity": runtime_manifest.get("treatments", {}).get("F0", {}).get("weight_identity"),
    "adapter_manifest_sha256": runtime_manifest.get("treatments", {}).get("F0", {}).get("adapter_manifest_sha256"),
    "training_versions": f0_adapter.get("run", {}).get("versions"),
    "note": "Pre-existing artifact; no model training was performed for this update.",
}

f1_adapter_path = Path("experiments/training/results_f1_author_split_v2/output_f1/pretrained_model/mem_adapter_manifest.json")
if f1_adapter_path.is_file():
    f1_adapter = read(f1_adapter_path)
    INPUTS[f1_adapter_path.as_posix()] = digest(ROOT / f1_adapter_path)
    from src.training.mem_adapter_artifact import validate_matched_adapters
    validate_matched_adapters(f0_adapter, f1_adapter)
    f1_runtime_path = Path("experiments/evaluation/review-f1-runtime-retry-20261115/run_manifest.json")
    f1_runtime = read(f1_runtime_path) if f1_runtime_path.is_file() else {}
    if f1_runtime_path.is_file():
        INPUTS[f1_runtime_path.as_posix()] = digest(ROOT / f1_runtime_path)
    snapshot["f1_v2_artifact"] = {
        "classification": "TRAINED_FULL_MEMORY_ADAPTER_ARTIFACT",
        "manifest_path": f1_adapter_path.as_posix(),
        "manifest_sha256": INPUTS[f1_adapter_path.as_posix()],
        "condition": f1_adapter.get("run", {}).get("arguments", {}).get("condition"),
        "training_steps": f1_adapter.get("run", {}).get("arguments", {}).get("steps"),
        "data_manifest_sha256": f1_adapter.get("run", {}).get("data_manifest_sha256"),
        "writer": f1_adapter.get("memory_provenance", [{}])[0].get("writer"),
        "weight_identity": f1_runtime.get("treatments", {}).get("F1", {}).get("weight_identity"),
        "adapter_manifest_sha256": f1_runtime.get("treatments", {}).get("F1", {}).get("adapter_manifest_sha256"),
        "training_versions": f1_adapter.get("run", {}).get("versions"),
        "validate_adapter": "PASS",
    }
    snapshot["matched_f0_f1_pair"] = {
        "status": "VALIDATED_MATCHED_PAIR",
        "seed": f0_adapter.get("run", {}).get("arguments", {}).get("seed"),
        "steps": f0_adapter.get("run", {}).get("arguments", {}).get("steps"),
        "effective_batch_size": f0_adapter.get("run", {}).get("effective_batch_size"),
        "data_manifest_sha256": f0_adapter.get("run", {}).get("data_manifest_sha256"),
        "f0_condition": "no_memory",
        "f1_condition": "memory",
        "f1_writer": f1_adapter.get("memory_provenance", [{}])[0].get("writer"),
    }

review_receipt_path = Path("docs/thesis/evidence/f0_f1_readiness_review_2026-11-15.json")
if review_receipt_path.is_file():
    INPUTS[review_receipt_path.as_posix()] = digest(ROOT / review_receipt_path)
    snapshot["f0_f1_readiness_review"] = read(review_receipt_path)
policy_environment = runtime_manifest.get("environments", {}).get("policy", {})
snapshot["smolvla_inference_environment"] = {
    "spec": "envs/smolvla.yml",
    "requirements": "envs/smolvla_inference.txt",
    "package_freeze": freeze_path.as_posix(),
    "package_freeze_sha256": INPUTS[freeze_path.as_posix()],
    "software_manifest_sha256": INPUTS[software_manifest_path.as_posix()],
    "observed": policy_environment,
    "cuda": runtime_manifest.get("cuda"),
    "source": "runtime-check manifest; package presence is not treated as policy-load success by itself.",
}
snapshot["v2_runtime_check"] = {
    "classification": "DIAGNOSTIC_ONLY_NO_SCORE",
    "manifest_path": runtime_manifest_path.as_posix(),
    "manifest_sha256": INPUTS[runtime_manifest_path.as_posix()],
    "status": runtime_manifest.get("status"),
    "completed_checks": runtime_manifest.get("completed_runtime_checks"),
    "completed_episodes": runtime_manifest.get("completed_episodes"),
    "actions_total": sum(row["actions"] for row in runtime_checks),
    "episodes": runtime_checks,
    "report_path": runtime_report.as_posix(),
    "report_sha256": INPUTS[runtime_report.as_posix()],
    "simulator_environment": runtime_manifest.get("environments", {}).get("simulator"),
    "policy_environment": policy_environment,
    "cuda": runtime_manifest.get("cuda"),
}

pilot_root = Path("experiments/evaluation/e0-b0-f0-pilot-20261115")
pilot_manifest_path = pilot_root / "run_manifest.json"
pilot_manifest = read(pilot_manifest_path)
if pilot_manifest.get("status") != "complete" or pilot_manifest.get("completed_episodes") != 4:
    raise ValueError("Expected the four-episode E0 B0/F0 development pilot to complete")
expected_pilot = {("T1", "B0"), ("T1", "F0"), ("T3", "B0"), ("T3", "F0")}
pilot_rows = []
pilot_episode_root = ROOT / pilot_root / "episodes"
for folder in sorted(pilot_episode_root.iterdir()):
    if not folder.is_dir():
        continue
    episode_relative = pilot_root / "episodes" / folder.name
    status = read(episode_relative / "status.json")
    if status.get("status") != "complete" or status.get("error") or status.get("logging_errors"):
        raise ValueError(f"Pilot episode did not complete cleanly: {folder.name}")
    summary_relative = episode_relative / "summary.json"
    summary = read(summary_relative)
    if digest(ROOT / summary_relative) != status.get("files_sha256", {}).get("summary.json"):
        raise ValueError(f"Pilot summary checksum mismatch: {folder.name}")
    for name, expected_hash in status.get("files_sha256", {}).items():
        relative = episode_relative / name
        actual_hash = digest(ROOT / relative)
        INPUTS[relative.as_posix()] = actual_hash
        if actual_hash != expected_hash:
            raise ValueError(f"Pilot artifact checksum mismatch: {relative}")
    result = summary["result"]
    actions_relative = episode_relative / "actions.jsonl"
    action_count = sum(1 for line in (ROOT / actions_relative).open(encoding="utf-8") if line.strip())
    if action_count != result.get("total_steps"):
        raise ValueError(f"Pilot action count does not match summary: {folder.name}")
    row = {
        "episode_id": result.get("episode_id"), "task": summary.get("task"),
        "condition": summary.get("condition"), "state_idx": summary.get("state_idx"),
        "source_demo": summary.get("source_demo"), "policy_seed": summary.get("policy_seed"),
        "initial_state_sha256": summary.get("initial_state_sha256"),
        "status": status.get("status"), "termination_reason": result.get("termination_reason"),
        "success": result.get("success"), "scr": result.get("scr"),
        "completed_subgoals": result.get("completed_subgoals"), "total_subgoals": result.get("total_subgoals"),
        "confirmed_cycles": result.get("confirmed_cycles"), "target_cycles": result.get("target_cycles"),
        "overshot": result.get("overshot"), "policy_actions": action_count,
        "mean_policy_inference_ms_per_query": result.get("avg_latency_ms"),
        "total_episode_wall_time_s": result.get("total_wall_time_s"),
        "decision_metrics_present": result.get("decision_metrics") is not None,
        "summary_path": summary_relative.as_posix(), "summary_sha256": INPUTS[summary_relative.as_posix()],
    }
    pilot_rows.append(row)
if {(row["task"], row["condition"]) for row in pilot_rows} != expected_pilot or len(pilot_rows) != 4:
    raise ValueError("E0 pilot roster mismatch")
pilot_report = pilot_root / "report.md"
INPUTS[pilot_report.as_posix()] = digest(ROOT / pilot_report)
if "NON-COMPARABLE_OFFICIAL_PAPER" not in (ROOT / pilot_report).read_text(encoding="utf-8"):
    raise ValueError("Pilot report is missing the custom/non-comparable classification")
pilot_counts = []
for task, condition in sorted(expected_pilot):
    rows = [row for row in pilot_rows if row["task"] == task and row["condition"] == condition]
    pilot_counts.append({
        "task": task, "condition": condition, "n": len(rows),
        "successes": sum(row["success"] is True for row in rows),
        "sr": sum(row["success"] is True for row in rows) / len(rows),
        "mean_scr": sum(float(row["scr"]) for row in rows) / len(rows),
        "overshot_count": sum(row["overshot"] is True for row in rows),
        "unique_state_hashes": len({row["initial_state_sha256"] for row in rows}),
    })
snapshot["e0_v2_pilot"] = {
    "classification": pilot_manifest.get("classification"),
    "comparability": pilot_manifest.get("comparability"),
    "manifest_path": pilot_manifest_path.as_posix(),
    "manifest_sha256": INPUTS[pilot_manifest_path.as_posix()],
    "status": pilot_manifest.get("status"), "phase": pilot_manifest.get("phase"),
    "expected_episodes": pilot_manifest.get("expected_episodes"),
    "completed_episodes": pilot_manifest.get("completed_episodes"),
    "split_lock_sha256": pilot_manifest.get("split_lock_sha256"),
    "protocol_sha256": pilot_manifest.get("protocol_sha256"),
    "source_revision": pilot_manifest.get("raw_revision"),
    "policy_environment": pilot_manifest.get("environments", {}).get("policy"),
    "simulator_environment": pilot_manifest.get("environments", {}).get("simulator"),
    "cuda": pilot_manifest.get("cuda"),
    "episodes": pilot_rows, "counts": pilot_counts,
    "report_path": pilot_report.as_posix(), "report_sha256": INPUTS[pilot_report.as_posix()],
    "interpretation": "DEVELOPMENT_ONLY; one seen state per task and condition; not a memory-effect estimate or official LIBERO score.",
}

for folder in ["docs", "src", "configs", "envs", "resources/manifests", "scripts"]:
    for p in (ROOT / folder).rglob("*"):
        if not p.is_file() or "thesis" in p.relative_to(ROOT).parts or "__pycache__" in p.parts:
            continue
        if p.suffix in [".md", ".py", ".json", ".yaml", ".yml", ".tex", ".bib", ".pdf"]:
            INPUTS[p.relative_to(ROOT).as_posix()] = digest(p)
save("snapshot.json", snapshot)
save("input_hashes.json", INPUTS)
print(json.dumps({"snapshot": str(OUT / "snapshot.json"), "historical_jsons_read": snapshot["historical_episode_jsons_read"],
                  "mem_test": snapshot["mem_test"]["counts"], "training_step": progress["step"], "adapter_files_verified": len(adapter_files),
                  "dataset_episode_counts": snapshot["training"]["dataset_episode_counts"],
                  "v2_runtime_checks": snapshot["v2_runtime_check"]["completed_checks"],
                  "e0_v2_pilot_episodes": snapshot["e0_v2_pilot"]["completed_episodes"]}, ensure_ascii=False, indent=2))
