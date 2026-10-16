"""Read-only analysis of the first 15 completed compact-memory episodes.

No model execution, simulator execution, benchmark writes, or synthetic data.
The output is a diagnostic audit, not an official LIBERO score.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.interfaces.text_memory import TextMemoryRenderer
from src.memory.models import EvidenceSource, MemoryEvent, MemoryQuery, MemoryStatus
from src.memory.observation_writer import ObservationMemoryWriter
from src.memory.retriever import DeterministicMemoryRetriever
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater

OUT = Path(__file__).parent
RUNS = {
    "compact": ROOT / "experiments/results/v2_memory/prompt_shoren_obs_text",
    "legacy": ROOT / "experiments/results/v2_memory/pilot_obs_text",
    "off_wsl": ROOT / "experiments/results/libero-10-wsl",
}
KEYS = [f"libero_10_task{t}/init_{i}" for t, n in [(0, 10), (1, 5)] for i in range(n)]
TOK_REV = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TOK_PATH = Path.home() / (".cache/huggingface/hub/models--HuggingFaceTB--"
                         f"SmolVLM2-500M-Video-Instruct/snapshots/{TOK_REV}/tokenizer.json")
backend = Tokenizer.from_file(str(TOK_PATH))
backend.no_padding()
backend.no_truncation()
inventory = {}


def read(path):
    data = path.read_bytes()
    inventory[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8")


def js(path):
    return json.loads(read(path))


def jl(path):
    return [json.loads(x) for x in read(path).splitlines() if x.strip()]


class TokenCounter:
    def encode(self, text, add_special_tokens=True):
        return backend.encode(text, add_special_tokens=add_special_tokens).ids


def count(text):
    return len(backend.encode(text if text.endswith("\n") else text + "\n").ids)


result = {
    "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
    "scope": "Fixed first 15 episodes: task0 init0..9; task1 init0..4. Live run may advance.",
    "classification": "LIBERO-derived diagnostic audit; NON-COMPARABLE_OFFICIAL_PAPER",
    "tokenizer_revision": TOK_REV,
    "tokenizer_sha256": hashlib.sha256(TOK_PATH.read_bytes()).hexdigest(),
    "runs": {}, "pairs": {}, "episodes": [],
}
episodes = {}
for run, directory in RUNS.items():
    episodes[run] = {k: js(directory / k / "episode.json") for k in KEYS
                     if (directory / k / "episode.json").exists()}
    es = episodes[run]
    result["runs"][run] = {
        "count": len(es), "successes": sum(x["success"] for x in es.values()),
        "by_task": {f"task{t}": {"count": sum(k.startswith(f"libero_10_task{t}/") for k in es),
                    "successes": sum(e["success"] for k, e in es.items() if k.startswith(f"libero_10_task{t}/"))}
                    for t in range(2)},
    }

for other in ["legacy", "off_wsl"]:
    common = [k for k in KEYS if k in episodes[other]]
    pair = {"keys": common, "compact_success": 0, "other_success": 0,
            "mismatches": [], "initial_action_comparison": []}
    for k in common:
        a, b = episodes["compact"][k], episodes[other][k]
        pair["compact_success"] += int(a["success"])
        pair["other_success"] += int(b["success"])
        for field in ["seed", "instruction", "initial_state_id", "execution_horizon_s", "checkpoint", "settling_snapshot"]:
            if a.get(field) != b.get(field):
                pair["mismatches"].append([k, field])
        for field in ["python_version", "robosuite_version", "model_config_sha256", "num_steps_wait", "policy_interface"]:
            if a["provenance"].get(field) != b["provenance"].get(field):
                pair["mismatches"].append([k, "provenance." + field])
        ma = jl(RUNS["compact"] / k / "model_output.jsonl")
        mb = jl(RUNS[other] / k / "model_output.jsonl")
        aa, ab = np.array(ma[0]["action"]), np.array(mb[0]["action"])
        pair["initial_action_comparison"].append({
            "key": k, "compact": aa.tolist(), "other": ab.tolist(),
            "xyz_l2_delta": float(np.linalg.norm(aa[:3] - ab[:3])),
            "full_l2_delta": float(np.linalg.norm(aa - ab)),
            "first10_action_rmse": float(np.sqrt(np.mean((np.array([x["action"] for x in ma[:10]])
                                                       - np.array([x["action"] for x in mb[:10]])) ** 2))),
        })
    result["pairs"][other] = pair

totals = Counter()
fact_counts = Counter()
for key in KEYS:
    e = episodes["compact"][key]
    traces = jl(RUNS["compact"] / key / "memory_trace.jsonl")
    outputs = jl(RUNS["compact"] / key / "model_output.jsonl")
    events = e["memory"]["snapshot"]["events"]
    ep_id = e["memory"]["snapshot"]["episode_id"]
    writer_store = EpisodeMemoryStore(ep_id)
    writer = ObservationMemoryWriter(writer_store, MemoryUpdater(writer_store),
        target_entity_names=tuple(x for x in e["memory"]["task_entities"] if x != "basket"),
        goal_container_name="basket")
    initial = e["settling_snapshot"]["post_settling"]
    writer.on_episode_start(ep_id, None, {"robot0_eef_pos": initial["eef_pos"],
                                         "robot0_gripper_qpos": initial["gripper_qpos"]})
    for out in outputs:
        writer.on_step(out["step"], (out["step"] + 1) * .05, np.array(out["action"]),
                       {"robot0_eef_pos": out["eef_pos_actual"], "robot0_gripper_qpos": out["gripper_qpos_actual"]})
    writer_replayed = writer_store.to_dict(e["num_steps"])["events"]
    totals["writer_episode_replay_mismatches"] += int(writer_replayed != events)
    store = EpisodeMemoryStore(ep_id)
    updater = MemoryUpdater(store)
    retriever = DeterministicMemoryRetriever()
    renderer = TextMemoryRenderer(max_total_tokens=48, compact_mode=True,
                                  tokenizer=TokenCounter(), strict_tokenizer=True)
    # Numeric event suffix is the creation order; the store serializes equal-step events lexically.
    events = sorted(events, key=lambda ev: (ev["step"], ev["timestamp"],
                                            int(re.search(r"_(\d+)_(?:goal_)?obs_", ev["event_id"]).group(1))))
    cursor = 0
    rec = {"key": key, "success": e["success"], "steps": e["num_steps"],
           "initial_prompt": traces[0]["policy_prompt"], "initial_tokens": count(traces[0]["policy_prompt"]),
           "instruction_tokens": count(e["instruction"]), "event_counts": dict(Counter(ev["event_type"] for ev in events)),
           "manipulated_object_ids": sorted({ev["object_id"] for ev in events if ev["event_type"] != "initial_observation"}),
           "diagnostics": e["diagnostics"], "event_telemetry": [], "obsolete_recovery_examples": [],
           "holding_telemetry_examples": [], "unique_prompts": [], "replay_mismatches": []}
    rec["writer_replay_matches"] = writer_replayed == events
    last_prompt = None
    for tr in traces:
        step = tr["step"]
        while cursor < len(events) and (events[cursor]["event_type"] == "initial_observation" or events[cursor]["step"] < step):
            d = dict(events[cursor])
            d["evidence_source"] = EvidenceSource(d["evidence_source"])
            d["validity"] = MemoryStatus(d["validity"])
            d["evidence_ids"] = tuple(d["evidence_ids"])
            updater.record(MemoryEvent(**d))
            cursor += 1
        retrieved = retriever.retrieve(store.snapshot(step), MemoryQuery(
            episode_id=ep_id, as_of_step=step, entity_ids=tuple(e["memory"]["task_entities"])))
        rr = renderer.render_with_audit(e["instruction"], retrieved)
        if rr.text != tr["policy_prompt"]:
            rec["replay_mismatches"].append({"step": step, "replayed": rr.text, "logged": tr["policy_prompt"]})
        totals["decisions"] += 1
        nt = count(tr["policy_prompt"])
        totals["token_count_mismatch"] += int(nt != tr["total_tokens_with_instruction"])
        totals["actual_over_48"] += int(nt > 48)
        totals["token_budget_fact_drops"] += int(tr["token_truncated"])
        totals["prompt_unchanged"] += int(tr["policy_prompt"] == e["instruction"])
        totals["basket_only"] += int(tr["facts_included"] == ["Basket is target destination."])
        fact_counts.update(tr["facts_included"])
        if tr["policy_prompt"] != last_prompt:
            rec["unique_prompts"].append({"step": step, "prompt": tr["policy_prompt"]})
            last_prompt = tr["policy_prompt"]
        if "Holding " in tr["policy_prompt"] and step:
            diag = outputs[step - 1]["contact_diagnostics"]
            totals["holding_decisions"] += 1
            if not diag["both_fingers_contact"] and diag["object_delta_z"] < .005:
                totals["holding_without_contact_or_lift"] += 1
                if len(rec["holding_telemetry_examples"]) < 3:
                    rec["holding_telemetry_examples"].append({"decision_step": step, "prompt": tr["policy_prompt"],
                                                              "preceding_telemetry": diag})
        if "Lost contact with" in tr["policy_prompt"]:
            totals["recovery_decisions"] += 1
            for ev in sorted(retrieved.events, key=lambda x: x.step, reverse=True):
                if ev.event_type != "observed_slip":
                    continue
                obj = store.get_object(ev.object_id)
                if obj and obj.last_updated_step > ev.step and obj.state == "released":
                    totals["obsolete_slip_after_newer_release"] += 1
                    if len(rec["obsolete_recovery_examples"]) < 3:
                        rec["obsolete_recovery_examples"].append({"decision_step": step, "slip_step": ev.step,
                            "newer_release_step": obj.last_updated_step, "prompt": tr["policy_prompt"]})
                break
    for ev in events:
        if ev["event_type"] == "initial_observation":
            continue
        out = outputs[ev["step"]]
        diag = out["contact_diagnostics"]
        row = {"step": ev["step"], "event": ev["event_type"], "object_id": ev["object_id"],
               "gap": sum(abs(x) for x in out["gripper_qpos_actual"]), "diag": diag}
        rec["event_telemetry"].append(row)
        if ev["event_type"] == "observed_grasp":
            totals["grasp_events"] += 1
            totals["grasp_without_both_contact"] += int(not diag["both_fingers_contact"])
            totals["grasp_target_distance_over_10cm"] += int(diag["dist_eef_to_object"] > .10)
        if ev["event_type"] == "observed_lift":
            totals["lift_events"] += 1
            totals["lift_target_dz_under_5mm"] += int(diag["object_delta_z"] < .005)
    totals["replay_mismatches"] += len(rec["replay_mismatches"])
    result["episodes"].append(rec)

result["totals"] = dict(totals)
result["fact_frequencies"] = dict(fact_counts.most_common())
for rel in ["src/interfaces/text_memory.py", "src/memory/observation_writer.py", "src/memory/retriever.py",
            "src/memory/updater.py", "src/memory/models.py", "src/evaluation/rollout.py",
            "src/evaluation/diagnostics.py", "src/models/smolvla/adapter.py", "scripts/run_benchmark.py",
            "resources/checkpoints/smolvla_libero/config.json", "resources/checkpoints/smolvla_libero/policy_preprocessor.json",
            "resources/manifests/models/smolvla_libero.yaml"]:
    read(ROOT / rel)
for d in RUNS.values():
    patch = d / "git_diff.patch"
    if patch.exists():
        read(patch)
    else:
        result.setdefault("missing_provenance_files", []).append(str(patch))
result["input_sha256"] = inventory
(OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ["scope", "runs", "totals", "fact_frequencies"]}, indent=2))
for other, pair in result["pairs"].items():
    print(other, "paired", len(pair["keys"]), pair["compact_success"], pair["other_success"], "mismatches", pair["mismatches"])
