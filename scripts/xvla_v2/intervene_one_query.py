#!/usr/bin/env python
"""Replay one X-VLA F1 memory query with a single fact removed.

The source is the completed X-VLA E2 T1 unseen-state-10 rollout. This is a
single-trajectory custom prompt counterfactual, not an efficacy estimate.
The source writer snapshot is installed before importing writer components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--writer-source-root", type=Path,
                           default=ROOT / "experiments/runtime/xvla_e2_memory_source_20261121")
    boot, _ = bootstrap.parse_known_args()
    from scripts.xvla_v2.memory_source import install
    install(boot.writer_source_root,
            ROOT / "experiments/training/relations-author-split-v2/shards/manifest.json")

import numpy as np

from scripts.evaluate_mem_adapter import code_identity, detector_launcher
from scripts.run_x_vla_libero import start_server, verify_adapter
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest, load_suite_source
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.evaluation.mem_eval_io import EpisodeRecorder, evaluation_roster, write_json
from src.memory.visual_pipeline import load_prompt_tokenizer
from src.memory.relation_pipeline import relation_pipeline
from src.models.xvla_mem import XVLAMemPolicy
from src.perception.relation_detector import ProcessRelationDetector
from src.simulator.xvla_mem_env import ProcessXVLAMemEnv
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file
from src.training.mem_adapter_artifact import verify_files


SOURCE = ROOT / "experiments/evaluation/xvla_e2_f1_mem_on_20261122/episodes/T1_F1_mem-on_unseen_state10"
DEFAULT_OUT = ROOT / "experiments/evaluation/xvla_e4_e5_one_query_20261123"
TARGET_STEP = 10
SOURCE_FACT = "bowl: 0 observed returns."


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def rgb_hash(obs):
    return {key: hashlib.sha256(np.ascontiguousarray(obs[key]).tobytes()).hexdigest()
            for key in ("agentview_image", "robot0_eye_in_hand_image")}


def verify_source(roster_row, adapter_dir):
    status = json.loads((SOURCE / "status.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8-sig"))
    if status.get("status") != "complete":
        raise ValueError("X-VLA E2 source episode is incomplete")
    for filename, digest in status["files_sha256"].items():
        if sha256_file(SOURCE / filename) != digest:
            raise ValueError("X-VLA E2 source artifact hash mismatch: " + filename)
    run_manifest_path = SOURCE.parent.parent / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8-sig"))
    arguments = run_manifest["arguments"]
    if run_manifest.get("status") != "complete" or arguments.get("arm") != "F1" or arguments.get("memory") != "on":
        raise ValueError("Source run is not a completed F1 memory-on E2 run")
    if (summary.get("task") != "T1" or summary.get("state_idx") != 10
            or summary.get("split") != "unseen" or summary.get("policy_seed") != 42
            or summary.get("initial_state_sha256") != roster_row["initial_state_sha256"]):
        raise ValueError("Source rollout identity differs from locked task/state/seed")
    if sha256_file(adapter_dir / "adapter_model.safetensors") != run_manifest["adapter_export"]["files"]["adapter_model.safetensors"]:
        raise ValueError("Intervention adapter weights differ from source rollout")
    queries = read_jsonl(SOURCE / "queries.jsonl")
    observations = read_jsonl(SOURCE / "observations.jsonl")
    actions = read_jsonl(SOURCE / "actions.jsonl")
    target = next((q for q in queries if q["step"] == TARGET_STEP), None)
    if target is None or target["render"].get("facts_included") != [SOURCE_FACT]:
        raise ValueError("Locked intervention query does not contain exactly the expected fact")
    if target["prompt"] != f'{roster_row["instruction"]} Memory: {SOURCE_FACT}':
        raise ValueError("Target prompt differs from the recorded one-fact prompt")
    return run_manifest, status, summary, queries, observations, actions, target


class OneQueryIntervention:
    def __init__(self, policy, baseline_queries, source_actions, instruction, arm):
        self.original_predict = policy.predict_action_chunk
        self.baseline_queries = {int(row["step"]): row for row in baseline_queries}
        self.source_actions = {int(row["step"]): np.asarray(row["action"], dtype=np.float32)
                               for row in source_actions}
        self.instruction, self.arm = instruction, arm
        self.step = 0
        self.target = None
        self.rows = []
        policy.predict_action_chunk = self.predict

    def observe(self, event, payload):
        if event == "observation":
            self.step = int(payload["step"])

    def predict(self, observation, prompt):
        source = self.baseline_queries.get(self.step)
        if source is None:
            raise ValueError(f"Query schedule differs from source at step {self.step}")
        if self.step < TARGET_STEP and prompt != source["prompt"]:
            raise ValueError("Prompt diverged before the locked intervention")
        actual_prompt = prompt
        if self.step == TARGET_STEP:
            if prompt != source["prompt"]:
                raise ValueError("Source prompt differs at intervention query")
            if self.arm == "remove_fact_one_query":
                actual_prompt = self.instruction
        chunk = np.asarray(self.original_predict(observation, actual_prompt))
        if self.step < TARGET_STEP and not np.array_equal(chunk, np.asarray(source["chunk"], dtype=chunk.dtype)):
            raise ValueError(f"Policy chunk prefix mismatch at query step {self.step}")
        if self.step == TARGET_STEP:
            source_chunk = np.asarray(source["chunk"], dtype=chunk.dtype)
            self.target = dict(step=self.step, arm=self.arm, original_prompt=prompt,
                actual_prompt=actual_prompt,
                source_fact=SOURCE_FACT,
                change_count=int(actual_prompt != prompt),
                source_chunk_sha256=hashlib.sha256(source_chunk.tobytes()).hexdigest(),
                branch_chunk_sha256=hashlib.sha256(chunk.tobytes()).hexdigest(),
                first_action_l2=float(np.linalg.norm(chunk[0] - source_chunk[0])),
                executed_prefix_l2=float(np.linalg.norm(chunk[:10] - source_chunk[:10])))
            self.rows.append(self.target)
        return chunk


class CheckedObserver:
    def __init__(self, recorder, intervention, source_observations):
        self.recorder = recorder
        self.intervention = intervention
        self.source_observations = {int(r["step"]): r for r in source_observations}

    def __call__(self, event, **payload):
        self.intervention.observe(event, payload)
        step = int(payload.get("step", -1))
        if event == "observation" and step <= TARGET_STEP:
            expected = self.source_observations[step]["rgb_sha256"]
            if rgb_hash(payload["observation"]) != expected:
                raise ValueError(f"Source RGB mismatch at observation {step}")
        if event == "action" and step < TARGET_STEP:
            expected = self.intervention.source_actions[step]
            if not np.array_equal(np.asarray(payload["action"], dtype=np.float32), expected):
                raise ValueError(f"Source action prefix mismatch at step {step}")
        if event == "query" and step == TARGET_STEP:
            row = self.intervention.target
            payload["prompt"] = row["actual_prompt"]
            payload["intervention"] = {"arm": row["arm"], "source_fact": SOURCE_FACT}
        self.recorder(event, **payload)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--adapter-dir", type=Path, default=ROOT / "experiments/training/xvla_v2_f1_step5000")
    p.add_argument("--checkpoint", type=Path, default=Path("/tmp/xvla_checkpoint_e2_f1_mem_on_20261122"))
    p.add_argument("--writer-source-root", type=Path,
                   default=ROOT / "experiments/runtime/xvla_e2_memory_source_20261121")
    p.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    p.add_argument("--sim-python", type=Path, default=Path(".venv_libero_reference_3813/bin/python"))
    p.add_argument("--inference-python", type=Path, default=Path(".venv_x_vla/bin/python"))
    p.add_argument("--detector-python", type=Path,
                   default=Path("/mnt/d/Thesis_26/.venv_memory/Scripts/python.exe"))
    p.add_argument("--detector-model", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    p.add_argument("--port", type=int, default=9440)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output_dir.absolute()
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    protocol, lock, roster = evaluation_roster(args.split_lock, "unseen", ["T1"], "10")
    if len(roster) != 1:
        raise ValueError("Unexpected locked state roster")
    row = roster[0]
    run_manifest, source_status, source_summary, source_queries, source_obs, source_actions, target = verify_source(
        row, args.adapter_dir)
    output.mkdir(parents=True, exist_ok=False)
    spec_path = ROOT / "resources/manifests/models/x_vla_libero.yaml"
    import yaml
    model_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    verify_files(args.checkpoint, model_spec["sha256"])
    verify_adapter(args.adapter_dir, model_spec)
    source_root = args.writer_source_root
    contract = json.loads((ROOT / "experiments/training/relations-author-split-v2/shards/manifest.json").read_text())
    memory_contract = contract["memory_provenance"][0]
    for name, digest in memory_contract["writer_code_sha256"].items():
        if sha256_file(source_root / name) != digest:
            raise ValueError("Writer snapshot hash mismatch: " + name)
    writer_receipt = json.loads((source_root / "memory_source_receipt.json").read_text())
    manifest = dict(classification="CUSTOM_XVLA_ONE_QUERY_PROMPT_COUNTERFACTUAL",
        comparability="NON_COMPARABLE_OFFICIAL_PAPER", status="preflight",
        source_run="experiments/evaluation/xvla_e2_f1_mem_on_20261122",
        source_episode=str(SOURCE.relative_to(ROOT).as_posix()),
        source_run_manifest_sha256=sha256_file(SOURCE.parent.parent / "run_manifest.json"),
        source_episode_status_sha256=sha256_file(SOURCE / "status.json"),
        source_episode_artifact_sha256=source_status["files_sha256"],
        source_outcome=source_summary["result"], task_state=row, seed=42,
        target_query=dict(step=TARGET_STEP, prompt=target["prompt"], fact=SOURCE_FACT),
        arms=["F1-original", "remove_fact_one_query"], protocol=protocol,
        protocol_sha256=source_digest(DEFAULT_PROTOCOL), split_lock_sha256=lock["lock_sha256"],
        checkpoint=dict(adapter=str(args.adapter_dir), adapter_sha256=sha256_file(args.adapter_dir/"adapter_model.safetensors"),
                        base_checkpoint=str(args.checkpoint)),
        memory_source_receipt=writer_receipt,
        rollout=dict(execution_horizon=10, max_policy_actions=1000, sim_seed=42,
                     controller="absolute OSC_POSE", action_chunk=30),
        interpretation="One failed unseen-state source trajectory. Diagnostic prompt sensitivity only; not a success efficacy estimate.")
    write_json(output / "run_manifest.json", manifest)
    shutil.copyfile(args.adapter_dir / "export_manifest.json", output / "adapter_manifest_F1.json")
    code = code_identity(output)
    manifest["code"] = code
    manifest["environments"] = {
        "orchestrator_python": sys.version,
        "simulator_python": str(args.sim_python),
        "inference_python": str(args.inference_python),
        "detector_python": str(args.detector_python),
        "writer_source_root": str(source_root),
    }
    write_json(output / "run_manifest.json", manifest)

    if args.preflight_only:
        manifest["status"] = "preflight_passed_no_rollout"
        write_json(output / "run_manifest.json", manifest)
        print(f"Preflight passed without rollout: {output}")
        return

    from transformers import BartTokenizerFast
    tokenizer = BartTokenizerFast.from_pretrained(str(args.checkpoint), local_files_only=True)
    prompt_tokenizer = load_prompt_tokenizer(ROOT / "experiments/training/assets_mem_adapter_v1/backbone")
    suite = load_suite_source(ROOT / "external/libero-mem")
    detector_exec = detector_launcher(args.detector_python, output)
    detector = ProcessRelationDetector(detector_exec, args.detector_model, device="cuda", timeout_seconds=args.timeout)
    if detector.provenance != memory_contract["detector"]:
        raise ValueError("Loaded detector provenance differs from F1 training contract")
    factory, renderer = relation_pipeline(suite, "T1", detector, "libero_raw", prompt_tokenizer)
    env = ProcessXVLAMemEnv("T1", split_lock=args.split_lock, split="unseen",
                            python_executable=str(args.sim_python), timeout_seconds=args.timeout)
    summaries = []
    try:
        for index, arm in enumerate(("F1-original", "remove_fact_one_query")):
            episode_id = f"T1_F1_unseen_state10_q10_{arm}"
            ep_dir = output / "episodes" / episode_id
            server_dir = output / "servers" / episode_id
            server_dir.mkdir(parents=True, exist_ok=False)
            metadata = dict(row, episode_id=episode_id, condition="F1-mem-on",
                            weight_arm="F1", memory="on", policy_seed=42,
                            intervention=arm, intervention_query=TARGET_STEP,
                            source_episode="T1_F1_mem-on_unseen_state10")
            recorder = EpisodeRecorder(ep_dir, metadata, save_video=True, fps=20)
            policy = XVLAMemPolicy(f"http://127.0.0.1:{args.port+index}/act", env, tokenizer,
                                    json.loads((args.adapter_dir/"export_manifest.json").read_text()), args.timeout)
            intervention = OneQueryIntervention(policy, source_queries, source_actions, row["instruction"], arm)
            observer = CheckedObserver(recorder, intervention, source_obs)
            random.seed(42)
            np.random.seed(42)
            try:
                with start_server(args.inference_python.absolute(), ROOT / "external/X-VLA", args.checkpoint.absolute(),
                                  server_dir, args.port+index, args.timeout, adapter=args.adapter_dir.absolute(), seed=42):
                    runner = LiberoMemRolloutRunner(env, policy, "F1", renderer=renderer,
                        writer_factory=factory, observer=observer, stream_memory_trace=True)
                    result = runner.run_episode(episode_id, row["instruction"], row["state_idx"])
                np.savez_compressed(ep_dir / "xvla_raw_chunks.npz", raw=np.stack(policy.raw_chunks))
                recorder.finish(result=result)
            except BaseException as exc:
                recorder.finish(error=exc)
                raise
            source_action_map = intervention.source_actions
            actual_actions = {int(r["step"]): np.asarray(r["action"], dtype=np.float32)
                               for r in read_jsonl(ep_dir / "actions.jsonl")}
            shared = sorted(source_action_map.keys() & actual_actions.keys())
            prefix = [s for s in shared if s < TARGET_STEP]
            exact_prefix = sum(np.array_equal(source_action_map[s], actual_actions[s]) for s in prefix)
            full_exact = sum(np.array_equal(source_action_map[s], actual_actions[s]) for s in shared)
            summary = dict(arm=arm, success=result.success, scr=result.scr, overshot=result.overshot,
                steps=result.total_steps, source_steps=len(source_action_map), shared_steps=len(shared),
                prefix_exact_actions=f"{exact_prefix}/{len(prefix)}", full_exact_actions=f"{full_exact}/{len(shared)}",
                intervention=intervention.target)
            summaries.append(summary)
            write_json(output / "summary.json", dict(episodes=summaries))
            del runner
        manifest.update(status="complete", completed_episodes=len(summaries))
        write_json(output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        env.close()
        detector.close()
    lines = ["# X-VLA single-query fact intervention", "",
             "Classification: `CUSTOM_XVLA_ONE_QUERY_PROMPT_COUNTERFACTUAL`; `NON_COMPARABLE_OFFICIAL_PAPER`.", "",
             "Source: completed X-VLA E2 F1 memory-on T1 unseen state 10, an unsuccessful 1,000-action rollout.",
             "At query step 10, the source prompt contained one writer-produced fact. Each branch replays from the same locked state and seed; the intervention branch removes that fact for exactly that query.",
             "This is a one-state diagnostic, not an estimate of success efficacy.", "",
             "| Arm | Success | SCR | Overshot | Steps | Prefix action parity | Full action parity | First-action L2 | Executed-prefix L2 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        intervention = item["intervention"] or {}
        lines.append(f'| {item["arm"]} | {item["success"]} | {item["scr"]:.3f} | {item["overshot"]} | {item["steps"]} | '
                     f'{item["prefix_exact_actions"]} | {item["full_exact_actions"]} | '
                     f'{intervention.get("first_action_l2", 0):.6f} | {intervention.get("executed_prefix_l2", 0):.6f} |')
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
