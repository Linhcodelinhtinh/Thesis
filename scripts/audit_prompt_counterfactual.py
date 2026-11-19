"""Run state-16 prompt interventions from the locked T1 F1 baseline.

This is a custom, non-comparable prompt intervention. It keeps the trained
F1 adapter and policy seed fixed, verifies the logged RGB and action prefix,
and changes only the prompt at one selected policy query.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from evaluate_mem_adapter import check_environments, code_identity
from src.evaluation.libero_mem_protocol import ROOT, load_protocol, load_suite_source
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.evaluation.mem_eval_io import EpisodeRecorder, evaluation_roster, write_json
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.relation_detector import ProcessRelationDetector
from src.memory.relation_pipeline import relation_pipeline
from src.models.smolvla.adapter import SmolVLAAdapter
from src.simulator.process_libero_mem_env import ProcessLiberoMemEnv
from src.training.mem_adapter_artifact import validate_adapter, verify_files
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file


TARGET_QUERIES = (630, 650, 670, 680)
INTERVENTIONS = ("F1-original", "F1-off-one-query", "evidence-fact-one-query")
DEFAULT_BASELINE = ROOT / "experiments/evaluation/e2_unseen_author_v2/batch-16-17/episodes/T1_F1_unseen_state16"
DEFAULT_ADAPTER = ROOT / "experiments/training/results_f1_author_split_v2/output_f1/pretrained_model"
DEFAULT_OUTPUT = ROOT / "experiments/evaluation/prompt_counterfactual_state16_20261118"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def digest_rgb(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def verify_baseline(directory, manifest, roster_row, adapter):
    status = json.loads((directory / "status.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    if status.get("status") != "complete":
        raise ValueError("The source F1 episode is not complete")
    for filename, expected_digest in status["files_sha256"].items():
        if sha256_file(directory / filename) != expected_digest:
            raise ValueError("Source episode artifact hash mismatch: " + filename)
    if summary["initial_state_sha256"] != roster_row["initial_state_sha256"]:
        raise ValueError("Source episode and locked initial state differ")
    if summary["policy_seed"] != manifest["arguments"]["policy_seed"]:
        raise ValueError("Source episode and policy seed differ")
    if summary["result"]["total_steps"] != 689 or not summary["result"]["overshot"]:
        raise ValueError("Unexpected source F1 reference outcome")
    expected_adapter_hash = adapter["adapter_files"]["adapter_model.safetensors"]
    if expected_adapter_hash != sha256_file(DEFAULT_ADAPTER / "adapter_model.safetensors"):
        raise ValueError("Source and current F1 checkpoints differ")
    return read_jsonl(directory / "queries.jsonl"), read_jsonl(directory / "observations.jsonl")


class PromptIntervention:
    def __init__(self, policy, predict_original, baseline_queries, target_step, arm, instruction, renderer):
        self.policy = policy
        self.baseline_queries = {row["step"]: row for row in baseline_queries}
        self.target_step = target_step
        self.arm = arm
        self.instruction = instruction
        self.renderer = renderer
        self.query_count = 0
        self.current_step = 0
        self.latest_observation = None
        self.latest_info = None
        self.latest_memory = None
        self.current_prompt = None
        self.current_intervention = None
        self.target_prompt = None
        self.target_summary = None
        self.query_summaries = []
        self._predict_original = predict_original
        policy.predict_action_chunk = self._predict

    def observe(self, event, payload):
        if event == "observation":
            self.current_step = int(payload["step"])
            self.latest_observation = payload["observation"]
            self.latest_info = payload["info"]
        elif event == "memory":
            self.latest_memory = payload["snapshot"]

    def _evidence_fact(self):
        if self.latest_observation is None or self.latest_info is None or self.latest_memory is None:
            raise ValueError("No current evidence is available at the intervention query")
        relation = self.latest_memory["relations"].get("bowl_1")
        aperture = float(np.abs(self.latest_observation["robot0_gripper_qpos"]).sum())
        if (not self.latest_info.get("final_predicate") or aperture < 0.055 or relation is None
                or relation["status"] != "confirmed" or relation["reference"] != "plate_1"
                or not relation["current_evidence_ids"] or relation["returns"] != 0
                or relation["history_complete"]):
            raise ValueError("The evidence fact is not supported by the current writer and simulator state")
        return dict(
            fact="Bowl on original plate. Gripper released. Return count uncertain.",
            simulator_predicate=bool(self.latest_info["final_predicate"]),
            relation_status=relation["status"],
            support_id=relation["reference"],
            current_evidence_ids=list(relation["current_evidence_ids"]),
            measured_gripper_aperture=aperture,
            observed_returns=relation["returns"],
            history_complete=relation["history_complete"],
        )

    def _predict(self, observation, original_prompt):
        step = self.current_step
        original_query = self.baseline_queries.get(step)
        if step < self.target_step:
            if original_query is None:
                raise ValueError(f"Pre-intervention query timing differs at step {step}")
            if original_prompt != original_query["prompt"]:
                raise ValueError("Pre-intervention prompt diverged from the source F1 rollout")

        prompt = original_prompt
        intervention = None
        if step == self.target_step:
            if original_query is None:
                raise ValueError("The intervention step is not a source F1 query")
            if original_prompt != original_query["prompt"]:
                raise ValueError("Source F1 prompt differs at the intervention point")
            if self.arm == "F1-off-one-query":
                prompt = self.instruction
                intervention = dict(kind=self.arm, evidence=None)
            elif self.arm == "evidence-fact-one-query":
                evidence = self._evidence_fact()
                prompt = f"{self.instruction} Memory: {evidence['fact']}"
                token_count = self.renderer.count_tokens(prompt)
                if token_count > self.renderer.max_total_tokens:
                    raise ValueError("Evidence intervention exceeds the locked prompt token budget")
                intervention = dict(kind=self.arm, evidence=evidence, total_tokens=token_count)
            else:
                intervention = dict(kind=self.arm, evidence=None)

        chunk = self._predict_original(observation, prompt)
        chunk = np.asarray(chunk)
        if step < self.target_step:
            expected_chunk = np.asarray(original_query["chunk"], dtype=chunk.dtype)
            if not np.array_equal(chunk, expected_chunk):
                raise ValueError(f"Policy/noise prefix mismatch at query {step}")

        self.current_prompt = prompt
        self.current_intervention = intervention
        self.query_count += 1
        if step != self.target_step:
            return chunk

        original_chunk = np.asarray(original_query["chunk"], dtype=chunk.dtype)
        self.target_prompt = prompt
        self.target_summary = dict(
            step=step,
            original_prompt=original_query["prompt"],
            actual_prompt=prompt,
            intervention=intervention,
            original_chunk_sha256=hashlib.sha256(original_chunk.tobytes()).hexdigest(),
            branch_chunk_sha256=hashlib.sha256(chunk.tobytes()).hexdigest(),
            first_action_l2=float(np.linalg.norm(chunk[0] - original_chunk[0])),
            prefix_action_l2=float(np.linalg.norm(chunk[:10] - original_chunk[:10])),
            original_z_action_sum=float(original_chunk[:10, 2].sum()),
            branch_z_action_sum=float(chunk[:10, 2].sum()),
            original_gripper_action_sum=float(original_chunk[:10, 6].sum()),
            branch_gripper_action_sum=float(chunk[:10, 6].sum()),
        )
        self.query_summaries.append(self.target_summary)
        return chunk


class AuditObserver:
    def __init__(self, recorder, intervention, reference_observations, target_step):
        self.recorder = recorder
        self.intervention = intervention
        self.reference_observations = {row["step"]: row for row in reference_observations}
        self.target_step = target_step

    def __call__(self, event, **payload):
        self.intervention.observe(event, payload)
        if event == "observation" and payload["step"] <= self.target_step:
            expected = self.reference_observations[payload["step"]]["rgb_sha256"]
            actual = {camera: digest_rgb(payload["observation"][camera])
                      for camera in ("agentview_image", "robot0_eye_in_hand_image")}
            if actual != expected:
                raise ValueError(f"RGB replay hash mismatch at observation {payload['step']}")
        elif event == "action" and payload["step"] < self.target_step:
            expected_action = np.asarray(self.intervention.baseline_actions[payload["step"]], dtype=np.float32)
            if not np.array_equal(np.asarray(payload["action"], dtype=np.float32), expected_action):
                raise ValueError(f"Action prefix mismatch at action {payload['step']}")
        elif event == "query" and payload["step"] == self.target_step:
            payload["prompt"] = self.intervention.current_prompt
            payload["intervention"] = self.intervention.current_intervention
            payload["render"] = dict(
                text=self.intervention.current_prompt,
                truncated=False,
                token_truncated=False,
                tokenizer_truncated=False,
                budget_tokens=self.intervention.renderer.max_total_tokens,
                total_tokens_with_instruction=self.intervention.renderer.count_tokens(
                    self.intervention.current_prompt),
                facts_included=([self.intervention.current_intervention["evidence"]["fact"]]
                                if self.intervention.current_intervention["evidence"] else []),
                memory_skipped_reason=(None if self.intervention.current_intervention["evidence"]
                                       else "prompt_intervention_f1_off"),
            )
        self.recorder(event, **payload)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "resources/checkpoints/smolvla_libero")
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--detector-model", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    parser.add_argument("--detector-python", default=str(ROOT / ".venv_memory/Scripts/python.exe"))
    parser.add_argument("--sim-python", default="/mnt/d/Thesis_26/.venv_libero_reference_3813/bin/python")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--targets", type=int, nargs="+", default=list(TARGET_QUERIES))
    parser.add_argument("--arms", choices=INTERVENTIONS[1:], nargs="+", default=list(INTERVENTIONS[1:]))
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output_dir.absolute()
    targets = tuple(args.targets)
    arms = tuple(args.arms)
    if not targets or any(target not in TARGET_QUERIES for target in targets):
        raise ValueError(f"Targets must be selected from {TARGET_QUERIES}")
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment directory: {output}")

    protocol, lock, roster = evaluation_roster(args.split_lock, "unseen", ["T1"], "16")
    if len(roster) != 1 or roster[0]["state_idx"] != 16:
        raise ValueError("Unexpected prompt-intervention roster")
    roster_row = roster[0]
    baseline_dir = DEFAULT_BASELINE
    baseline_manifest = json.loads((baseline_dir.parent.parent / "run_manifest.json").read_text(encoding="utf-8-sig"))
    adapter = validate_adapter(args.adapter_dir, args.base_checkpoint)
    baseline_queries, baseline_observations = verify_baseline(baseline_dir, baseline_manifest, roster_row, adapter)
    baseline_actions = {row["step"]: row["action"] for row in read_jsonl(baseline_dir / "actions.jsonl")}
    for target in targets:
        if target not in {row["step"] for row in baseline_queries}:
            raise ValueError(f"Missing source query {target}")
    for target in targets:
        record = next(row for row in baseline_observations if row["step"] == target)
        if not record["info"]["final_predicate"]:
            raise ValueError(f"The source bowl is not on its support at query {target}")

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        classification="CUSTOM_PROMPT_COUNTERFACTUAL_LIBERO_MEM",
        comparability="NON-COMPARABLE_OFFICIAL_PAPER",
        status="preflight",
        task=roster_row,
        protocol_id=protocol["protocol_id"],
        split_lock_sha256=lock["lock_sha256"],
        source_run="experiments/evaluation/e2_unseen_author_v2/batch-16-17",
        source_episode="T1_F1_unseen_state16",
        source_episode_artifact_sha256=json.loads((baseline_dir / "status.json").read_text(
            encoding="utf-8-sig"))["files_sha256"],
        checkpoint=dict(
            adapter_path=str(args.adapter_dir),
            adapter_sha256=sha256_file(args.adapter_dir / "adapter_model.safetensors"),
            adapter_manifest_sha256=sha256_file(args.adapter_dir / "mem_adapter_manifest.json"),
            base_checkpoint_path=str(args.base_checkpoint),
            memory_contract=adapter["memory_provenance"][0],
        ),
        policy_seed=42,
        target_queries=list(targets),
        interventions=["F1-original", *arms],
        source_rgb_hashes={str(step): next(row["rgb_sha256"] for row in baseline_observations
                                           if row["step"] == step) for step in targets},
        code_identity=code_identity(output),
        protocol=protocol,
        horizon=protocol["rollout"]["horizon_policy_actions"],
        execution_horizon=protocol["rollout"]["execution_horizon"],
        seeds=dict(policy=42, simulator=protocol["reset"]["seed"]),
    )
    write_json(output / "split.lock.json", lock)
    shutil.copyfile(args.adapter_dir / "mem_adapter_manifest.json", output / "adapter_manifest_F1.json")
    write_json(output / "run_manifest.json", manifest)

    env_manifest = check_environments(argparse.Namespace(
        use_wsl=True,
        wsl_distro=args.wsl_distro,
        sim_python=args.sim_python,
        detector_python=args.detector_python,
        save_video=True,
        conditions=["F1"],
    ), protocol)
    if not env_manifest["policy"]["packages"]["torch"]:
        raise RuntimeError("The pinned policy environment is unavailable")
    verify_files(args.detector_model, adapter["memory_provenance"][0]["detector"]["files"])
    manifest["environments"] = env_manifest
    manifest["status"] = "running"
    write_json(output / "run_manifest.json", manifest)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA policy execution required")
    policy = SmolVLAAdapter()
    policy.load(str(args.adapter_dir), base_checkpoint=str(args.base_checkpoint), device="cuda", strict=True)
    predict_original = policy.predict_action_chunk
    contract = adapter["memory_provenance"][0]
    detector = ProcessRelationDetector(args.detector_python, args.detector_model,
                                       device="cuda", timeout_seconds=args.worker_timeout)
    suite = load_suite_source(ROOT / "external/libero-mem")
    factory, renderer = relation_pipeline(suite, "T1", detector, "libero_raw", policy.tokenizer,
                                          contract["writer_config"]["strategy"])
    env = ProcessLiberoMemEnv(task_name="T1", split_lock=args.split_lock, split="unseen",
                              python_executable=args.sim_python, use_wsl=True,
                              wsl_distro=args.wsl_distro, timeout_seconds=args.worker_timeout)

    baseline_result = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8-sig"))["result"]
    rows = []
    for target in targets:
        source_query = next(row for row in baseline_queries if row["step"] == target)
        rows.append(dict(
            task="T1", task_name=roster_row["task_name"], arm="F1-original",
            query_step=target, episode_id="T1_F1_unseen_state16",
            success=baseline_result["success"], overshot=baseline_result["overshot"],
            scr=baseline_result["scr"], steps=baseline_result["total_steps"],
            mean_policy_latency_ms=baseline_result["avg_latency_ms"],
            wall_time_s=baseline_result["total_wall_time_s"],
            intervention=dict(
                step=target, original_prompt=source_query["prompt"],
                actual_prompt=source_query["prompt"], intervention=dict(kind="F1-original"),
                original_chunk_sha256=hashlib.sha256(
                    np.asarray(source_query["chunk"], dtype=np.float32).tobytes()).hexdigest(),
                branch_chunk_sha256=hashlib.sha256(
                    np.asarray(source_query["chunk"], dtype=np.float32).tobytes()).hexdigest(),
                first_action_l2=0.0, prefix_action_l2=0.0,
                original_z_action_sum=float(np.asarray(source_query["chunk"], dtype=np.float32)[:10, 2].sum()),
                branch_z_action_sum=float(np.asarray(source_query["chunk"], dtype=np.float32)[:10, 2].sum()),
                original_gripper_action_sum=float(np.asarray(source_query["chunk"], dtype=np.float32)[:10, 6].sum()),
                branch_gripper_action_sum=float(np.asarray(source_query["chunk"], dtype=np.float32)[:10, 6].sum()),
                source_episode="T1_F1_unseen_state16",
            ),
        ))
    write_json(output / "counterfactual_summary.json", rows)
    try:
        for target in targets:
            for arm in arms:
                episode_id = f"T1_F1_state16_q{target}_{arm}"
                episode_dir = output / "episodes" / episode_id
                metadata = dict(
                    task="T1", task_name=roster_row["task_name"], split="unseen", state_idx=16,
                    initial_state_sha256=roster_row["initial_state_sha256"], episode_id=episode_id,
                    condition="F1", prompt_intervention=arm, intervention_query=target,
                    policy_seed=42, baseline_episode="T1_F1_unseen_state16",
                )
                recorder = EpisodeRecorder(episode_dir, metadata, save_video=True,
                                           fps=protocol["simulation"]["control_freq"])
                intervention = PromptIntervention(policy, predict_original, baseline_queries, target, arm,
                                                  roster_row["instruction"], renderer)
                intervention.baseline_actions = baseline_actions
                observer = AuditObserver(recorder, intervention, baseline_observations, target)
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                try:
                    runner = LiberoMemRolloutRunner(
                        env, policy, condition="F1",
                        execution_horizon=protocol["rollout"]["execution_horizon"],
                        max_steps=protocol["rollout"]["horizon_policy_actions"],
                        renderer=renderer, writer_factory=factory, observer=observer,
                        stream_memory_trace=True,
                    )
                    result = runner.run_episode(episode_id, roster_row["instruction"], 16)
                    result.prompt_history[target // 10] = intervention.target_prompt
                    write_json(episode_dir / "counterfactual_queries.json", intervention.query_summaries)
                    recorder.finish(result=result)
                except BaseException as exc:
                    recorder.finish(error=exc)
                    raise
                rows.append(dict(
                    task="T1", task_name=roster_row["task_name"], arm=arm,
                    query_step=target, episode_id=episode_id,
                    success=result.success, overshot=result.overshot,
                    scr=result.scr, steps=result.total_steps,
                    mean_policy_latency_ms=result.avg_latency_ms,
                    wall_time_s=result.total_wall_time_s,
                    intervention=intervention.target_summary,
                ))
                write_json(output / "counterfactual_summary.json", rows)
                print(f"{episode_id}: success={result.success} SCR={result.scr:.3f} "
                      f"overshot={result.overshot} steps={result.total_steps}", flush=True)
                del runner
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        manifest["status"] = "complete"
        write_json(output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        env.close()
        detector.close()

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        import csv
        fields = ("task_name", "arm", "query_step", "episode_id", "success", "overshot", "scr",
                  "steps", "mean_policy_latency_ms", "wall_time_s")
        writer_csv = csv.DictWriter(stream, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows({key: row[key] for key in fields} for row in rows)
    report_lines = [
        "# F1 prompt counterfactuals on T1 state 16",
        "",
        "NON-COMPARABLE_OFFICIAL_PAPER",
        "",
        "Each intervention changes one prompt at the selected query. Subsequent queries return to the F1 renderer.",
        "Every branch must match the source RGB hashes and F1 action chunks through the intervention query.",
        "See `counterfactual_summary.json` and per-episode `counterfactual_queries.json` for the branch results.",
        "",
        "| Query | Intervention | SCR | Overshot | Success | Steps | Mean policy latency (ms) |",
        "|---:|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['query_step']} | {row['arm']} | {row['scr']:.3f} | {row['overshot']} | "
            f"{row['success']} | {row['steps']} | {row['mean_policy_latency_ms']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
