#!/usr/bin/env python
"""One-query prompt intervention at the successful D2 event requery.

This is a single-state, custom counterfactual. It replays the actual T1 seen
state-7 event run and changes only the prompt at event query step 142. Exact
RGB, prompt, and action-prefix checks are required through the intervention.
The F1-original rerun is a deterministic-replay control; F1-off removes the
memory text for that one query. Neither branch is an efficacy estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_prompt_counterfactual import (  # noqa: E402
    AuditObserver,
    PromptIntervention,
    read_jsonl,
)
from evaluate_mem_adapter import (  # noqa: E402
    check_environments,
    code_identity,
    detector_launcher,
)
from src.evaluation.libero_mem_protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
    load_protocol,
    load_suite_source,
    source_digest,
)
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner  # noqa: E402
from src.evaluation.mem_eval_io import EpisodeRecorder, evaluation_roster, write_json  # noqa: E402
from src.evaluation.memory_supervisor import (  # noqa: E402
    DEFAULT_DECISION_PROTOCOL,
    execution_treatment,
    load_decision_protocol,
)
from src.memory.relation_writer_versions import (  # noqa: E402
    make_relation_pipeline,
    versioned_relation_source_hashes,
)
from src.perception.relation_detector import ProcessRelationDetector  # noqa: E402
from src.simulator.process_libero_mem_env import ProcessLiberoMemEnv  # noqa: E402
from src.training.mem_adapter_artifact import validate_adapter, verify_files  # noqa: E402
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK  # noqa: E402
from src.utils.file_hash import sha256_file  # noqa: E402


SOURCE_RUN = ROOT / "experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl"
SOURCE_EPISODE = SOURCE_RUN / "episodes/T1_F1_seen_state7"
DEFAULT_ADAPTER = ROOT / "experiments/training/results_f1_author_split_v2/output_f1/pretrained_model"
DEFAULT_BASE_CHECKPOINT = ROOT / "resources/checkpoints/smolvla_libero"
DEFAULT_DETECTOR_MODEL = ROOT / "resources/models/grounding-dino-tiny"
DEFAULT_OUTPUT = ROOT / "experiments/evaluation/d2_success_prompt_intervention_20261121"
TARGET_STEP = 142
ARMS = ("F1-original", "F1-off-one-query")


def verify_source(adapter, adapter_path, protocol, lock, roster_row):
    run_manifest = json.loads((SOURCE_RUN / "run_manifest.json").read_text(encoding="utf-8-sig"))
    status_path = SOURCE_EPISODE / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    summary = json.loads((SOURCE_EPISODE / "summary.json").read_text(encoding="utf-8-sig"))

    if run_manifest.get("status") != "complete":
        raise ValueError("D2 source run is incomplete")
    if run_manifest.get("decision_protocol_sha256") != sha256_file(DEFAULT_DECISION_PROTOCOL):
        raise ValueError("D2 decision protocol hash differs from the current protocol")
    if run_manifest.get("protocol_sha256") != source_digest(DEFAULT_PROTOCOL):
        raise ValueError("D2 LIBERO-Mem protocol hash differs from the current protocol")
    if run_manifest.get("split_lock_sha256") != lock["lock_sha256"]:
        raise ValueError("D2 source and current split locks differ")
    if run_manifest.get("split_lock_file_sha256") != sha256_file(DEFAULT_AUTHOR_MEM_LOCK):
        raise ValueError("D2 split-lock file bytes differ from the current lock")
    if run_manifest.get("arguments", {}).get("policy_seed") != 42:
        raise ValueError("Unexpected D2 policy seed")

    treatment = run_manifest.get("treatments", {}).get("F1", {})
    if treatment.get("memory_source") != "rgb_relations_v1" or not treatment.get("writer_override"):
        raise ValueError("D2 source did not use the expected explicit relation-writer override")
    current_writer_hashes = versioned_relation_source_hashes(ROOT, "rgb_relations_v1")
    if current_writer_hashes != treatment.get("runtime_writer_source_sha256"):
        raise ValueError("Runtime relation-writer source hashes differ from the D2 source run")

    for name, expected in status.get("files_sha256", {}).items():
        path = SOURCE_EPISODE / name
        if sha256_file(path) != expected:
            raise ValueError("D2 source episode hash mismatch: " + name)
    if status.get("status") != "complete":
        raise ValueError("Successful D2 source episode is incomplete")
    if summary.get("initial_state_sha256") != roster_row["initial_state_sha256"]:
        raise ValueError("D2 source and current locked state differ")
    if summary.get("policy_seed") != 42:
        raise ValueError("D2 source episode seed differs")
    result = summary.get("result", {})
    if not result.get("success") or result.get("total_steps") != 1000 or result.get("overshot"):
        raise ValueError("D2 source is not the recorded successful T1 state-7 event episode")
    if adapter["adapter_files"]["adapter_model.safetensors"] != treatment.get("weight_identity"):
        raise ValueError("Current adapter weights differ from the D2 source run")
    if sha256_file(adapter_path / "mem_adapter_manifest.json") != treatment.get("adapter_manifest_sha256"):
        raise ValueError("Current adapter manifest differs from the D2 source run")

    queries = read_jsonl(SOURCE_EPISODE / "queries.jsonl")
    observations = read_jsonl(SOURCE_EPISODE / "observations.jsonl")
    actions = read_jsonl(SOURCE_EPISODE / "actions.jsonl")
    decisions = read_jsonl(SOURCE_EPISODE / "decisions.jsonl")
    target_query = next((row for row in queries if row["step"] == TARGET_STEP), None)
    target_decision = next((row for row in decisions if row["step"] == TARGET_STEP), None)
    if target_query is None or target_decision is None:
        raise ValueError("D2 source is missing the locked step-142 event query/decision")
    if target_decision.get("decision") != "requery" or not target_decision.get("changes"):
        raise ValueError("Step 142 is not a logged evidence-backed requery")
    if not any(change.get("after", {}).get("evidence_ids") for change in target_decision["changes"]):
        raise ValueError("Step-142 requery lacks evidence IDs")
    if target_query.get("prompt") == roster_row["instruction"]:
        raise ValueError("Step-142 source prompt does not contain memory")
    return run_manifest, status, summary, queries, observations, actions, target_query, target_decision


def action_comparison(source_actions, branch_dir, target_step):
    branch_actions = read_jsonl(branch_dir / "actions.jsonl")
    source_by_step = {int(row["step"]): np.asarray(row["action"], dtype=np.float32) for row in source_actions}
    branch_by_step = {int(row["step"]): np.asarray(row["action"], dtype=np.float32) for row in branch_actions}
    shared = sorted(source_by_step.keys() & branch_by_step.keys())
    prefix = [step for step in shared if step < target_step]
    matches_prefix = sum(np.array_equal(source_by_step[step], branch_by_step[step]) for step in prefix)
    full_matches = sum(np.array_equal(source_by_step[step], branch_by_step[step]) for step in shared)
    return dict(
        source_action_steps=len(source_by_step),
        branch_action_steps=len(branch_by_step),
        shared_action_steps=len(shared),
        pre_intervention_steps=len(prefix),
        pre_intervention_exact_matches=matches_prefix,
        full_trajectory_exact_matches=full_matches,
        post_intervention_exact_matches=sum(
            np.array_equal(source_by_step[step], branch_by_step[step])
            for step in shared if step >= target_step
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--detector-python", default=str(ROOT / ".venv_memory_wsl/bin/python"))
    parser.add_argument("--sim-python", default=str(ROOT / ".venv_libero_reference_3813/bin/python"))
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output_dir.absolute()
    if output.exists():
        raise FileExistsError(f"Preserving existing intervention output: {output}")

    protocol, lock, roster = evaluation_roster(args.split_lock, "seen", ["T1"], "7")
    if len(roster) != 1 or roster[0]["state_idx"] != 7:
        raise ValueError("Unexpected D2 success-state roster")
    roster_row = roster[0]
    decision_protocol = load_decision_protocol()
    event_treatment = execution_treatment(decision_protocol, "event", "projected")

    # The D2 run explicitly used the same relation-writer override recorded
    # below. Validate artifacts strictly, but let the checked override establish
    # runtime writer source identity as it did in D2.
    adapter = validate_adapter(args.adapter_dir, args.base_checkpoint, verify_provenance_files=False)
    (run_manifest, source_status, source_summary, baseline_queries,
     baseline_observations, baseline_actions, target_query, target_decision) = verify_source(
        adapter, args.adapter_dir, protocol, lock, roster_row
    )

    output.mkdir(parents=True, exist_ok=False)
    source_writer_hashes = run_manifest["treatments"]["F1"]["runtime_writer_source_sha256"]
    intervention_spec = dict(
        classification="CUSTOM_D2_SUCCESS_STATE_ONE_QUERY_COUNTERFACTUAL_NOT_EFFICACY",
        comparability="NON-COMPARABLE_OFFICIAL_PAPER",
        status="preflight",
        source_run=str(SOURCE_RUN.relative_to(ROOT).as_posix()),
        source_episode=str(SOURCE_EPISODE.relative_to(ROOT).as_posix()),
        source_run_manifest_sha256=sha256_file(SOURCE_RUN / "run_manifest.json"),
        source_episode_status_sha256=sha256_file(SOURCE_EPISODE / "status.json"),
        source_episode_artifact_sha256=source_status["files_sha256"],
        source_success=dict(
            success=source_summary["result"]["success"],
            scr=source_summary["result"]["scr"],
            overshot=source_summary["result"]["overshot"],
            steps=source_summary["result"]["total_steps"],
        ),
        roster=roster_row,
        split_lock_sha256=lock["lock_sha256"],
        protocol_sha256=source_digest(DEFAULT_PROTOCOL),
        decision_protocol_sha256=sha256_file(DEFAULT_DECISION_PROTOCOL),
        checkpoint=dict(
            adapter_path=str(args.adapter_dir),
            adapter_model_sha256=sha256_file(args.adapter_dir / "adapter_model.safetensors"),
            adapter_manifest_sha256=sha256_file(args.adapter_dir / "mem_adapter_manifest.json"),
            base_checkpoint_path=str(args.base_checkpoint),
            memory_contract=adapter["memory_provenance"][0],
        ),
        intervention=dict(
            target_step=TARGET_STEP,
            source_prompt=target_query["prompt"],
            source_chunk_sha256=target_query["chunk_sha256"],
            source_decision=target_decision,
            arms=list(ARMS),
            changed_field="memory text in exactly one policy query",
            prefix_integrity="exact RGB and action/chunk checks through step 142",
        ),
        runtime_writer_source_sha256=source_writer_hashes,
        policy_seed=42,
        simulator_seed=protocol["reset"]["seed"],
        decision_mode="event",
        decision_exposure="projected",
        code_identity=code_identity(output),
        interpretation_limit=(
            "One successful seen-state source episode and two one-query branches. "
            "Even exact prefix reproduction cannot establish population-level decision or success efficacy."
        ),
    )
    write_json(output / "split.lock.json", lock)
    shutil.copyfile(args.adapter_dir / "mem_adapter_manifest.json", output / "adapter_manifest_F1.json")
    write_json(output / "run_manifest.json", intervention_spec)

    env_args = argparse.Namespace(
        use_wsl=False,
        wsl_distro=args.wsl_distro,
        sim_python=args.sim_python,
        detector_python=args.detector_python,
        save_video=True,
        conditions=["F1"],
    )
    environments = check_environments(env_args, protocol)
    if not environments["policy"]["packages"]["torch"]:
        raise RuntimeError("Pinned policy environment is unavailable")
    verify_files(args.detector_model, adapter["memory_provenance"][0]["detector"]["files"])
    intervention_spec["environments"] = environments
    intervention_spec["status"] = "running"
    write_json(output / "run_manifest.json", intervention_spec)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA policy execution required; no CPU fallback")
    from src.models.smolvla.adapter import SmolVLAAdapter

    policy = SmolVLAAdapter()
    policy.load(
        str(args.adapter_dir),
        base_checkpoint=str(args.base_checkpoint),
        device="cuda",
        strict=True,
        verify_provenance_files=False,
    )
    predict_original = policy.predict_action_chunk
    contract = adapter["memory_provenance"][0]
    detector_exec = detector_launcher(args.detector_python, output)
    detector = ProcessRelationDetector(detector_exec, args.detector_model, device="cuda",
                                       timeout_seconds=args.worker_timeout)
    suite = load_suite_source(ROOT / "external/libero-mem")
    factory, renderer = make_relation_pipeline(
        suite, "T1", detector, "libero_raw", policy.tokenizer,
        contract["writer_config"]["strategy"], "rgb_relations_v1",
    )
    env = ProcessLiberoMemEnv(
        task_name="T1",
        split_lock=args.split_lock,
        split="seen",
        python_executable=args.sim_python,
        use_wsl=False,
        wsl_distro=args.wsl_distro,
        timeout_seconds=args.worker_timeout,
    )

    summaries = []
    try:
        for arm in ARMS:
            episode_id = f"T1_F1_seen_state7_q{TARGET_STEP}_{arm}"
            episode_dir = output / "episodes" / episode_id
            metadata = dict(
                task="T1",
                task_name=roster_row["task_name"],
                split="seen",
                state_idx=7,
                initial_state_sha256=roster_row["initial_state_sha256"],
                episode_id=episode_id,
                condition="F1",
                decision_mode="event",
                decision_exposure="projected",
                prompt_intervention=arm,
                intervention_query=TARGET_STEP,
                policy_seed=42,
                baseline_episode="T1_F1_seen_state7",
            )
            recorder = EpisodeRecorder(episode_dir, metadata, save_video=True,
                                       fps=protocol["simulation"]["control_freq"])
            intervention = PromptIntervention(
                policy,
                predict_original,
                baseline_queries,
                TARGET_STEP,
                arm,
                roster_row["instruction"],
                renderer,
            )
            intervention.baseline_actions = {row["step"]: row["action"] for row in baseline_actions}
            observer = AuditObserver(recorder, intervention, baseline_observations, TARGET_STEP)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            try:
                runner = LiberoMemRolloutRunner(
                    env,
                    policy,
                    condition="F1",
                    execution_horizon=protocol["rollout"]["execution_horizon"],
                    max_steps=protocol["rollout"]["horizon_policy_actions"],
                    renderer=renderer,
                    writer_factory=factory,
                    observer=observer,
                    stream_memory_trace=True,
                    execution_treatment=event_treatment,
                    memory_writer_override="rgb_relations_v1",
                    runtime_writer_source_sha256=source_writer_hashes,
                    runtime_writer_config=run_manifest["treatments"]["F1"]["runtime_writer_config"],
                )
                result = runner.run_episode(episode_id, roster_row["instruction"], 7)
                write_json(episode_dir / "counterfactual_queries.json", intervention.query_summaries)
                recorder.finish(result=result)
            except BaseException as exc:
                recorder.finish(error=exc)
                raise

            comparison = action_comparison(baseline_actions, episode_dir, TARGET_STEP)
            summary = dict(
                arm=arm,
                episode_id=episode_id,
                success=result.success,
                scr=result.scr,
                overshot=result.overshot,
                completed_subgoals=result.completed_subgoals,
                total_subgoals=result.total_subgoals,
                steps=result.total_steps,
                termination_reason=result.termination_reason,
                mean_policy_latency_ms=result.avg_latency_ms,
                wall_time_s=result.total_wall_time_s,
                query_count=(result.decision_metrics or {}).get("queries"),
                early_requeries=(result.decision_metrics or {}).get("early_requeries"),
                intervention=intervention.target_summary,
                action_comparison_to_source=comparison,
            )
            summaries.append(summary)
            write_json(output / "counterfactual_summary.json", summaries)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        intervention_spec["status"] = "complete"
        intervention_spec["completed_episodes"] = len(summaries)
        intervention_spec["outcome_summary_sha256"] = sha256_file(output / "counterfactual_summary.json")
        write_json(output / "run_manifest.json", intervention_spec)
    except BaseException as exc:
        intervention_spec.update(status="failed", error=repr(exc))
        write_json(output / "run_manifest.json", intervention_spec)
        raise
    finally:
        env.close()
        detector.close()

    report_lines = [
        "# D2 successful-state one-query prompt intervention",
        "",
        "Classification: `CUSTOM_D2_SUCCESS_STATE_ONE_QUERY_COUNTERFACTUAL_NOT_EFFICACY`.",
        "",
        "Source: the completed T1 seen-state 7 D2 event-projected rollout, which succeeded at step 1,000.",
        "At query step 142, both branches are required to match source RGB and all action chunks before the intervention.",
        "`F1-original` repeats the evidence-bearing prompt; `F1-off-one-query` removes memory text at that query only.",
        "Both branches keep the D2 event supervisor, checkpoint, writer, state, and seed fixed.",
        "This two-branch single-state counterfactual is diagnostic and does not estimate efficacy.",
        "",
        "| Arm | Success | SCR | Overshot | Steps | Queries | Early re-queries | Exact action prefix | Exact post-intervention actions | Wall time (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cmp = row["action_comparison_to_source"]
        report_lines.append(
            f"| {row['arm']} | {row['success']} | {row['scr']:.3f} | {row['overshot']} | "
            f"{row['steps']} | {row['query_count']} | {row['early_requeries']} | "
            f"{cmp['pre_intervention_exact_matches']}/{cmp['pre_intervention_steps']} | "
            f"{cmp['post_intervention_exact_matches']}/{max(cmp['shared_action_steps']-cmp['pre_intervention_steps'],0)} | "
            f"{row['wall_time_s']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
