#!/usr/bin/env python
"""Reconcile bounded-size F0/F1 rollout batches against the declared T1/T3 roster."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.evaluation.mem_eval_io import evaluation_roster
from src.training.mem_adapter_artifact import validate_matched_adapters
from src.utils.file_hash import sha256_file

ARMS = ("F0", "F1")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(row, seed):
    return row["task"], row["initial_state_sha256"], seed


def aggregate(batches, lock, roster, seeds, split):
    planned = {identity(row, seed): row for row in roster for seed in seeds}
    if len(planned) != len(roster) * len(seeds):
        raise ValueError("The campaign has duplicate task/state/seed units")
    outcomes, timings, provenance = {}, {}, None
    attempts, adapter_hashes, adapters, loaded = [], {}, {}, {}
    for root in batches:
        manifest = read_json(root / "run_manifest.json")
        args = manifest["arguments"]
        seed = args["policy_seed"]
        if (manifest["classification"] != "CUSTOM_V2_LIBERO_MEM"
                or manifest["comparability"] != "NON-COMPARABLE_OFFICIAL_PAPER"
                or manifest["status"] not in ("complete", "failed")
                or args["runtime_check"] or args["preflight_only"]
                or seed not in seeds or args["split"] != split
                or manifest["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL)
                or manifest["split_lock_sha256"] != lock["lock_sha256"]):
            raise ValueError(f"Run differs from the declared pilot campaign: {root}")
        treatments = manifest["treatments"]
        present_arms = [arm for arm in ARMS if arm in args["conditions"]]
        if not present_arms or len(set(args["conditions"])) != len(args["conditions"]):
            raise ValueError(f"No primary arm or duplicate condition: {root}")
        for arm in present_arms:
            path = root / f"adapter_manifest_{arm}.json"
            digest = sha256_file(path)
            if digest != treatments[arm]["adapter_manifest_sha256"] or digest != adapter_hashes.get(arm, digest):
                raise ValueError(f"Missing or changed {arm} provenance: {root}")
            adapter_hashes[arm] = digest
            adapters[arm] = read_json(path)
            if treatments[arm]["weight_identity"] != adapters[arm]["adapter_files"]["adapter_model.safetensors"]:
                raise ValueError(f"Wrong {arm} weight identity: {root}")
            if adapters[arm]["run"]["arguments"]["condition"] != ("no_memory" if arm == "F0" else "memory"):
                raise ValueError(f"Wrong training condition: {root}")
            if arm == "F1" and adapters[arm]["memory_provenance"][0]["writer"] != "rgb_relations_v1":
                raise ValueError(f"This campaign requires the relation writer: {root}")
            runtime = manifest.get("loaded_policies", {}).get(arm)
            if runtime is not None:
                if arm in loaded and loaded[arm] != runtime:
                    raise ValueError(f"Loaded policy identity differs for {arm}: {root}")
                loaded[arm] = runtime
        contract = dict(base=manifest["base"], source=manifest["source_sha256"],
                        assets=manifest["local_assets_lock_sha256"], environments=manifest["environments"],
                        cuda=manifest["cuda"], save_video=args["save_video"],
                        stream_memory_trace=args.get("stream_memory_trace", False))
        # F0-only retries have no detector process; compare that runtime when F1 is present.
        contract["environments"] = dict(contract["environments"])
        detector_runtime = contract["environments"].pop("detector", None)
        if "F1" in present_arms:
            if detector_runtime is None:
                raise ValueError(f"Missing F1 detector runtime: {root}")
            if provenance is not None and provenance.get("detector", detector_runtime) != detector_runtime:
                raise ValueError(f"Detector runtime differs: {root}")
        if provenance is None:
            provenance = contract
        elif {key: value for key, value in provenance.items() if key != "detector"} != contract:
            raise ValueError(f"Model, environment, logging or source identity differs: {root}")
        if detector_runtime is not None:
            provenance["detector"] = detector_runtime
        batch_roster = {identity(row, seed): row for row in manifest["roster"]}
        if (len(batch_roster) != len(manifest["roster"])
                or any(planned.get(key) != row for key, row in batch_roster.items())
                or manifest["expected_episodes"] != len(batch_roster) * len(args["conditions"])):
            raise ValueError(f"Batch roster differs from the campaign: {root}")
        statuses = Counter()
        for key, row in batch_roster.items():
            for arm in args["conditions"]:
                episode_id = f"{row['task']}_{arm}_{split}_state{row['state_idx']}"
                directory = root / "episodes" / episode_id
                status_path = directory / "status.json"
                status = read_json(status_path) if status_path.is_file() else None
                if status is not None and (identity(status, status["policy_seed"]) != key
                                           or status["condition"] != arm or status["episode_id"] != episode_id):
                    raise ValueError(f"Episode status identity differs from the roster: {directory}")
                statuses[status["status"] if status else "not_started"] += 1
                if not status or status["status"] != "complete":
                    continue
                if arm in ARMS and arm not in manifest.get("loaded_policies", {}):
                    raise ValueError(f"Missing loaded policy provenance: {directory}")
                path = directory / "summary.json"
                if sha256_file(path) != status["files_sha256"]["summary.json"]:
                    raise ValueError(f"Episode summary checksum mismatch: {path}")
                summary = read_json(path)
                if (identity(summary, summary["policy_seed"]) != key or summary["condition"] != arm
                        or summary["episode_id"] != episode_id
                        or summary["result"]["condition"] != arm or type(summary["result"]["success"]) is not bool):
                    raise ValueError(f"Episode summary identity differs from the roster: {path}")
                if arm in ARMS:
                    if (key, arm) in outcomes:
                        raise ValueError(f"Duplicate completed episode for {key}, {arm}")
                    outcomes[key, arm] = int(summary["result"]["success"])
                    timings[key, arm] = (summary["result"]["avg_latency_ms"],
                                         summary["result"]["total_wall_time_s"])
        if statuses["complete"] != manifest["completed_episodes"] or (
                manifest["status"] == "complete" and statuses["complete"] != manifest["expected_episodes"]):
            raise ValueError(f"Completed episode count mismatch: {root}")
        attempts.append(dict(run=str(root), status=manifest["status"], episodes=dict(statuses),
                             run_manifest_sha256=sha256_file(root / "run_manifest.json"),
                             error=manifest.get("error")))

    if all(arm in adapters for arm in ARMS):
        validate_matched_adapters(adapters["F0"], adapters["F1"])
    per_task = {}
    for task in sorted({key[0] for key in planned}):
        keys = [key for key in planned if key[0] == task]
        pairs = [(outcomes.get((key, "F0")), outcomes.get((key, "F1"))) for key in keys]
        observed = [f1 - f0 for f0, f1 in pairs if f0 is not None and f1 is not None]
        timing_by_arm = {}
        for arm in ARMS:
            values = [timings[key, arm] for key in keys if (key, arm) in timings]
            timing_by_arm[arm] = dict(episodes=len(values),
                median_episode_avg_policy_query_ms=median(v[0] for v in values) if values else None,
                median_rollout_wall_time_s=median(v[1] for v in values) if values else None)
        per_task[task] = dict(planned_pairs=len(pairs), complete_pairs=len(observed),
            f0_complete=sum(f0 is not None for f0, _ in pairs),
            f1_complete=sum(f1 is not None for _, f1 in pairs),
            f0_successes=sum(f0 == 1 for f0, _ in pairs), f1_successes=sum(f1 == 1 for _, f1 in pairs),
            observed_pair_delta=sum(observed) / len(observed) if observed else None, timing=timing_by_arm,
            lower_bound=sum((f1 if f1 is not None else 0) - (f0 if f0 is not None else 1)
                            for f0, f1 in pairs) / len(pairs),
            upper_bound=sum((f1 if f1 is not None else 1) - (f0 if f0 is not None else 0)
                            for f0, f1 in pairs) / len(pairs))
    complete = all(row["planned_pairs"] == row["complete_pairs"] for row in per_task.values())
    lower = sum(row["lower_bound"] for row in per_task.values()) / len(per_task)
    upper = sum(row["upper_bound"] for row in per_task.values()) / len(per_task)
    classification = ("CUSTOM_V2_LIBERO_MEM_RAW_UNSEEN" if split == "unseen"
                      else "CUSTOM_V2_LIBERO_MEM_DEVELOPMENT")
    return dict(classification=classification, split=split, seeds=seeds,
                split_lock_sha256=lock["lock_sha256"], complete_paired_roster=complete,
                adapter_manifest_sha256=adapter_hashes, loaded_policies=loaded,
                state_groups=len({key[1] for key in planned}), per_task=per_task,
                macro_delta=lower if complete else None, missing_outcome_bounds=[lower, upper],
                attempts=attempts, inference="Descriptive paired outcomes; missing-outcome bounds are not confidence intervals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--split", choices=("seen", "unseen", "val", "test"), default="seen")
    parser.add_argument("--tasks", nargs="+", default=["T1", "T3"])
    parser.add_argument("--states", required=True)
    parser.add_argument("--policy-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.policy_seeds) != len(set(args.policy_seeds)):
        parser.error("Rollout seeds must be distinct")
    _, lock, roster = evaluation_roster(args.split_lock, args.split, args.tasks, args.states)
    report = aggregate(args.runs, lock, roster, args.policy_seeds, args.split)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Paired roster complete: {report['complete_paired_roster']}. Report: {args.output}")


if __name__ == "__main__":
    main()
