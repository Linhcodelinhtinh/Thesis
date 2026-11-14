#!/usr/bin/env python
"""Compare F1 decision treatments on paired seen states, including abstentions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.evaluation.mem_eval_io import evaluation_roster, write_json
from src.evaluation.memory_supervisor import DEFAULT_DECISION_PROTOCOL, execution_treatment, load_decision_protocol
from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_episode(directory, status, arm, key, roster_row, treatment):
    if (status["task"], status["initial_state_sha256"], status["policy_seed"]) != key:
        raise ValueError(f"Episode identity differs from roster: {directory}")
    if (status["condition"] != "F1" or status["split"] != "seen" or status["decision_arm"] != arm
            or status["source_demo"] != roster_row["source_demo"]):
        raise ValueError(f"Episode metadata differs from treatment: {directory}")
    artifacts = status["files_sha256"]
    required = {"summary.json", "queries.jsonl", "actions.jsonl", "trajectory.npz", "memory_trace.jsonl"}
    if not required <= artifacts.keys():
        raise ValueError(f"Missing required episode artifacts: {directory}")
    for name in required | ({"decisions.jsonl"} & artifacts.keys()):
        if sha256_file(directory / name) != artifacts.get(name):
            raise ValueError(f"Missing or changed episode artifact {name}: {directory}")
    result = read_json(directory / "summary.json")["result"]
    metrics = result["decision_metrics"]
    if (not 1 <= metrics["queries"] <= treatment["max_policy_queries"]
            or metrics["abstained"] != (result["termination_reason"] == "decision_abstention")
            or (metrics["abstained"] and result["success"])):
        raise ValueError(f"Inconsistent decision result: {directory}")
    return dict(success=int(result["success"]), steps=result["total_steps"],
                wall_time_s=result["total_wall_time_s"], overshot=int(result["overshot"]), **metrics)


def compare(treatment, control, planned, bootstrap):
    common = sorted(treatment.keys() & control.keys())
    tasks = sorted({key[0] for key in planned})
    deltas = defaultdict(list)
    by_state = defaultdict(list)
    bounds = defaultdict(list)
    for key in planned:
        task, state, _ = key
        a, b = treatment.get(key), control.get(key)
        bounds[task].append(((a["success"] if a else 0) - (b["success"] if b else 1),
                             (a["success"] if a else 1) - (b["success"] if b else 0)))
        if a is not None and b is not None:
            delta = a["success"] - b["success"]
            deltas[task].append(delta)
            by_state[state].append((task, delta))
    observed = mean(map(mean, deltas.values())) if common else None
    interval = None
    if len(common) == len(planned) and len(by_state) > 1:
        rng = random.Random(bootstrap["seed"])
        groups = list(by_state.values())
        samples = []
        for _ in range(bootstrap["resamples"]):
            drawn = defaultdict(list)
            for group in rng.choices(groups, k=len(groups)):
                for task, delta in group:
                    drawn[task].append(delta)
            if all(task in drawn for task in tasks):
                samples.append(mean(mean(drawn[task]) for task in tasks))
        if samples:
            samples.sort()
            interval = [samples[int((len(samples) - 1) * q)] for q in (.025, .975)]
    return dict(pairs=len(common), state_groups=len(by_state), observed_task_macro_success_delta=observed,
                missing_outcome_bounds=[mean(mean(row[i] for row in bounds[task]) for task in tasks)
                                        for i in (0, 1)],
                complete_roster_state_bootstrap_95=interval,
                observed_query_delta=mean(treatment[key]["queries"] - control[key]["queries"] for key in common)
                if common else None,
                observed_policy_ms_delta=mean(treatment[key]["policy_total_ms"] - control[key]["policy_total_ms"]
                                              for key in common) if common else None)


def shadow_parity(shadow, baseline):
    def prompts(directory):
        with (directory / "queries.jsonl").open(encoding="utf-8") as rows:
            for line in rows:
                query = json.loads(line)
                yield query["step"], query["prompt"], query["chunk_sha256"]

    with (np.load(shadow / "trajectory.npz", allow_pickle=False) as a,
          np.load(baseline / "trajectory.npz", allow_pickle=False) as b):
        return (list(prompts(shadow)) == list(prompts(baseline))
                and all(np.array_equal(a[key], b[key])
                        for key in ("actions", "query_steps", "action_chunks")))


def aggregate(runs, roster, seeds, arms):
    protocol = load_decision_protocol()
    planned = {(row["task"], row["initial_state_sha256"], seed): row
               for row in roster for seed in seeds}
    if len(planned) != len(roster) * len(seeds):
        raise ValueError("Campaign contains repeated task/state/seed identities")
    complete, attempts = defaultdict(dict), []
    episode_paths = defaultdict(dict)
    provenance = None
    for run in runs:
        manifest = read_json(run / "run_manifest.json")
        args = manifest["arguments"]
        arm = execution_treatment(protocol, args.get("decision_mode"), args.get("decision_exposure")).arm
        if (arm not in arms or args["split"] != "seen" or args["conditions"] != ["F1"]
                or args["policy_seed"] not in seeds or args["runtime_check"] or args["preflight_only"]
                or manifest["status"] not in ("complete", "failed")
                or manifest["classification"] != "CUSTOM_V2_LIBERO_MEM"
                or manifest["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL)
                or manifest["decision_protocol_sha256"] != source_digest(DEFAULT_DECISION_PROTOCOL)
                or manifest["split_lock_sha256"] != protocol["split_lock_sha256"]):
            raise ValueError(f"Run differs from the declared decision campaign: {run}")
        if "F1" not in manifest.get("loaded_policies", {}):
            attempts.append(dict(run=str(run), arm=arm, status="failed_before_policy_load"))
            continue
        adapter = sha256_file(run / "adapter_manifest_F1.json")
        if adapter != manifest["treatments"]["F1"]["adapter_manifest_sha256"]:
            raise ValueError(f"Adapter receipt hash differs: {run}")
        identity = dict(adapter=adapter, base=manifest["base"], loaded=manifest["loaded_policies"]["F1"],
                        assets=manifest["local_assets_lock_sha256"], runtime=manifest["environments"],
                        cuda=manifest["cuda"], source=manifest["source_sha256"],
                        video=args["save_video"], streaming=args["stream_memory_trace"])
        if provenance is None:
            provenance = identity
        elif provenance != identity:
            raise ValueError(f"Checkpoint, runtime, source or logging differs: {run}")
        if len(manifest["roster"]) != manifest["expected_episodes"]:
            raise ValueError(f"Unexpected batch size: {run}")
        visited = set()
        for row in manifest["roster"]:
            key = (row["task"], row["initial_state_sha256"], args["policy_seed"])
            if key in visited or planned.get(key) != row:
                raise ValueError(f"Unexpected roster row in {run}")
            visited.add(key)
            directory = run / "episodes" / f"{row['task']}_F1_seen_state{row['state_idx']}"
            status_path = directory / "status.json"
            status = read_json(status_path) if status_path.is_file() else {"status": "missing"}
            attempts.append(dict(run=str(run), arm=arm, unit=key, status=status["status"]))
            if status["status"] != "complete":
                continue
            if key in complete[arm]:
                raise ValueError(f"Repeated complete unit for {arm}: {key}")
            complete[arm][key] = validate_episode(directory, status, arm, key, row, protocol)
            episode_paths[arm][key] = directory
    metrics, paired = {}, {}
    for arm in arms:
        rows = complete[arm]
        metrics[arm] = dict(completed=len(rows), planned=len(planned),
                            means={key: mean(row[key] for row in rows.values())
                                   for key in next(iter(rows.values()), {})},
                            by_task={})
        for task in sorted({key[0] for key in planned}):
            successes = [row["success"] for key, row in rows.items() if key[0] == task]
            n = sum(key[0] == task for key in planned)
            metrics[arm]["by_task"][task] = dict(
                bounds=[sum(successes) / n, (sum(successes) + n - len(successes)) / n],
                observed_success=mean(successes) if successes else None)
        if arm != "text":
            paired[arm] = compare(rows, complete["text"], planned, protocol["descriptive_bootstrap"])
    comparisons = {arm: compare(complete[arm], complete["periodic"], planned, protocol["descriptive_bootstrap"])
                   for arm in arms if arm.startswith("event-") and "periodic" in arms}
    parity = {}
    for arm in arms:
        if arm.startswith("shadow-"):
            baseline, shadow = episode_paths["text"], episode_paths[arm]
            common = sorted(baseline.keys() & shadow.keys())
            differences = [key for key in common if not shadow_parity(shadow[key], baseline[key])]
            parity[arm] = dict(pairs=len(common), identical=len(common) - len(differences),
                               different_units=differences)
    return dict(classification="DECISION_DEVELOPMENT_CUSTOM_V2", arms=metrics, versus_text=paired,
                versus_periodic=comparisons, shadow_parity=parity,
                attempts=attempts, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    parser.add_argument("--tasks", nargs="+", choices=("T1", "T3"), default=["T1", "T3"])
    parser.add_argument("--states", default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--arms", nargs="+", default=["text", "shadow-projected", "event-projected", "periodic"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.arms)) != len(args.arms) or "text" not in args.arms:
        parser.error("Choose a new output and distinct arms including text")
    _, lock, roster = evaluation_roster(args.split_lock, "seen", args.tasks, args.states)
    if lock["lock_sha256"] != load_decision_protocol()["split_lock_sha256"]:
        raise ValueError("Use the decision protocol's seen state lock")
    write_json(args.output, aggregate(args.runs, roster, args.seeds, args.arms))


if __name__ == "__main__":
    main()
