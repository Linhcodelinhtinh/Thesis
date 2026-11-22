#!/usr/bin/env python
"""Offline counterfactual: when would memory_halt_v1 have fired in a finished memory-on run?

Reads only recorded artifacts of each episode (memory_trace.jsonl, observations.jsonl,
queries.jsonl). The halt rule is evaluated on the recorded causal writer snapshot and
measured gripper aperture at every recorded query boundary, exactly as the online
wrapper would see them. Upstream progress (eval_result) is read only to classify the
timing afterwards; it never feeds the rule.

This is a timing diagnostic, NOT a success estimate: after a real halt the trajectory
diverges from the recording.

Categories per episode
  halt_before_completion   rule fired before the upstream milestone count reached target
                           (benign if the bowl then rests on the plate; harmful if the
                           writer over-counted)
  halt_before_overshoot    fired after completion and before the irreversible overshoot
  halt_after_overshoot     fired too late
  no_halt                  rule never fired (writer under-count or relation never confirmed)
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.memory_halt import HaltConfig, evaluate_halt


def snapshot_from_dict(raw):
    relations = {key: SimpleNamespace(returns=int(v["returns"]), status=v["status"], reference=v["reference"],
                                      armed=bool(v["armed"]))
                 for key, v in raw["relations"].items()}
    return SimpleNamespace(as_of_observation=int(raw["as_of_observation"]),
                           profile=SimpleNamespace(count_cycles=bool(raw["profile"]["count_cycles"])),
                           relations=relations)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay_episode(directory, config_kwargs):
    directory = Path(directory)
    observations = {row["step"]: row for row in read_jsonl(directory / "observations.jsonl")}
    first = next(iter(observations.values()))["info"]["eval_result"]
    target = int(first["target_cycles"])
    config = HaltConfig(target, **config_kwargs)
    snapshots = {}
    with (directory / "memory_trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            snapshots[row["step"]] = row["snapshot"]
    query_steps = [row["step"] for row in read_jsonl(directory / "queries.jsonl")]
    completion = overshoot = None
    for step in sorted(observations):
        result = observations[step]["info"].get("eval_result") or {}
        if completion is None and result.get("completed_subgoals", 0) >= target:
            completion = step
        if overshoot is None and result.get("overshot"):
            overshoot = step
    streak, halt_step, first_count_step, trace = 0, None, None, []
    for step in query_steps:
        if step not in snapshots or step not in observations:
            raise ValueError(f"{directory.name}: missing snapshot/observation at query step {step}")
        grip = observations[step]["raw_proprioception_pos_quatxyzw_gripper"][7:9]
        record = evaluate_halt(snapshot_from_dict(snapshots[step]), {"robot0_gripper_qpos": grip}, config)
        if first_count_step is None and (record["observed_returns"] or 0) >= target:
            first_count_step = step
        streak = streak + 1 if record["halt_condition"] else 0
        trace.append(dict(step=step, reason=record["reason"], returns=record["observed_returns"]))
        if streak >= config.confirm_queries:
            halt_step = step
            break
    max_returns = max((rel["returns"] for snap in snapshots.values() for rel in snap["relations"].values()),
                      default=0)
    if halt_step is None:
        category = "no_halt"
    elif completion is None or halt_step < completion:
        category = "halt_before_completion"
    elif overshoot is None or halt_step < overshoot:
        category = "halt_before_overshoot"
    else:
        category = "halt_after_overshoot"
    return dict(episode=directory.name, target_cycles=target, completion_step=completion,
                overshoot_step=overshoot, halt_step=halt_step, first_query_with_target_count=first_count_step,
                writer_max_returns=max_returns, category=category,
                margin_to_overshoot=(overshoot - halt_step if halt_step is not None and overshoot is not None else None),
                last_reasons=trace[-3:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="Run directories containing episodes/")
    p.add_argument("--confirm-queries", type=int, default=1)
    p.add_argument("--no-support-check", action="store_true")
    p.add_argument("--allow-closed-gripper", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    kwargs = dict(confirm_queries=a.confirm_queries, require_on_support=not a.no_support_check,
                  require_gripper_open=not a.allow_closed_gripper)
    rows = []
    for run in a.runs:
        for episode in sorted((run / "episodes").iterdir()):
            if not (episode / "memory_trace.jsonl").is_file():
                continue
            row = replay_episode(episode, kwargs)
            row["run"] = run.name
            rows.append(row)
    summary = {}
    for row in rows:
        task = row["episode"].split("_", 1)[0]
        bucket = summary.setdefault(task, {})
        bucket[row["category"]] = bucket.get(row["category"], 0) + 1
    report = dict(classification="OFFLINE_HALT_TIMING_NOT_SUCCESS", rule=HaltConfig(1, **kwargs).to_dict(),
                  episodes=len(rows), by_task=summary, rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(episodes=len(rows), by_task=summary), indent=2))


if __name__ == "__main__":
    main()
