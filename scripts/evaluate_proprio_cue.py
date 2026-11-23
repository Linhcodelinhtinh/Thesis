"""Replay the proprioceptive touchdown cue on the development demonstrations and the E2 rollouts.

Reads the replay inputs (gripper aperture and end-effector position, no detector), the author milestones, and the
episode event steps of a writer comparison, so the cue can be scored alone and fused with a vision writer's returns.
It reads only and writes one new JSON file.
"""
import argparse
from dataclasses import replace
import gzip
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audit_libero_mem_writer import match_events
from scripts.compare_memory_writers import bootstrap
from scripts.extract_writer_replay_inputs import sha256_file
from src.memory.proprio_cycle_cue import ProprioCueConfig, fuse_returns, touchdown_steps

TOLERANCE = 20


def load_development(inputs, indexes):
    data = json.loads((inputs / "development_gripper.json").read_text(encoding="utf-8"))
    if data["classification"] != "REPLAY_INPUTS_NOT_A_RESULT":
        raise ValueError("Unexpected development input file")
    report = json.loads((inputs / "extraction_report.json").read_text(encoding="utf-8"))
    if not all(row["hdf5_sha256_verified"] for row in report["development"]):
        raise ValueError("The extraction did not verify the demonstration HDF5 files")
    episodes = {}
    for index in map(Path, indexes):
        for row in json.loads((index / "milestone-audit.json").read_text(encoding="utf-8"))["episodes"]:
            stored = data["episodes"][row["episode"]]
            if "eef_pos" not in stored:
                raise ValueError("These replay inputs carry no end-effector position")
            episodes[row["episode"]] = dict(task=stored["task"], milestones=row["milestones"],
                                            aperture=np.abs(np.asarray(stored["gripper_qpos"], dtype=float)).sum(axis=1),
                                            eef=np.asarray(stored["eef_pos"], dtype=float))
    return episodes


def load_rollouts(inputs):
    episodes = {}
    with gzip.open(inputs / "rollout_replay_inputs.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            episodes[row["episode"]] = dict(task=row["task"], milestones=row["milestones"],
                                            aperture=np.abs(np.asarray(row["gripper_qpos"], dtype=float)).sum(axis=1),
                                            eef=np.asarray(row["eef_pos"], dtype=float))
    return episodes


def score(episodes, events_of):
    """Counts [tp, fp, fn] per episode, overall and by task."""
    counts = {}
    for name, row in episodes.items():
        match = match_events(sorted(events_of(name, row)), row["milestones"], TOLERANCE)
        counts[name] = [match["true_positive"], match["false_positive"], match["false_negative"]]
    by_task = {}
    for name, row in episodes.items():
        by_task.setdefault(row["task"], []).append(counts[name])
    summarize = lambda rows: dict(zip(("tp", "fp", "fn"), map(int, np.sum(rows, axis=0))),
                                  ci95=bootstrap(rows), episodes=len(rows))
    return dict(all=summarize(list(counts.values())), **{task: summarize(rows) for task, rows in sorted(by_task.items())})


def lift_cycles(episodes, config):
    """Closed-gripper phases that lifted at least the configured rise, to size the negatives."""
    count = 0
    for row in episodes.values():
        closed_from = None
        for step, (aperture, position) in enumerate(zip(row["aperture"], row["eef"])):
            if closed_from is None and aperture <= config.closed_aperture_meters:
                closed_from, base, peak = step, position[2], position[2]
            elif closed_from is not None:
                if aperture >= config.released_aperture_meters:
                    count += int(peak - base >= config.lift_meters)
                    closed_from = None
                else:
                    base = min(base, position[2]) if step - closed_from < config.grasp_height_frames else base
                    peak = max(peak, position[2])
    return count


def evaluate(episodes, vision, config):
    cue = {name: touchdown_steps(row["aperture"], row["eef"], config) for name, row in episodes.items()}
    result = dict(lift_cycles_at_least_lift=lift_cycles(episodes, config),
                  proprio=score(episodes, lambda n, r: cue[n]))
    for writer, returns in vision.items():
        result[writer] = score(episodes, lambda n, r: returns[n])
        result[f"{writer}+proprio"] = score(episodes, lambda n, r: fuse_returns(returns[n], cue[n]))
    return result


def sensitivity(episodes, base):
    grid = {}
    for lift, height, xy in itertools.product((.05, .08, .1), (.02, .03, .04), (.04, .06, .1)):
        config = replace(base, lift_meters=lift, touchdown_height_meters=height, touchdown_xy_meters=xy)
        cue = {name: touchdown_steps(row["aperture"], row["eef"], config) for name, row in episodes.items()}
        by_task = score(episodes, lambda n, r: cue[n])
        grid[f"lift={lift},height={height},xy={xy}"] = {key: {k: value[k] for k in ("tp", "fp", "fn")}
                                                        for key, value in by_task.items()}
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--development-index", type=Path, nargs="+", required=True)
    parser.add_argument("--comparison-episodes", type=Path, required=True, help="episodes.json.gz of a writer comparison")
    parser.add_argument("--writers", nargs="+", default=["v3", "v4"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = ProprioCueConfig.load()
    comparison = json.loads(gzip.open(args.comparison_episodes, "rt", encoding="utf-8").read())["episodes"]
    sets = dict(development=load_development(args.inputs, args.development_index), e2=load_rollouts(args.inputs))
    report = dict(classification="RECORDED_REPLAY_NON_COMPARABLE", config=config.to_dict(), tolerance=TOLERANCE,
                  inputs=dict(comparison_episodes_sha256=sha256_file(args.comparison_episodes),
                              **{name: sha256_file(args.inputs / name) for name in
                                 ("development_gripper.json", "rollout_replay_inputs.jsonl.gz", "extraction_report.json")}),
                  note="Thresholds were set while looking at the development demonstrations, so only E2 is held out; "
                       "E2 has 3 author milestones, so it checks false positives, not recall")
    for name, episodes in sets.items():
        vision = {writer: {e: comparison[name][writer][e]["returns"] for e in episodes} for writer in args.writers}
        report[name] = evaluate(episodes, vision, config)
        report[name]["sensitivity"] = sensitivity(episodes, config)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for name in sets:
        print(name, {key: value["all"] for key, value in report[name].items() if isinstance(value, dict) and "all" in value})


if __name__ == "__main__":
    main()
