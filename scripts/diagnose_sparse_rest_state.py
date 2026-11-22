"""Why does the sparse writer spend most frames in dense mode? Count the first failing rest condition.

The sparse writer only skips frames while the dense writer is "at rest". This replays the dense V3
writer on the archived evidence and, for every frame, records the first condition of that rest
test that fails, so the share of frames that could ever be skipped is explicit. It reads only.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.compare_memory_writers import (CONVENTION, PROFILE, SYNTHETIC, RecordedDetector, development_jobs,
                                            parse_evidence, rollout_jobs, synthetic_observation)
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_sparse_config import SparseWriterConfig

CONDITIONS = (
    ("prefix", lambda step, reducer, config: step < config.dense_prefix_steps),
    ("no_reducer", lambda step, reducer, config: reducer is None),
    ("status_not_confirmed", lambda step, reducer, config: reducer.status != "confirmed"),
    ("no_reference", lambda step, reducer, config: reducer.reference is None),
    ("not_ready_or_no_origin", lambda step, reducer, config: not reducer.ready),
    ("departure_active", lambda step, reducer, config: reducer.departure is not None),
    ("pending_lift", lambda step, reducer, config: bool(reducer.pending)),
    ("gap_active", lambda step, reducer, config: bool(reducer.gap_active)),
)


def classify(evidence, grip, episode, config):
    fed = {step: replace(item, image_hashes=SYNTHETIC[step % 256]) for step, item in enumerate(evidence)}
    writer = BowlCycleWriterV3(episode, PROFILE, RecordedDetector(fed), CONVENTION, config=BowlCycleConfig.load())
    counts = Counter()
    for step in range(len(evidence)):
        writer.step(step, synthetic_observation(step, grip[step]), None if step == 0 else np.zeros(7))
        reducer = writer.reducers.get("bowl_1")
        label = next((name for name, failed in CONDITIONS if failed(step, reducer, config)), "at_rest")
        if label == "status_not_confirmed":
            # Separate frames the engaged-gripper rule already covers from open-gripper frames whose
            # relation was merely not confirmed (detection gaps, unsettled or candidate states).
            engaged = float(np.abs(np.asarray(grip[step], dtype=float)).sum()) < config.engaged_aperture_meters
            label += ":gripper_engaged" if engaged else f":gripper_open:{reducer.status}"
        counts[label] += 1
    return counts


def run(jobs):
    config = SparseWriterConfig.load()
    totals, by_task = Counter(), {}
    for episode, task, payloads, grip, _ in jobs:
        counts = classify(parse_evidence(payloads), grip, episode, config)
        totals.update(counts)
        by_task.setdefault(task, Counter()).update(counts)
    frames = sum(totals.values())
    share = lambda counter: {name: round(count / sum(counter.values()), 4) for name, count in sorted(counter.items())}
    return dict(frames=frames, share=share(totals), by_task={task: share(c) for task, c in sorted(by_task.items())},
                counts=dict(totals))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--development-index", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ["v3"]
    development = list(development_jobs(args.development_index, args.inputs, names))[:-1]
    e2 = list(rollout_jobs(args.inputs, names))[:-1]
    report = dict(classification="RECORDED_WRITER_REPLAY_NON_COMPARABLE",
                  note="First failing condition of the sparse writer's rest test, per frame, under the dense V3 writer",
                  development=run(development), e2=run(e2))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key]["share"] for key in ("development", "e2")}, indent=2))


if __name__ == "__main__":
    main()
