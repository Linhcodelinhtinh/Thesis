"""Compare return-event recognition on recorded detections and gripper aperture."""
import argparse
import json
from pathlib import Path
import sys
from itertools import groupby
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audit_libero_mem_writer import match_events
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.models import EvidenceSource, V2MemoryEvent
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer import BowlCycleReducer, BowlCycleWriter
from src.memory.relation_cycle_writer_v3 import BowlCycleReducerV3, BowlCycleWriterV3
from src.memory.relation_writer_versions import versioned_relation_source_hashes
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.relation_cycle_tracking import BowlCycleTracker
from src.perception.visual_evidence import CAMERAS, VisualEvidence
from src.utils.file_hash import sha256_file


PROFILE = MemoryProfile("bowl", 1, "plate", 3, "on", count_cycles=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def check_hash(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"Source checksum mismatch: {path}")


def replay(episode, frames, writer_version=BowlCycleWriter.VERSION, config=None):
    """Use the selected live association and reducer on archived inputs."""
    config = config or (BowlCycleConfig.load() if writer_version == BowlCycleWriterV3.VERSION else
                        BowlCycleConfig())
    writer_v3 = writer_version == BowlCycleWriterV3.VERSION
    if writer_version not in (BowlCycleWriter.VERSION, BowlCycleWriterV3.VERSION):
        raise ValueError(f"Unsupported cycle writer for replay: {writer_version}")
    tracker = BowlCycleTracker(PROFILE, config=config)
    reducer = BowlCycleReducerV3(config) if writer_v3 else BowlCycleReducer(config)
    startup_open = True
    previous_step = -1
    returns, serial, length = [], 0, 0
    for step, (payload, grip) in enumerate(frames):
        evidence = payload if isinstance(payload, VisualEvidence) else VisualEvidence.from_dict(payload)
        if (evidence.step != step or evidence.source != EvidenceSource.PREDICTED_RGB_TRACKER
                or (evidence.model_id, evidence.revision) != (MODEL_ID, REVISION)
                or set(evidence.views) != set(CAMERAS) or set(evidence.image_hashes) != set(CAMERAS)):
            raise ValueError(f"Invalid evidence at {episode}/{step}")
        grip = np.asarray(grip, dtype=float)
        if grip.shape != (2,) or not np.isfinite(grip).all():
            raise ValueError(f"Invalid gripper state at {episode}/{step}")
        aperture = float(np.abs(grip).sum())
        if step != previous_step + 1 or aperture < config.released_aperture_meters:
            startup_open = False
        tracker.allow_late_anchors = writer_v3 and startup_open
        if writer_v3:
            reducer.allow_late_origin = startup_open
        tracks = tracker.observe(evidence)
        status, reference, ids = reducer.measure(step, tracks["bowl_1"], tracks["plate_1"],
                                                 aperture, tracker.wrist_near)

        def emit(kind, ref, start, evidence_ids):
            nonlocal serial
            serial += 1
            if kind == "return":
                returns.append(step)
            return V2MemoryEvent(f"{episode}:{serial}", kind, start, step, evidence.source,
                                 evidence_ids=tuple(dict.fromkeys(evidence_ids)), details={"reference": ref})

        reducer.update(step, status, reference, ids, emit)
        if writer_v3 and (
            tracker.anchors.get("bowl_1") is None or tracker.previous.get("bowl_1") is None
        ):
            startup_open = False
        length = step + 1
        previous_step = step
    return returns, length


def rollout_inputs(trace, observations):
    for step, (frame, observation) in enumerate(zip(rows(trace), rows(observations), strict=True)):
        if frame["step"] != step or observation["step"] != step:
            raise ValueError(f"Non-contiguous steps in {trace}")
        yield frame["signals"]["visual_evidence"], observation["raw_proprioception_pos_quatxyzw_gripper"][7:9]


def rollout_episodes(root, writer_version=BowlCycleWriter.VERSION, config=None):
    directories = sorted(root.glob("**/episodes/T[13]_F1_unseen_state*"))
    if not directories:
        raise ValueError(f"No T1/T3 F1 unseen traces under {root}")
    for directory in directories:
        status_path = directory / "status.json"
        status = read(status_path)
        if status["status"] != "complete":
            continue
        if (status["episode_id"] != directory.name or status["condition"] != "F1"
                or status["split"] != "unseen" or status["task"] not in ("T1", "T3")):
            raise ValueError(f"Unexpected episode identity: {directory}")
        trace, observations = (directory / name for name in ("memory_trace.jsonl", "observations.jsonl"))
        for path in (trace, observations):
            check_hash(path, status["files_sha256"][path.name])
        proposed, length = replay(directory.name, rollout_inputs(trace, observations), writer_version, config)
        previous, baseline, milestones = 0, [], []
        for step, (frame, observation) in enumerate(zip(rows(trace), rows(observations), strict=True)):
            baseline.extend(step for event in frame["signals"]["new_events"] if event["type"] == "return")
            count = observation["info"]["eval_result"]["confirmed_cycles"]
            if count not in (previous, previous + 1):
                raise ValueError(f"Invalid benchmark progress in {directory}")
            if count > previous:
                milestones.append(step)
            previous = count
        yield dict(episode=directory.name, task=status["task"], frames=length, milestones=milestones,
                   baseline=baseline, proposed=proposed, source_receipt_sha256=sha256_file(status_path))


def development_episodes(index, data_dir, writer_version=BowlCycleWriter.VERSION, config=None,
                         frame_cache=None):
    import h5py
    import ijson
    from src.training.memory_data import validate_episode

    manifest_path, trace, detections = (index / name for name in
                                       ("manifest.json", "memory_trace.jsonl", "detections.jsonl"))
    manifest = read(manifest_path)
    if (manifest["status"] != "complete" or manifest["writer"] != "rgb_relations_v1"
            or manifest["source_kind"] != "author_refined"):
        raise ValueError(f"Expected a complete author-refined relation index: {index}")
    for path in (trace, detections):
        check_hash(path, manifest["artifacts_sha256"][path.name])
    expected = {row["id"]: row["frames"] for row in manifest["episodes"]}
    task, _ = next(iter(expected)).rsplit("_demo_", 1)
    task_alias = "T" + task.split("_")[2]
    if task_alias not in ("T1", "T3"):
        raise ValueError("Development replay requires a T1/T3 index")
    hdf5 = data_dir / (task + "_demo.hdf5")
    check_hash(hdf5, manifest["sources"][hdf5.name])
    predictions = {}
    with h5py.File(hdf5, "r") as source:
        for episode, records in groupby(rows(detections), key=lambda row: row["episode_id"]):
            episode_task, demo = episode.rsplit("_demo_", 1)
            if episode in predictions or episode_task != task or episode not in expected:
                raise ValueError(f"Unexpected detection episode: {episode}")
            _, robot, _ = validate_episode(source[f"data/demo_{demo}"])
            if frame_cache is None:
                frames = ((row["evidence"], grip)
                          for row, grip in zip(records, robot[:, :2], strict=True))
            else:
                frames = [(VisualEvidence.from_dict(row["evidence"]), grip)
                          for row, grip in zip(records, robot[:, :2], strict=True)]
                frame_cache[episode] = frames
            proposed, length = replay(episode, frames, writer_version, config)
            if length != expected[episode]:
                raise ValueError(f"Incorrect frame count: {episode}")
            predictions[episode] = proposed
    if predictions.keys() != expected.keys():
        raise ValueError("Incomplete detection roster")

    baseline = {}
    for episode, records in groupby(rows(trace), key=lambda row: row["episode_id"]):
        if episode in baseline or episode not in expected:
            raise ValueError(f"Unexpected baseline episode: {episode}")
        baseline[episode] = [row["step"] for row in records for event in row["signals"]["new_events"]
                             if event["type"] == "return"]
    metainfo = data_dir / "metainfo.json"
    check_hash(metainfo, manifest["sources"][metainfo.name])
    reference = {}
    vocabulary = {"0": 0, **{f"{name}: bowl 1 on plate 1": i
                             for i, name in enumerate(("one", "two", "three"), 1)}}
    with metainfo.open("rb") as stream:
        for demo, labels in ijson.kvitems(stream, task, use_float=True):
            episode = f"{task}_{demo}"
            if episode not in expected:
                continue
            counts = [vocabulary[str(frame["bowl 1"][2])] for frame in labels["exo_boxes"]]
            if len(counts) != expected[episode] or any(b not in (a, a + 1) for a, b in zip(counts, counts[1:])):
                raise ValueError(f"Unexpected milestone sequence: {episode}")
            reference[episode] = [i for i, (a, b) in enumerate(zip(counts, counts[1:]), 1) if b > a]
    if reference.keys() != baseline.keys() or reference.keys() != expected.keys():
        raise ValueError("Incomplete baseline or author milestone roster")
    receipt = sha256_file(manifest_path)
    for episode, length in expected.items():
        yield dict(episode=episode, task=task_alias, frames=length, milestones=reference[episode],
                   baseline=baseline[episode], proposed=predictions[episode], source_receipt_sha256=receipt)


def score(episodes, tolerance):
    summaries = {}
    for row in episodes:
        for arm in ("baseline", "proposed"):
            row[arm + "_match"] = match_events(row[arm], row["milestones"], tolerance)
    for task in ("T1", "T3", "all"):
        selected = [row for row in episodes if task in (row["task"], "all")]
        if not selected:
            continue
        summaries[task] = {"episodes": len(selected), "reference_events": sum(len(row["milestones"]) for row in selected)}
        for arm in ("baseline", "proposed"):
            matches = [row[arm + "_match"] for row in selected]
            totals = {key: sum(match[key] for match in matches)
                      for key in ("true_positive", "false_positive", "false_negative")}
            delays = [pair["delay"] for match in matches for pair in match["matches"]]
            summaries[task][arm] = dict(
                totals, false_positive_per_episode=totals["false_positive"] / len(selected),
                min_delay=min(delays, default=None), median_delay=median(delays) if delays else None,
                max_delay=max(delays, default=None),
            )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--rollout-root", type=Path)
    sources.add_argument("--development-index", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--tolerance", type=int, default=20)
    parser.add_argument("--writer", choices=(BowlCycleWriter.VERSION, BowlCycleWriterV3.VERSION),
                        default=BowlCycleWriter.VERSION)
    parser.add_argument("--cycle-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("Tolerance must be nonnegative")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    config = (BowlCycleConfig.load(args.cycle_config) if args.cycle_config else
              BowlCycleConfig.load() if args.writer == BowlCycleWriterV3.VERSION else BowlCycleConfig())
    if args.rollout_root:
        episodes = list(rollout_episodes(args.rollout_root, args.writer, config))
    else:
        episodes = [row for path in args.development_index for row in
                    development_episodes(path, args.data_dir, args.writer, config)]
    if not episodes or len({row["episode"] for row in episodes}) != len(episodes):
        raise ValueError("Replay requires a nonempty, unique episode roster")
    summary = score(episodes, args.tolerance)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(dict(classification="RECORDED_WRITER_REPLAY_NON_COMPARABLE",
                           tolerance=args.tolerance, writer=args.writer, summary=summary, episodes=episodes,
                           writer_config=config.to_dict(),
                           writer_code_sha256=versioned_relation_source_hashes(ROOT, args.writer)),
                      stream, indent=2)
            stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
