"""Extract the compact inputs that a recorded-detection writer replay needs.

A replay of a memory writer needs, per frame, the recorded detector evidence and the measured
gripper state. The development indexes keep the evidence (``detections.jsonl``) but not the
gripper state, which lives in the author HDF5 demos; the F1 rollouts keep both in two large JSONL
files. This script reads those large sources once, streams them, and writes small files that can
be copied around. It reads only; it never changes a source.

Development gripper state is cross-checked against every ``gripper:<step>`` record the writer
retained in ``memory_trace.jsonl``, so a mismatch between the HDF5 and the run that produced the
detections fails loudly instead of silently shifting the replay.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def aperture(qpos):
    """The aperture the writers use: sum of the absolute finger positions, in float64."""
    return float(np.abs(np.asarray(qpos, dtype=float)).sum())


def extract_development(index, data_dir, verify_hdf5=False):
    """Return (episodes, report) for one development index of the T1 or T3 task."""
    import h5py

    index, data_dir = Path(index), Path(data_dir)
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete development index: {index}")
    expected = {row["id"]: row["frames"] for row in manifest["episodes"]}
    task = next(iter(expected)).rsplit("_demo_", 1)[0]
    alias = "T" + task.split("_")[2]
    hdf5 = data_dir / (task + "_demo.hdf5")
    if verify_hdf5 and sha256_file(hdf5) != manifest["sources"][hdf5.name]:
        raise ValueError(f"HDF5 checksum differs from the index manifest: {hdf5}")
    episodes = {}
    with h5py.File(hdf5, "r") as source:
        for episode, frames in expected.items():
            name, demo = episode.rsplit("_demo_", 1)
            if name != task:
                raise ValueError(f"Episode {episode} does not belong to {task}")
            full = np.asarray(source[f"data/demo_{demo}/robot_states"][:], dtype=np.float32)
            if full.shape != (frames, 9) or not np.isfinite(full).all():
                raise ValueError(f"Unexpected robot state for {episode}: {full.shape}, expected {frames} frames")
            # robot_states = [gripper(2), eef_pos(3), eef_quat(4)]: the quaternion must be unit length.
            if not np.allclose(np.linalg.norm(full[:, 5:9], axis=1), 1, atol=1e-4):
                raise ValueError(f"robot_states layout differs from [gripper, eef_pos, eef_quat]: {episode}")
            robot = full[:, :2]
            episodes[episode] = dict(task=alias, frames=frames, gripper_qpos=robot.tolist(),
                                     eef_pos=full[:, 2:5].tolist())
    checked = mismatched = 0
    worst = 0.0
    seen = set()
    for row in rows(index / "memory_trace.jsonl"):
        episode, step = row["episode_id"], row["step"]
        if episode not in episodes:
            raise ValueError(f"Trace episode is not in the manifest: {episode}")
        seen.add(episode)
        record = row["snapshot"]["evidence"].get(f"gripper:{step}")
        if record is None:
            continue
        checked += 1
        difference = abs(record["aperture"] - aperture(episodes[episode]["gripper_qpos"][step]))
        worst = max(worst, difference)
        mismatched += difference > 1e-9
    if seen != set(expected):
        raise ValueError("The trace does not cover the manifest roster")
    if mismatched:
        raise ValueError(f"{mismatched}/{checked} retained apertures differ from the HDF5 in {index}; "
                         f"worst difference {worst:.3g}")
    report = dict(index=str(index), task=alias, episodes=len(episodes),
                  frames=sum(expected.values()), apertures_cross_checked=checked,
                  worst_aperture_difference=worst, manifest_sha256=sha256_file(index / "manifest.json"),
                  hdf5=hdf5.name, hdf5_sha256_verified=bool(verify_hdf5))
    return episodes, report


def extract_rollouts(root, output):
    """Write one JSON line per complete F1 episode: evidence, gripper state and reference counts."""
    root, output = Path(root), Path(output)
    directories = sorted(root.glob("**/episodes/T[13]_F1_unseen_state*"))
    if not directories:
        raise ValueError(f"No T1/T3 F1 unseen traces under {root}")
    written, frames = [], 0
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for directory in directories:
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "complete":
                continue
            if (status["episode_id"] != directory.name or status["condition"] != "F1"
                    or status["split"] != "unseen" or status["task"] not in ("T1", "T3")):
                raise ValueError(f"Unexpected episode identity: {directory}")
            trace, observations = directory / "memory_trace.jsonl", directory / "observations.jsonl"
            for path in (trace, observations):
                if sha256_file(path) != status["files_sha256"][path.name]:
                    raise ValueError(f"Source checksum mismatch: {path}")
            evidence, recorded, milestones, grip, eef = [], [], [], [], []
            previous = 0
            for step, (frame, observation) in enumerate(zip(rows(trace), rows(observations), strict=True)):
                if frame["step"] != step or observation["step"] != step:
                    raise ValueError(f"Non-contiguous steps in {directory}")
                evidence.append(frame["signals"]["visual_evidence"])
                recorded.extend(step for event in frame["signals"]["new_events"] if event["type"] == "return")
                proprio = observation["raw_proprioception_pos_quatxyzw_gripper"]
                if len(proprio) != 9 or abs(sum(q * q for q in proprio[3:7]) - 1) > 1e-3:
                    raise ValueError(f"Unexpected proprioception layout in {directory}")
                grip.append(proprio[7:9])
                eef.append(proprio[:3])
                count = observation["info"]["eval_result"]["confirmed_cycles"]
                if count not in (previous, previous + 1):
                    raise ValueError(f"Invalid benchmark progress in {directory}")
                if count > previous:
                    milestones.append(step)
                previous = count
            stream.write(json.dumps(dict(
                episode=directory.name, task=status["task"], frames=len(grip), gripper_qpos=grip, eef_pos=eef,
                evidence=evidence, recorded_return_steps=recorded, milestones=milestones,
                source_receipt_sha256=sha256_file(directory / "status.json"))) + "\n")
            written.append(directory.name)
            frames += len(grip)
    return dict(root=str(root), episodes=len(written), frames=frames, file=output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-index", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-hdf5-sha256", action="store_true",
                        help="Also hash the multi-GB HDF5 files against the index manifest")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(development=[], rollouts=None)
    gripper = {}
    for index in args.development_index:
        episodes, entry = extract_development(index, args.data_dir, args.verify_hdf5_sha256)
        if gripper.keys() & episodes.keys():
            raise ValueError("Development indexes overlap")
        gripper.update(episodes)
        report["development"].append(entry)
        print(json.dumps(entry), flush=True)
    with (args.output_dir / "development_gripper.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(classification="REPLAY_INPUTS_NOT_A_RESULT", episodes=gripper), stream)
    if args.rollout_root:
        report["rollouts"] = extract_rollouts(args.rollout_root, args.output_dir / "rollout_replay_inputs.jsonl.gz")
        print(json.dumps(report["rollouts"]), flush=True)
    with (args.output_dir / "extraction_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("done", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
