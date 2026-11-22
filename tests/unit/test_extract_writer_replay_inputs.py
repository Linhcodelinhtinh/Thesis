"""The replay-input extractor, on tiny synthetic sources (unit fixtures, not benchmark data)."""
import gzip
import json

import numpy as np
import pytest

from scripts.extract_writer_replay_inputs import (aperture, extract_development, extract_rollouts, main,
                                                  sha256_file)

h5py = pytest.importorskip("h5py")
TASK = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
FRAMES = {f"{TASK}_demo_1": 5, f"{TASK}_demo_2": 4}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def make_index(tmp_path, shift=0.0):
    index, data = tmp_path / "index-T1", tmp_path / "data"
    index.mkdir()
    data.mkdir()
    rng = np.random.default_rng(0)
    qpos = {}
    with h5py.File(data / f"{TASK}_demo.hdf5", "w") as target:
        for episode, frames in FRAMES.items():
            robot = rng.uniform(-.04, .04, size=(frames, 9)).astype(np.float32)
            target[f"data/demo_{episode.rsplit('_demo_', 1)[1]}/robot_states"] = robot
            qpos[episode] = robot[:, :2]
    (index / "manifest.json").write_text(json.dumps(dict(
        status="complete", episodes=[dict(id=episode, frames=frames) for episode, frames in FRAMES.items()],
        sources={})), encoding="utf-8")
    trace = []
    for episode, frames in FRAMES.items():
        for step in range(frames):
            # Like the writer, keep the aperture of only some frames.
            evidence = {f"gripper:{step}": dict(step=step, aperture=aperture(qpos[episode][step]) + shift)} \
                if step % 2 == 0 else {}
            trace.append(dict(episode_id=episode, step=step, snapshot=dict(evidence=evidence)))
    write_jsonl(index / "memory_trace.jsonl", trace)
    return index, data, qpos


def test_development_gripper_state_is_extracted_and_cross_checked(tmp_path):
    index, data, qpos = make_index(tmp_path)
    episodes, report = extract_development(index, data)
    assert set(episodes) == set(FRAMES) and episodes[f"{TASK}_demo_1"]["task"] == "T1"
    np.testing.assert_array_equal(np.array(episodes[f"{TASK}_demo_2"]["gripper_qpos"], dtype=np.float32),
                                  qpos[f"{TASK}_demo_2"])
    assert report["apertures_cross_checked"] == 3 + 2 and report["worst_aperture_difference"] == 0.0


def test_a_trace_from_a_different_run_fails_loudly(tmp_path):
    index, data, _ = make_index(tmp_path, shift=1e-3)
    with pytest.raises(ValueError, match="differ from the HDF5"):
        extract_development(index, data)


def test_a_frame_count_that_disagrees_with_the_manifest_fails(tmp_path):
    index, data, _ = make_index(tmp_path)
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    manifest["episodes"][0]["frames"] = 6
    (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected gripper state"):
        extract_development(index, data)


def make_rollout(root, name="T1_F1_unseen_state0", frames=4, tamper=False):
    directory = root / "run" / "episodes" / name
    directory.mkdir(parents=True)
    trace, observations = [], []
    for step in range(frames):
        event = [dict(type="return")] if step == 3 else []
        trace.append(dict(step=step, signals=dict(visual_evidence=dict(step=step), new_events=event)))
        observations.append(dict(step=step, raw_proprioception_pos_quatxyzw_gripper=[0] * 7 + [.02, -.02],
                                 info=dict(eval_result=dict(confirmed_cycles=int(step >= 2)))))
    write_jsonl(directory / "memory_trace.jsonl", trace)
    write_jsonl(directory / "observations.jsonl", observations)
    status = dict(status="complete", episode_id=name, condition="F1", split="unseen", task="T1",
                  files_sha256={path: sha256_file(directory / path)
                                for path in ("memory_trace.jsonl", "observations.jsonl")})
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    if tamper:
        with (directory / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")


def test_rollout_inputs_keep_evidence_gripper_and_reference_counts(tmp_path):
    make_rollout(tmp_path)
    summary = extract_rollouts(tmp_path, tmp_path / "rollouts.jsonl.gz")
    assert summary["episodes"] == 1 and summary["frames"] == 4
    with gzip.open(tmp_path / "rollouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        episode = json.loads(stream.readline())
    assert episode["milestones"] == [2] and episode["recorded_return_steps"] == [3]
    assert episode["gripper_qpos"][0] == [.02, -.02] and len(episode["evidence"]) == 4


def test_rollout_checksum_mismatch_is_refused(tmp_path):
    make_rollout(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_rollouts(tmp_path, tmp_path / "rollouts.jsonl.gz")


def test_command_line_writes_a_report_and_refuses_an_existing_output(tmp_path):
    index, data, _ = make_index(tmp_path)
    output = tmp_path / "out"
    main(["--development-index", str(index), "--data-dir", str(data), "--output-dir", str(output)])
    assert json.loads((output / "extraction_report.json").read_text())["development"][0]["episodes"] == 2
    assert set(json.loads((output / "development_gripper.json").read_text())["episodes"]) == set(FRAMES)
    with pytest.raises(FileExistsError):
        main(["--development-index", str(index), "--data-dir", str(data), "--output-dir", str(output)])
