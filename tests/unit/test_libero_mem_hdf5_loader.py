"""Loader contract fixtures; no simulator or benchmark score is synthesized."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from src.training.libero_mem_hdf5_loader import LiberoMemHDF5Parser


@pytest.fixture
def fixture_data(tmp_path):
    task = "unit_task"
    n = 4
    with h5py.File(tmp_path / f"{task}_demo.hdf5", "w") as file:
        group = file.create_group("data/demo_1")
        group["actions"] = np.zeros((n, 7), dtype=np.float32)
        robot = np.tile([.04, -.04, 0., 0., 1.2, 0., 0., 0., 1.], (n, 1))
        group["robot_states"] = robot
        obs = group.create_group("obs")
        obs["ee_pos"] = robot[:, 2:5]
        obs["gripper_states"] = robot[:, :2]
        obs["agentview_rgb"] = np.full((n, 16, 16, 3), 61, dtype=np.uint8)
        obs["eye_in_hand_rgb"] = np.full((n, 16, 16, 3), 97, dtype=np.uint8)
    meta = {task: {"demo_1": {"initial_state": [1., 2., 3.], "task_description": "unit instruction",
            "success": True, "exo_boxes": [{"bowl 1": [20, [.1, .2, .3, .4], "3"]}] * n}}}
    (tmp_path / "metainfo.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path, task


def test_exact_resolution_and_missing_file(fixture_data):
    root, task = fixture_data
    parser = LiberoMemHDF5Parser(root)
    assert parser.resolve_hdf5_path(task).is_file()
    with pytest.raises(FileNotFoundError):
        parser.resolve_hdf5_path("unit")  # no fuzzy alias fallback


def test_initial_state_grouping_cross_task_and_missing_state_fails():
    parser = LiberoMemHDF5Parser()
    state = np.arange(10, dtype=np.float64)
    assert parser.is_val_episode("T1_demo_1", state) == parser.is_val_episode("T3_demo_84", state.astype('>f8'))
    with pytest.raises(ValueError, match="Initial state required"):
        parser.is_val_episode("demo_1")


def test_metadata_is_not_predicted_evidence_or_fake_oracle(fixture_data):
    root, task = fixture_data
    episode = next(LiberoMemHDF5Parser(root).iter_task(task))
    assert episode.instruction == "unit instruction"
    for obs in episode.observations:
        assert set(obs) == {"agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_gripper_qpos", "robot0_eef_quat"}
        assert np.all(obs["agentview_image"] == 61)
        assert np.all(obs["robot0_eye_in_hand_image"] == 97)
        np.testing.assert_equal(obs["robot0_eef_quat"], [0., 0., 0., 1.])


def test_missing_state_cannot_be_fabricated_from_ee_ori(fixture_data):
    root, task = fixture_data
    with h5py.File(root / f"{task}_demo.hdf5", "a") as file:
        del file["data/demo_1/robot_states"]
        file["data/demo_1/obs/ee_ori"] = np.zeros((4, 3))
    with pytest.raises(KeyError):
        list(LiberoMemHDF5Parser(root).iter_task(task))


def test_metadata_cache_cannot_cross_directories(fixture_data, tmp_path):
    root, task = fixture_data
    first = LiberoMemHDF5Parser(root).parse_task(task)
    meta_path = root / "metainfo.json"
    meta = json.loads(meta_path.read_text())
    meta[task]["demo_1"]["task_description"] = "changed unit instruction"
    meta_path.write_text(json.dumps(meta))
    second = LiberoMemHDF5Parser(root).parse_task(task)
    assert first[0].instruction != second[0].instruction
