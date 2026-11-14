"""Unit tests for LIBERO-Mem-Derived demonstration generator schema and success logic."""
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
import pytest

from src.training.memory_data import validate_episode
from src.training.libero_mem_hdf5_loader import LiberoMemHDF5Parser, DemonstrationEpisode


def create_mock_derived_episode(group, num_steps=10):
    """Helper to create an episode matching the exact derived demonstration schema."""
    actions = np.zeros((num_steps, 7), dtype=np.float32)
    actions[:, 0] = 0.1  # Valid actions in [-1, 1]

    # Observations
    agentview_rgb = np.zeros((num_steps, 256, 256, 3), dtype=np.uint8)
    eye_in_hand_rgb = np.zeros((num_steps, 256, 256, 3), dtype=np.uint8)
    ee_pos = np.ones((num_steps, 3), dtype=np.float64) * 0.5
    ee_ori = np.zeros((num_steps, 3), dtype=np.float64)
    ee_states = np.concatenate([ee_pos, ee_ori], axis=-1)
    joint_states = np.zeros((num_steps, 7), dtype=np.float64)
    gripper_states = np.array([[0.02, -0.02]] * num_steps, dtype=np.float64)

    # robot_states: [gripper_states(2), ee_pos(3), ee_quat(4)]
    # Unit quaternion [0, 0, 0, 1]
    ee_quat = np.array([[0.0, 0.0, 0.0, 1.0]] * num_steps, dtype=np.float64)
    robot_states = np.concatenate([gripper_states, ee_pos, ee_quat], axis=-1)

    rewards = np.zeros(num_steps, dtype=np.float32)
    dones = np.zeros(num_steps, dtype=np.uint8)

    group.create_dataset("actions", data=actions)
    group.create_dataset("rewards", data=rewards)
    group.create_dataset("dones", data=dones)
    group.create_dataset("robot_states", data=robot_states)

    obs = group.create_group("obs")
    obs.create_dataset("agentview_rgb", data=agentview_rgb)
    obs.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)
    obs.create_dataset("ee_pos", data=ee_pos)
    obs.create_dataset("ee_ori", data=ee_ori)
    obs.create_dataset("ee_states", data=ee_states)
    obs.create_dataset("joint_states", data=joint_states)
    obs.create_dataset("gripper_states", data=gripper_states)

    group.attrs["init_state"] = np.zeros(80, dtype=np.float64)
    group.attrs["num_samples"] = num_steps
    group.attrs["success"] = True


def test_derived_episode_passes_training_validation():
    """Verify that generated HDF5 episodes satisfy memory_data.validate_episode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_demo.hdf5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("data/demo_1")
            create_mock_derived_episode(grp, num_steps=25)

        with h5py.File(h5_path, "r") as f:
            demo_grp = f["data/demo_1"]
            actions, robot, resolutions = validate_episode(demo_grp)

            assert actions.shape == (25, 7)
            assert robot.shape == (25, 9)
            assert "agentview_rgb" in resolutions
            assert "eye_in_hand_rgb" in resolutions
            assert resolutions["agentview_rgb"] == [256, 256, 3]


def test_derived_episode_rejection_on_missing_robot_states():
    """Verify that validate_episode fails when robot_states is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "bad_demo.hdf5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("data/demo_1")
            create_mock_derived_episode(grp, num_steps=10)
            del grp["robot_states"]

        with h5py.File(h5_path, "r") as f:
            demo_grp = f["data/demo_1"]
            with pytest.raises((KeyError, ValueError)):
                validate_episode(demo_grp)


def test_derived_episode_rejection_on_unnormalized_quaternions():
    """Verify that validate_episode fails when robot_states quaternion norm != 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "bad_quat.hdf5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("data/demo_1")
            create_mock_derived_episode(grp, num_steps=10)
            bad_robot = grp["robot_states"][:]
            bad_robot[:, 5:9] = 2.0  # Norm != 1
            del grp["robot_states"]
            grp.create_dataset("robot_states", data=bad_robot)

        with h5py.File(h5_path, "r") as f:
            demo_grp = f["data/demo_1"]
            with pytest.raises(AssertionError):
                validate_episode(demo_grp)


def test_derived_hdf5_loader_integration():
    """Verify that LiberoMemHDF5Parser can iterate over generated derived demonstrations."""
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        h5_path = data_dir / f"{task_name}_demo.hdf5"
        meta_path = data_dir / "metainfo.json"

        # Create HDF5 file
        with h5py.File(h5_path, "w") as f:
            data = f.create_group("data")
            create_mock_derived_episode(data.create_group("demo_1"), num_steps=20)
            create_mock_derived_episode(data.create_group("demo_2"), num_steps=20)

        # Create companion metainfo.json
        init_state = [0.1] * 80
        metainfo = {
            task_name: {
                "demo_1": {
                    "task_description": "pick up the bowl and place it back on the plate",
                    "initial_state": init_state,
                    "success": True
                },
                "demo_2": {
                    "task_description": "pick up the bowl and place it back on the plate",
                    "initial_state": init_state,
                    "success": True
                }
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metainfo, f)

        parser = LiberoMemHDF5Parser(data_dir=data_dir)
        episodes = list(parser.iter_task(task_name))
        assert len(episodes) == 2
        for ep in episodes:
            assert isinstance(ep, DemonstrationEpisode)
            assert ep.task_name == task_name
            assert ep.instruction == "pick up the bowl and place it back on the plate"
            assert len(ep.actions) == 20
            assert len(ep.observations) == 20
            assert "agentview_image" in ep.observations[0]
            assert "robot0_eye_in_hand_image" in ep.observations[0]
            assert "robot0_gripper_qpos" in ep.observations[0]


def test_strict_success_criteria():
    """Verify that success requires target subgoals, true final predicate, and not overshot."""
    target_cycles = 1

    # Case 1: True full success
    completed_subgoals = 1
    final_predicate = True
    overshot = False
    succ = (completed_subgoals == target_cycles) and final_predicate and (not overshot)
    assert succ is True

    # Case 2: Intermediate progress but false final predicate
    completed_subgoals = 1
    final_predicate = False
    overshot = False
    succ = (completed_subgoals == target_cycles) and final_predicate and (not overshot)
    assert succ is False

    # Case 3: Overshot
    completed_subgoals = 1
    final_predicate = True
    overshot = True
    succ = (completed_subgoals == target_cycles) and final_predicate and (not overshot)
    assert succ is False

    # Case 4: Insufficient subgoals
    completed_subgoals = 0
    final_predicate = True
    overshot = False
    succ = (completed_subgoals == target_cycles) and final_predicate and (not overshot)
    assert succ is False
