"""Unit tests for LiberoMemHDF5Parser."""

from pathlib import Path
import numpy as np
import pytest

from src.training.libero_mem_hdf5_loader import DemonstrationEpisode, LiberoMemHDF5Parser

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "libero_mem"


def test_resolve_hdf5_path():
    parser = LiberoMemHDF5Parser(data_dir=DATA_DIR)
    task1 = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    path1 = parser.resolve_hdf5_path(task1)
    assert path1.exists()
    assert path1.name.endswith(".hdf5")

    task3 = "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times"
    path3 = parser.resolve_hdf5_path(task3)
    assert path3.exists()


def test_resolve_missing_hdf5_fails_fast():
    parser = LiberoMemHDF5Parser(data_dir=DATA_DIR)
    with pytest.raises(FileNotFoundError):
        parser.resolve_hdf5_path("NONEXISTENT_TASK_ABC")


def test_deterministic_split_assignment():
    parser = LiberoMemHDF5Parser(val_ratio=0.1, seed=42)
    # Same episode ID should always produce same split assignment
    is_val_1 = parser.is_val_episode("demo_001")
    is_val_2 = parser.is_val_episode("demo_001")
    assert is_val_1 == is_val_2

    # Check approximate distribution over 1000 IDs
    val_count = sum(parser.is_val_episode(f"ep_{i}") for i in range(1000))
    # Roughly 10% +/- 4%
    assert 60 <= val_count <= 140


@pytest.mark.skipif(
    not (DATA_DIR / "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5").exists(),
    reason="LIBERO-Mem pilot dataset not downloaded",
)
def test_parse_task_demonstrations():
    parser = LiberoMemHDF5Parser(data_dir=DATA_DIR, val_ratio=0.2, seed=42)
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    episodes = parser.parse_task(task_name=task_name, max_episodes=2)

    assert len(episodes) == 2
    for ep in episodes:
        assert isinstance(ep, DemonstrationEpisode)
        assert ep.task_name == "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
        assert len(ep.instruction) > 0

        # Actions contract
        assert isinstance(ep.actions, np.ndarray)
        assert ep.actions.ndim == 2
        assert ep.actions.shape[1] == 7
        assert np.all(np.isfinite(ep.actions))

        # Initial state contract
        if ep.initial_state is not None:
            assert ep.initial_state.shape == (90,)
            assert np.all(np.isfinite(ep.initial_state))

        # Observations contract
        T = len(ep.actions)
        assert len(ep.observations) == T
        for obs in ep.observations:
            assert "agentview_image" in obs
            assert obs["agentview_image"].shape == (256, 256, 3)
            assert "robot0_eye_in_hand_image" in obs
            assert obs["robot0_eye_in_hand_image"].shape == (256, 256, 3)

            assert "robot0_eef_pos" in obs
            assert obs["robot0_eef_pos"].shape == (3,)
            assert "robot0_gripper_qpos" in obs
            assert obs["robot0_gripper_qpos"].shape == (2,)
            assert "robot0_eef_quat" in obs
            assert obs["robot0_eef_quat"].shape == (4,)

            assert "oracle_signals" in obs
            assert "is_on_support" in obs["oracle_signals"]
            assert "is_held" in obs["oracle_signals"]


@pytest.mark.skipif(
    not (DATA_DIR / "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5").exists(),
    reason="LIBERO-Mem pilot dataset not downloaded",
)
def test_parse_task_split_filter():
    parser = LiberoMemHDF5Parser(data_dir=DATA_DIR, val_ratio=0.5, seed=42)
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    train_eps = parser.parse_task(task_name=task_name, max_episodes=10, split_filter="train")
    val_eps = parser.parse_task(task_name=task_name, max_episodes=10, split_filter="val")

    assert all(ep.split == "train" for ep in train_eps)
    assert all(ep.split == "val" for ep in val_eps)
    assert len(train_eps) + len(val_eps) <= 10
