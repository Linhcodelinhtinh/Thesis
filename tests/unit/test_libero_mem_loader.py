"""Unit tests for LiberoMemBDDLLoader and Robosuite compatibility layer."""

from pathlib import Path
import numpy as np
import pytest

from src.simulator.libero_mem_loader import (
    LiberoMemBDDLLoader,
    apply_robosuite_compatibility_shims,
    setup_libero_mem_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_setup_libero_mem_paths():
    libero_root = setup_libero_mem_paths()
    assert libero_root.exists()
    assert (libero_root / "libero" / "libero" / "bddl_files" / "libero_mem").exists()


def test_apply_robosuite_shims():
    # Should not raise exception
    apply_robosuite_compatibility_shims()


def test_loader_resolve_bddl_path():
    loader = LiberoMemBDDLLoader()
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    bddl_path = loader.resolve_bddl_path(task_name)
    assert bddl_path.exists()
    assert bddl_path.suffix == ".bddl"
    assert "KITCHEN_SCENE1_1" in bddl_path.name


def test_loader_resolve_missing_task_fails_fast():
    loader = LiberoMemBDDLLoader()
    with pytest.raises(FileNotFoundError):
        loader.resolve_bddl_path("NONEXISTENT_TASK_XYZ")


def test_loader_initial_state_from_metainfo():
    loader = LiberoMemBDDLLoader()
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    init_state = loader.get_initial_state(task_name, demo_idx=1)
    assert init_state is not None
    assert isinstance(init_state, np.ndarray)
    assert init_state.shape == (90,)


def test_loader_grounding_boxes_from_metainfo():
    loader = LiberoMemBDDLLoader()
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    boxes = loader.get_grounding_boxes(task_name, demo_idx=1, step=0)
    assert isinstance(boxes, dict)
    assert len(boxes) > 0


@pytest.mark.skipif(
    not (REPO_ROOT / "external" / "libero-mem").exists(),
    reason="external/libero-mem not present",
)
def test_loader_create_real_env_and_step():
    loader = LiberoMemBDDLLoader()
    task_name = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
    env = loader.create_env(task_name=task_name, camera_height=128, camera_width=128)
    try:
        init_state = loader.get_initial_state(task_name, demo_idx=1)
        if init_state is not None:
            env.set_init_state(init_state)

        obs = env.reset()
        assert "agentview_image" in obs
        assert "robot0_eye_in_hand_image" in obs
        assert obs["agentview_image"].shape == (128, 128, 3)

        # Step 7-dim zero action
        zero_action = np.zeros(7, dtype=np.float32)
        next_obs, reward, done, info = env.step(zero_action)
        assert "agentview_image" in next_obs
        assert next_obs["agentview_image"].shape == (128, 128, 3)
    finally:
        env.close()
