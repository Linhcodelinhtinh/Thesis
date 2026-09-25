"""Unit tests for SmolVLAAdapter (Phase 5 Hardening).

Verifies fail-fast behavior (no zero-action fallback), observation mapping
to LeRobot raw features, 8D state vector calculation, and postprocessing.
Runs completely offline (< 1s).
"""

import numpy as np
import pytest
import torch

from src.models.smolvla.adapter import SmolVLAAdapter


def test_smolvla_adapter_init_and_specs():
    adapter = SmolVLAAdapter(chunk_size=50, action_dim=7)
    assert adapter.chunk_size == 50
    assert adapter.action_dim == 7
    assert adapter.queue_size == 0
    assert adapter._is_loaded is False

    obs_spec = adapter.observation_spec
    assert "agentview_image" in obs_spec
    assert "robot0_eye_in_hand_image" in obs_spec
    assert "robot0_eef_pos" in obs_spec
    assert "robot0_eef_quat" in obs_spec
    assert "robot0_gripper_qpos" in obs_spec

    act_spec = adapter.action_spec
    assert act_spec["shape"] == (7,)
    assert act_spec["chunk_size"] == 50
    assert act_spec["range"] == (-1.0, 1.0)


def test_smolvla_adapter_fails_fast_when_not_loaded():
    """Verify that calling predict_action_chunk on unloaded adapter fails loudly."""
    adapter = SmolVLAAdapter()
    sample_obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }

    # Zero-tolerance for fallback: MUST raise RuntimeError
    with pytest.raises(RuntimeError, match="not loaded.*Fail-fast per AGENTS.md Rule 10"):
        adapter.predict_action_chunk(sample_obs, instruction="pick up can")

    with pytest.raises(RuntimeError, match="not loaded.*Fail-fast per AGENTS.md Rule 10"):
        adapter.select_action(sample_obs, instruction="pick up can")


def test_smolvla_adapter_raw_feature_mapping():
    """Verify mapping of LIBERO obs to LeRobot format with 8D state and (3, 256, 256) images."""
    adapter = SmolVLAAdapter()
    sample_obs = {
        "agentview_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.5], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.03, -0.03], dtype=np.float32),
    }

    raw = adapter.build_raw_features(sample_obs, instruction="place alphabet soup in basket")

    assert "observation.images.image" in raw
    assert "observation.images.image2" in raw
    assert "observation.state" in raw
    assert raw["task"] == "place alphabet soup in basket"

    # Images must be (3, 256, 256) float tensors in [0, 1]
    cam1 = raw["observation.images.image"]
    assert isinstance(cam1, torch.Tensor)
    assert cam1.shape == (3, 256, 256)
    assert 0.0 <= cam1.min().item() and cam1.max().item() <= 1.0

    cam2 = raw["observation.images.image2"]
    assert isinstance(cam2, torch.Tensor)
    assert cam2.shape == (3, 256, 256)

    # State must be 8D: pos(3) + axis_angle(3) + gripper(2)
    state = raw["observation.state"]
    assert isinstance(state, torch.Tensor)
    assert state.shape == (8,)
    assert torch.allclose(state[:3], torch.tensor([0.1, -0.2, 0.5]))
    assert torch.allclose(state[3:6], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(state[6:8], torch.tensor([0.03, -0.03]))


def test_smolvla_adapter_postprocess():
    adapter = SmolVLAAdapter(chunk_size=5, action_dim=7)

    # Tensor input from postprocessor
    raw_tensor = torch.tensor([[[0.1 * i] * 7 for i in range(5)]], dtype=torch.float32)  # (1, 5, 7)
    processed = adapter.postprocess(raw_tensor)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (5, 7)
    assert np.isclose(processed[1, 0], 0.1)

    info = adapter.validate_interface()
    assert info["model_type"] == "SmolVLAAdapter"
    assert info["action_dim"] == 7
    assert info["state_dim"] == 8
    assert info["is_loaded"] is False


def test_smolvla_adapter_mock_pipeline_execution():
    """Verify execution when official policy and processors are injected."""
    adapter = SmolVLAAdapter(chunk_size=5, action_dim=7)

    class MockLeRobotPolicy:
        def __init__(self):
            self.called = False

        def predict_action_chunk(self, batch):
            self.called = True
            return torch.ones((1, 5, 7), dtype=torch.float32) * 0.5

    class MockProcessor:
        def __call__(self, x):
            return x

    adapter.policy = MockLeRobotPolicy()
    adapter.preprocessor = MockProcessor()
    adapter.postprocessor = MockProcessor()
    adapter._is_loaded = True

    sample_obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }

    chunk = adapter.predict_action_chunk(sample_obs, "test task")
    assert chunk.shape == (5, 7)
    assert np.allclose(chunk, 0.5)

    # Test FIFO queue action selection
    action_0 = adapter.select_action(sample_obs, "test task")
    assert action_0.shape == (7,)
    assert adapter.queue_size == 4


def test_smolvla_adapter_image_180_rotation():
    """Verify that agentview and wrist images are rotated 180 degrees ([::-1, ::-1])."""
    adapter = SmolVLAAdapter()
    
    # Create an asymmetric 128x128 image with top-left pixel = 255 and rest 0
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[0, 0, :] = 255  # Top-left corner
    
    sample_obs = {
        "agentview_image": img.copy(),
        "robot0_eye_in_hand_image": img.copy(),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }

    raw = adapter.build_raw_features(sample_obs, "test task")
    
    # In a 180-degree rotation, top-left pixel (0, 0) moves to bottom-right (255, 255)
    cam1 = raw["observation.images.image"]  # (3, 256, 256)
    cam2 = raw["observation.images.image2"]
    
    # Top-left should now be 0, and bottom-right should be non-zero (due to resize interpolation)
    assert cam1[:, 0, 0].max().item() == 0.0
    assert cam1[:, -1, -1].max().item() > 0.0
    assert cam2[:, 0, 0].max().item() == 0.0
    assert cam2[:, -1, -1].max().item() > 0.0

