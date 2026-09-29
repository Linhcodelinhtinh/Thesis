"""Tests enforcing mandatory shape contracts for SmolVLA and LIBERO interfaces (Phase 5).

Verifies:
- observation_state runtime_dim is strictly 8 with exact semantics
  [eef_pos_x, eef_pos_y, eef_pos_z, eef_axis_x, eef_axis_y, eef_axis_z, gripper_qpos_0, gripper_qpos_1]
- Source of truth normalizer safetensors matches 8D shape across all statistical tensors.
- Fail-fast rejection of invalid observation and action shapes.
- Action chunk output shape is strictly (chunk_size, 7).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.model_manifest import (
    audit_model_interface,
    load_model_manifest,
    verify_normalizer_safetensors_shape,
)
from src.models.smolvla.adapter import SmolVLAAdapter


MANIFEST_PATH = Path("resources/manifests/models/smolvla_libero.yaml")
SAFETENSORS_PATH = Path("resources/checkpoints/smolvla_libero/policy_preprocessor_step_5_normalizer_processor.safetensors")


def test_manifest_observation_state_shape_contract():
    """Verify manifest observation_state schema and values."""
    manifest = load_model_manifest(MANIFEST_PATH)
    assert manifest.observation_state is not None

    obs_state = manifest.observation_state
    assert obs_state["runtime_dim"] == 8
    expected_semantics = [
        "eef_pos_x",
        "eef_pos_y",
        "eef_pos_z",
        "eef_axis_x",
        "eef_axis_y",
        "eef_axis_z",
        "gripper_qpos_0",
        "gripper_qpos_1",
    ]
    assert obs_state["semantics"] == expected_semantics
    assert "policy_preprocessor_step_5_normalizer_processor.safetensors" in obs_state["source_of_truth"]

    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "PASS"
    assert audit["is_valid"] is True
    assert audit["observation_state"]["runtime_dim"] == 8


def test_safetensors_observation_state_shape_parity():
    """Verify ground truth weights in safetensors strictly conform to 8D observation.state."""
    assert SAFETENSORS_PATH.exists(), f"Safetensors file not found at {SAFETENSORS_PATH}"

    res = verify_normalizer_safetensors_shape(SAFETENSORS_PATH, expected_dim=8)
    assert res["status"] == "PASS"
    assert res["is_valid"] is True

    # Validate essential normalizer parameter shapes
    tensors = res["state_tensors"]
    assert tensors["observation.state.mean"] == (8,)
    assert tensors["observation.state.std"] == (8,)
    assert tensors["observation.state.min"] == (8,)
    assert tensors["observation.state.max"] == (8,)
    assert tensors["observation.state.q01"] == (8,)
    assert tensors["observation.state.q10"] == (8,)
    assert tensors["observation.state.q50"] == (8,)
    assert tensors["observation.state.q90"] == (8,)
    assert tensors["observation.state.q99"] == (8,)


def test_adapter_observation_state_shape_generation():
    """Verify adapter build_raw_features produces exact 8D tensor from simulator obs."""
    adapter = SmolVLAAdapter(chunk_size=50, action_dim=7)

    # 1. From standard LIBERO low-dim observation fields
    obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }
    raw = adapter.build_raw_features(obs, "pick up object")
    state = raw["observation.state"]

    assert isinstance(state, torch.Tensor)
    assert state.shape == (8,)
    assert state.dtype == torch.float32

    # 2. From pre-assembled 8D observation.state numpy array
    obs_direct_np = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation.state": np.arange(8, dtype=np.float32),
    }
    raw_np = adapter.build_raw_features(obs_direct_np, "pick up object")
    assert raw_np["observation.state"].shape == (8,)
    assert torch.equal(raw_np["observation.state"], torch.arange(8, dtype=torch.float32))

    # 3. From pre-assembled 8D observation.state torch Tensor
    obs_direct_torch = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation.state": torch.ones(8, dtype=torch.float32),
    }
    raw_torch = adapter.build_raw_features(obs_direct_torch, "pick up object")
    assert raw_torch["observation.state"].shape == (8,)


def test_adapter_observation_state_shape_rejection():
    """Verify that any deviation from 8D state raises ValueError fail-fast."""
    adapter = SmolVLAAdapter()

    # 6D observation.state (e.g. pos + axis without gripper)
    obs_6d = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="expected 8D vector.*got shape torch.Size\\(\\[6\\]\\)"):
        adapter.build_raw_features(obs_6d, "instruction")

    # 7D observation.state
    obs_7d = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation.state": np.zeros(7, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="expected 8D vector.*got shape torch.Size\\(\\[7\\]\\)"):
        adapter.build_raw_features(obs_7d, "instruction")

    # Malformed pos shape (e.g. 2D instead of 3D)
    obs_bad_pos = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Expected robot0_eef_pos of shape \\(3,\\)"):
        adapter.build_raw_features(obs_bad_pos, "instruction")

    # Malformed gripper shape (e.g. 1D instead of 2D)
    obs_bad_grip = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Expected robot0_gripper_qpos of shape \\(2,\\)"):
        adapter.build_raw_features(obs_bad_grip, "instruction")


def test_adapter_action_chunk_shape_contract():
    """Verify action chunk shape output contracts."""
    adapter = SmolVLAAdapter(chunk_size=50, action_dim=7)

    # Valid batched tensor output (1, 50, 7) -> (50, 7)
    tensor_3d = torch.zeros((1, 50, 7), dtype=torch.float32)
    out_3d = adapter.postprocess(tensor_3d)
    assert out_3d.shape == (50, 7)
    assert out_3d.dtype == np.float32

    # Valid unbatched tensor output (50, 7) -> (50, 7)
    tensor_2d = torch.zeros((50, 7), dtype=torch.float32)
    out_2d = adapter.postprocess(tensor_2d)
    assert out_2d.shape == (50, 7)

    # Action dim mismatch: e.g. 6D or 8D actions
    tensor_wrong_dim = torch.zeros((1, 50, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        adapter.postprocess(tensor_wrong_dim)


def test_adapter_interface_validation_contract():
    """Verify that adapter.validate_interface() matches the audited observation_state contract."""
    adapter = SmolVLAAdapter(chunk_size=50, action_dim=7)
    info = adapter.validate_interface()

    assert info["state_dim"] == 8
    assert info["action_dim"] == 7
    assert info["chunk_size"] == 50
    assert info["invert_gripper_action"] is True
    assert info["gripper_action_polarity"] == "INVERTED_RLDS_TO_ROBOSUITE"
    assert "observation_state" in info

    contract = info["observation_state"]
    assert contract["runtime_dim"] == 8
    assert len(contract["semantics"]) == 8
    assert contract["semantics"] == [
        "eef_pos_x", "eef_pos_y", "eef_pos_z",
        "eef_axis_x", "eef_axis_y", "eef_axis_z",
        "gripper_qpos_0", "gripper_qpos_1"
    ]
    assert "policy_preprocessor_step_5_normalizer_processor.safetensors" in contract["source_of_truth"]


def test_smolvla_gripper_polarity_inversion():
    """Verify ADR-0009 gripper action polarity inversion."""
    # When invert_gripper_action=True (default)
    adapter = SmolVLAAdapter(chunk_size=2, action_dim=7, invert_gripper_action=True)

    # Raw model unnormalized output: step 0 has gripper +1.0 (Open in RLDS), step 1 has -1.0 (Close in RLDS)
    raw_unnorm = torch.tensor([
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
    ], dtype=torch.float32)

    sim_actions = adapter.postprocess(raw_unnorm)
    # Robosuite expects -1.0 for Open and +1.0 for Close
    assert np.isclose(sim_actions[0, -1], -1.0)
    assert np.isclose(sim_actions[1, -1], 1.0)

    # When invert_gripper_action=False (direct passthrough)
    adapter_direct = SmolVLAAdapter(chunk_size=2, action_dim=7, invert_gripper_action=False)
    sim_direct = adapter_direct.postprocess(raw_unnorm)
    assert np.isclose(sim_direct[0, -1], 1.0)
    assert np.isclose(sim_direct[1, -1], -1.0)


def test_smolvla_dual_action_telemetry_cache():
    """Verify caching of raw normalized chunk, unnormalized chunk, and executed chunk."""
    adapter = SmolVLAAdapter(chunk_size=2, action_dim=7, invert_gripper_action=True)

    class MockPolicy:
        def predict_action_chunk(self, batch):
            # Model returns normalized tensor
            return torch.ones((1, 2, 7), dtype=torch.float32) * 2.0

    class MockUnnormalizer:
        def __call__(self, x):
            # Unnormalizer converts to [-1, 1] range: e.g. gripper = 0.8
            arr = torch.zeros((1, 2, 7), dtype=torch.float32)
            arr[..., -1] = 0.8  # RLDS Open
            return arr

    adapter.policy = MockPolicy()
    adapter.preprocessor = lambda x: x
    adapter.postprocessor = MockUnnormalizer()
    adapter._is_loaded = True

    sample_obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }

    executed_chunk = adapter.predict_action_chunk(sample_obs, "test task")
    assert executed_chunk.shape == (2, 7)
    # Gripper inverted: 0.8 -> -0.8
    assert np.isclose(executed_chunk[0, -1], -0.8)

    # Verify telemetry cache
    tel = adapter.get_last_telemetry()
    assert tel["raw_normalized_chunk"] is not None
    assert tel["raw_normalized_chunk"].shape == (2, 7)
    assert np.allclose(tel["raw_normalized_chunk"], 2.0)

    assert tel["unnormalized_chunk"] is not None
    assert tel["unnormalized_chunk"].shape == (2, 7)
    assert np.isclose(tel["unnormalized_chunk"][0, -1], 0.8)

    assert tel["executed_chunk"] is not None
    assert np.isclose(tel["executed_chunk"][0, -1], -0.8)

