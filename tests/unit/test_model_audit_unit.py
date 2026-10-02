"""Fast unit tests for Phase 4: Model Audit Framework and Base VLAPolicy."""

import numpy as np
import pytest

from src.models.base import VLAPolicy
from src.models.model_manifest import audit_model_interface, load_model_manifest
from src.models.registry import get_model_class, list_models, register_model


class MockPolicy(VLAPolicy):
    """Concrete mock policy for testing VLAPolicy queue and interface methods."""

    def load(self, checkpoint_path: str, **kwargs):
        pass

    def preprocess(self, obs, instruction):
        return {"instruction": instruction}

    def predict_action_chunk(self, obs, instruction):
        # Returns a predictable chunk of shape (chunk_size, 7)
        return np.ones((self.chunk_size, self.action_dim), dtype=np.float32) * 0.1

    def postprocess(self, output):
        return output

    def validate_interface(self):
        return {"status": "PASS"}

    @property
    def observation_spec(self):
        return {"agentview_image": (128, 128, 3)}


def test_vla_policy_queue_management():
    """Verify that select_action correctly manages the internal action queue."""
    policy = MockPolicy(chunk_size=5, action_dim=7)
    assert policy.queue_size == 0

    obs = {"agentview_image": np.zeros((128, 128, 3))}
    instruction = "pick up object"

    # Step 1: Queue should be populated with 5 items, first item popped -> 4 remaining
    act1 = policy.select_action(obs, instruction)
    assert act1.shape == (7,)
    assert policy.queue_size == 4

    # Steps 2-5: Pop remaining 4 actions
    for _ in range(4):
        policy.select_action(obs, instruction)
    assert policy.queue_size == 0

    # Step 6: Queue was empty, triggers new predict_action_chunk -> 4 remaining
    policy.select_action(obs, instruction)
    assert policy.queue_size == 4

    # Reset clears queue
    policy.reset()
    assert policy.queue_size == 0


def test_model_registry():
    """Verify model registration and retrieval."""
    @register_model("mock_model_v1")
    class RegisteredMock(MockPolicy):
        pass

    assert "mock_model_v1" in list_models()
    cls = get_model_class("mock_model_v1")
    assert cls == RegisteredMock


def test_smolvla_manifest_audit():
    """Verify loading and auditing the official smolvla_libero manifest."""
    manifest_path = "resources/manifests/models/smolvla_libero.yaml"
    manifest = load_model_manifest(manifest_path)

    assert manifest.model_id == "lerobot/smolvla_libero"
    assert manifest.action_dim == 7
    assert manifest.chunk_size == 50
    assert manifest.image_resolution == (256, 256)
    assert "agentview" in manifest.camera_mapping
    assert "robot0_eye_in_hand" in manifest.camera_mapping

    # Observation state audit contract
    assert manifest.observation_state is not None
    assert manifest.observation_state["runtime_dim"] == 8
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
    assert manifest.observation_state["semantics"] == expected_semantics
    assert manifest.observation_state["source_of_truth"] == [
        "policy_preprocessor_step_5_normalizer_processor.safetensors"
    ]

    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "PASS"
    assert audit["is_valid"] is True
    assert audit["observation_state"]["runtime_dim"] == 8
    # Verify immutable 40-char commit SHA
    assert manifest.revision == "31d453f7edd78c839a8bbc39744a292686daf0de"
    assert len(manifest.revision) == 40
    # Verify file inventory present
    assert "model.safetensors" in manifest.files
    assert "policy_preprocessor.json" in manifest.files
    # Verify camera3 discrepancy was flagged in audit warnings
    assert any("camera3" in w for w in audit["warnings"])


def test_normalizer_safetensors_8d_state_shapes():
    """Verify that the official normalizer safetensors contains exact 8D observation.state statistics."""
    from src.models.model_manifest import verify_normalizer_safetensors_shape
    from pathlib import Path

    safetensors_path = "resources/checkpoints/smolvla_libero/policy_preprocessor_step_5_normalizer_processor.safetensors"
    res = verify_normalizer_safetensors_shape(safetensors_path, expected_dim=8)

    assert res["status"] == "PASS"
    assert res["is_valid"] is True
    assert len(res["issues"]) == 0

    # Ensure all core statistics are strictly 8-dimensional
    assert res["state_tensors"]["observation.state.mean"] == (8,)
    assert res["state_tensors"]["observation.state.std"] == (8,)
    assert res["state_tensors"]["observation.state.min"] == (8,)
    assert res["state_tensors"]["observation.state.max"] == (8,)
    assert res["state_tensors"]["observation.state.q01"] == (8,)
    assert res["state_tensors"]["observation.state.q99"] == (8,)


def test_observation_state_audit_failures():
    """Verify that invalid observation_state configurations fail the audit."""
    manifest_path = "resources/manifests/models/smolvla_libero.yaml"
    manifest = load_model_manifest(manifest_path)

    # Corrupt runtime_dim to 6
    manifest.observation_state["runtime_dim"] = 6
    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "FAIL"
    assert audit["is_valid"] is False
    assert any("runtime_dim 6" in issue for issue in audit["issues"])

    # Corrupt semantics
    manifest.observation_state["runtime_dim"] = 8
    manifest.observation_state["semantics"] = ["eef_x", "eef_y"]
    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "FAIL"
    assert any("semantics mismatch" in issue for issue in audit["issues"])

    # Corrupt source of truth
    manifest.observation_state["semantics"] = [
        "eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_axis_x",
        "eef_axis_y", "eef_axis_z", "gripper_qpos_0", "gripper_qpos_1"
    ]
    manifest.observation_state["source_of_truth"] = ["wrong_file.json"]
    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "FAIL"
    assert any("source_of_truth must include" in issue for issue in audit["issues"])


def test_normalizer_safetensors_shape_mismatch():
    """Verify that expecting wrong dimension on 8D safetensors reports failure."""
    from src.models.model_manifest import verify_normalizer_safetensors_shape

    safetensors_path = "resources/checkpoints/smolvla_libero/policy_preprocessor_step_5_normalizer_processor.safetensors"
    # Expecting 6D on 8D safetensors must fail
    res = verify_normalizer_safetensors_shape(safetensors_path, expected_dim=6)
    assert res["status"] == "FAIL"
    assert res["is_valid"] is False
    assert len(res["issues"]) > 0
    assert any("shape mismatch: expected (6,), got (8,)" in issue for issue in res["issues"])


def test_checkpoint_integrity_verification(tmp_path):
    """Verify cryptographic hash checking against mock and real files."""
    from src.models.model_manifest import verify_checkpoint_integrity
    import hashlib

    manifest_path = "resources/manifests/models/smolvla_libero.yaml"
    manifest = load_model_manifest(manifest_path)

    # Create dummy files matching expected hashes
    cp_dir = tmp_path / "mock_checkpoint"
    cp_dir.mkdir()

    for fname, meta in manifest.files.items():
        if "policy_preprocessor.json" == fname:
            # Create a file and set hash
            test_content = b'{"steps": []}'
            (cp_dir / fname).write_bytes(test_content)
            manifest.files[fname]["sha256"] = hashlib.sha256(test_content).hexdigest()

    res = verify_checkpoint_integrity(cp_dir, manifest, require_all=False)
    assert res["file_results"]["policy_preprocessor.json"]["status"] == "MATCH"
    assert res["file_results"]["policy_preprocessor.json"]["passed"] is True


def test_minivla_manifest_audit():
    """Verify loading and auditing official minivla_libero90 manifest (Candidate B)."""
    manifest_path = "resources/manifests/models/minivla_libero90.yaml"
    manifest = load_model_manifest(manifest_path)

    assert manifest.model_id == "Stanford-ILIAD/minivla-libero90-prismatic"
    assert manifest.action_dim == 7
    assert manifest.chunk_size == 1
    assert manifest.image_resolution == (224, 224)
    assert "agentview" in manifest.camera_mapping
    assert "robot0_eye_in_hand" in manifest.camera_mapping
    assert len(manifest.revision) == 40

    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "PASS"
    assert audit["is_valid"] is True


def test_minivla_vq_manifest_audit():
    """Verify loading and auditing official minivla_vq_libero90 manifest (Candidate C)."""
    manifest_path = "resources/manifests/models/minivla_vq_libero90.yaml"
    manifest = load_model_manifest(manifest_path)

    assert manifest.model_id == "Stanford-ILIAD/minivla-vq-libero90-prismatic"
    assert manifest.action_dim == 7
    assert manifest.chunk_size == 10
    assert manifest.image_resolution == (224, 224)
    assert "agentview" in manifest.camera_mapping
    assert "robot0_eye_in_hand" in manifest.camera_mapping
    assert len(manifest.revision) == 40

    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "PASS"
    assert audit["is_valid"] is True


