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

    audit = audit_model_interface(manifest)
    assert audit["audit_status"] == "PASS"
    assert audit["is_valid"] is True
    # Verify immutable 40-char commit SHA
    assert manifest.revision == "31d453f7edd78c839a8bbc39744a292686daf0de"
    assert len(manifest.revision) == 40
    # Verify file inventory present
    assert "model.safetensors" in manifest.files
    assert "policy_preprocessor.json" in manifest.files
    # Verify camera3 discrepancy was flagged in audit warnings
    assert any("camera3" in w for w in audit["warnings"])


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

