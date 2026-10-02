"""Unit tests for MiniVLA Adapter and Components (Phase 9).

Verifies fail-fast behavior (no zero-action fallback), extra_action_tokenizer
discretization/detokenization, SigLIP 224px observation preprocessing,
prompt formatting, and dynamic gripper polarity.
Runs completely offline (< 1s).
"""

import numpy as np
import pytest
import torch

from src.models.minivla.action_tokenizer import ExtraActionTokenizer
from src.models.minivla.adapter import MiniVLAAdapter, MiniVLAVQAdapter
from src.models.minivla.config import MiniVLAConfig
from src.models.minivla.postprocess import MiniVLAPostprocessor
from src.models.minivla.preprocess import format_instruction_prompt, resize_and_normalize_image
from src.models.registry import get_model_class, list_models


def test_minivla_registration():
    """Verify that minivla models are registered properly."""
    models = list_models()
    assert "minivla_libero90" in models
    assert "minivla_vq_libero90" in models

    cls_b = get_model_class("minivla_libero90")
    assert cls_b == MiniVLAAdapter

    cls_c = get_model_class("minivla_vq_libero90")
    assert cls_c == MiniVLAVQAdapter


def test_minivla_adapter_init_and_specs():
    """Test specifications of MiniVLA (Candidate B)."""
    adapter = MiniVLAAdapter(chunk_size=1, action_dim=7)
    assert adapter.chunk_size == 1
    assert adapter.action_dim == 7
    assert adapter.queue_size == 0
    assert adapter.loaded is False

    obs_spec = adapter.observation_spec
    assert "agentview_image" in obs_spec
    assert obs_spec["agentview_image"]["shape"] == (224, 224, 3)

    act_spec = adapter.action_spec
    assert act_spec["shape"] == (7,)
    assert act_spec["chunk_size"] == 1
    assert act_spec["range"] == (-1.0, 1.0)
    assert act_spec["tokenizer_type"] == "extra_action_tokenizer"


def test_minivla_adapter_fails_fast_when_not_loaded():
    """Verify that calling predict_action_chunk on unloaded adapter fails loudly."""
    adapter = MiniVLAAdapter()
    sample_obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
    }

    with pytest.raises(RuntimeError, match="not loaded.*Fail-fast per AGENTS.md Rule 10"):
        adapter.predict_action_chunk(sample_obs, instruction="pick up the bowl")

    with pytest.raises(RuntimeError, match="not loaded.*Fail-fast per AGENTS.md Rule 10"):
        adapter.select_action(sample_obs, instruction="pick up the bowl")


def test_format_instruction_prompt():
    """Test Prismatic / OpenVLA instruction prompt template."""
    prompt = format_instruction_prompt("pick up the soup and place it in the basket")
    expected = "In: What action should the robot take to pick up the soup and place it in the basket?\nOut:"
    assert prompt == expected

    # Test whitespace stripping
    prompt2 = format_instruction_prompt("  open the cabinet drawer  \n")
    assert prompt2 == "In: What action should the robot take to open the cabinet drawer?\nOut:"


def test_resize_and_normalize_image():
    """Test SigLIP 224x224 image preprocessing."""
    # Test random uint8 image
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    tensor = resize_and_normalize_image(img, target_size=(224, 224))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # SigLIP normalization with mean=0.5, std=0.5 transforms [0, 1] -> [-1, 1]
    assert tensor.min().item() >= -1.05
    assert tensor.max().item() <= 1.05


def test_extra_action_tokenizer_encode_decode():
    """Test ExtraActionTokenizer round-trip encoding and decoding matching upstream Stanford-ILIAD."""
    tokenizer = ExtraActionTokenizer(
        n_bins=256,
        action_dim=7,
        tokenizer_len=152192,
    )

    # Normalized actions in [-1.0, 1.0]
    original_actions = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    token_ids = tokenizer.encode_actions_to_tokens(original_actions)

    assert token_ids.shape == (7,)
    assert np.all(token_ids >= 152192 - 256)
    assert np.all(token_ids < 152192)

    # Decode tokens back
    decoded_actions = tokenizer.decode_token_ids_to_actions(token_ids)
    assert decoded_actions.shape == (7,)

    # Maximum quantization error for 256 bins across span 2.0 is 2.0 / 256 ≈ 0.0078
    assert np.allclose(original_actions, decoded_actions, atol=0.015)


def test_extra_action_tokenizer_unnormalization():
    """Test action unnormalization respecting dimension mask (gripper untouched)."""
    stats = {
        "libero_90": {
            "action": {
                "q01": [-0.5, -0.5, -0.2, -0.1, -0.1, -0.1, 0.0],
                "q99": [0.5, 0.5, 0.4, 0.1, 0.1, 0.1, 1.0],
                "mean": [0.0] * 7,
                "std": [0.25] * 7,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    tokenizer = ExtraActionTokenizer(
        n_bins=256,
        action_dim=7,
        dataset_statistics=stats,
    )
    assert tokenizer.has_stats is True

    # -1.0 on continuous dimensions (0..5) maps to q01
    # Gripper dimension (index 6, mask=False) remains normalized (-1.0)
    norm_min = np.full(7, -1.0, dtype=np.float32)
    unnorm_min = tokenizer.unnormalize_actions(norm_min, method="quantile")
    assert np.allclose(unnorm_min[:6], tokenizer.q01[:6], atol=1e-5)
    assert np.isclose(unnorm_min[6], -1.0)  # Gripper remains unchanged per upstream mask

    # +1.0 on continuous dimensions maps to q99; gripper remains +1.0
    norm_max = np.full(7, 1.0, dtype=np.float32)
    unnorm_max = tokenizer.unnormalize_actions(norm_max, method="quantile")
    assert np.allclose(unnorm_max[:6], tokenizer.q99[:6], atol=1e-5)
    assert np.isclose(unnorm_max[6], 1.0)


def test_minivla_postprocessor_dynamic_gripper_polarity():
    """Verify dynamic gripper polarity inversion."""
    tokenizer = ExtraActionTokenizer(n_bins=256, action_dim=7, tokenizer_len=152192)

    # DIRECT polarity
    post_direct = MiniVLAPostprocessor(tokenizer, gripper_polarity="DIRECT")
    # INVERTED polarity
    post_inverted = MiniVLAPostprocessor(tokenizer, gripper_polarity="INVERTED")

    # Tokens representing zero action, but positive gripper
    tokens = np.full(7, 152192 - 128, dtype=np.int64)
    # Set gripper token
    tokens[-1] = 152192 - 255

    action_direct = post_direct.postprocess_tokens(tokens)
    action_inverted = post_inverted.postprocess_tokens(tokens)

    # Gripper channel should be inverted
    assert np.isclose(action_direct[0, -1], -action_inverted[0, -1])


def test_minivla_validate_interface():
    """Test interface auditing output."""
    adapter = MiniVLAAdapter()
    audit = adapter.validate_interface()

    assert audit["model_id"] == "Stanford-ILIAD/minivla-libero90-prismatic"
    assert audit["action_dim"] == 7
    assert audit["chunk_size"] == 1
    assert audit["action_tokenizer"] == "extra_action_tokenizer"
    assert audit["n_bins"] == 256
    assert audit["image_resolution"] == [224, 224]
    assert audit["is_loaded"] is False
