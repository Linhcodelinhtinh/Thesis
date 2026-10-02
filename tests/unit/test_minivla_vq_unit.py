"""Unit tests for MiniVLA-VQ Residual-VQ Tokenizer and Adapter (Phase 9).

Verifies codebook decoding, multi-codebook residual reconstruction,
action chunk horizon H, unnormalization, and VLAPolicy contract.
Runs completely offline (< 1s).
"""

import numpy as np
import pytest

from src.models.minivla.adapter import MiniVLAVQAdapter
from src.models.minivla.residual_vq import ResidualVQActionTokenizer
from src.models.registry import get_model_class, list_models


def test_residual_vq_tokenizer_initialization():
    """Test ResidualVQActionTokenizer setup with custom codebooks."""
    chunk_horizon = 10
    action_dim = 7
    num_codebooks = 4
    codebook_size = 256

    # Create synthetic orthogonal codebook stages
    codebooks = np.zeros((num_codebooks, codebook_size, chunk_horizon, action_dim), dtype=np.float32)
    for k in range(num_codebooks):
        # Stage k adds (k + 1) * 0.1 to index
        for v in range(codebook_size):
            codebooks[k, v, :, :] = (k + 1) * 0.05 * (v / codebook_size)

    tokenizer = ResidualVQActionTokenizer(
        chunk_horizon=chunk_horizon,
        action_dim=action_dim,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        codebooks=codebooks,
    )

    assert tokenizer.chunk_horizon == 10
    assert tokenizer.action_dim == 7
    assert tokenizer.num_codebooks == 4
    assert tokenizer.codebook_size == 256


def test_residual_vq_token_decoding_and_reconstruction():
    """Test reconstructing an action chunk from token IDs across codebooks."""
    chunk_horizon = 10
    action_dim = 7
    num_codebooks = 2
    codebook_size = 256

    codebooks = np.zeros((num_codebooks, codebook_size, chunk_horizon, action_dim), dtype=np.float32)
    # Stage 0 contributes 0.2
    codebooks[0, 10, :, :] = 0.2
    # Stage 1 contributes 0.05
    codebooks[1, 20, :, :] = 0.05

    tokenizer = ResidualVQActionTokenizer(
        chunk_horizon=chunk_horizon,
        action_dim=action_dim,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        extra_tokens_offset=1000,
        codebooks=codebooks,
    )

    # Token IDs for indices [10, 20]
    token_ids = np.array([1010, 1020], dtype=np.int64)
    chunk = tokenizer.decode_tokens_to_actions(token_ids)

    assert chunk.shape == (10, 7)
    # Reconstructed should be sum: 0.2 + 0.05 = 0.25
    assert np.allclose(chunk, 0.25, atol=1e-5)


def test_residual_vq_unnormalization():
    """Test unnormalizing action chunks from Residual-VQ."""
    stats = {
        "libero_90": {
            "action": {
                "q01": [-1.0, -1.0, 0.0, -0.5, -0.5, -0.5, -1.0],
                "q99": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
            }
        }
    }
    tokenizer = ResidualVQActionTokenizer(
        chunk_horizon=5,
        action_dim=7,
        dataset_statistics=stats,
    )
    assert tokenizer.has_stats is True

    # Zero normalized chunk corresponds to midpoint of q01 and q99
    zero_chunk = np.zeros((5, 7), dtype=np.float32)
    unnorm = tokenizer.unnormalize_actions(zero_chunk, method="quantile")

    expected_mid = (np.array(stats["libero_90"]["action"]["q01"]) + np.array(stats["libero_90"]["action"]["q99"])) / 2.0
    assert np.allclose(unnorm, expected_mid, atol=1e-5)


def test_minivla_vq_adapter_specs():
    """Verify MiniVLAVQAdapter specs and registration."""
    cls_vq = get_model_class("minivla_vq_libero90")
    adapter = cls_vq(chunk_size=10, action_dim=7)

    assert adapter.chunk_size == 10
    assert adapter.action_dim == 7
    act_spec = adapter.action_spec
    assert act_spec["chunk_size"] == 10
    assert act_spec["tokenizer_type"] == "libero_vq_extra_action_tokenizer"

    audit = adapter.validate_interface()
    assert audit["chunk_size"] == 10
    assert audit["action_tokenizer"] == "libero_vq_extra_action_tokenizer"
