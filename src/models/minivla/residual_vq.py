"""Residual-VQ Action Tokenizer for MiniVLA-VQ (Phase 9).

Implements the official Stanford-ILIAD OpenVLA-mini Residual-VQ action tokenizer
(libero_vq_extra_action_tokenizer):
- Residual Vector Quantization with K codebooks and chunk horizon H.
- Multi-codebook token decoding into continuous action chunks (H, action_dim).
- Action unnormalization using dataset statistics (q01/q99 quantile or mean/std).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ResidualVQActionTokenizer:
    """Action tokenizer for MiniVLA using Residual Vector Quantization (RVQ).

    In RVQ, an action chunk of horizon H and dimension D is reconstructed
    hierarchically across K codebook stages:
        A_norm = sum_{k=0}^{K-1} Codebook_k[index_k]
    The normalized action chunk is then unnormalized using dataset statistics.
    """

    def __init__(
        self,
        chunk_horizon: int = 10,
        action_dim: int = 7,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        extra_tokens_offset: int = 151665,
        codebooks: Optional[np.ndarray] = None,
        dataset_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Residual-VQ Action Tokenizer.

        Args:
            chunk_horizon: Number of timesteps per action chunk (H).
            action_dim: Action dimension per timestep (default 7 for Franka Panda).
            num_codebooks: Number of residual quantization levels (K).
            codebook_size: Vocabulary size per codebook (V).
            extra_tokens_offset: Base token ID offset in language model vocabulary.
            codebooks: Optional pre-loaded codebook array of shape
                (num_codebooks, codebook_size, chunk_horizon, action_dim)
                or (num_codebooks, codebook_size, chunk_horizon * action_dim).
            dataset_statistics: Training dataset normalization statistics.
        """
        self.chunk_horizon = chunk_horizon
        self.action_dim = action_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.extra_tokens_offset = extra_tokens_offset
        self.dataset_statistics = dataset_statistics or {}

        # Initialize or reshape codebooks
        self.flat_dim = chunk_horizon * action_dim
        if codebooks is not None:
            if codebooks.shape == (num_codebooks, codebook_size, chunk_horizon, action_dim):
                self.codebooks = codebooks.astype(np.float32)
            elif codebooks.shape == (num_codebooks, codebook_size, self.flat_dim):
                self.codebooks = codebooks.reshape(
                    num_codebooks, codebook_size, chunk_horizon, action_dim
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"Unexpected codebook shape: {codebooks.shape}. Expected "
                    f"({num_codebooks}, {codebook_size}, {chunk_horizon}, {action_dim}) or "
                    f"({num_codebooks}, {codebook_size}, {self.flat_dim})."
                )
        else:
            # Default zero initialization (will be populated on load)
            self.codebooks = np.zeros(
                (num_codebooks, codebook_size, chunk_horizon, action_dim),
                dtype=np.float32,
            )

        # Extract unnormalization stats for libero_90 / libero
        libero_stats = (
            self.dataset_statistics.get("libero_90", {}).get("action", {})
            or self.dataset_statistics.get("action", {})
        )
        if libero_stats:
            self.q01 = np.array(libero_stats.get("q01", [-1.0] * action_dim), dtype=np.float32)
            self.q99 = np.array(libero_stats.get("q99", [1.0] * action_dim), dtype=np.float32)
            self.mean = np.array(libero_stats.get("mean", [0.0] * action_dim), dtype=np.float32)
            self.std = np.array(libero_stats.get("std", [1.0] * action_dim), dtype=np.float32)
            self.has_stats = True
        else:
            self.q01 = np.full(action_dim, -1.0, dtype=np.float32)
            self.q99 = np.full(action_dim, 1.0, dtype=np.float32)
            self.mean = np.zeros(action_dim, dtype=np.float32)
            self.std = np.ones(action_dim, dtype=np.float32)
            self.has_stats = False

    def decode_token_ids_to_codebook_indices(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert language model token IDs to codebook indices.

        Args:
            token_ids: Array of token IDs (..., num_codebooks).

        Returns:
            Codebook indices in [0, codebook_size - 1].
        """
        indices = token_ids - self.extra_tokens_offset
        return np.clip(indices, 0, self.codebook_size - 1).astype(np.int64)

    def decode_codebook_indices_to_actions(
        self,
        codebook_indices: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct normalized action chunk from residual codebook indices.

        Args:
            codebook_indices: Array of shape (..., num_codebooks) containing indices.

        Returns:
            Normalized action chunk of shape (..., chunk_horizon, action_dim) in [-1, 1].
        """
        indices = np.asarray(codebook_indices, dtype=np.int64)
        if indices.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"Expected last dimension of indices to be num_codebooks={self.num_codebooks}, "
                f"got {indices.shape[-1]}."
            )

        # Sum vectors across codebooks: sum_{k=0}^{K-1} codebooks[k, index[k]]
        batch_shape = indices.shape[:-1]
        reconstructed = np.zeros(
            batch_shape + (self.chunk_horizon, self.action_dim), dtype=np.float32
        )

        for k in range(self.num_codebooks):
            k_indices = indices[..., k]
            # Clip index within codebook range
            k_indices = np.clip(k_indices, 0, self.codebook_size - 1)
            reconstructed += self.codebooks[k, k_indices]

        # Clip normalized action chunk to [-1, 1]
        return np.clip(reconstructed, -1.0, 1.0)

    def decode_tokens_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode raw language model token IDs directly to normalized action chunks.

        Args:
            token_ids: Array of token IDs (..., num_codebooks).

        Returns:
            Normalized action chunk (..., chunk_horizon, action_dim) in [-1, 1].
        """
        indices = self.decode_token_ids_to_codebook_indices(token_ids)
        return self.decode_codebook_indices_to_actions(indices)

    def unnormalize_actions(
        self,
        actions_normalized: np.ndarray,
        method: str = "quantile",
    ) -> np.ndarray:
        """Convert normalized action chunks [-1, 1] into robot execution space.

        Args:
            actions_normalized: Array of shape (..., chunk_horizon, action_dim) in [-1, 1].
            method: 'quantile' (q01/q99) or 'mean_std'.

        Returns:
            Unnormalized action chunk ready for robot controller.
        """
        if not self.has_stats:
            return actions_normalized.copy()

        if method == "quantile":
            # Scale from [-1, 1] -> [0, 1] -> [q01, q99]
            unit = (actions_normalized + 1.0) / 2.0
            unnorm = unit * (self.q99 - self.q01) + self.q01
            return np.clip(unnorm, self.q01, self.q99)
        elif method == "mean_std":
            return actions_normalized * self.std + self.mean
        else:
            return actions_normalized.copy()
