"""Action Tokenizer for MiniVLA (Phase 9).

Implements the official Stanford-ILIAD OpenVLA-mini extra_action_tokenizer:
- Discretizes continuous action [-1, 1] into 256 bins.
- Maps between token IDs and normalized actions.
- Unnormalizes actions using dataset statistics (q01/q99 or mean/std).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ExtraActionTokenizer:
    """Action tokenizer for MiniVLA using dedicated action bin tokens."""

    def __init__(
        self,
        n_bins: int = 256,
        action_dim: int = 7,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        extra_tokens_offset: int = 151665,
        dataset_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.action_range = action_range
        self.extra_tokens_offset = extra_tokens_offset
        self.dataset_statistics = dataset_statistics or {}

        # Precompute bin centers in [-1.0, 1.0]
        self.bin_centers = np.linspace(action_range[0], action_range[1], n_bins, dtype=np.float32)
        # Precompute bin boundaries for quantization
        self.bin_edges = np.linspace(action_range[0], action_range[1], n_bins + 1, dtype=np.float32)

        # Extract unnormalization stats for libero_90
        libero_stats = self.dataset_statistics.get("libero_90", {}).get("action", {})
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

    def encode_actions_to_tokens(self, actions: np.ndarray) -> np.ndarray:
        """Quantize normalized actions [-1, 1] into token IDs.

        Args:
            actions: (..., action_dim) array of continuous actions.

        Returns:
            (..., action_dim) array of token IDs.
        """
        clipped = np.clip(actions, self.action_range[0], self.action_range[1])
        # Find bin indices: 0 to n_bins - 1
        bin_indices = np.digitize(clipped, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        token_ids = bin_indices + self.extra_tokens_offset
        return token_ids.astype(np.int64)

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode predicted token IDs into continuous normalized actions [-1, 1].

        Args:
            token_ids: (..., action_dim) array of token IDs.

        Returns:
            (..., action_dim) array of continuous normalized actions.
        """
        bin_indices = token_ids - self.extra_tokens_offset
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        actions_normalized = self.bin_centers[bin_indices]
        return actions_normalized

    def unnormalize_actions(self, actions_normalized: np.ndarray, method: str = "quantile") -> np.ndarray:
        """Convert normalized actions [-1, 1] into original environment scale.

        Args:
            actions_normalized: (..., action_dim) in [-1, 1].
            method: 'quantile' using q01/q99 or 'mean_std'.

        Returns:
            Unnormalized actions ready for robot controller.
        """
        if not self.has_stats:
            return actions_normalized.copy()

        if method == "quantile":
            # Scale from [-1, 1] to [0, 1], then to [q01, q99]
            unit_actions = (actions_normalized + 1.0) / 2.0
            unnorm = unit_actions * (self.q99 - self.q01) + self.q01
            # Clip outlier values
            return np.clip(unnorm, self.q01, self.q99)
        elif method == "mean_std":
            return actions_normalized * self.std + self.mean
        else:
            return actions_normalized.copy()
