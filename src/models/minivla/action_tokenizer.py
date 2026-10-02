"""Action Tokenizer for MiniVLA (Phase 9).

Strictly adheres to official Stanford-ILIAD OpenVLA-mini extra_action_tokenizer specification:
- Discretizes continuous action [-1, 1] into 256 bins with 255 bin centers.
- Maps token IDs in REVERSE order from the vocabulary tail:
    token_id = tokenizer_len - digitized_action
    digitized_action = tokenizer_len - token_id
- Unnormalizes actions using dataset statistics (q01/q99) while respecting the dimension mask:
    Gripper dimension (mask=False) is preserved in normalized scale [-1, 1], not scaled to [0, 1].
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ExtraActionTokenizer:
    """Action tokenizer for MiniVLA matching Stanford-ILIAD OpenVLA-mini exact specification."""

    def __init__(
        self,
        n_bins: int = 256,
        action_dim: int = 7,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        tokenizer_len: int = 152192,  # Qwen2.5 base (151936) + 256 extra action tokens
        dataset_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.min_action = float(action_range[0])
        self.max_action = float(action_range[1])
        self.tokenizer_len = tokenizer_len
        self.dataset_statistics = dataset_statistics or {}

        # Upstream bins and bin centers: 256 bins -> 255 bin centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins, dtype=np.float32)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Action token index bounds
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

        # Extract unnormalization stats for libero_90
        libero_stats = self.dataset_statistics.get("libero_90", {}).get("action", {}) or self.dataset_statistics.get("action", {})
        if libero_stats:
            self.q01 = np.array(libero_stats.get("q01", [-1.0] * action_dim), dtype=np.float32)
            self.q99 = np.array(libero_stats.get("q99", [1.0] * action_dim), dtype=np.float32)
            self.mean = np.array(libero_stats.get("mean", [0.0] * action_dim), dtype=np.float32)
            self.std = np.array(libero_stats.get("std", [1.0] * action_dim), dtype=np.float32)
            # Upstream dataset_statistics includes boolean mask: [True, True, True, True, True, True, False]
            mask_raw = libero_stats.get("mask", [True] * (action_dim - 1) + [False])
            self.mask = np.array(mask_raw, dtype=bool)
            self.has_stats = True
        else:
            self.q01 = np.full(action_dim, -1.0, dtype=np.float32)
            self.q99 = np.full(action_dim, 1.0, dtype=np.float32)
            self.mean = np.zeros(action_dim, dtype=np.float32)
            self.std = np.ones(action_dim, dtype=np.float32)
            self.mask = np.array([True] * (action_dim - 1) + [False], dtype=bool)
            self.has_stats = False

    def encode_actions_to_tokens(self, actions: np.ndarray) -> np.ndarray:
        """Quantize normalized actions [-1, 1] into token IDs matching upstream Stanford-ILIAD.

        Discretizes actions into [1, n_bins] and maps in reverse order from tokenizer_len.
        """
        clipped = np.clip(actions, self.min_action, self.max_action)
        # np.digitize returns indices in [1, n_bins]
        discretized_action = np.digitize(clipped, self.bins)
        token_ids = self.tokenizer_len - discretized_action
        return token_ids.astype(np.int64)

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode predicted token IDs into continuous normalized actions [-1, 1] per upstream openvla-mini."""
        token_ids = np.asarray(token_ids)
        discretized_actions = self.tokenizer_len - token_ids
        discretized_actions = np.clip(discretized_actions - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized_actions]

    def unnormalize_actions(self, actions_normalized: np.ndarray, method: str = "quantile") -> np.ndarray:
        """Convert normalized actions into environment scale respecting mask (dim 6 gripper untouched).

        Per official upstream OpenVLA implementation:
            actions = np.where(
                mask,
                0.5 * (normalized + 1.0) * (q99 - q01) + q01,
                normalized
            )
        """
        if not self.has_stats:
            return actions_normalized.copy()

        if method == "quantile":
            # Unnormalize dimensions where mask is True; keep dimensions where mask is False (e.g. gripper)
            unnorm_continuous = 0.5 * (actions_normalized + 1.0) * (self.q99 - self.q01) + self.q01
            # Clip outlier values on continuous dimensions
            unnorm_continuous = np.clip(unnorm_continuous, self.q01, self.q99)
            actions = np.where(self.mask, unnorm_continuous, actions_normalized)
            return actions.astype(np.float32)
        elif method == "mean_std":
            unnorm_mean_std = actions_normalized * self.std + self.mean
            actions = np.where(self.mask, unnorm_mean_std, actions_normalized)
            return actions.astype(np.float32)
        else:
            return actions_normalized.copy()
