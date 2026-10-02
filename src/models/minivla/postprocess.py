"""Postprocessing for MiniVLA Models (Phase 9).

Converts raw language model outputs (token sequences or VQ codebook indices)
into valid robot actions:
- Detokenization to normalized action values [-1, 1].
- Unnormalization using dataset statistics.
- Dynamic gripper polarity handling (never hardcoded +1/-1).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.models.minivla.action_tokenizer import ExtraActionTokenizer
from src.models.minivla.residual_vq import ResidualVQActionTokenizer


class MiniVLAPostprocessor:
    """Postprocessor for converting model token outputs into unnormalized robot actions."""

    def __init__(
        self,
        tokenizer: Union[ExtraActionTokenizer, ResidualVQActionTokenizer],
        gripper_polarity: str = "DIRECT",  # "DIRECT" or "INVERTED"
        unnorm_method: str = "quantile",
        action_dim: int = 7,
    ) -> None:
        """Initialize postprocessor.

        Args:
            tokenizer: Action tokenizer instance (ExtraActionTokenizer or ResidualVQActionTokenizer).
            gripper_polarity: Gripper action direction convention ('DIRECT' or 'INVERTED').
            unnorm_method: Unnormalization method ('quantile' or 'mean_std').
            action_dim: Dimension of action vector (default 7).
        """
        self.tokenizer = tokenizer
        self.gripper_polarity = gripper_polarity.upper()
        self.unnorm_method = unnorm_method
        self.action_dim = action_dim

    def postprocess_tokens(
        self,
        tokens_or_indices: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Convert token IDs or codebook indices into unnormalized action chunk.

        Args:
            tokens_or_indices: Array or Tensor of predicted token IDs or codebook indices.

        Returns:
            np.ndarray of shape (chunk_size, action_dim) ready for robot execution.
        """
        if isinstance(tokens_or_indices, torch.Tensor):
            arr = tokens_or_indices.detach().cpu().numpy()
        else:
            arr = np.asarray(tokens_or_indices)

        # 1. Detokenize to normalized continuous actions in [-1, 1]
        if isinstance(self.tokenizer, ExtraActionTokenizer):
            # Shape: (..., action_dim)
            if arr.ndim == 1:
                if arr.shape[0] != self.action_dim:
                    raise ValueError(
                        f"Expected 1D token array of length {self.action_dim}, got {arr.shape[0]}"
                    )
                # 1-step chunk: shape (1, 7)
                norm_action = self.tokenizer.decode_token_ids_to_actions(arr).reshape(1, self.action_dim)
            elif arr.ndim == 2:
                # Shape (chunk_size, action_dim)
                norm_action = self.tokenizer.decode_token_ids_to_actions(arr)
            else:
                raise ValueError(f"Unexpected token array shape: {arr.shape}")

            # 2. Unnormalize
            unnorm_action = self.tokenizer.unnormalize_actions(
                norm_action, method=self.unnorm_method
            )

        elif isinstance(self.tokenizer, ResidualVQActionTokenizer):
            # Tokenizer expects indices or tokens
            if arr.shape[-1] == self.tokenizer.num_codebooks:
                # Decode codebook indices or token IDs directly
                norm_action = self.tokenizer.decode_tokens_to_actions(arr)
            else:
                raise ValueError(
                    f"Expected last dimension {self.tokenizer.num_codebooks} for VQ tokens, "
                    f"got {arr.shape[-1]}"
                )

            # Ensure 2D shape (chunk_horizon, action_dim)
            if norm_action.ndim == 3 and norm_action.shape[0] == 1:
                norm_action = norm_action[0]

            # 2. Unnormalize
            unnorm_action = self.tokenizer.unnormalize_actions(
                norm_action, method=self.unnorm_method
            )

        else:
            raise TypeError(f"Unsupported action tokenizer type: {type(self.tokenizer)}")

        # 3. Dynamic Gripper Polarity Alignment
        final_actions = unnorm_action.copy()
        if self.gripper_polarity == "INVERTED":
            # Invert the gripper channel (last dimension)
            final_actions[..., -1] = -final_actions[..., -1]

        # Ensure return type is float32
        return final_actions.astype(np.float32)
