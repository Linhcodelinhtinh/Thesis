"""Preprocessing for MiniVLA Models (Phase 9).

Implements official Stanford-ILIAD OpenVLA-mini preprocessing:
- SigLIP-224px visual observation transforms (bilinear resize to 224x224, normalize).
- Language prompt formatting per Prismatic VLM standard:
  "In: What action should the robot take to {instruction}?\\nOut:"
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


def format_instruction_prompt(instruction: str) -> str:
    """Format language instruction into official Prismatic/OpenVLA prompt.

    Args:
        instruction: Raw natural language instruction (e.g. 'pick up the soup').

    Returns:
        Formatted prompt string.
    """
    clean_inst = instruction.strip()
    return f"In: What action should the robot take to {clean_inst}?\nOut:"


def resize_and_normalize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Resize RGB image to target size and apply SigLIP normalization.

    Args:
        image: RGB array (H, W, 3), uint8 [0, 255] or float [0, 1].
        target_size: (H, W) target resolution (default 224x224 for SigLIP).
        mean: Channel-wise mean for standardization (default (0.5, 0.5, 0.5)).
        std: Channel-wise std for standardization (default (0.5, 0.5, 0.5)).

    Returns:
        torch.Tensor of shape (3, H, W) normalized to float32.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array (H, W, C), got ndim={image.ndim}")

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            uint8_img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        uint8_img = image.astype(np.uint8)

    # Resize using PIL Bilinear interpolation
    pil_img = Image.fromarray(uint8_img)
    resized_pil = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_arr = np.array(resized_pil, dtype=np.float32) / 255.0  # [0.0, 1.0]

    # Convert to CHW tensor: (3, H, W)
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

    # Normalize with mean and std: (x - mean) / std
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    normalized = (tensor - mean_t) / std_t

    return normalized


class MiniVLAPreprocessor:
    """Preprocessor for MiniVLA visual observations and language instructions."""

    def __init__(
        self,
        image_resolution: Tuple[int, int] = (224, 224),
        primary_camera_key: str = "agentview_image",
        wrist_camera_key: str = "robot0_eye_in_hand_image",
        device: str = "cpu",
    ) -> None:
        self.image_resolution = image_resolution
        self.primary_camera_key = primary_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.device = device

    def preprocess_observation(
        self,
        obs: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """Convert raw environment observation into model input dictionary.

        Args:
            obs: Raw observation dictionary containing camera views and proprioception.
            instruction: Task description text.

        Returns:
            Dictionary with formatted prompt, image tensors, and metadata.
        """
        if self.primary_camera_key not in obs:
            raise KeyError(
                f"Missing required primary camera key '{self.primary_camera_key}' in obs. "
                f"Available keys: {list(obs.keys())}"
            )

        # Primary camera (agentview)
        primary_img = obs[self.primary_camera_key]
        pixel_values = resize_and_normalize_image(
            primary_img, target_size=self.image_resolution
        )

        inputs: Dict[str, Any] = {
            "pixel_values": pixel_values.unsqueeze(0).to(self.device),  # (1, 3, H, W)
            "prompt": format_instruction_prompt(instruction),
            "instruction": instruction,
        }

        # Optional wrist camera
        if self.wrist_camera_key in obs and obs[self.wrist_camera_key] is not None:
            wrist_img = obs[self.wrist_camera_key]
            wrist_pixel_values = resize_and_normalize_image(
                wrist_img, target_size=self.image_resolution
            )
            inputs["wrist_pixel_values"] = wrist_pixel_values.unsqueeze(0).to(self.device)

        return inputs
