"""SmolVLA Model Adapter for LIBERO Evaluation (Phase 5 Hardening).

Acts strictly as a bridge between LIBERO environment observations and official
LeRobot SmolVLA policy:
  LiberoEnv
      ↓
  observation mapper (LIBERO observation ↕ LeRobot feature representation)
      ↓
  official LeRobot preprocessor (Rename, BatchDim, NewLine, Tokenizer, Device, Normalizer)
      ↓
  official SmolVLAPolicy (Flow-Matching, Action Sampling)
      ↓
  official postprocessor (UnnormalizerProcessorStep, DeviceProcessorStep)
      ↓
  environment action (7D continuous OSC)

Per instructions, the adapter does NOT reimplement model internals (flow-matching,
tokenization, normalization, etc.), delegating execution strictly to official
LeRobot implementations. All zero-action fallbacks have been eliminated per
AGENTS.md Rule 10 (Fail-Fast).
"""

import types
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.models.base import VLAPolicy
from src.models.registry import register_model
from src.utils.transforms import quat2axisangle


def _ensure_pyav_compatibility() -> None:
    """Monkey-patch PyAV 14+ incompatibility with legacy LeRobot datasets import."""
    try:
        import av
        if not hasattr(av, "option"):
            av.option = types.SimpleNamespace(Option=object)
    except ImportError:
        pass


def _resize_rgb_image(img: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize RGB image to target (H, W) using bilinear interpolation, returning float32 [0, 1]."""
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image array (H, W, C), got ndim={img.ndim}")

    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            uint8_img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        uint8_img = img.astype(np.uint8)

    pil_img = Image.fromarray(uint8_img)
    resized_pil = pil_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0
    return resized_arr


@register_model("smolvla_libero")
class SmolVLAAdapter(VLAPolicy):
    """Adapter for official SmolVLA (lerobot/smolvla_libero) evaluated on LIBERO."""

    def __init__(self, chunk_size: int = 50, action_dim: int = 7) -> None:
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self.checkpoint_path: Optional[str] = None
        self.device: str = "cpu"
        self.policy: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.postprocessor: Optional[Any] = None
        self._is_loaded: bool = False

    def load(
        self,
        checkpoint_path: str = "lerobot/smolvla_libero",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Load official SmolVLA policy and official pre/post-processors from checkpoint.

        Args:
            checkpoint_path: HuggingFace Hub repo id or local path.
            device: Target execution device ('cuda', 'cpu').
            **kwargs: Extra arguments passed to from_pretrained.
        """
        _ensure_pyav_compatibility()

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path

        # 1. Load official SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path, **kwargs)
        self.policy.to(self.device)
        self.policy.eval()

        # 2. Load official preprocessor and postprocessor pipelines
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": self.device}},
        )

        self._is_loaded = True

    def reset(self) -> None:
        """Reset internal action queue and underlying policy recurrent state."""
        super().reset()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def build_raw_features(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert LIBERO simulator observations into the raw dictionary format for LeRobot preprocessor.

        Computes:
          - observation.images.image: agentview resized to (3, 256, 256) float tensor in [0, 1]
          - observation.images.image2: wrist camera resized to (3, 256, 256) float tensor in [0, 1]
          - observation.state: 8D state vector [pos (3,), axis_angle (3,), gripper_qpos (2,)]
          - task: language instruction string
        """
        raw_features: Dict[str, Any] = {}

        # 1. Images: Convert to channel-first float tensors (3, 256, 256)
        if "agentview_image" in obs:
            agentview_arr = _resize_rgb_image(obs["agentview_image"], (256, 256))
            raw_features["observation.images.image"] = (
                torch.from_numpy(agentview_arr).permute(2, 0, 1).float()
            )
        elif "observation.images.image" in obs:
            raw_features["observation.images.image"] = obs["observation.images.image"]
        elif "observation.images.camera1" in obs:
            raw_features["observation.images.image"] = obs["observation.images.camera1"]

        if "robot0_eye_in_hand_image" in obs:
            wrist_arr = _resize_rgb_image(obs["robot0_eye_in_hand_image"], (256, 256))
            raw_features["observation.images.image2"] = (
                torch.from_numpy(wrist_arr).permute(2, 0, 1).float()
            )
        elif "observation.images.image2" in obs:
            raw_features["observation.images.image2"] = obs["observation.images.image2"]
        elif "observation.images.camera2" in obs:
            raw_features["observation.images.image2"] = obs["observation.images.camera2"]

        # 2. State: Compute 8D vector [pos(3), axis_angle(3), gripper(2)]
        if "observation.state" in obs:
            state_val = obs["observation.state"]
            if isinstance(state_val, np.ndarray):
                raw_features["observation.state"] = torch.from_numpy(state_val).float()
            else:
                raw_features["observation.state"] = state_val.float()
        elif "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).flatten()
            axis_angle = quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32)

            if "robot0_gripper_qpos" in obs:
                gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).flatten()
            else:
                # Default open gripper values if gripper_qpos not provided
                gripper = np.array([0.02, -0.02], dtype=np.float32)

            state_8d = np.concatenate([pos, axis_angle, gripper])
            raw_features["observation.state"] = torch.from_numpy(state_8d).float()

        # 3. Language instruction
        raw_features["task"] = instruction
        return raw_features

    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Preprocess observation through official LeRobot preprocessor pipeline.

        Fails fast if preprocessor is not loaded.
        """
        raw_features = self.build_raw_features(obs, instruction)

        if self.preprocessor is not None:
            return self.preprocessor(raw_features)

        # If running in offline test mode without loaded checkpoint
        if not self._is_loaded:
            raise RuntimeError(
                "SmolVLA policy and preprocessor are not loaded. Call load(checkpoint_path) "
                "before calling preprocess. Fail-fast per AGENTS.md Rule 10."
            )
        return raw_features

    def predict_action_chunk(
        self, obs: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """First-Class API: Generate full action chunk via official SmolVLA pipeline.

        Executes:
          1. official preprocessor (Rename, BatchDim, Tokenizer, Device, Normalizer)
          2. official SmolVLAPolicy forward pass
          3. official postprocessor (UnnormalizerProcessorStep, DeviceProcessorStep)

        Fails fast if policy or processors are not loaded.
        """
        if not self._is_loaded or self.policy is None or self.postprocessor is None:
            raise RuntimeError(
                "SmolVLA policy is not loaded. Call load(checkpoint_path) before calling "
                "predict_action_chunk. Fail-fast per AGENTS.md Rule 10 (No zero-action fallback)."
            )

        batch = self.preprocess(obs, instruction)

        with torch.no_grad():
            if hasattr(self.policy, "predict_action_chunk"):
                raw_chunk = self.policy.predict_action_chunk(batch)
            elif hasattr(self.policy, "select_actions"):
                raw_chunk = self.policy.select_actions(batch)
            else:
                raw_chunk = self.policy(batch)

            # Apply official unnormalizer postprocessor
            unnormalized_actions = self.postprocessor(raw_chunk)

        chunk = self.postprocess(unnormalized_actions)
        return chunk

    def postprocess(self, output: Any) -> np.ndarray:
        """Format postprocessor output into numpy array of shape (chunk_size, action_dim)."""
        if isinstance(output, torch.Tensor):
            actions = output.detach().cpu().numpy()
        elif isinstance(output, np.ndarray):
            actions = output
        else:
            actions = np.asarray(output, dtype=np.float32)

        # Squeeze batch dimension if present: (1, 50, 7) -> (50, 7)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)

        if actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected (*, {self.action_dim}), got {actions.shape}"
            )

        return actions.astype(np.float32)

    def validate_interface(self) -> Dict[str, Any]:
        """Return interface audit details for this adapter."""
        return {
            "model_type": "SmolVLAAdapter",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "cameras": ["observation.images.camera1", "observation.images.camera2"],
            "camera_resolution": [256, 256],
            "state_dim": 8,
            "state_semantics": [
                "eef_pos_x", "eef_pos_y", "eef_pos_z",
                "eef_axis_x", "eef_axis_y", "eef_axis_z",
                "gripper_qpos_0", "gripper_qpos_1",
            ],
            "has_official_preprocessor": self.preprocessor is not None,
            "has_official_postprocessor": self.postprocessor is not None,
            "queue_size": self.queue_size,
        }

    @property
    def observation_spec(self) -> Dict[str, Any]:
        """Specification of expected observation dictionary from LIBERO."""
        return {
            "agentview_image": {"shape": (128, 128, 3), "dtype": "uint8_or_float"},
            "robot0_eye_in_hand_image": {"shape": (128, 128, 3), "dtype": "uint8_or_float"},
            "robot0_eef_pos": {"shape": (3,), "dtype": "float32"},
            "robot0_eef_quat": {"shape": (4,), "dtype": "float32"},
            "robot0_gripper_qpos": {"shape": (2,), "dtype": "float32"},
        }
