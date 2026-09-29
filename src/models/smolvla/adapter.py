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


_ensure_pyav_compatibility()


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
    """Adapter for official SmolVLA (lerobot/smolvla_libero) evaluated on LIBERO.

    Delegates environment-level transformation (image 180° rotation, 8D state extraction)
    strictly to official LeRobot LiberoProcessorStep, and policy-level transformations
    (tokenization, normalizer, batching) to official LeRobot PolicyProcessorPipeline.
    """

    def __init__(
        self,
        chunk_size: int = 50,
        action_dim: int = 7,
        invert_gripper_action: bool = True,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self.checkpoint_path: Optional[str] = None
        self.device: str = "cpu"
        self.policy: Optional[Any] = None
        self.env_preprocessor: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.postprocessor: Optional[Any] = None
        self._is_loaded: bool = False
        self.invert_gripper_action: bool = invert_gripper_action

        # Telemetry cache for dual-action observability (Phase 2)
        self.last_raw_normalized_chunk: Optional[np.ndarray] = None
        self.last_unnormalized_chunk: Optional[np.ndarray] = None
        self.last_executed_chunk: Optional[np.ndarray] = None

    def _ensure_env_preprocessor(self) -> Any:
        """Lazily initialize official LeRobot LiberoProcessorStep."""
        if self.env_preprocessor is None:
            _ensure_pyav_compatibility()
            from lerobot.processor.env_processor import LiberoProcessorStep
            self.env_preprocessor = LiberoProcessorStep()
        return self.env_preprocessor

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

        # 2. Instantiate official LiberoProcessorStep (environment preprocessor)
        self.env_preprocessor = self._ensure_env_preprocessor()

        # 3. Load official policy preprocessor and postprocessor pipelines
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
        """Convert LIBERO simulator observations using official LiberoProcessorStep.

        Delegates image rotation (180° / flip H,W) and 8D state vector formulation
        strictly to official LeRobot LiberoProcessorStep per AGENTS.md Rule 3 & 7.

        Computes:
          - observation.images.image: agentview rotated 180° via LiberoProcessorStep (3, 256, 256)
          - observation.images.image2: wrist rotated 180° via LiberoProcessorStep (3, 256, 256)
          - observation.state: 8D state vector [pos (3,), axis_angle (3,), gripper_qpos (2,)]
          - task: language instruction string
        """
        env_step = self._ensure_env_preprocessor()
        raw_obs: Dict[str, Any] = {}

        # 1. Images: Convert to unflipped (1, 3, 256, 256) float tensors in [0, 1] for LiberoProcessorStep
        if "agentview_image" in obs:
            agentview_arr = _resize_rgb_image(obs["agentview_image"], (256, 256))
            raw_obs["observation.images.image"] = (
                torch.from_numpy(agentview_arr).permute(2, 0, 1).unsqueeze(0).float()
            )
        elif "observation.images.image" in obs:
            img = obs["observation.images.image"]
            raw_obs["observation.images.image"] = img.unsqueeze(0) if img.dim() == 3 else img
        elif "observation.images.camera1" in obs:
            img = obs["observation.images.camera1"]
            raw_obs["observation.images.image"] = img.unsqueeze(0) if img.dim() == 3 else img

        if "robot0_eye_in_hand_image" in obs:
            wrist_arr = _resize_rgb_image(obs["robot0_eye_in_hand_image"], (256, 256))
            raw_obs["observation.images.image2"] = (
                torch.from_numpy(wrist_arr).permute(2, 0, 1).unsqueeze(0).float()
            )
        elif "observation.images.image2" in obs:
            img = obs["observation.images.image2"]
            raw_obs["observation.images.image2"] = img.unsqueeze(0) if img.dim() == 3 else img
        elif "observation.images.camera2" in obs:
            img = obs["observation.images.camera2"]
            raw_obs["observation.images.image2"] = img.unsqueeze(0) if img.dim() == 3 else img

        # 2. State: Structure as observation.robot_state for LiberoProcessorStep or pass 8D state
        if "observation.state" in obs:
            state_val = obs["observation.state"]
            if isinstance(state_val, np.ndarray):
                state_tensor = torch.from_numpy(state_val).float()
            elif isinstance(state_val, torch.Tensor):
                state_tensor = state_val.float()
            else:
                state_tensor = torch.as_tensor(state_val, dtype=torch.float32)

            if state_tensor.shape[-1] != 8:
                raise ValueError(
                    f"Observation state dimension mismatch: expected 8D vector, got shape {state_tensor.shape}. "
                    "SmolVLA audit contract requires [eef_pos(3), eef_axis(3), gripper_qpos(2)] (runtime_dim=8)."
                )
            raw_obs["observation.state"] = state_tensor.unsqueeze(0) if state_tensor.dim() == 1 else state_tensor
        elif "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            pos_t = torch.as_tensor(obs["robot0_eef_pos"], dtype=torch.float32).flatten()
            quat_t = torch.as_tensor(obs["robot0_eef_quat"], dtype=torch.float32).flatten()
            if pos_t.numel() != 3:
                raise ValueError(f"Expected robot0_eef_pos of shape (3,), got shape ({pos_t.numel()},)")
            if quat_t.numel() != 4:
                raise ValueError(f"Expected robot0_eef_quat of shape (4,), got shape ({quat_t.numel()},)")

            if "robot0_gripper_qpos" in obs:
                gripper_t = torch.as_tensor(obs["robot0_gripper_qpos"], dtype=torch.float32).flatten()
                if gripper_t.numel() != 2:
                    raise ValueError(f"Expected robot0_gripper_qpos of shape (2,), got shape ({gripper_t.numel()},)")
            else:
                gripper_t = torch.tensor([0.02, -0.02], dtype=torch.float32)

            raw_obs["observation.robot_state"] = {
                "eef": {"pos": pos_t.reshape(1, 3), "quat": quat_t.reshape(1, 4)},
                "gripper": {"qpos": gripper_t.reshape(1, 2)},
            }

        # 3. Instruction
        raw_obs["task"] = instruction

        # 4. Process through official LeRobot LiberoProcessorStep
        processed_obs = env_step._process_observation(raw_obs)

        # 5. Format features for policy preprocessor (unbatched shapes (3, 256, 256) and (8,))
        features: Dict[str, Any] = {}
        if "observation.images.image" in processed_obs:
            features["observation.images.image"] = processed_obs["observation.images.image"].squeeze(0)
        if "observation.images.image2" in processed_obs:
            features["observation.images.image2"] = processed_obs["observation.images.image2"].squeeze(0)
        if "observation.state" in processed_obs:
            features["observation.state"] = processed_obs["observation.state"].squeeze(0)
        features["task"] = instruction

        return features

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

            # Cache raw normalized chunk for telemetry (Phase 2)
            if isinstance(raw_chunk, torch.Tensor):
                raw_norm_arr = raw_chunk.detach().cpu().numpy().copy()
            else:
                raw_norm_arr = np.asarray(raw_chunk, dtype=np.float32).copy()

            if raw_norm_arr.ndim == 3 and raw_norm_arr.shape[0] == 1:
                raw_norm_arr = raw_norm_arr[0]
            self.last_raw_normalized_chunk = raw_norm_arr

            # Apply official unnormalizer postprocessor
            unnormalized_actions = self.postprocessor(raw_chunk)

            # Cache unnormalized chunk before gripper inversion
            raw_unnorm = self.postprocess(unnormalized_actions, apply_gripper_inversion=False)
            self.last_unnormalized_chunk = raw_unnorm.copy()

        chunk = self.postprocess(unnormalized_actions, apply_gripper_inversion=True)
        self.last_executed_chunk = chunk.copy()
        return chunk

    def postprocess(self, output: Any, apply_gripper_inversion: bool = True) -> np.ndarray:
        """Format postprocessor output into numpy array of shape (chunk_size, action_dim).

        Applies ADR-0009 gripper polarity inversion when apply_gripper_inversion and
        self.invert_gripper_action are True (+1 Open -> -1 Open for Robosuite).
        """
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

        actions_out = actions.astype(np.float32).copy()
        if apply_gripper_inversion and self.invert_gripper_action:
            # ADR-0009: Invert gripper polarity from RLDS (+1=Open, -1=Close) to Robosuite (-1=Open, +1=Close)
            actions_out[..., -1] = -1.0 * actions_out[..., -1]

        return actions_out

    def get_last_telemetry(self) -> Dict[str, Optional[np.ndarray]]:
        """Return dual-action telemetry (raw normalized, unnormalized, executed) for last inference."""
        return {
            "raw_normalized_chunk": self.last_raw_normalized_chunk,
            "unnormalized_chunk": self.last_unnormalized_chunk,
            "executed_chunk": self.last_executed_chunk,
        }

    def validate_interface(self) -> Dict[str, Any]:
        """Return interface audit details for this adapter."""
        return {
            "model_type": "SmolVLAAdapter",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "invert_gripper_action": self.invert_gripper_action,
            "gripper_action_polarity": (
                "INVERTED_RLDS_TO_ROBOSUITE" if self.invert_gripper_action else "DIRECT"
            ),
            "cameras": ["observation.images.camera1", "observation.images.camera2"],
            "camera_resolution": [256, 256],
            "observation_state": {
                "runtime_dim": 8,
                "semantics": [
                    "eef_pos_x", "eef_pos_y", "eef_pos_z",
                    "eef_axis_x", "eef_axis_y", "eef_axis_z",
                    "gripper_qpos_0", "gripper_qpos_1",
                ],
                "source_of_truth": [
                    "policy_preprocessor_step_5_normalizer_processor.safetensors"
                ],
            },
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
