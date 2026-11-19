"""MiniVLA Candidate Adapter (Phase 9).

Integrates Stanford-ILIAD OpenVLA-mini (Candidate B & Candidate C):
- Candidate B: minivla_libero90 (Qwen2.5-0.5B + DinoSigLIP 224px, extra_action_tokenizer, chunk_size=1)
  Official loader: prismatic.models.load.load_vla
  Preprocessing: np.flipud -> LANCZOS 224x224 -> DinoSigLIP transforms
  Action: 7-dim quantile unnormalization, [0, 1] -> [-1, +1], sign binarization, inverted gripper for Robosuite
- Candidate C: minivla_vq_libero90 (Residual-VQ action chunking, chunk_size=H)
  Fail-closed pending release of upstream VQ-VAE codebook assets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import sys
import numpy as np
import torch

# Ensure upstream openvla-mini package is on sys.path
_UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "external" / "openvla-mini"
if _UPSTREAM_ROOT.exists() and str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

from src.models.minivla.prismatic_compat import apply_prismatic_shims
apply_prismatic_shims()

from src.models.base import VLAPolicy
from src.models.registry import register_model
from src.models.minivla.config import MiniVLAConfig
from src.models.minivla.preprocess import MiniVLAPreprocessor, get_libero_image
from src.models.minivla.action_tokenizer import ExtraActionTokenizer
from src.models.minivla.residual_vq import ResidualVQActionTokenizer
from src.models.minivla.postprocess import (
    MiniVLAPostprocessor,
    invert_gripper_action,
    normalize_gripper_action,
)

MINIVLA_OFFICIAL_RUNTIME_VERIFIED = True
MINIVLA_EVALUATION_BLOCK_REASON = None


def _find_checkpoint_pt(ckpt_dir: Path, checkpoint_file: Optional[str] = None) -> Optional[Path]:
    """Resolve an explicitly named checkpoint or the official resume checkpoint."""
    candidates = sorted((ckpt_dir / "checkpoints").glob("*.pt")) + sorted(ckpt_dir.glob("*.pt"))
    if checkpoint_file is not None:
        requested = Path(checkpoint_file)
        possible_paths = (
            [requested]
            if requested.is_absolute()
            else [ckpt_dir / requested, ckpt_dir / "checkpoints" / requested]
        )
        matches = [p.resolve() for p in possible_paths if p.is_file() and p.suffix == ".pt"]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Explicit MiniVLA checkpoint '{checkpoint_file}' does not resolve to exactly one .pt file "
                f"under '{ckpt_dir}'."
            )
        return matches[0]

    unique_candidates = sorted({p.resolve() for p in candidates})
    if not unique_candidates:
        return None
    if len(unique_candidates) > 1:
        # Check if official config resume pointer exists
        official_pointer = ckpt_dir / "checkpoints" / "step-060000-epoch-26-loss=0.1209.pt"
        if official_pointer.exists():
            return official_pointer.resolve()
        names = [p.relative_to(ckpt_dir.resolve()).as_posix() for p in unique_candidates]
        raise ValueError(
            "Multiple MiniVLA checkpoints are present; select the official evaluation checkpoint explicitly. "
            f"No mtime-based selection is permitted. Candidates: {names}"
        )
    return unique_candidates[0]


@register_model("minivla_libero90")
class MiniVLAAdapter(VLAPolicy):
    """Adapter for official MiniVLA-LIBERO90 (Candidate B, extra_action_tokenizer).

    Autoregressively generates 7 action tokens per step, mapped to continuous actions
    in [-1, 1] via 256-bin discretization and unnormalized via dataset statistics.
    """

    def __init__(
        self,
        chunk_size: int = 1,
        action_dim: int = 7,
        config: Optional[MiniVLAConfig] = None,
        invert_gripper_action: bool = True,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self.config = config or MiniVLAConfig(
            model_id="Stanford-ILIAD/minivla-libero90-prismatic",
            action_dim=action_dim,
            chunk_size=chunk_size,
        )
        self.device = self.config.device
        self.invert_gripper_action = invert_gripper_action

        # Initialize subcomponents
        self.preprocessor = MiniVLAPreprocessor(
            image_resolution=self.config.image_resolution,
            device=self.device,
            orientation=getattr(self.config, "image_orientation", "flipud"),
        )
        self.action_tokenizer = ExtraActionTokenizer(
            n_bins=self.config.n_action_bins,
            action_dim=self.action_dim,
            dataset_statistics=self.config.dataset_statistics,
        )
        gripper_polarity = "INVERTED" if invert_gripper_action else self.config.gripper_polarity
        self.postprocessor = MiniVLAPostprocessor(
            tokenizer=self.action_tokenizer,
            gripper_polarity=gripper_polarity,
            action_dim=self.action_dim,
        )

        self.model: Optional[Any] = None
        self._is_loaded: bool = False

    @property
    def observation_spec(self) -> Dict[str, Any]:
        """Specification of expected observations for MiniVLA."""
        return {
            "agentview_image": {
                "shape": (self.config.image_resolution[0], self.config.image_resolution[1], 3),
                "dtype": "uint8",
                "description": "Primary third-person agentview RGB camera observation",
            },
            "robot0_eye_in_hand_image": {
                "shape": (self.config.image_resolution[0], self.config.image_resolution[1], 3),
                "dtype": "uint8",
                "optional": True,
                "description": "Wrist eye-in-hand RGB camera observation",
            },
        }

    @property
    def action_spec(self) -> Dict[str, Any]:
        """Specification of output action dimensions, range, and semantics."""
        return {
            "shape": (self.action_dim,),
            "chunk_size": self.chunk_size,
            "range": (-1.0, 1.0),
            "semantics": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            "tokenizer_type": self.config.action_tokenizer_type,
        }

    def load(
        self,
        checkpoint_path: str = "Stanford-ILIAD/minivla-libero90-prismatic",
        device: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Load MiniVLA model weights, tokenizer, and dataset statistics using official Prismatic loader.

        Fails loudly if checkpoint path is invalid or weights are corrupt (AGENTS.md Rule 10).
        """
        if not MINIVLA_OFFICIAL_RUNTIME_VERIFIED:
            raise RuntimeError(
                "MiniVLA evaluation is blocked pending official runtime parity. "
                + (MINIVLA_EVALUATION_BLOCK_REASON or "")
            )

        if device is not None:
            self.device = device
            self.config.device = device
            self.preprocessor.device = device

        # Warm up CUDA early to initialize context and prevent driver sleep on hybrid GPUs
        if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    _ = torch.empty(1, device=self.device)
                except Exception:
                    pass

        ckpt = Path(checkpoint_path)

        # Allow passing direct .pt checkpoint file or directory
        if ckpt.is_file() and ckpt.suffix == ".pt":
            pt_file = ckpt.resolve()
            ckpt_dir = ckpt.parents[1] if ckpt.parent.name == "checkpoints" else ckpt.parent
            self.config = MiniVLAConfig.from_checkpoint(ckpt_dir, device=self.device)
        else:
            # Handle Hugging Face Hub IDs or local directory aliases
            if not ckpt.exists():
                for cand_name in [
                    checkpoint_path.split("/")[-1],
                    "minivla_libero90",
                    "minivla-libero90-prismatic",
                ]:
                    candidate_dir = Path("resources/checkpoints") / cand_name
                    if candidate_dir.exists():
                        ckpt = candidate_dir
                        break

            if not ckpt.exists():
                raise FileNotFoundError(
                    f"MiniVLA checkpoint directory not found: {checkpoint_path}. "
                    f"Fail-fast per AGENTS.md Rule 10 (no fallback allowed)."
                )

            # Load configuration & statistics
            self.config = MiniVLAConfig.from_checkpoint(ckpt, device=self.device)

            # Locate .pt weights file
            pt_file = _find_checkpoint_pt(ckpt, checkpoint_file=checkpoint_file)

        if pt_file is None or not pt_file.exists():
            raise FileNotFoundError(
                f"MiniVLA checkpoint directory '{ckpt}' does not contain any valid .pt model weights file!\n"
                f"Expected .pt checkpoint under '{ckpt}/checkpoints/step-*.pt'.\n"
                f"Please ensure complete checkpoint download per AGENTS.md Rule 10."
            )

        # Reinitialize preprocessor and tokenizer with checkpoint config & stats
        orientation = kwargs.get("image_orientation", getattr(self.config, "image_orientation", "flipud"))
        self.preprocessor = MiniVLAPreprocessor(
            image_resolution=self.config.image_resolution,
            device=self.device,
            orientation=orientation,
        )
        self.action_tokenizer = ExtraActionTokenizer(
            n_bins=self.config.n_action_bins,
            action_dim=self.action_dim,
            tokenizer_len=self.config.tokenizer_len,
            dataset_statistics=self.config.dataset_statistics,
        )
        polarity = "INVERTED" if self.invert_gripper_action else self.config.gripper_polarity
        self.postprocessor = MiniVLAPostprocessor(
            tokenizer=self.action_tokenizer,
            gripper_polarity=polarity,
            action_dim=self.action_dim,
        )

        # Load PyTorch Model Architecture & Weights via official Prismatic loader
        print(f"[MiniVLA] Loading official Prismatic model from: {pt_file}...")
        try:
            from prismatic.models.load import load_vla
        except ImportError as exc:
            raise ImportError(
                f"Failed to import official Prismatic loader: {exc}. "
                f"Ensure external/openvla-mini is available and on sys.path."
            ) from exc

        vla = load_vla(
            model_id_or_path=str(pt_file),
            load_for_training=False,
        )

        # Cast to device and appropriate precision (bfloat16 in VRAM)
        target_device = torch.device(self.device)
        if target_device.type == "cuda" and torch.cuda.is_available():
            vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype, device=target_device)
            vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype, device=target_device)
            vla.to(dtype=vla.llm_backbone.half_precision_dtype, device=target_device)
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        else:
            vla.to(target_device)

        self.model = vla
        self._is_loaded = True
        print(f"[MiniVLA] Model successfully loaded (Device: {self.device}, Action Dim: {self.action_dim}).")

    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert raw environment observations into model input features."""
        return self.preprocessor.preprocess_observation(obs, instruction)

    def predict_action_chunk(
        self, obs: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """Generate and return an action chunk using official Prismatic pipeline.

        Fails fast if model is not loaded.
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError(
                f"Model '{self.config.model_id}' is not loaded. "
                f"Cannot predict actions. Fail-fast per AGENTS.md Rule 10."
            )

        inputs = self.preprocess(obs, instruction)
        pil_image = inputs["image"]

        with torch.inference_mode():
            # Query official Prismatic OpenVLA model
            # predict_action handles prompt building, tokenization, image transform, and action unnormalization
            raw_action = self.model.predict_action(
                pil_image,
                instruction,
                unnorm_key="libero_90",
            )

        # Apply official LIBERO gripper postprocessing
        # 1. Normalize gripper [0, 1] -> [-1, +1] and binarize
        action = normalize_gripper_action(raw_action, binarize=True)

        # 2. Invert gripper sign for Robosuite environment
        if self.invert_gripper_action:
            action = invert_gripper_action(action)

        chunk = action.reshape(self.chunk_size, self.action_dim).astype(np.float32)

        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Action chunk shape mismatch: expected ({self.chunk_size}, {self.action_dim}), "
                f"got {chunk.shape}."
            )
        return chunk

    def postprocess(self, output: Any) -> np.ndarray:
        """Detokenize and unnormalize raw model output tokens into robot actions."""
        return self.postprocessor.postprocess_tokens(output)

    def validate_interface(self) -> Dict[str, Any]:
        """Audit and report interface details."""
        return {
            "model_id": self.config.model_id,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "action_tokenizer": self.config.action_tokenizer_type,
            "n_bins": self.config.n_action_bins,
            "image_resolution": list(self.config.image_resolution),
            "gripper_polarity": self.postprocessor.gripper_polarity,
            "has_statistics": self.action_tokenizer.has_stats,
            "is_loaded": self._is_loaded,
            "model_active": self.model is not None,
            "evaluation_ready": MINIVLA_OFFICIAL_RUNTIME_VERIFIED,
            "evaluation_block_reason": MINIVLA_EVALUATION_BLOCK_REASON,
        }


@register_model("minivla_vq_libero90")
class MiniVLAVQAdapter(MiniVLAAdapter):
    """Adapter for official MiniVLA-VQ-LIBERO90 (Candidate C, Residual-VQ chunking).

    Predicts discrete codebook indices representing action chunks of horizon H,
    reconstructed hierarchically across K codebook stages.
    """

    def __init__(
        self,
        chunk_size: int = 10,
        action_dim: int = 7,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        config: Optional[MiniVLAConfig] = None,
        invert_gripper_action: bool = True,
    ) -> None:
        cfg = config or MiniVLAConfig(
            model_id="Stanford-ILIAD/minivla-vq-libero90-prismatic",
            action_tokenizer_type="libero_vq_extra_action_tokenizer",
            action_dim=action_dim,
            chunk_size=chunk_size,
        )
        super().__init__(
            chunk_size=chunk_size,
            action_dim=action_dim,
            config=cfg,
            invert_gripper_action=invert_gripper_action,
        )
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # Replace tokenizer with ResidualVQActionTokenizer
        self.action_tokenizer = ResidualVQActionTokenizer(
            chunk_horizon=chunk_size,
            action_dim=action_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            dataset_statistics=self.config.dataset_statistics,
        )
        polarity = "INVERTED" if invert_gripper_action else self.config.gripper_polarity
        self.postprocessor = MiniVLAPostprocessor(
            tokenizer=self.action_tokenizer,
            gripper_polarity=polarity,
            action_dim=self.action_dim,
        )

    def load(
        self,
        checkpoint_path: str = "Stanford-ILIAD/minivla-vq-libero90-prismatic",
        device: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Load MiniVLA-VQ model weights and Residual-VQ tokenizer.

        Fails loudly if VQ codebook or VQ-VAE weights are missing (AGENTS.md Rule 10).
        """
        super().load(checkpoint_path=checkpoint_path, device=device, checkpoint_file=checkpoint_file, **kwargs)

        # Restore ResidualVQActionTokenizer with checkpoint statistics
        self.action_tokenizer = ResidualVQActionTokenizer(
            chunk_horizon=self.chunk_size,
            action_dim=self.action_dim,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            dataset_statistics=self.config.dataset_statistics,
        )
        polarity = "INVERTED" if self.invert_gripper_action else self.config.gripper_polarity
        self.postprocessor = MiniVLAPostprocessor(
            tokenizer=self.action_tokenizer,
            gripper_polarity=polarity,
            action_dim=self.action_dim,
        )

        # Enforce fail-fast check: Candidate C requires trained VQ codebook
        if np.all(self.action_tokenizer.codebooks == 0):
            raise FileNotFoundError(
                "Candidate C (MiniVLA-VQ) requires external VQ-VAE model weights from Stanford-ILIAD/pretrain_vq. "
                "Current codebook is unpopulated (all zeros). "
                "Fail-fast per AGENTS.md Rule 2 & 10 (no zero-codebook fallback allowed)."
            )
