"""MiniVLA Model Adapter for LIBERO Evaluation (Phase 9).

Integrates Stanford-ILIAD OpenVLA-mini (Candidate B & Candidate C):
- Candidate B: minivla_libero90 (Qwen2.5-0.5B + SigLIP 224px, extra_action_tokenizer, chunk_size=1)
- Candidate C: minivla_vq_libero90 (Residual-VQ action chunking, chunk_size=H)

Adheres strictly to the VLAPolicy interface, AGENTS.md fail-fast rules,
and Stanford-ILIAD provenance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import glob
import json
import numpy as np
import torch

from src.models.base import VLAPolicy
from src.models.registry import register_model
from src.models.minivla.config import MiniVLAConfig
from src.models.minivla.preprocess import MiniVLAPreprocessor
from src.models.minivla.action_tokenizer import ExtraActionTokenizer
from src.models.minivla.residual_vq import ResidualVQActionTokenizer
from src.models.minivla.postprocess import MiniVLAPostprocessor
from src.models.minivla.prismatic_model import MiniVLAPrismaticModel


def _find_checkpoint_pt(ckpt_dir: Path) -> Optional[Path]:
    """Find the latest or primary .pt weights file in the checkpoint directory."""
    # 1. Look in checkpoints/ subdirectory
    sub_ckpts = list((ckpt_dir / "checkpoints").glob("*.pt"))
    if sub_ckpts:
        # Prefer step with highest step number or lowest loss
        sub_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sub_ckpts[0]

    # 2. Look in root of ckpt_dir
    root_ckpts = list(ckpt_dir.glob("*.pt"))
    if root_ckpts:
        root_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return root_ckpts[0]

    return None


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
        invert_gripper_action: bool = False,
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
        **kwargs: Any,
    ) -> None:
        """Load MiniVLA model weights, tokenizer, and dataset statistics.

        Fails loudly if checkpoint path is invalid or weights are corrupt (AGENTS.md Rule 10).
        """
        if device is not None:
            self.device = device
            self.config.device = device
            self.preprocessor.device = device

        ckpt = Path(checkpoint_path)

        # Handle Hugging Face Hub IDs
        if not ckpt.exists() and checkpoint_path.startswith("Stanford-ILIAD/"):
            # Check default local cache locations
            for cand_name in [
                checkpoint_path.split("/")[-1],
                "minivla_libero90",
                "minivla-libero90-prismatic",
            ]:
                candidate_dir = Path("resources/checkpoints") / cand_name
                if candidate_dir.exists() and _find_checkpoint_pt(candidate_dir) is not None:
                    ckpt = candidate_dir
                    break
            else:
                print(f"[MiniVLA] Attempting to download/cache snapshot for '{checkpoint_path}'...")
                try:
                    from huggingface_hub import snapshot_download
                    local_download_dir = Path("resources/checkpoints") / checkpoint_path.split("/")[-1]
                    local_path = snapshot_download(
                        repo_id=checkpoint_path,
                        local_dir=str(local_download_dir),
                        local_dir_use_symlinks=False,
                    )
                    ckpt = Path(local_path)
                except Exception as exc:
                    raise FileNotFoundError(
                        f"MiniVLA checkpoint '{checkpoint_path}' is not present locally and automatic "
                        f"download failed: {exc}. Please download manually using:\n"
                        f"  huggingface-cli download {checkpoint_path} --local-dir resources/checkpoints/{checkpoint_path.split('/')[-1]}\n"
                        f"Fail-fast per AGENTS.md Rule 10 (no mock execution allowed)."
                    ) from exc

        if not ckpt.exists():
            raise FileNotFoundError(
                f"MiniVLA checkpoint directory not found: {checkpoint_path}. "
                f"Fail-fast per AGENTS.md Rule 10 (no fallback allowed)."
            )

        # Load configuration & statistics
        self.config = MiniVLAConfig.from_checkpoint(ckpt, device=self.device)

        # Reinitialize tokenizer with checkpoint stats
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

        # Locate .pt weights file
        pt_file = _find_checkpoint_pt(ckpt)
        if pt_file is None or not pt_file.exists():
            raise FileNotFoundError(
                f"MiniVLA checkpoint directory '{ckpt}' does not contain any valid .pt model weights file!\n"
                f"Expected .pt checkpoint under '{ckpt}/checkpoints/step-*.pt'.\n"
                f"Please ensure complete checkpoint download per AGENTS.md Rule 10."
            )

        # Load PyTorch Model Architecture & Weights
        print(f"[MiniVLA] Loading model weights from: {pt_file}...")
        self.model = MiniVLAPrismaticModel.from_prismatic_checkpoint(
            checkpoint_path=pt_file,
            device=self.device,
            dtype=torch.float32,
        )
        self._is_loaded = True
        print(f"[MiniVLA] Model successfully loaded (Device: {self.device}, Action Dim: {self.action_dim}).")

    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert raw environment observations into model input features."""
        return self.preprocessor.preprocess_observation(obs, instruction)

    def predict_action_chunk(
        self, obs: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """Generate and return an action chunk.

        Fails fast if model is not loaded.
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError(
                f"Model '{self.config.model_id}' is not loaded. "
                f"Cannot predict actions. Fail-fast per AGENTS.md Rule 10."
            )

        inputs = self.preprocess(obs, instruction)

        # Full PyTorch model inference
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            generated_ids = self.model.generate_actions(
                dino_pixel_values=pixel_values,
                siglip_pixel_values=pixel_values,
                instruction=instruction,
                max_action_tokens=self.action_dim,
            )
            chunk = self.postprocess(generated_ids)

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
        invert_gripper_action: bool = False,
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
        **kwargs: Any,
    ) -> None:
        """Load MiniVLA-VQ model weights and Residual-VQ tokenizer.

        Fails loudly if VQ codebook or VQ-VAE weights are missing (AGENTS.md Rule 10).
        """
        super().load(checkpoint_path=checkpoint_path, device=device, **kwargs)

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
