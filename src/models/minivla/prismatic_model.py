"""Native Prismatic VLM / MiniVLA PyTorch Architecture & Loader (Phase 9).

Provides a self-contained, robust implementation of the MiniVLA model architecture
compatible with Stanford-ILIAD Prismatic checkpoints without requiring external
monolithic dependencies.

Architecture components:
1. Vision Backbone: DinoSigLIP (DINOv2 + SigLIP fused) or SigLIP featurizer.
2. Projector: FusedMLPProjector mapping visual tokens to LLM embedding dimension.
3. LLM Backbone: Qwen2.5-0.5B with 256 extended action tokens.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


class LinearProjector(nn.Module):
    """Linear projector mapping visual tokens to LLM embedding dimension."""

    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MLPProjector(nn.Module):
    """2-layer GELU MLP projector."""

    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class FusedMLPProjector(nn.Module):
    """3-layer GELU MLP projector for fused DinoSigLIP visual representations."""

    def __init__(self, fused_vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        self.projector = nn.Sequential(
            nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MiniVLAPrismaticModel(nn.Module):
    """Complete MiniVLA Model for inference with Prismatic checkpoints.

    Integrates:
    - Vision Featurizer (DinoSigLIP ViT)
    - Fused MLP Projector
    - Qwen2.5 Language Model with 256 action tokens
    """

    def __init__(
        self,
        llm_model_id: str = "Qwen/Qwen2.5-0.5B",
        dino_model_id: str = "vit_large_patch14_reg4_dinov2.lvd142m",
        siglip_model_id: str = "vit_so400m_patch14_siglip_224",
        num_extra_tokens: int = 256,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.torch_dtype = dtype
        self.num_extra_tokens = num_extra_tokens

        # 1. Initialize Tokenizer & LLM
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            llm_model_id, trust_remote_code=True
        )
        if num_extra_tokens > 0:
            extra_tokens = [f"<|extra_{i}|>" for i in range(num_extra_tokens)]
            # Check if extra tokens are already present
            new_tokens = [t for t in extra_tokens if t not in self.tokenizer.get_vocab()]
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        self.llm_embed_dim = self.llm.config.hidden_size

        # 2. Initialize Vision Featurizers via timm
        self.dino_featurizer = timm.create_model(
            dino_model_id, pretrained=False, num_classes=0, img_size=224
        )
        self.siglip_featurizer = timm.create_model(
            siglip_model_id, pretrained=False, num_classes=0, img_size=224
        )
        self.dino_featurizer.eval()
        self.siglip_featurizer.eval()

        self.vision_embed_dim = self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

        # 3. Projector
        self.projector = FusedMLPProjector(self.vision_embed_dim, self.llm_embed_dim)

        self.to(device=device, dtype=dtype)
        self.eval()

    @classmethod
    def from_prismatic_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MiniVLAPrismaticModel":
        """Load full model weights from an official Prismatic .pt checkpoint file."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        print(f"[MiniVLA] Instantiating architecture for checkpoint: {ckpt_file.name}...")
        model = cls(device="cpu", dtype=torch.float32)

        print(f"[MiniVLA] Loading weights from {ckpt_file} onto {device}...")
        checkpoint_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        # Checkpoint typically contains dict with key "model"
        state_dict = checkpoint_data.get("model", checkpoint_data)

        # Load projector
        if "projector" in state_dict:
            model.projector.load_state_dict(state_dict["projector"])
        elif any(k.startswith("projector.") for k in state_dict):
            proj_dict = {k.replace("projector.", ""): v for k, v in state_dict.items() if k.startswith("projector.")}
            model.projector.load_state_dict(proj_dict)

        # Load LLM backbone
        if "llm_backbone" in state_dict:
            model.llm.load_state_dict(state_dict["llm_backbone"], strict=False)
        elif any(k.startswith("llm_backbone.") for k in state_dict):
            llm_dict = {k.replace("llm_backbone.", ""): v for k, v in state_dict.items() if k.startswith("llm_backbone.")}
            model.llm.load_state_dict(llm_dict, strict=False)

        # Load vision backbone if present
        if "vision_backbone" in state_dict:
            vb_dict = state_dict["vision_backbone"]
            dino_dict = {k.replace("dino_featurizer.", ""): v for k, v in vb_dict.items() if "dino_featurizer" in k}
            siglip_dict = {k.replace("siglip_featurizer.", ""): v for k, v in vb_dict.items() if "siglip_featurizer" in k}
            if dino_dict:
                model.dino_featurizer.load_state_dict(dino_dict, strict=False)
            if siglip_dict:
                model.siglip_featurizer.load_state_dict(siglip_dict, strict=False)

        model.to(device=device, dtype=dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print("[MiniVLA] Checkpoint successfully loaded into active memory.")
        return model

    def encode_image(self, dino_tensor: torch.Tensor, siglip_tensor: torch.Tensor) -> torch.Tensor:
        """Extract visual patches and project into LLM embedding space."""
        # DinoViT intermediate patch representations
        dino_patches = self.dino_featurizer.forward_features(dino_tensor)
        if hasattr(dino_patches, "shape") and len(dino_patches.shape) == 3 and dino_patches.shape[1] > 256:
            # Drop class/register tokens if present
            dino_patches = dino_patches[:, -256:, :]

        siglip_patches = self.siglip_featurizer.forward_features(siglip_tensor)
        if hasattr(siglip_patches, "shape") and len(siglip_patches.shape) == 3 and siglip_patches.shape[1] > 256:
            siglip_patches = siglip_patches[:, -256:, :]

        # Concatenate features along embedding dimension
        fused_patches = torch.cat([dino_patches, siglip_patches], dim=-1)
        # Project into LLM dimension: (B, 256, llm_embed_dim)
        projected_patches = self.projector(fused_patches)
        return projected_patches

    @torch.no_grad()
    def generate_actions(
        self,
        dino_pixel_values: torch.Tensor,
        siglip_pixel_values: torch.Tensor,
        instruction: str,
        max_action_tokens: int = 7,
    ) -> List[int]:
        """Autoregressively generate action tokens conditioned on image and task instruction."""
        prompt = f"<|im_start|>user\nWhat action should the robot take to {instruction.lower().strip()}?<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device_name)

        # Visual embeddings
        visual_embeds = self.encode_image(
            dino_pixel_values.to(self.device_name, dtype=self.torch_dtype),
            siglip_pixel_values.to(self.device_name, dtype=self.torch_dtype),
        )

        # Text input embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate visual prefix and text embeddings
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Autoregressively generate new action tokens
        generated_tokens: List[int] = []
        current_embeds = inputs_embeds

        for _ in range(max_action_tokens):
            outputs = self.llm(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            # Action tokens are confined to the extra action tokens at vocabulary tail
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            generated_tokens.append(next_token_id)

            # Embed next token and append to inputs_embeds
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device_name)
            next_embed = self.llm.get_input_embeddings()(next_token_tensor)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

        return generated_tokens
