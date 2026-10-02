"""MiniVLA Model Adapter Package for LIBERO Evaluation (Phase 9).

Integrates Stanford-ILIAD OpenVLA-mini (Candidate B & C) based on Qwen2.5-0.5B,
SigLIP 224px, and extra_action_tokenizer / Residual-VQ tokenizer.
"""

from src.models.minivla.adapter import MiniVLAAdapter, MiniVLAVQAdapter
from src.models.minivla.config import MiniVLAConfig
from src.models.minivla.action_tokenizer import ExtraActionTokenizer
from src.models.minivla.residual_vq import ResidualVQActionTokenizer
from src.models.minivla.preprocess import MiniVLAPreprocessor
from src.models.minivla.postprocess import MiniVLAPostprocessor

__all__ = [
    "MiniVLAAdapter",
    "MiniVLAVQAdapter",
    "MiniVLAConfig",
    "ExtraActionTokenizer",
    "ResidualVQActionTokenizer",
    "MiniVLAPreprocessor",
    "MiniVLAPostprocessor",
]
