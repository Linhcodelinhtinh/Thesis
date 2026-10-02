"""Model adapters and inference interfaces."""

from src.models.base import VLAPolicy
from src.models.registry import get_model_class, list_models, register_model

# Import adapters to trigger registration
import src.models.smolvla.adapter
import src.models.minivla.adapter

__all__ = [
    "VLAPolicy",
    "register_model",
    "get_model_class",
    "list_models",
]
