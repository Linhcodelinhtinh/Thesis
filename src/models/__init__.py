"""Model adapters and inference interfaces."""

import logging
from src.models.base import VLAPolicy
from src.models.registry import get_model_class, list_models, register_model

logger = logging.getLogger(__name__)

# Import adapters to trigger registration in environments where their dependencies are installed.
# In isolated environments (per AGENTS.md Section 3), specific model dependencies
# (e.g. transformers version requirements) reside in dedicated envs (smolvla.yml, minivla.yml).
try:
    import src.models.smolvla.adapter
except Exception as exc:
    logger.debug("SmolVLA adapter not registered in current environment: %s", exc)

try:
    import src.models.minivla.adapter
except Exception as exc:
    logger.debug("MiniVLA adapter not registered in current environment: %s", exc)

__all__ = [
    "VLAPolicy",
    "register_model",
    "get_model_class",
    "list_models",
]
