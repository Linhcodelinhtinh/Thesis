"""Model Registry for VLA policy adapters (Phase 4).

Provides centralized registration and retrieval of VLA policy classes.
"""

from typing import Callable, Dict, List, Type

from src.models.base import VLAPolicy

_MODEL_REGISTRY: Dict[str, Type[VLAPolicy]] = {}


def register_model(name: str) -> Callable[[Type[VLAPolicy]], Type[VLAPolicy]]:
    """Decorator to register a VLAPolicy subclass with a canonical name."""
    def decorator(cls: Type[VLAPolicy]) -> Type[VLAPolicy]:
        if not issubclass(cls, VLAPolicy):
            raise TypeError(f"Class {cls.__name__} must inherit from VLAPolicy.")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[VLAPolicy]:
    """Retrieve a registered VLAPolicy class by name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """List all registered model adapter names."""
    return sorted(list(_MODEL_REGISTRY.keys()))
