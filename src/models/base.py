"""Base interface for Vision-Language-Action (VLA) policies (Phase 4).

Defines the standard contract for VLA policy adapters, featuring first-class
chunk prediction (predict_action_chunk), internal action queue management
(select_action), state resetting, and interface auditing per SRS.md and
IMPLEMENTATION_PLAN.md.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class VLAPolicy(ABC):
    """Abstract Base Class for all VLA policy adapters.

    Guarantees a unified interface for evaluation, supporting both chunk-level
    predictions (predict_action_chunk) and step-by-step queue execution
    (select_action).
    """

    def __init__(self, chunk_size: int = 50, action_dim: int = 7) -> None:
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self._action_queue: Deque[np.ndarray] = deque()
        self._last_chunk: Optional[np.ndarray] = None
        self._step_count: int = 0

    @abstractmethod
    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        """Load model weights and processors from a local path or repository."""
        pass

    def reset(self) -> None:
        """Reset internal buffers, temporal history, and action queues."""
        self._action_queue.clear()
        self._last_chunk = None
        self._step_count = 0

    @abstractmethod
    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert raw simulator observations into model input features."""
        pass

    @abstractmethod
    def predict_action_chunk(
        self, obs: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """First-Class API: Generate and return a full action chunk.

        Args:
            obs: Raw simulator observation dictionary.
            instruction: Task language instruction text.

        Returns:
            np.ndarray of shape (chunk_size, action_dim).
        """
        pass

    def select_action(
        self, obs: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """Step-by-step action selection managing an internal FIFO action queue.

        If the queue is empty, generates a new action chunk, fills the queue,
        and pops the first action.

        Returns:
            Single 1D action vector of shape (action_dim,).
        """
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(obs, instruction)
            if chunk.shape != (self.chunk_size, self.action_dim):
                raise ValueError(
                    f"Action chunk shape mismatch: expected ({self.chunk_size}, {self.action_dim}), "
                    f"got {chunk.shape}."
                )
            self._last_chunk = chunk
            for action in chunk:
                self._action_queue.append(action)

        self._step_count += 1
        return self._action_queue.popleft()

    @abstractmethod
    def postprocess(self, output: Any) -> np.ndarray:
        """Unnormalize and format raw model outputs into robot action space."""
        pass

    @abstractmethod
    def validate_interface(self) -> Dict[str, Any]:
        """Audit and report interface details (inputs, outputs, normalization)."""
        pass

    @property
    @abstractmethod
    def observation_spec(self) -> Dict[str, Any]:
        """Specification of expected observation keys, shapes, and dtypes."""
        pass

    @property
    def action_spec(self) -> Dict[str, Any]:
        """Specification of output action dimensions, range, and semantics."""
        return {
            "shape": (self.action_dim,),
            "chunk_size": self.chunk_size,
            "range": (-1.0, 1.0),
            "semantics": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        }

    @property
    def queue_size(self) -> int:
        """Current number of pending actions in the internal execution queue."""
        return len(self._action_queue)
