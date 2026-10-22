"""Causal, online object and relation tracking for V2 memory pipelines.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 6:
- Only consumes observations o_t and executed actions a_{<t}.
- Does NOT access privileged simulator internals, future actions, or oracle subgoals.
- Maintains episode-local object identity with immutable origin anchors.
- Applies hysteresis (dwell thresholds) to prevent debounce/jitter in relation changes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.memory.models import IdentityStatus, VisibilityStatus


@dataclass
class TrackedInstance:
    """State of an observed physical entity across time."""

    object_id: str
    semantic_name: str
    origin_anchor: str
    identity_status: IdentityStatus = IdentityStatus.TRACKED
    visibility: VisibilityStatus = VisibilityStatus.VISIBLE
    last_known_pos: Optional[np.ndarray] = None
    last_observed_step: int = -1
    container_id: Optional[str] = None
    is_held: bool = False
    is_on_support: bool = True
    support_id: Optional[str] = None
    consecutive_on_support_steps: int = 0
    consecutive_departed_steps: int = 0
    consecutive_held_steps: int = 0


@dataclass
class TrackingObservation:
    """Parsed telemetry frame at step t."""

    step: int
    eef_pos: np.ndarray
    gripper_gap: float
    last_action: Optional[np.ndarray] = None
    object_detections: Dict[str, np.ndarray] = field(default_factory=dict)


class CausalObjectTracker:
    """Online, causal perception tracker evaluating object-support relationships.

    Estimates:
    - Object departure from support
    - Object return to support
    - Gripper grasp and lift
    - Occlusion and container containment
    """

    def __init__(
        self,
        target_object_id: str = "bowl_A",
        target_semantic_name: str = "bowl",
        origin_anchor: str = "bowl initially on the left plate",
        support_id: str = "plate_A",
        support_semantic_name: str = "plate",
        support_origin_pos: Optional[np.ndarray] = None,
        dwell_steps: int = 3,
        xy_threshold: float = 0.08,
        z_threshold: float = 0.06,
        grasp_gap_threshold: float = 0.045,
    ) -> None:
        self.target_object_id = target_object_id
        self.target_semantic_name = target_semantic_name
        self.origin_anchor = origin_anchor
        self.support_id = support_id
        self.support_semantic_name = support_semantic_name
        self.dwell_steps = dwell_steps
        self.xy_threshold = xy_threshold
        self.z_threshold = z_threshold
        self.grasp_gap_threshold = grasp_gap_threshold

        # Default support position on table if not provided
        self.support_pos = (
            np.array(support_origin_pos, dtype=np.float32)
            if support_origin_pos is not None
            else np.array([-0.05, 0.20, 0.85], dtype=np.float32)
        )

        self.target = TrackedInstance(
            object_id=self.target_object_id,
            semantic_name=self.target_semantic_name,
            origin_anchor=self.origin_anchor,
            last_known_pos=self.support_pos.copy(),
            support_id=self.support_id,
            is_on_support=True,
        )

        self._prev_eef_pos: Optional[np.ndarray] = None
        self._prev_gripper_gap: Optional[float] = None
        self._last_step = -1

    def reset(self, initial_observation: Optional[Dict[str, Any]] = None) -> None:
        """Reset tracker state cleanly at episode boundary."""
        self.target = TrackedInstance(
            object_id=self.target_object_id,
            semantic_name=self.target_semantic_name,
            origin_anchor=self.origin_anchor,
            last_known_pos=self.support_pos.copy(),
            support_id=self.support_id,
            is_on_support=True,
        )
        self._prev_eef_pos = None
        self._prev_gripper_gap = None
        self._last_step = -1

        if initial_observation is not None:
            self.update(0, initial_observation, None)

    def update(
        self,
        step: int,
        observation: Dict[str, Any],
        last_action: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Process observation at step t with executed action a_{t-1}.

        Returns diagnostic signals for event verification.
        """
        if step <= self._last_step and self._last_step != -1:
            raise ValueError(f"Step out of order: step={step} <= last_step={self._last_step}")
        self._last_step = step

        # Extract proprioception
        eef_pos = np.asarray(observation.get("robot0_eef_pos", [0.0, 0.0, 1.0]), dtype=np.float32)
        gripper_qpos = np.asarray(observation.get("robot0_gripper_qpos", [0.05, 0.05]), dtype=np.float32)
        gripper_gap = float(np.sum(np.abs(gripper_qpos[:2])))

        # Check if gripper closed
        gripper_closed = gripper_gap < self.grasp_gap_threshold
        if last_action is not None and len(last_action) >= 7:
            gripper_command_close = last_action[-1] > 0.0
        else:
            gripper_command_close = gripper_closed

        # Grounding evidence check:
        # Object location must be supported by detection/segmentation or verified grasp
        detected_pos = observation.get("detected_objects", {}).get(self.target_object_id)
        has_grounding_evidence = detected_pos is not None

        if has_grounding_evidence:
            current_obj_pos = np.asarray(detected_pos, dtype=np.float32)
            self.target.visibility = VisibilityStatus.VISIBLE
            self.target.identity_status = IdentityStatus.TRACKED
            evidence_id = f"obs_{step}_{self.target_object_id}_detected"
        elif self.target.is_held:
            # If held, object position updates with gripper only if held was previously confirmed
            current_obj_pos = eef_pos.copy()
            self.target.visibility = VisibilityStatus.VISIBLE
            has_grounding_evidence = True
            evidence_id = f"obs_{step}_{self.target_object_id}_kinematic_held"
        else:
            # No visual or kinematic evidence: object cannot be confirmed
            current_obj_pos = self.support_pos.copy()
            self.target.visibility = VisibilityStatus.UNKNOWN
            self.target.identity_status = IdentityStatus.UNKNOWN
            evidence_id = None

        self.target.last_known_pos = current_obj_pos.copy()
        self.target.last_observed_step = step

        if not has_grounding_evidence:
            self.target.consecutive_on_support_steps = 0
            self.target.consecutive_departed_steps = 0
            self.target.consecutive_held_steps = 0
            return {
                "step": step,
                "eef_pos": eef_pos,
                "gripper_gap": gripper_gap,
                "gripper_closed": gripper_closed,
                "object_pos": current_obj_pos,
                "support_pos": self.support_pos,
                "dist_xy": 0.0,
                "dist_z": 0.0,
                "has_grounding_evidence": False,
                "evidence_id": None,
                "raw_on_support": False,
                "raw_held": False,
                "is_held": False,
                "is_departed_candidate": False,
                "is_return_candidate": False,
                "consecutive_on_support": 0,
                "consecutive_departed": 0,
            }

        # Compute distance to support
        d_xy = float(np.linalg.norm(current_obj_pos[:2] - self.support_pos[:2]))
        d_z = float(abs(current_obj_pos[2] - self.support_pos[2]))
        near_support = d_xy <= self.xy_threshold and d_z <= self.z_threshold

        # Compute distance between EEF and object
        d_eef_obj = float(np.linalg.norm(eef_pos - current_obj_pos))
        close_to_eef = d_eef_obj <= 0.07

        # Held evaluation: close to EEF + gripper closed
        raw_held = close_to_eef and gripper_closed
        if raw_held:
            self.target.consecutive_held_steps += 1
        else:
            self.target.consecutive_held_steps = 0

        # Update held status with hysteresis
        if self.target.consecutive_held_steps >= self.dwell_steps:
            self.target.is_held = True
        elif not raw_held and not gripper_closed:
            self.target.is_held = False

        # Support evaluation
        raw_on_support = near_support and not (self.target.is_held and d_z > 0.04)
        if raw_on_support:
            self.target.consecutive_on_support_steps += 1
            self.target.consecutive_departed_steps = 0
        else:
            self.target.consecutive_departed_steps += 1
            self.target.consecutive_on_support_steps = 0

        # Transition support state with hysteresis
        is_departed_candidate = self.target.consecutive_departed_steps >= self.dwell_steps
        is_return_candidate = self.target.consecutive_on_support_steps >= self.dwell_steps

        signals = {
            "step": step,
            "eef_pos": eef_pos,
            "gripper_gap": gripper_gap,
            "gripper_closed": gripper_closed,
            "object_pos": current_obj_pos,
            "support_pos": self.support_pos,
            "dist_xy": d_xy,
            "dist_z": d_z,
            "raw_on_support": raw_on_support,
            "raw_held": raw_held,
            "is_held": self.target.is_held,
            "is_departed_candidate": is_departed_candidate,
            "is_return_candidate": is_return_candidate,
            "consecutive_on_support": self.target.consecutive_on_support_steps,
            "consecutive_departed": self.target.consecutive_departed_steps,
        }

        self._prev_eef_pos = eef_pos.copy()
        self._prev_gripper_gap = gripper_gap
        return signals
