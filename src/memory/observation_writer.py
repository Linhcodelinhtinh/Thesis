"""Stage B Observation-Derived Memory Writer.

Extracts physical events and object state estimates purely from runtime
observations (proprioception, EEF kinematics, gripper states, actions)
WITHOUT accessing ground-truth simulation internals or oracle contacts.
Marked with EvidenceSource.OBSERVATION_TRACKER per docs/V2_MEMORY_EXECUTION_PLAN.md.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.memory.models import (
    EvidenceSource,
    MemoryEvent,
    MemoryStatus,
    ObjectMemory,
)
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater


class ObservationMemoryWriter:
    """Non-oracle event writer estimating state from observable telemetry.

    Observes:
    - End-effector position (robot0_eef_pos)
    - Gripper joint width (robot0_gripper_qpos)
    - Gripper command action (action[-1])
    All generated facts are tagged with EvidenceSource.OBSERVATION_TRACKER
    and realistic calibrated confidence (0.70 - 0.85).
    """

    def __init__(
        self,
        store: EpisodeMemoryStore,
        updater: MemoryUpdater,
        target_entity_names: Tuple[str, ...],
        goal_container_name: Optional[str] = None,
    ) -> None:
        self.store = store
        self.updater = updater

        clean_goal = goal_container_name.strip().lower() if goal_container_name else None
        self.goal_container_name = clean_goal

        # Ensure goal container is distinct from target items
        filtered_targets: List[str] = []
        for name in target_entity_names:
            c = name.strip().lower()
            if c and c != clean_goal and c not in filtered_targets:
                filtered_targets.append(c)
        self.target_entity_names = tuple(filtered_targets)

        self._episode_id: Optional[str] = None
        self._target_state: str = "idle"
        self._is_grasped: bool = False
        self._is_lifted: bool = False
        self._is_released: bool = False
        self._eef_z_at_grasp: Optional[float] = None
        self._prev_finger_gap: Optional[float] = None
        self._blocked_counter: int = 0
        self._grasped_object_id: Optional[str] = None
        self._grasped_label: Optional[str] = None
        self._event_counter: int = 0

    def on_episode_start(
        self,
        episode_id: str,
        env: Any,
        obs: Dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize trackers without fabricating synthetic visual detections."""
        self._episode_id = episode_id
        self._target_state = "idle"
        self._is_grasped = False
        self._is_lifted = False
        self._is_released = False
        self._eef_z_at_grasp = None
        self._prev_finger_gap = None
        self._blocked_counter = 0
        self._grasped_object_id = None
        self._grasped_label = None
        self._event_counter = 0

    def on_step(
        self,
        step: int,
        timestamp: float,
        action: np.ndarray,
        next_obs: Dict[str, Any],
    ) -> List[MemoryEvent]:
        """Estimate manipulation events purely from proprioception and actions."""
        if self._episode_id is None:
            return []

        recorded_events: List[MemoryEvent] = []

        # Observable signals
        gripper_closing = bool(len(action) > 0 and action[-1] >= 0.0)
        gripper_opening = bool(len(action) > 0 and action[-1] < 0.0)

        # Gripper finger joint separation (Panda fingers qpos)
        gripper_qpos = next_obs.get("robot0_gripper_qpos")
        eef_pos = next_obs.get("robot0_eef_pos")
        if gripper_qpos is None or eef_pos is None:
            # Cannot infer physical manipulation events without observable proprioception
            return []

        q_arr = np.asarray(gripper_qpos).flatten()
        if len(q_arr) < 2:
            return []
        finger_gap = float(abs(q_arr[0]) + abs(q_arr[1]))

        # EEF height
        if len(eef_pos) < 3:
            return []
        curr_eef_z = float(eef_pos[2])

        # Track persistent obstruction:
        # Fingers are in graspable width (15mm to 65mm), closing command is active,
        # AND finger motion has stalled (gap not changing rapidly).
        # In free-air closure, Panda fingers move at ~15-25 mm/step.
        gap_stalled = False
        if self._prev_finger_gap is not None:
            gap_velocity = abs(finger_gap - self._prev_finger_gap)
            if gap_velocity < 0.003:  # Stalled against obstruction
                gap_stalled = True

        fingers_in_range = 0.015 <= finger_gap <= 0.065
        eef_near_table = curr_eef_z <= 0.95

        if gripper_closing and fingers_in_range and gap_stalled and eef_near_table:
            self._blocked_counter += 1
        elif not gripper_closing or not fingers_in_range:
            self._blocked_counter = 0

        # Require stalled gap across consecutive steps under close command to confirm obstruction
        is_obstructed = (self._blocked_counter >= 1)

        # Heuristic 1: ESTIMATED GRASP
        if not self._is_grasped and is_obstructed and eef_near_table:
            # Identity determination:
            # If multiple targets exist, proprioception alone cannot resolve identity!
            if len(self.target_entity_names) == 1:
                clean_target = self.target_entity_names[0]
                label = clean_target.replace("_", " ")
                conf = 0.70
                val = MemoryStatus.CONFIRMED
            else:
                # Multi-target task: Do NOT blindly assign to the first target entity!
                clean_target = "unidentified_object"
                label = "an object"
                conf = 0.50
                val = MemoryStatus.UNCERTAIN

            self._is_grasped = True
            self._is_released = False
            self._target_state = "grasped"
            self._eef_z_at_grasp = curr_eef_z
            self._grasped_object_id = clean_target
            self._grasped_label = label
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_grasp",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_grasp",
                object_id=clean_target,
                semantic_label=label,
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=conf,
                object_state_before="idle",
                object_state_after="grasped",
                validity=val,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Heuristic 2: ESTIMATED LIFT
        # Was grasped, and EEF height has risen by at least 0.04m while fingers remain blocked
        if self._is_grasped and not self._is_lifted and self._eef_z_at_grasp is not None:
            delta_z = curr_eef_z - self._eef_z_at_grasp
            if delta_z >= 0.04 and fingers_in_range:
                self._is_lifted = True
                self._target_state = "lifted"
                self._event_counter += 1
                target_id = self._grasped_object_id or "unidentified_object"
                label = self._grasped_label or "an object"
                conf = 0.70 if target_id != "unidentified_object" else 0.50
                val = MemoryStatus.CONFIRMED if target_id != "unidentified_object" else MemoryStatus.UNCERTAIN
                evt = MemoryEvent(
                    event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_lift",
                    episode_id=self._episode_id,
                    step=step,
                    timestamp=timestamp,
                    event_type="observed_lift",
                    object_id=target_id,
                    semantic_label=label,
                    evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                    confidence=conf,
                    object_state_before="grasped",
                    object_state_after="lifted",
                    validity=val,
                )
                self.updater.record(evt)
                recorded_events.append(evt)

        # Heuristic 3: ESTIMATED CONTACT LOSS / SLIP
        # Fingers snap shut (gap collapsed < 0.015m) despite closing command while holding/lifted
        if (self._is_grasped or self._is_lifted) and gripper_closing and (finger_gap < 0.015):
            prev_state = self._target_state
            self._is_grasped = False
            self._is_lifted = False
            self._eef_z_at_grasp = None
            self._blocked_counter = 0
            self._target_state = "contact_lost"
            target_id = self._grasped_object_id or "unidentified_object"
            label = self._grasped_label or "an object"
            conf = 0.70 if target_id != "unidentified_object" else 0.50
            val = MemoryStatus.CONFIRMED if target_id != "unidentified_object" else MemoryStatus.UNCERTAIN
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_slip",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_slip",
                object_id=target_id,
                semantic_label=label,
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=conf,
                object_state_before=prev_state,
                object_state_after="contact_lost",
                validity=val,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Heuristic 4: ESTIMATED RELEASE / PLACEMENT ATTEMPT
        # Previously lifted or grasped, and gripper opening command is issued
        if (self._is_lifted or self._is_grasped) and gripper_opening and not self._is_released:
            self._is_released = True
            self._is_grasped = False
            self._is_lifted = False
            self._eef_z_at_grasp = None
            self._blocked_counter = 0
            prev_state = self._target_state
            self._target_state = "released"
            target_id = self._grasped_object_id or "unidentified_object"
            label = self._grasped_label or "an object"
            conf = 0.70 if target_id != "unidentified_object" else 0.50
            val = MemoryStatus.CONFIRMED if target_id != "unidentified_object" else MemoryStatus.UNCERTAIN
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_release",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_release",
                object_id=target_id,
                semantic_label=label,
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=conf,
                object_state_before=prev_state,
                object_state_after="released",
                validity=val,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        self._prev_finger_gap = finger_gap
        return recorded_events
