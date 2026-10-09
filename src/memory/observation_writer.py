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
        self._target_state: str = "detected"
        self._is_grasped: bool = False
        self._is_lifted: bool = False
        self._is_released: bool = False
        self._eef_z_at_grasp: Optional[float] = None
        self._event_counter: int = 0

    def on_episode_start(
        self,
        episode_id: str,
        env: Any,
        obs: Dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Register initial observed entities from task specification."""
        self._episode_id = episode_id
        self._target_state = "detected"
        self._is_grasped = False
        self._is_lifted = False
        self._is_released = False
        self._eef_z_at_grasp = None
        self._event_counter = 0

        # Register target objects as visually detected
        for entity in self.target_entity_names:
            clean_name = entity.strip().lower()
            if not clean_name:
                continue
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{episode_id}_{step}_{self._event_counter}_obs_init",
                episode_id=episode_id,
                step=step,
                timestamp=timestamp,
                event_type="initial_observation",
                object_id=clean_name,
                semantic_label=clean_name.replace("_", " "),
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=0.80,
                object_state_before=None,
                object_state_after="detected",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)

        # Register goal container if distinct
        if self.goal_container_name:
            clean_goal = self.goal_container_name.strip().lower()
            self._event_counter += 1
            evt_goal = MemoryEvent(
                event_id=f"evt_{episode_id}_{step}_{self._event_counter}_goal_obs_init",
                episode_id=episode_id,
                step=step,
                timestamp=timestamp,
                event_type="initial_observation",
                object_id=clean_goal,
                semantic_label=clean_goal.replace("_", " "),
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=0.80,
                object_state_before=None,
                object_state_after="target_destination",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt_goal)

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
        target_name = self.target_entity_names[0] if self.target_entity_names else "target_object"
        clean_target = target_name.strip().lower()

        # Observable signals
        gripper_closing = bool(len(action) > 0 and action[-1] >= 0.0)
        gripper_opening = bool(len(action) > 0 and action[-1] < 0.0)

        # Gripper finger joint separation (Panda fingers qpos)
        gripper_qpos = next_obs.get("robot0_gripper_qpos")
        finger_gap = 0.0
        if gripper_qpos is not None:
            q_arr = np.asarray(gripper_qpos).flatten()
            if len(q_arr) >= 2:
                finger_gap = float(abs(q_arr[0]) + abs(q_arr[1]))

        # EEF height
        eef_pos = next_obs.get("robot0_eef_pos")
        curr_eef_z = float(eef_pos[2]) if eef_pos is not None and len(eef_pos) >= 3 else 0.0

        # Heuristic 1: ESTIMATED GRASP
        # Closing command active, but fingers are blocked (gap > 0.02m) rather than fully snapped shut
        fingers_blocked = finger_gap > 0.02
        if not self._is_grasped and gripper_closing and fingers_blocked:
            self._is_grasped = True
            self._target_state = "grasped"
            self._eef_z_at_grasp = curr_eef_z
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_grasp",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_grasp",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=0.75,
                object_state_before="detected",
                object_state_after="grasped",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Heuristic 2: ESTIMATED LIFT
        # Was grasped, and EEF height has risen by at least 0.04m while fingers remain blocked
        if self._is_grasped and not self._is_lifted and self._eef_z_at_grasp is not None:
            delta_z = curr_eef_z - self._eef_z_at_grasp
            if delta_z >= 0.04 and fingers_blocked:
                self._is_lifted = True
                self._target_state = "lifted"
                self._event_counter += 1
                evt = MemoryEvent(
                    event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_lift",
                    episode_id=self._episode_id,
                    step=step,
                    timestamp=timestamp,
                    event_type="observed_lift",
                    object_id=clean_target,
                    semantic_label=clean_target.replace("_", " "),
                    evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                    confidence=0.70,
                    object_state_before="grasped",
                    object_state_after="lifted",
                    validity=MemoryStatus.CONFIRMED,
                )
                self.updater.record(evt)
                recorded_events.append(evt)

        # Heuristic 3: ESTIMATED CONTACT LOSS / SLIP
        # Fingers snap shut (gap collapsed) despite closing command while holding/lifted
        if (self._is_grasped or self._is_lifted) and gripper_closing and (finger_gap < 0.015):
            prev_state = self._target_state
            self._is_grasped = False
            self._target_state = "contact_lost"
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_slip",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_slip",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=0.75,
                object_state_before=prev_state,
                object_state_after="contact_lost",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Heuristic 4: ESTIMATED RELEASE / PLACEMENT ATTEMPT
        # Previously lifted, and gripper opening command is issued
        if self._is_lifted and gripper_opening and not self._is_released:
            self._is_released = True
            prev_state = self._target_state
            self._target_state = "released"
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_obs_release",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="observed_release",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=0.70,
                object_state_before=prev_state,
                object_state_after="released",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        return recorded_events
