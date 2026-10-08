"""Stage A Deterministic Oracle Memory Writer.

Extracts simulator-grounded physical events and object states directly
from MuJoCo simulation telemetry and diagnostics, marking all records
with EvidenceSource.ORACLE_SIMULATOR per docs/V2_MEMORY_EXECUTION_PLAN.md.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.memory.models import (
    EvidenceSource,
    MemoryEvent,
    MemoryStatus,
    ObjectMemory,
)
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater


class OracleMemoryWriter:
    """Deterministic, simulator-grounded event writer for Stage A validation.

    Tracks object state transitions (on_table -> grasped -> lifted -> placed)
    and fixture transitions (closed -> opened, off -> turned_on).
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
        self.target_entity_names = target_entity_names
        self.goal_container_name = goal_container_name

        self._episode_id: Optional[str] = None
        self._target_state: str = "on_table"
        self._is_grasped: bool = False
        self._is_lifted: bool = False
        self._is_placed: bool = False
        self._fixture_state: Optional[str] = None
        self._event_counter: int = 0

    def on_episode_start(
        self,
        episode_id: str,
        env: Any,
        obs: Dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize episode-local tracking and register initial entities."""
        self._episode_id = episode_id
        self._target_state = "on_table"
        self._is_grasped = False
        self._is_lifted = False
        self._is_placed = False
        self._event_counter = 0

        # Register target objects
        for entity in self.target_entity_names:
            clean_name = entity.strip().lower()
            if not clean_name:
                continue
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{episode_id}_{step}_{self._event_counter}_init",
                episode_id=episode_id,
                step=step,
                timestamp=timestamp,
                event_type="initial_detection",
                object_id=clean_name,
                semantic_label=clean_name.replace("_", " "),
                evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                confidence=1.0,
                object_state_before=None,
                object_state_after="on_table",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)

        # Register goal fixture/container if specified
        if self.goal_container_name:
            clean_goal = self.goal_container_name.strip().lower()
            if "drawer" in clean_goal:
                self._fixture_state = "closed"
            elif "stove" in clean_goal:
                self._fixture_state = "off"
            else:
                self._fixture_state = "empty"

            self._event_counter += 1
            evt_goal = MemoryEvent(
                event_id=f"evt_{episode_id}_{step}_{self._event_counter}_goal_init",
                episode_id=episode_id,
                step=step,
                timestamp=timestamp,
                event_type="initial_detection",
                object_id=clean_goal,
                semantic_label=clean_goal.replace("_", " "),
                evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                confidence=1.0,
                object_state_before=None,
                object_state_after=self._fixture_state,
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt_goal)

    def on_step(
        self,
        step: int,
        timestamp: float,
        action: np.ndarray,
        next_obs: Dict[str, Any],
        contact_info: Optional[Dict[str, Any]] = None,
        lift_info: Optional[Dict[str, Any]] = None,
        success: bool = False,
    ) -> List[MemoryEvent]:
        """Inspect step telemetry and record any verified physical state transitions."""
        if self._episode_id is None:
            return []

        recorded_events: List[MemoryEvent] = []
        target_name = self.target_entity_names[0] if self.target_entity_names else "target_object"
        clean_target = target_name.strip().lower()

        gripper_closing = bool(len(action) > 0 and action[-1] >= 0.0)
        both_contact = bool(contact_info.get("both_fingers_contact", False)) if contact_info else False
        is_lifted_phys = bool(lift_info.get("is_lifted", False)) if lift_info else False

        # Transition 1: GRASP
        if not self._is_grasped and both_contact and gripper_closing:
            self._is_grasped = True
            self._target_state = "grasped"
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_grasp",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="grasp",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                confidence=1.0,
                object_state_before="on_table",
                object_state_after="grasped",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Transition 2: LIFT
        if self._is_grasped and not self._is_lifted and is_lifted_phys:
            self._is_lifted = True
            self._target_state = "lifted_in_air"
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_lift",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="lift",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                confidence=1.0,
                object_state_before="grasped",
                object_state_after="lifted_in_air",
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

        # Transition 3: CONTACT LOSS / SLIP
        if (self._is_grasped or self._is_lifted) and not both_contact and not success:
            if self._target_state != "dropped_on_table":
                prev_state = self._target_state
                self._is_grasped = False
                self._target_state = "dropped_on_table"
                self._event_counter += 1
                evt = MemoryEvent(
                    event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_slip",
                    episode_id=self._episode_id,
                    step=step,
                    timestamp=timestamp,
                    event_type="slip",
                    object_id=clean_target,
                    semantic_label=clean_target.replace("_", " "),
                    evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                    confidence=1.0,
                    object_state_before=prev_state,
                    object_state_after="dropped_on_table",
                    validity=MemoryStatus.CONFIRMED,
                )
                self.updater.record(evt)
                recorded_events.append(evt)

        # Transition 4: SUCCESSFUL PLACEMENT
        if success and not self._is_placed:
            self._is_placed = True
            final_container = self.goal_container_name or "goal_container"
            prev_state = self._target_state
            self._target_state = f"in_{final_container}"
            self._event_counter += 1
            evt = MemoryEvent(
                event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_place",
                episode_id=self._episode_id,
                step=step,
                timestamp=timestamp,
                event_type="place",
                object_id=clean_target,
                semantic_label=clean_target.replace("_", " "),
                evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                confidence=1.0,
                object_state_before=prev_state,
                object_state_after=f"in_{final_container}",
                target_id=final_container,
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)
            recorded_events.append(evt)

            # Also update goal container state
            if self.goal_container_name:
                self._event_counter += 1
                evt_goal = MemoryEvent(
                    event_id=f"evt_{self._episode_id}_{step}_{self._event_counter}_goal_done",
                    episode_id=self._episode_id,
                    step=step,
                    timestamp=timestamp,
                    event_type="receive_object",
                    object_id=final_container,
                    semantic_label=final_container.replace("_", " "),
                    evidence_source=EvidenceSource.ORACLE_SIMULATOR,
                    confidence=1.0,
                    object_state_before=self._fixture_state,
                    object_state_after=f"contains_{clean_target}",
                    validity=MemoryStatus.CONFIRMED,
                )
                self.updater.record(evt_goal)
                recorded_events.append(evt_goal)

        return recorded_events
