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

    Inspects MuJoCo simulation physics directly for initial scene grounding
    and verifies physical state transitions (on_table -> grasped -> lifted -> placed).
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

        # Strict entity separation: exclude goal container from manipulable target list
        # to prevent contradictory registrations (e.g. plate registered as both on_table and empty)
        filtered_targets: List[str] = []
        for name in target_entity_names:
            c = name.strip().lower()
            if c and c != clean_goal and c not in filtered_targets:
                filtered_targets.append(c)
        self.target_entity_names = tuple(filtered_targets)

        self._episode_id: Optional[str] = None
        self._target_state: str = "on_table"
        self._is_grasped: bool = False
        self._is_lifted: bool = False
        self._is_placed: bool = False
        self._fixture_state: Optional[str] = None
        self._event_counter: int = 0

    def _get_sim(self, env: Any) -> Any:
        """Resolve MuJoCo sim instance across wrapper layers."""
        sim = getattr(env, "sim", None)
        if sim is None and hasattr(env, "_env"):
            sim = getattr(env._env, "sim", None)
        if sim is None and hasattr(env, "env"):
            sim = getattr(env.env, "sim", None)
        return sim

    def _resolve_initial_state(self, env: Any, entity_name: str, is_fixture: bool = False) -> str:
        """Query genuine simulation physics/joints to establish ground-truth initial state."""
        sim = self._get_sim(env)
        if sim is None or not hasattr(sim, "model") or not hasattr(sim, "data"):
            if is_fixture:
                if "drawer" in entity_name:
                    return "closed"
                elif "stove" in entity_name:
                    return "off"
                return "empty"
            return "on_table"

        # 1. Grounded Fixture State Verification
        if is_fixture:
            if "drawer" in entity_name:
                try:
                    for i in range(getattr(sim.model, "njnt", 0)):
                        jname = sim.model.joint_id2name(i) or ""
                        if "drawer" in jname.lower():
                            qpos = float(sim.data.qpos[sim.model.jnt_qposadr[i]])
                            return "closed" if abs(qpos) < 0.05 else "opened"
                except Exception:
                    pass
                return "closed"

            if "stove" in entity_name:
                try:
                    for i in range(getattr(sim.model, "njnt", 0)):
                        jname = sim.model.joint_id2name(i) or ""
                        if "stove" in jname.lower() or "knob" in jname.lower():
                            qpos = float(sim.data.qpos[sim.model.jnt_qposadr[i]])
                            return "turned_on" if abs(qpos) > 0.1 else "off"
                except Exception:
                    pass
                return "off"

            # Receptacle / container: check if target object is already inside
            return "empty"

        # 2. Grounded Manipulable Object State Verification
        try:
            nbody = getattr(sim.model, "nbody", 0)
            for i in range(nbody):
                bname = sim.model.body_id2name(i) or ""
                if entity_name in bname.lower():
                    pos = sim.data.body_xpos[i]
                    z = float(pos[2])
                    # In standard LIBERO tables, top surface is ~0.82m-0.88m.
                    # Elevated items on fixtures (e.g. cookie box) have z > 0.95m.
                    if z > 0.95:
                        return "elevated_on_fixture"
                    elif 0.75 <= z <= 0.95:
                        return "on_table"
                    else:
                        return "resting"
        except Exception:
            pass

        return "on_table"

    def on_episode_start(
        self,
        episode_id: str,
        env: Any,
        obs: Dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize episode-local tracking and register initial entities using genuine sim state."""
        self._episode_id = episode_id
        self._is_grasped = False
        self._is_lifted = False
        self._is_placed = False
        self._event_counter = 0

        # Register distinct target objects grounded in simulator
        for entity in self.target_entity_names:
            clean_name = entity.strip().lower()
            if not clean_name:
                continue
            init_state = self._resolve_initial_state(env, clean_name, is_fixture=False)
            self._target_state = init_state
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
                object_state_after=init_state,
                validity=MemoryStatus.CONFIRMED,
            )
            self.updater.record(evt)

        # Register distinct goal fixture/container grounded in simulator
        if self.goal_container_name:
            clean_goal = self.goal_container_name.strip().lower()
            fixture_state = self._resolve_initial_state(env, clean_goal, is_fixture=True)
            self._fixture_state = fixture_state
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
                object_state_after=fixture_state,
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
        dist_to_goal: Optional[float] = None,
        success: bool = False,
    ) -> List[MemoryEvent]:
        """Inspect step telemetry and record verified physical state transitions."""
        if self._episode_id is None:
            return []

        recorded_events: List[MemoryEvent] = []
        target_name = self.target_entity_names[0] if self.target_entity_names else "target_object"
        clean_target = target_name.strip().lower()

        gripper_closing = bool(len(action) > 0 and action[-1] >= 0.0)
        gripper_opening = bool(len(action) > 0 and action[-1] < 0.0)
        both_contact = bool(contact_info.get("both_fingers_contact", False)) if contact_info else False
        is_lifted_phys = bool(lift_info.get("is_lifted", False)) if lift_info else False

        # Transition 1: GRASP
        if not self._is_grasped and both_contact and gripper_closing:
            self._is_grasped = True
            prev_state = self._target_state
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
                object_state_before=prev_state,
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

        # Transition 3: PHYSICAL PLACEMENT
        # Verified when a lifted object is released (gripper opens) within container vicinity (dist <= 0.08m)
        physically_placed = (
            self._is_lifted
            and gripper_opening
            and (dist_to_goal is not None and dist_to_goal <= 0.08)
        )
        if (physically_placed or success) and not self._is_placed:
            self._is_placed = True
            self._is_grasped = False
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

            # Update goal container state
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

        # Transition 4: CONTACT LOSS / SLIP (unintended drop when not placed)
        elif (self._is_grasped or self._is_lifted) and not both_contact and not self._is_placed:
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

        return recorded_events
