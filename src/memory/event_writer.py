"""FSM Cycle Counter and Schema V2 Event Writer.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 6.4:
- FSM transitions:
  support_observed -> departure_candidate -> departure_confirmed ->
  return_candidate -> return_confirmed (count += 1) -> support_observed.
- Confirmation hysteresis to reject debounce and sensor noise.
- Strict causal timing: occurred_at (physical onset) vs available_at (confirmation step).
- Isolated from simulator internals, oracle rewards, and evaluator subgoals.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.memory.models import (
    CycleCounter,
    EvidenceSource,
    IdentityStatus,
    ObjectIdentity,
    RelationFact,
    RelationStatus,
    SchemaV2EpisodeMemory,
    V2MemoryEvent,
    VisibilityStatus,
)
from src.perception.object_tracker import CausalObjectTracker


class FSMState(str, Enum):
    """Cycle FSM progression states."""

    SUPPORT_OBSERVED = "support_observed"
    DEPARTURE_CANDIDATE = "departure_candidate"
    DEPARTURE_CONFIRMED = "departure_confirmed"
    RETURN_CANDIDATE = "return_candidate"
    RETURN_CONFIRMED = "return_confirmed"


class LiberoMemEventWriter:
    """Online event writer and cycle counter tracking manipulation episodes."""

    def __init__(
        self,
        episode_id: str,
        target_object_id: str = "bowl_A",
        target_semantic_name: str = "bowl",
        origin_anchor: str = "bowl initially on the left plate",
        support_id: str = "plate_A",
        support_semantic_name: str = "plate",
        tracker: Optional[CausalObjectTracker] = None,
        dwell_steps: int = 3,
    ) -> None:
        self.episode_id = episode_id
        self.target_object_id = target_object_id
        self.target_semantic_name = target_semantic_name
        self.origin_anchor = origin_anchor
        self.support_id = support_id
        self.support_semantic_name = support_semantic_name
        self.dwell_steps = dwell_steps

        self.cycle_key = f"{self.target_object_id}_{self.support_id}"

        self.tracker = (
            tracker
            if tracker is not None
            else CausalObjectTracker(
                target_object_id=self.target_object_id,
                target_semantic_name=self.target_semantic_name,
                origin_anchor=self.origin_anchor,
                support_id=self.support_id,
                support_semantic_name=self.support_semantic_name,
                dwell_steps=self.dwell_steps,
            )
        )

        self.fsm_state = FSMState.SUPPORT_OBSERVED
        self.confirmed_cycles = 0
        self.in_progress = False
        self.history_complete = True
        self.uncertain_count = 0

        self._candidate_departure_step: Optional[int] = None
        self._candidate_return_step: Optional[int] = None

        self._events: List[V2MemoryEvent] = []
        self._relations: List[RelationFact] = []
        self._cycle_event_ids: List[str] = []

        self._current_step = 0
        self._last_consumed_action = -1

    def _add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        status: RelationStatus,
        occurred_at: int,
        available_at: int,
        evidence_ids: Tuple[str, ...],
        supersedes: Optional[str] = None,
    ) -> None:
        fact = RelationFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            status=status,
            occurred_at=occurred_at,
            available_at=available_at,
            evidence_ids=evidence_ids,
            supersedes=supersedes,
        )
        self._relations.append(fact)

    def _add_event(
        self,
        event_type: str,
        occurred_at: int,
        available_at: int,
        source: EvidenceSource = EvidenceSource.PREDICTED_RGB_TRACKER,
        details: Optional[Dict[str, Any]] = None,
        evidence_ids: Tuple[str, ...] = (),
    ) -> str:
        event_id = f"evt_{len(self._events) + 1}_{event_type}"
        event = V2MemoryEvent(
            id=event_id,
            type=event_type,
            occurred_at=occurred_at,
            available_at=available_at,
            source=source,
            details=details,
            evidence_ids=evidence_ids,
        )
        self._events.append(event)
        return event_id

    def step(
        self,
        step: int,
        observation: Dict[str, Any],
        last_action: Optional[np.ndarray] = None,
    ) -> SchemaV2EpisodeMemory:
        if step > 0 and (step - self._current_step > 1):
            self.history_complete = False

        self._current_step = step
        self._last_consumed_action = step - 1 if step > 0 else -1

        signals = self.tracker.update(step, observation, last_action)
        has_grounding = signals.get("has_grounding_evidence", True)
        evidence_id = signals.get("evidence_id") or f"obs_{step}_{self.target_object_id}"

        # Ingest initial relation at step 0 only if supported by evidence
        if step == 0 and has_grounding and signals.get("raw_on_support"):
            self._add_relation(
                subject=self.target_object_id,
                predicate="on",
                obj=self.support_id,
                status=RelationStatus.CONFIRMED_BY_WRITER,
                occurred_at=0,
                available_at=0,
                evidence_ids=(evidence_id,),
            )

        if not has_grounding:
            # Without observation grounding, mark history incomplete and do not confirm transitions or cycles
            self.history_complete = False
            return self.get_snapshot()

        is_departed = signals["is_departed_candidate"]
        is_return = signals["is_return_candidate"]

        # FSM Transition Logic
        if self.fsm_state == FSMState.SUPPORT_OBSERVED:
            if not signals["raw_on_support"]:
                self.fsm_state = FSMState.DEPARTURE_CANDIDATE
                self._candidate_departure_step = step

        elif self.fsm_state == FSMState.DEPARTURE_CANDIDATE:
            if is_departed:
                self.fsm_state = FSMState.DEPARTURE_CONFIRMED
                self.in_progress = True
                occ = self._candidate_departure_step if self._candidate_departure_step is not None else step
                evt_id = self._add_event(
                    event_type="departed_support",
                    occurred_at=occ,
                    available_at=step,
                    details={"cycle_index": self.confirmed_cycles + 1},
                    evidence_ids=(evidence_id,),
                )
                self._add_relation(
                    subject=self.target_object_id,
                    predicate="on",
                    obj=self.support_id,
                    status=RelationStatus.RETRACTED,
                    occurred_at=occ,
                    available_at=step,
                    evidence_ids=(evt_id, evidence_id),
                )
            elif signals["raw_on_support"]:
                # Transient jitter; return to resting
                self.fsm_state = FSMState.SUPPORT_OBSERVED
                self._candidate_departure_step = None

        elif self.fsm_state == FSMState.DEPARTURE_CONFIRMED:
            if signals["raw_on_support"]:
                self.fsm_state = FSMState.RETURN_CANDIDATE
                self._candidate_return_step = step

        elif self.fsm_state == FSMState.RETURN_CANDIDATE:
            if is_return:
                self.fsm_state = FSMState.RETURN_CONFIRMED
                self.confirmed_cycles += 1
                self.in_progress = False

                occ = self._candidate_return_step if self._candidate_return_step is not None else step
                evt_id = self._add_event(
                    event_type="returned_to_support",
                    occurred_at=occ,
                    available_at=step,
                    details={"cycle_count": self.confirmed_cycles},
                    evidence_ids=(evidence_id,),
                )
                self._cycle_event_ids.append(evt_id)

                self._add_relation(
                    subject=self.target_object_id,
                    predicate="on",
                    obj=self.support_id,
                    status=RelationStatus.CONFIRMED_BY_WRITER,
                    occurred_at=occ,
                    available_at=step,
                    evidence_ids=(evt_id, evidence_id),
                )
                # Rearm for next potential cycle
                self.fsm_state = FSMState.SUPPORT_OBSERVED
                self._candidate_departure_step = None
                self._candidate_return_step = None
            elif not signals["raw_on_support"]:
                # Object left support again before dwell confirmed
                self.fsm_state = FSMState.DEPARTURE_CONFIRMED
                self._candidate_return_step = None

        return self.get_snapshot()

    def get_snapshot(self) -> SchemaV2EpisodeMemory:
        """Produce immutable Schema V2 episodic memory snapshot as of current step."""
        obj_identity = ObjectIdentity(
            object_id=self.target_object_id,
            origin_anchor=self.origin_anchor,
            identity_status=self.tracker.target.identity_status,
            visibility=self.tracker.target.visibility,
            current_location_desc=f"on {self.support_semantic_name}" if self.fsm_state in (FSMState.SUPPORT_OBSERVED, FSMState.DEPARTURE_CANDIDATE) else "displaced",
            last_confirmed_in=self.tracker.target.container_id,
        )

        cycle_counter = CycleCounter(
            cycle_key=self.cycle_key,
            object_id=self.target_object_id,
            support_id=self.support_id,
            confirmed=self.confirmed_cycles,
            in_progress=self.in_progress,
            history_complete=self.history_complete,
            event_ids=tuple(self._cycle_event_ids),
            uncertain_count=self.uncertain_count,
        )

        return SchemaV2EpisodeMemory(
            episode_id=self.episode_id,
            as_of_observation=self._current_step,
            last_consumed_action=self._last_consumed_action,
            history_complete=self.history_complete,
            objects={self.target_object_id: obj_identity},
            relations=tuple(self._relations),
            cycles={self.cycle_key: cycle_counter},
            events=tuple(self._events),
        )
