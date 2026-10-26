"""Causal, evidence-backed T1/T3 lift/return counter, independent of the goal.

Counts are writer-confirmed visual estimates, not official contact truth.
"""
from enum import Enum
import numpy as np
from src.memory.models import (CycleCounter, EvidenceSource, ObjectIdentity,
                               RelationFact, RelationStatus, SchemaV2EpisodeMemory, V2MemoryEvent)
from src.perception.object_tracker import CausalObjectTracker


class FSMState(str, Enum):
    UNINITIALIZED = "uninitialized"
    SUPPORT_OBSERVED = "support_observed"
    DEPARTURE_CANDIDATE = "departure_candidate"
    DEPARTURE_CONFIRMED = "departure_confirmed"
    RETURN_CANDIDATE = "return_candidate"
    RETURN_CONFIRMED = "return_confirmed"


class LiberoMemEventWriter:
    VERSION = "rgb_cycle_v3"

    def __init__(self, episode_id, target_object_id="bowl_A", target_semantic_name="bowl",
                 origin_anchor="bowl on its initially observed plate", support_id="plate_A",
                 support_semantic_name="plate", tracker=None, dwell_steps=3):
        if dwell_steps < 2:
            raise ValueError("dwell_steps must be at least 2")
        self.episode_id = episode_id
        self.target_object_id = target_object_id
        self.origin_anchor = origin_anchor
        self.support_id = support_id
        self.cycle_key = f"{target_object_id}_{support_id}"
        self.tracker = tracker if tracker is not None else CausalObjectTracker(
            target_object_id, target_semantic_name, origin_anchor, support_id,
            support_semantic_name, dwell_steps=dwell_steps)
        self.dwell_steps = dwell_steps
        self.fsm_state = FSMState.UNINITIALIZED
        self.confirmed_cycles = 0
        self.in_progress = False
        self.history_complete = True
        self.uncertain_count = 0  # unobserved intervals, NOT inferred missed cycles
        self._events = []
        self._relations = []
        self._cycle_event_ids = []
        self._current_step = -1
        self._last_consumed_action = -1
        self._candidate_kind = None
        self._candidate_frames = []
        self._in_gap = False
        self._initialized = False
        self.last_signals = None

    def _event(self, kind, frames, details=None):
        evidence = tuple(dict.fromkeys(eid for _, ids in frames for eid in ids))
        if not evidence:
            raise ValueError("Confirmed event requires RGB evidence")
        event = V2MemoryEvent(
            id=f"{self.episode_id}:evt:{len(self._events)}:{kind}", type=kind,
            occurred_at=frames[0][0], available_at=self._current_step,
            source=EvidenceSource.PREDICTED_RGB_TRACKER, evidence_ids=evidence,
            details=dict(details or {}, object_id=self.target_object_id, support_id=self.support_id,
                         writer_version=self.VERSION, confirmation="visual_estimate"),
        )
        self._events.append(event)
        return event

    def _relation(self, event, status):
        self._relations.append(RelationFact(
            subject=self.target_object_id, predicate="on", object=self.support_id, status=status,
            occurred_at=event.occurred_at, available_at=event.available_at,
            evidence_ids=(event.id,) + event.evidence_ids,
        ))

    def _clear_candidate(self):
        self._candidate_kind = None
        self._candidate_frames = []

    def _dwell(self, kind, signals):
        if self._candidate_kind != kind:
            self._candidate_kind = kind
            self._candidate_frames = []
        self._candidate_frames.append((self._current_step, tuple(signals["evidence_ids"])))
        return len(self._candidate_frames) >= self.dwell_steps

    def step(self, step, observation, last_action=None, last_action_index=None):
        if step < 0 or step <= self._current_step:
            raise ValueError(f"Step out of order: {step} <= {self._current_step}")
        if last_action is not None:
            action = np.asarray(last_action)
            if step == 0 or action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Only a finite, already executed 7D action may accompany a transition")
            consumed = step - 1 if last_action_index is None else last_action_index
            if not isinstance(consumed, (int, np.integer)) or not 0 <= consumed <= step:
                raise ValueError("Invalid consumed action index")
        else:
            if last_action_index is not None:
                raise ValueError("Action index supplied without an action")
            consumed = -1
        skipped = step != self._current_step + 1
        signals = self.tracker.update(step, observation, last_action)
        self._current_step = step
        self._last_consumed_action = int(consumed)
        self.last_signals = signals
        unknown = not signals["has_grounding_evidence"]
        if skipped or unknown:
            self.history_complete = False
            if not self._in_gap:
                self.uncertain_count += 1
            self._in_gap = True
            self._clear_candidate()
            self.fsm_state = (FSMState.DEPARTURE_CONFIRMED if self.in_progress else
                              FSMState.SUPPORT_OBSERVED if self._initialized else FSMState.UNINITIALIZED)
            if unknown:
                return self.get_snapshot()
        self._in_gap = False
        rest = signals["raw_on_support"] and signals["stable"] and signals["gripper_released"]
        if not self._initialized:
            if not signals["raw_on_support"]:
                # An unverified departure during initialization may contain a
                # missed cycle; later acquisition cannot establish exact zero.
                self.history_complete = False
            if rest and self._dwell("initial_support", signals):
                event = self._event("initial_support_observed", self._candidate_frames)
                self._relation(event, RelationStatus.CONFIRMED_BY_WRITER)
                self._initialized = True
                self.fsm_state = FSMState.SUPPORT_OBSERVED
                self._clear_candidate()
            elif not rest:
                self._clear_candidate()
        elif not self.in_progress:
            if signals["raw_lifted"]:
                self.fsm_state = FSMState.DEPARTURE_CANDIDATE
                if self._dwell("departure", signals):
                    event = self._event("departed_support", self._candidate_frames)
                    self._relation(event, RelationStatus.RETRACTED)
                    self.in_progress = True
                    self.fsm_state = FSMState.DEPARTURE_CONFIRMED
                    self._clear_candidate()
            else:
                self._clear_candidate()
                self.fsm_state = FSMState.SUPPORT_OBSERVED
        else:
            if rest:
                self.fsm_state = FSMState.RETURN_CANDIDATE
                if self._dwell("return", signals):
                    self.confirmed_cycles += 1
                    event = self._event("returned_to_support", self._candidate_frames,
                                        {"cycle_count": self.confirmed_cycles})
                    self._cycle_event_ids.append(event.id)
                    self._relation(event, RelationStatus.CONFIRMED_BY_WRITER)
                    self.in_progress = False
                    self.fsm_state = FSMState.SUPPORT_OBSERVED
                    self._clear_candidate()
            else:
                self._clear_candidate()
                self.fsm_state = FSMState.DEPARTURE_CONFIRMED
        return self.get_snapshot()

    def get_snapshot(self):
        signals = self.last_signals or {}
        location = None
        if signals.get("has_grounding_evidence"):
            location = "near original plate" if signals["raw_on_support"] else "away from original plate"
        identity = ObjectIdentity(
            self.target_object_id, self.origin_anchor, self.tracker.target.identity_status,
            self.tracker.target.visibility, current_location_desc=location,
            metadata={"source": EvidenceSource.PREDICTED_RGB_TRACKER.value, "writer_version": self.VERSION},
        )
        cycles = {}
        if self._initialized:
            cycles[self.cycle_key] = CycleCounter(
                self.cycle_key, self.target_object_id, self.support_id, self.confirmed_cycles,
                self.in_progress, self.history_complete, tuple(self._cycle_event_ids), self.uncertain_count,
            )
        return SchemaV2EpisodeMemory(
            self.episode_id, max(0, self._current_step), self._last_consumed_action,
            self.history_complete and self._current_step >= 0, {self.target_object_id: identity},
            tuple(self._relations), cycles, tuple(self._events),
        )
