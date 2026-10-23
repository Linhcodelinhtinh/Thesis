"""Privileged Oracle Memory Writer for Diagnostic Comparisons (Branch O1).

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 8.3 & Section 13:
- Consumes simulator ground truth / exact physical states up to step t.
- Strictly CAUSAL: never accesses future states (>t) or future actions.
- Explicitly tags all facts with EvidenceSource.ORACLE_SIMULATOR.
- Does NOT provide "next subgoal" shortcuts or action recommendations.
- Used exclusively for upper-bound diagnostic comparison against predicted memory.
"""

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


class LiberoMemOracleWriter:
    """Privileged diagnostic writer accessing simulator ground truth."""

    def __init__(
        self,
        episode_id: str,
        target_object_id: str = "bowl_A",
        target_semantic_name: str = "bowl",
        origin_anchor: str = "bowl initially on the left plate",
        support_id: str = "plate_A",
        support_semantic_name: str = "plate",
    ) -> None:
        self.episode_id = episode_id
        self.target_object_id = target_object_id
        self.target_semantic_name = target_semantic_name
        self.origin_anchor = origin_anchor
        self.support_id = support_id
        self.support_semantic_name = support_semantic_name

        self.cycle_key = f"{self.target_object_id}_{self.support_id}"
        self.confirmed_cycles = 0
        self.in_progress = False

        self._events: List[V2MemoryEvent] = []
        self._relations: List[RelationFact] = []
        self._cycle_event_ids: List[str] = []

        self._was_on_support = True
        self._departure_step: Optional[int] = None
        self._current_step = 0
        self._last_consumed_action = -1
        self._container_id: Optional[str] = None
        self._is_occluded: bool = False

    def step(
        self,
        step: int,
        oracle_signals: Dict[str, Any],
    ) -> SchemaV2EpisodeMemory:
        """Process ground-truth simulation signals at step t.

        Expected oracle_signals:
        - `is_on_support`: bool (ground truth physical contact / predicate)
        - `is_held`: bool (ground truth contact with gripper)
        - `object_pos`: np.ndarray
        - `container_id`: Optional[str]
        - `is_occluded`: bool
        """
        if not isinstance(oracle_signals, dict):
            raise TypeError("oracle_signals must be a dictionary")

        self._current_step = step
        self._last_consumed_action = step - 1 if step > 0 else -1

        if "is_on_support" not in oracle_signals:
            raise KeyError("oracle_signals missing required ground truth key 'is_on_support'")

        is_on_support = bool(oracle_signals["is_on_support"])
        is_held = bool(oracle_signals.get("is_held", False))
        
        if "container_id" in oracle_signals:
            self._container_id = oracle_signals["container_id"]
        if "is_occluded" in oracle_signals:
            self._is_occluded = bool(oracle_signals["is_occluded"])

        # Departure detection
        if self._was_on_support and not is_on_support:
            self._was_on_support = False
            self.in_progress = True
            self._departure_step = step
            evt_id = f"oracle_evt_{len(self._events) + 1}_departed_support"
            self._events.append(
                V2MemoryEvent(
                    id=evt_id,
                    type="departed_support",
                    occurred_at=step,
                    available_at=step,
                    source=EvidenceSource.ORACLE_SIMULATOR,
                    probability=1.0,
                    details={"cycle_index": self.confirmed_cycles + 1},
                )
            )
            self._cycle_event_ids.append(evt_id)
            self._relations.append(
                RelationFact(
                    subject=self.target_object_id,
                    predicate="on",
                    object=self.support_id,
                    status=RelationStatus.RETRACTED,
                    occurred_at=step,
                    available_at=step,
                    evidence_ids=(evt_id,),
                )
            )

        # Return detection
        elif not self._was_on_support and is_on_support:
            self._was_on_support = True
            self.in_progress = False
            self.confirmed_cycles += 1
            evt_id = f"oracle_evt_{len(self._events) + 1}_returned_to_support"
            self._events.append(
                V2MemoryEvent(
                    id=evt_id,
                    type="returned_to_support",
                    occurred_at=step,
                    available_at=step,
                    source=EvidenceSource.ORACLE_SIMULATOR,
                    probability=1.0,
                    details={"cycle_count": self.confirmed_cycles},
                )
            )
            self._cycle_event_ids.append(evt_id)
            self._relations.append(
                RelationFact(
                    subject=self.target_object_id,
                    predicate="on",
                    object=self.support_id,
                    status=RelationStatus.CONFIRMED_BY_WRITER,
                    occurred_at=step,
                    available_at=step,
                    evidence_ids=(evt_id,),
                )
            )

        return self.get_snapshot()

    def get_snapshot(
        self,
        is_occluded: Optional[bool] = None,
        container_id: Optional[str] = None,
    ) -> SchemaV2EpisodeMemory:
        """Produce immutable privileged Schema V2 episodic memory snapshot."""
        effective_occluded = self._is_occluded if is_occluded is None else is_occluded
        effective_container = self._container_id if container_id is None else container_id

        obj_identity = ObjectIdentity(
            object_id=self.target_object_id,
            origin_anchor=self.origin_anchor,
            identity_status=IdentityStatus.TRACKED,
            visibility=VisibilityStatus.OCCLUDED if effective_occluded else VisibilityStatus.VISIBLE,
            current_location_desc=f"on {self.support_semantic_name}" if self._was_on_support else "displaced",
            last_confirmed_in=effective_container,
            metadata={"source": EvidenceSource.ORACLE_SIMULATOR},
        )

        cycle_counter = CycleCounter(
            cycle_key=self.cycle_key,
            object_id=self.target_object_id,
            support_id=self.support_id,
            confirmed=self.confirmed_cycles,
            in_progress=self.in_progress,
            history_complete=True,
            event_ids=tuple(self._cycle_event_ids),
            uncertain_count=0,
        )

        return SchemaV2EpisodeMemory(
            episode_id=self.episode_id,
            as_of_observation=self._current_step,
            last_consumed_action=self._last_consumed_action,
            history_complete=True,
            objects={self.target_object_id: obj_identity},
            relations=tuple(self._relations),
            cycles={self.cycle_key: cycle_counter},
            events=tuple(self._events),
        )
