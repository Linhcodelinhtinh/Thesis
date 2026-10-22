"""Evaluation metrics, subgoal completion tracking, and overshoot diagnostics for LIBERO-Mem.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 12:
- Subgoal Completion Rate (SCR): completed subgoals / total sequence subgoals.
- Ordered sequence progress: validates subgoals in strict chronological order.
- Repeated predicate interval detection: ensures distinct cycle events (T3).
- Overshoot detection: flags trajectories exceeding the target cycle count.
- Isolated from policy execution; does not mutate environment state during evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SubgoalDefinition:
    """Specification of an ordered subgoal in a LIBERO-Mem task."""

    index: int
    name: str
    target_predicate: str  # e.g. "on", "lifted"
    subject: str
    target_object: str
    cycle_index: int


@dataclass
class SubgoalEvaluationResult:
    """Comprehensive evaluation record for an episode trajectory."""

    task_name: str
    total_subgoals: int
    completed_subgoals: int
    scr: float  # Subgoal Completion Rate [0.0, 1.0]
    full_success: bool
    overshot: bool
    confirmed_cycles: int
    target_cycles: int
    subgoal_timestamps: Dict[int, int] = field(default_factory=dict)
    termination_reason: str = "horizon_reached"


class LiberoMemEvaluator:
    """Evaluates task execution progress and conformance for LIBERO-Mem.

    Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md and upstream BDDL:
    - BDDL for repeated manipulation (e.g. T3) specifies a Sequence of On predicates:
      (Sequence (On bowl plate) (On bowl plate) (On bowl plate))
      Therefore, official sequence subgoals correspond to the target cycle milestones (e.g. 3 for T3, 1 for T1).
    - Requires sustained dwell steps (consecutive steps satisfying condition)
      to avoid accepting single-frame instantaneous sensor noise or transient triggers.
    - Strict interval semantics: between two consecutive On milestones, the object
      must depart / be lifted for at least dwell_steps, followed by resting on support
      for at least dwell_steps.
    - Overshoot detection: if after satisfying all target cycles the agent continues
      and lifts the object again for dwell_steps, overshot is flagged and full_success
      is invalidated.
    """

    def __init__(
        self,
        task_name: str,
        target_cycles: int = 1,
        target_object: str = "bowl",
        support_object: str = "plate",
        dwell_steps: int = 2,
    ) -> None:
        self.task_name = task_name
        self.target_cycles = target_cycles
        self.target_object = target_object
        self.support_object = support_object
        self.dwell_steps = max(1, dwell_steps)

        self.subgoals = self._build_subgoal_definitions()
        self.reset()

    def _build_subgoal_definitions(self) -> List[SubgoalDefinition]:
        """Construct ordered subgoal list matching official BDDL Sequence goals."""
        subgoals: List[SubgoalDefinition] = []
        for cycle in range(1, self.target_cycles + 1):
            subgoals.append(
                SubgoalDefinition(
                    index=cycle - 1,
                    name=f"cycle_{cycle}_on_support",
                    target_predicate="on",
                    subject=self.target_object,
                    target_object=self.support_object,
                    cycle_index=cycle,
                )
            )
        return subgoals

    def reset(self) -> None:
        """Reset evaluation state for a new episode."""
        self._current_subgoal_idx = 0
        self._subgoal_timestamps: Dict[int, int] = {}
        self._completed_cycles = 0
        self._overshot = False
        self._consecutive_lifted = 0
        self._consecutive_on_support = 0
        self._cycle_departure_confirmed = False
        self._step_count = 0

    def evaluate_step(
        self,
        step: int,
        is_on_support: bool,
        is_lifted: bool,
    ) -> SubgoalEvaluationResult:
        """Process physical predicates at step t and advance sequence progress.

        Strict interval and dwell semantics:
        - To confirm departure/lift: object must be lifted and not on support for >= dwell_steps consecutive steps.
        - To confirm return on support: object must have confirmed departure, then be on support and not lifted for >= dwell_steps consecutive steps.
        - Overshoot: if all cycles are completed, another confirmed departure/lift triggers overshoot.
        """
        self._step_count = step

        # Track consecutive dwell counts
        if is_lifted and not is_on_support:
            self._consecutive_lifted += 1
            self._consecutive_on_support = 0
        elif is_on_support and not is_lifted:
            self._consecutive_on_support += 1
            self._consecutive_lifted = 0
        else:
            self._consecutive_lifted = 0
            self._consecutive_on_support = 0

        # Subgoal progression
        if self._current_subgoal_idx < len(self.subgoals):
            # Phase 1: Wait for confirmed departure / lift
            if not self._cycle_departure_confirmed:
                if self._consecutive_lifted >= self.dwell_steps:
                    self._cycle_departure_confirmed = True

            # Phase 2: If departure is confirmed, wait for confirmed return on support
            else:
                if self._consecutive_on_support >= self.dwell_steps:
                    target_sg = self.subgoals[self._current_subgoal_idx]
                    self._subgoal_timestamps[target_sg.index] = step
                    self._current_subgoal_idx += 1
                    self._completed_cycles += 1
                    self._cycle_departure_confirmed = False  # Reset for next cycle
        else:
            # All target cycles completed; check for overshoot
            if self._consecutive_lifted >= self.dwell_steps:
                self._overshot = True

        full_success = (self._current_subgoal_idx == len(self.subgoals)) and not self._overshot
        scr = self._current_subgoal_idx / len(self.subgoals) if self.subgoals else 0.0

        return SubgoalEvaluationResult(
            task_name=self.task_name,
            total_subgoals=len(self.subgoals),
            completed_subgoals=self._current_subgoal_idx,
            scr=scr,
            full_success=full_success,
            overshot=self._overshot,
            confirmed_cycles=self._completed_cycles,
            target_cycles=self.target_cycles,
            subgoal_timestamps=dict(self._subgoal_timestamps),
            termination_reason="success" if full_success else ("overshot" if self._overshot else "in_progress"),
        )
