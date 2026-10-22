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
    """Evaluates task execution progress and conformance for LIBERO-Mem."""

    def __init__(
        self,
        task_name: str,
        target_cycles: int = 1,
        target_object: str = "bowl",
        support_object: str = "plate",
    ) -> None:
        self.task_name = task_name
        self.target_cycles = target_cycles
        self.target_object = target_object
        self.support_object = support_object

        self.subgoals = self._build_subgoal_definitions()
        self.reset()

    def _build_subgoal_definitions(self) -> List[SubgoalDefinition]:
        """Construct ordered subgoal list matching BDDL sequence."""
        subgoals: List[SubgoalDefinition] = []
        idx = 0
        for cycle in range(1, self.target_cycles + 1):
            subgoals.append(
                SubgoalDefinition(
                    index=idx,
                    name=f"cycle_{cycle}_lift",
                    target_predicate="lifted",
                    subject=self.target_object,
                    target_object=self.support_object,
                    cycle_index=cycle,
                )
            )
            idx += 1
            subgoals.append(
                SubgoalDefinition(
                    index=idx,
                    name=f"cycle_{cycle}_return",
                    target_predicate="on",
                    subject=self.target_object,
                    target_object=self.support_object,
                    cycle_index=cycle,
                )
            )
            idx += 1
        return subgoals

    def reset(self) -> None:
        """Reset evaluation state for a new episode."""
        self._current_subgoal_idx = 0
        self._subgoal_timestamps: Dict[int, int] = {}
        self._completed_cycles = 0
        self._overshot = False
        self._is_currently_on_support = True
        self._is_currently_lifted = False
        self._step_count = 0

    def evaluate_step(
        self,
        step: int,
        is_on_support: bool,
        is_lifted: bool,
    ) -> SubgoalEvaluationResult:
        """Process physical predicates at step t and advance sequence progress.

        Strict interval semantics:
        - To satisfy 'lifted': object must have been on support, then lift is confirmed.
        - To satisfy 'on' (return): object must have been lifted, then return is confirmed.
        """
        self._step_count = step

        if self._current_subgoal_idx < len(self.subgoals):
            target_sg = self.subgoals[self._current_subgoal_idx]

            if target_sg.target_predicate == "lifted":
                if is_lifted and not is_on_support:
                    self._subgoal_timestamps[target_sg.index] = step
                    self._current_subgoal_idx += 1
                    self._is_currently_lifted = True
                    self._is_currently_on_support = False

            elif target_sg.target_predicate == "on":
                if is_on_support and not is_lifted and self._is_currently_lifted:
                    self._subgoal_timestamps[target_sg.index] = step
                    self._current_subgoal_idx += 1
                    self._completed_cycles += 1
                    self._is_currently_lifted = False
                    self._is_currently_on_support = True

        else:
            # All subgoals satisfied; check for overshoot
            if is_lifted and not is_on_support:
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
            termination_reason="success" if full_success else "in_progress",
        )
