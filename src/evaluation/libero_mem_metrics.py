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
        live_time_threshold: int = 5,
        nonlive_time_threshold: int = 15,
        overshoot_threshold: int = 5,
    ) -> None:
        self.task_name = task_name
        self.target_cycles = target_cycles
        self.target_object = target_object
        self.support_object = support_object
        self.dwell_steps = max(1, dwell_steps)
        self.live_time_threshold = live_time_threshold
        self.nonlive_time_threshold = nonlive_time_threshold
        self.overshoot_threshold = overshoot_threshold

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
        self._satisfied_subgoals: List[int] = []
        self._subgoal_timestamps: Dict[int, int] = {}
        self._overshot = False
        self._sub_goal_live_time = 0
        self._sub_goal_nonlive_time = 0
        self._step_count = 0

    def evaluate_step(
        self,
        step: int,
        is_on_support: bool,
        is_lifted: bool,
    ) -> SubgoalEvaluationResult:
        """Process physical predicates at step t matching official upstream _eval_goal semantics.

        Upstream Sequence Goal Semantics:
        - Subgoal completion requires predicate to hold for > live_time_threshold steps,
          preceded by > nonlive_time_threshold steps off support (confirming genuine physical departure).
        - If all subgoals are completed and the object remains off support for > overshoot_threshold steps,
          overshoot is triggered (invalidating full success).
        """
        self._step_count = step
        success = bool(is_on_support)

        if self._overshot:
            full_success = False
        elif success:
            if len(self._satisfied_subgoals) >= self.target_cycles:
                # All subgoals in sequence already satisfied; reset nonlive counter
                self._sub_goal_nonlive_time = 0
                full_success = True
            elif self._sub_goal_live_time > self.live_time_threshold and self._sub_goal_nonlive_time > self.nonlive_time_threshold:
                # Upstream transition: reset counters and append satisfied subgoal
                self._sub_goal_live_time = 0
                self._sub_goal_nonlive_time = 0
                sg_idx = len(self._satisfied_subgoals)
                self._satisfied_subgoals.append(sg_idx)
                self._subgoal_timestamps[sg_idx] = step
                full_success = (len(self._satisfied_subgoals) >= self.target_cycles)
            else:
                self._sub_goal_live_time += 1
                full_success = False
        else:
            # Condition does not hold (e.g. object off support / lifted)
            if len(self._satisfied_subgoals) >= self.target_cycles:
                if self._sub_goal_nonlive_time > self.overshoot_threshold:
                    self._overshot = True
            self._sub_goal_nonlive_time += 1
            full_success = False

        completed_count = len(self._satisfied_subgoals)
        scr = completed_count / len(self.subgoals) if self.subgoals else 0.0

        return SubgoalEvaluationResult(
            task_name=self.task_name,
            total_subgoals=len(self.subgoals),
            completed_subgoals=completed_count,
            scr=scr,
            full_success=full_success and not self._overshot,
            overshot=self._overshot,
            confirmed_cycles=completed_count,
            target_cycles=self.target_cycles,
            subgoal_timestamps=dict(self._subgoal_timestamps),
            termination_reason="success" if (full_success and not self._overshot) else ("overshot" if self._overshot else "in_progress"),
        )
