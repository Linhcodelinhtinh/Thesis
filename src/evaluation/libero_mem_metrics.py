"""Read-only metric projection of the pinned upstream T1/T3 evaluator."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SubgoalEvaluationResult:
    task_name: str
    total_subgoals: int
    completed_subgoals: int
    scr: float
    full_success: bool
    overshot: bool
    confirmed_cycles: int
    target_cycles: int
    subgoal_timestamps: Dict[int, int] = field(default_factory=dict)
    termination_reason: str = "in_progress"


class LiberoMemEvaluator:
    """Never replace upstream predicates/counters or call _check_success for logging.

    The environment adapter owns increment timing. This observer only reads the
    resulting state. `full_success` is terminal validity, not an intermediate
    upstream True or a transient completed sequence before the horizon.
    """
    def __init__(self, task_name, target_cycles):
        if target_cycles not in (1, 3):
            raise ValueError("Locked evaluator supports T1/T3 only")
        self.task_name, self.target_cycles = task_name, target_cycles
        self.reset()

    def reset(self):
        self._timestamps = {}
        self._previous_count = 0
        self._step = -1

    def observe(self, step, upstream_env, final_predicate, terminal=False):
        if step <= self._step:
            raise ValueError("Evaluation observations must be strictly ordered")
        self._step = step
        completed = len(upstream_env._satisfied_subgoals)
        if not self._previous_count <= completed <= self.target_cycles:
            raise ValueError("Unexpected upstream sequence progress/reset")
        for index in range(self._previous_count, completed):
            self._timestamps[index] = step
        self._previous_count = completed
        overshot = bool(upstream_env._overshot)
        success = terminal and completed == self.target_cycles and bool(final_predicate) and not overshot
        reason = "overshot" if overshot else "success" if success else "horizon_reached" if terminal else "in_progress"
        return SubgoalEvaluationResult(self.task_name, self.target_cycles, completed,
            completed / self.target_cycles, success, overshot, completed, self.target_cycles,
            dict(self._timestamps), reason)
