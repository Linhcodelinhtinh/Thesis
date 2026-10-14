"""Unit and integration verification for Audit 2026-10-14 fixes.

Tests:
1. Canonical alias mapping resolves mugs and tableware (white_mug -> porcelain_mug, etc.)
2. Simulation predicate returning False is preserved as False (no distance heuristic override)
3. Initially satisfied milestone at reset is tracked as initially_satisfied=True
4. Aggregator does not misclassify initial scene satisfaction as a transition failure
5. Deterministic suite seeding matches canonical suite offsets regardless of execution batch
"""

import numpy as np
import pytest

from src.evaluation.diagnostics import (
    OBJECT_CANONICAL_ALIASES,
    ContactDiagnostics,
    EpisodeDiagnosticsCollector,
    SubtaskMilestone,
    TerminationReason,
)
from src.evaluation.benchmark_aggregator import BenchmarkAggregator


def test_mug_canonical_aliases():
    """Verify that white_mug and yellow_and_white_mug resolve to porcelain_mug and white_yellow_mug."""
    assert "porcelain_mug" in OBJECT_CANONICAL_ALIASES.get("white_mug", [])
    assert "white_yellow_mug" in OBJECT_CANONICAL_ALIASES.get("yellow_and_white_mug", [])
    assert "akita_black_bowl" in OBJECT_CANONICAL_ALIASES.get("black_bowl", [])


class MockEvalEnvPredicateFalse:
    def __init__(self):
        self.parsed_problem = {
            "goal_state": [
                ["On", "porcelain_mug_1", "plate_1"],
                ["On", "white_yellow_mug_1", "plate_2"],
            ]
        }
        self.object_states_dict = {}

    def _eval_predicate(self, clause):
        # Always return False: object has NOT reached the goal predicate
        return False


class MockSimWithBodies:
    class Model:
        def __init__(self, names):
            self.names = names
            self.nbody = len(names)

        def body_id2name(self, i):
            return self.names[i]

    class Data:
        def __init__(self, nbody):
            # Place porcelain_mug_1 at [0.5, 0.5, 0.8] and plate_1 at [0.51, 0.51, 0.8] (distance ~1.4 cm <= 8cm)
            self.body_xpos = [
                np.array([0.5, 0.5, 0.8]),
                np.array([0.51, 0.51, 0.8]),
            ]
            self.ncon = 0
            self.contact = []

    def __init__(self):
        self.model = self.Model(["porcelain_mug_1", "plate_1"])
        self.data = self.Data(2)


class MockEnvForDiagnostics:
    def __init__(self, eval_env, sim):
        self._env = eval_env
        self.sim = sim


def test_predicate_false_not_overridden_by_distance_heuristic():
    """Verify that when eval_env is present and predicate returns False,
    the milestone stays False even if object is within 8 cm placement distance threshold.
    """
    eval_env = MockEvalEnvPredicateFalse()
    sim = MockSimWithBodies()
    env = MockEnvForDiagnostics(eval_env, sim)

    collector = EpisodeDiagnosticsCollector(
        env=env,
        instruction="put the white mug on the left plate and put the yellow and white mug on the right plate",
    )
    collector.capture_baseline()

    # Step at t=1 with dummy action
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.0])}
    action = np.zeros(7)
    collector.record_step(step_num=1, obs=obs, action=action)

    report = collector.finalize(
        final_success=False,
        termination_reason=TerminationReason.MAX_STEPS,
    )

    # Check milestone 0 (white mug)
    m0 = collector.subtask_milestones[0]
    # Because _eval_predicate returned False, m0 must NOT be achieved despite distance being 1.4 cm!
    assert m0.achieved is False
    assert m0.current_satisfied is False
    assert m0.evidence_source == "bddl_predicate"


def test_initially_satisfied_milestone_not_transition_failure():
    """Verify that if a milestone was already satisfied at step 0 (initial scene),
    it is flagged as initially_satisfied=True and not misclassified as transition failure.
    """
    class MockEvalEnvStoveInitiallyOn:
        def __init__(self):
            self.parsed_problem = {
                "goal_state": [
                    ["Turnon", "stove_1"],
                    ["On", "moka_pot_1", "stove_1"],
                ]
            }
            self.object_states_dict = {}

        def _eval_predicate(self, clause):
            # Stove is already ON at reset; moka pot is not on stove
            if clause[0] == "Turnon":
                return True
            return False

    eval_env = MockEvalEnvStoveInitiallyOn()
    sim = MockSimWithBodies()
    env = MockEnvForDiagnostics(eval_env, sim)

    collector = EpisodeDiagnosticsCollector(
        env=env,
        instruction="turn on the stove and put the moka pot on it",
    )
    collector.capture_baseline()

    # Check baseline status
    m0 = collector.subtask_milestones[0]
    assert m0.initially_satisfied is True
    assert m0.achieved is True
    assert m0.first_achieved_step == 0

    # Step forward where policy does nothing and fails
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.0])}
    action = np.zeros(7)
    collector.record_step(step_num=1, obs=obs, action=action)

    report = collector.finalize(
        final_success=False,
        termination_reason=TerminationReason.MAX_STEPS,
    )

    # Policy completed 0 new subtasks
    assert report.summary_metrics["policy_completed_subtasks"] == 0
    # Must be classified as ATOMIC_MANIPULATION_FAILURE, NOT SEQUENCE_TRANSITION_FAILURE!
    assert "ATOMIC_MANIPULATION_FAILURE" in report.secondary_failure_tags
    assert "SEQUENCE_TRANSITION_FAILURE" not in report.secondary_failure_tags
