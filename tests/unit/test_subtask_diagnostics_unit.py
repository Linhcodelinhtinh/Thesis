"""Unit tests for Subtask Milestone Diagnostics (Phase 11).

Verifies:
- BDDL AST goal decomposition into subtask milestone predicates.
- Natural language decomposition for multi-stage compositional tasks.
- Step-by-step milestone progress tracking and first-achieved step (t_milestone).
- Subtask completion rate calculation.
- Sequential survival steps determination.
- Failure classification: ATOMIC_MANIPULATION_FAILURE vs SEQUENCE_TRANSITION_FAILURE.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from src.evaluation.diagnostics import (
    DiagnosticsConfig,
    EpisodeDiagnosticsCollector,
    SubtaskMilestone,
    TerminationReason,
    extract_subtask_milestones,
)


class MockEnvWithParsedProblem:
    """Mock environment exposing BDDL parsed_problem AST."""

    def __init__(self, goal_ast: List[Any], language_instruction: str = "") -> None:
        self.parsed_problem = {"goal": goal_ast}
        self.language_instruction = language_instruction
        self.sim = None


def test_extract_subtask_milestones_from_bddl_ast():
    """Verify decomposing BDDL goal AST (and (in a b) (in c b)) into milestones."""
    goal_ast = [
        "and",
        ["in", "alphabet_soup_1", "basket_1"],
        ["in", "tomato_sauce_1", "basket_1"],
    ]
    env = MockEnvWithParsedProblem(goal_ast)
    milestones = extract_subtask_milestones(env)

    assert len(milestones) == 2
    assert milestones[0].target_object == "alphabet_soup_1"
    assert milestones[1].target_object == "tomato_sauce_1"
    assert "in" in milestones[0].description


def test_extract_subtask_milestones_from_libero_goal_state():
    """Verify extracting milestones from official LIBERO parsed_problem['goal_state'] format."""
    goal_state = [
        ["in", "alphabet_soup_1", "basket_1"],
        ["in", "tomato_sauce_1", "basket_1"],
    ]
    mock_env = type("MockLiberoEnv", (), {
        "parsed_problem": {"goal_state": goal_state},
        "language_instruction": "put both the alphabet soup and the tomato sauce in the basket",
        "sim": None,
    })()
    milestones = extract_subtask_milestones(mock_env)

    assert len(milestones) == 2
    assert milestones[0].name == "milestone_0_in_alphabet_soup_1_basket_1"
    assert milestones[0].clause == ["in", "alphabet_soup_1", "basket_1"]
    assert milestones[1].name == "milestone_1_in_tomato_sauce_1_basket_1"
    assert milestones[1].clause == ["in", "tomato_sauce_1", "basket_1"]


def test_subtask_milestone_evaluation_via_eval_predicate():
    """Verify authentic _eval_predicate evaluation during step progression."""
    achieved_clauses = set()

    class MockLiberoProblemEnv:
        def __init__(self):
            self.parsed_problem = {
                "goal_state": [
                    ["open", "cabinet_middle_drawer_1"],
                    ["in", "black_bowl_1", "cabinet_middle_drawer_1"],
                ]
            }
            self.object_states_dict = {"cabinet_middle_drawer_1": object(), "black_bowl_1": object()}
            self.sim = None

        def _eval_predicate(self, state):
            return tuple(state) in achieved_clauses

    env = MockLiberoProblemEnv()
    collector = EpisodeDiagnosticsCollector(
        env=env,
        instruction="put the black bowl in the middle drawer and open it",
    )
    assert len(collector.subtask_milestones) == 2
    assert collector.subtask_milestones[0].achieved is False

    sample_obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.5])}
    dummy_act = np.zeros(7, dtype=np.float32)

    # Step 0: nothing achieved
    collector.record_step(0, sample_obs, dummy_act)
    assert collector.subtask_milestones[0].achieved is False

    # Simulate drawer opened at step 1
    achieved_clauses.add(("open", "cabinet_middle_drawer_1"))
    collector.record_step(1, sample_obs, dummy_act)
    assert collector.subtask_milestones[0].achieved is True
    assert collector.subtask_milestones[0].first_achieved_step == 1
    assert collector.subtask_milestones[1].achieved is False


def test_extract_subtask_milestones_from_language():
    """Verify natural language decomposition across compositional patterns."""
    # Pattern A: dual pick and place
    inst_a = "put both the cream cheese box and the butter in the basket"
    m_a = extract_subtask_milestones(None, instruction=inst_a)
    assert len(m_a) == 2
    assert "cream_cheese_box" in m_a[0].target_object
    assert "butter" in m_a[1].target_object

    # Pattern B: mechanism and placement (stove)
    inst_b = "turn on the stove and put the moka pot on it"
    m_b = extract_subtask_milestones(None, instruction=inst_b)
    assert len(m_b) == 2
    assert "stove" in m_b[0].target_object
    assert "moka_pot" in m_b[1].target_object

    # Pattern C: placement and close (drawer)
    inst_c = "put the black bowl in the bottom drawer of the cabinet and close it"
    m_c = extract_subtask_milestones(None, instruction=inst_c)
    assert len(m_c) == 2
    assert "close" in m_c[1].description.lower()

    # Pattern D: microwave
    inst_d = "put the yellow and white mug in the microwave and close it"
    m_d = extract_subtask_milestones(None, instruction=inst_d)
    assert len(m_d) == 2

    # Pattern E: single object task fallback
    inst_e = "pick up the alphabet soup and place it in the basket"
    m_e = extract_subtask_milestones(None, instruction=inst_e)
    assert len(m_e) == 1
    assert m_e[0].name == "milestone_0_complete_task"


def test_subtask_milestone_progression_tracking():
    """Verify tracking t_milestone, completion rates, and survival steps."""
    inst = "put both the alphabet soup and the tomato sauce in the basket"
    collector = EpisodeDiagnosticsCollector(
        env=None,
        instruction=inst,
        task_name="libero_10_0",
    )
    assert len(collector.subtask_milestones) == 2
    assert collector.subtask_milestones[0].achieved is False
    assert collector.subtask_milestones[1].achieved is False

    # Simulate step records
    sample_obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.5]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }
    dummy_action = np.zeros(7, dtype=np.float32)

    for step in range(10):
        collector.record_step(step, sample_obs, dummy_action)

    # Manually mark milestone 0 achieved at step 5
    collector.subtask_milestones[0].achieved = True
    collector.subtask_milestones[0].first_achieved_step = 5

    # Finalize with failure (timeout)
    report = collector.finalize(
        final_success=False,
        termination_reason=TerminationReason.MAX_STEPS,
    )

    assert report.termination_reason == "MAX_STEPS"
    # Milestone 0 achieved, milestone 1 failed -> SEQUENCE_TRANSITION_FAILURE
    assert "SEQUENCE_TRANSITION_FAILURE" in report.secondary_failure_tags
    assert report.summary_metrics["subtask_completion_rate"] == 0.5
    assert report.summary_metrics["sequential_survival_steps"] == 5
    assert report.summary_metrics["completed_subtasks"] == 1
    assert report.summary_metrics["total_subtasks"] == 2


def test_atomic_manipulation_failure_tag():
    """Verify that failing before any subtask milestone yields ATOMIC_MANIPULATION_FAILURE."""
    inst = "put both the cream cheese box and the butter in the basket"
    collector = EpisodeDiagnosticsCollector(
        env=None,
        instruction=inst,
    )

    sample_obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.5]),
    }
    dummy_action = np.zeros(7, dtype=np.float32)
    for step in range(5):
        collector.record_step(step, sample_obs, dummy_action)

    # 0 milestones achieved
    report = collector.finalize(
        final_success=False,
        termination_reason=TerminationReason.MAX_STEPS,
    )

    assert "ATOMIC_MANIPULATION_FAILURE" in report.secondary_failure_tags
    assert report.summary_metrics["subtask_completion_rate"] == 0.0
    assert report.summary_metrics["completed_subtasks"] == 0


def test_full_success_marks_all_milestones():
    """Verify that task success marks all subtask milestones as achieved."""
    inst = "turn on the stove and put the moka pot on it"
    collector = EpisodeDiagnosticsCollector(
        env=None,
        instruction=inst,
    )

    sample_obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.5])}
    dummy_action = np.zeros(7, dtype=np.float32)
    for step in range(8):
        collector.record_step(step, sample_obs, dummy_action)

    report = collector.finalize(
        final_success=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    assert report.termination_reason == "SUCCESS"
    assert report.summary_metrics["subtask_completion_rate"] == 1.0
    assert report.summary_metrics["completed_subtasks"] == 2
    assert report.summary_metrics["sequential_survival_steps"] == 8
    assert len(report.subtask_milestones) == 2
    assert all(m["achieved"] is True for m in report.subtask_milestones)
