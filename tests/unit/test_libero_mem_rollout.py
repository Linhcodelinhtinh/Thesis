"""Closed-loop wiring tests with explicit unit fixtures, no simulation imports."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.visual_pipeline import visual_writer_factory
from src.training.libero_mem_data import LiberoMemDataProcessor
from test_rgb_cycle_writer import frame


class UnitEnv:
    task_name = "UNIT_FIXTURE_NOT_BENCHMARK"
    def reset(self, initial_state_id=0):
        self.step_id = 0
        return frame(0)
    def step(self, action):
        self.step_id += 1
        t = self.step_id
        result = SimpleNamespace(full_success=False, scr=0., completed_subgoals=0, total_subgoals=1,
            overshot=False, confirmed_cycles=0, target_cycles=1, termination_reason="unit_fixture")
        return frame(t, lift=.15 if 4 <= t < 8 else 0., gap=.005 if 4 <= t < 8 else .08), 0., t == 15, {"eval_result": result}


class UnitPolicy:
    def reset(self):
        self.calls = []
    def predict_action_chunk(self, observation, prompt):
        self.calls.append(prompt)
        return np.zeros((50, 7), dtype=np.float32)


def test_rollout_and_preparation_share_identical_causal_prompts():
    factory = visual_writer_factory(None, "libero_mem_refined")  # typed fixture detections in observations
    policy = UnitPolicy()
    runner = LiberoMemRolloutRunner(UnitEnv(), policy, "F1", execution_horizon=1, max_steps=15,
        writer_factory=factory, renderer=LiberoMemTextRenderer())
    result = runner.run_episode("unit", "lift bowl")
    observations = [frame(t, lift=.15 if 4 <= t < 8 else 0., gap=.005 if 4 <= t < 8 else .08) for t in range(16)]
    samples = LiberoMemDataProcessor(writer_factory=factory).process_episode("unit", "lift bowl", observations, np.zeros((16, 7)))
    assert result.prompt_history == [s.rendered_prompt for s in samples[:15]]
    assert len(result.memory_trace) == 16
    assert result.memory_trace[-1]["snapshot"]["cycles"]["bowl_A_plate_A"]["confirmed"] == 1
    assert result.memory_trace[-1]["snapshot"]["last_consumed_action"] == 14


def test_memory_off_does_not_construct_or_query_a_writer():
    def forbidden(**kwargs):
        raise AssertionError("Writer touched in no-memory mode")
    runner = LiberoMemRolloutRunner(UnitEnv(), UnitPolicy(), "F0", max_steps=15, writer_factory=forbidden)
    result = runner.run_episode("unit", "  original instruction\n")
    assert set(result.prompt_history) == {"  original instruction\n"}
    assert result.memory_trace == []


def test_predicted_rollout_requires_configured_backend():
    runner = LiberoMemRolloutRunner(UnitEnv(), UnitPolicy(), "F1")
    with pytest.raises(ValueError, match="RGB writer_factory"):
        runner.run_episode("unit", "lift bowl")
