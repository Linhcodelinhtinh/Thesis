"""Unit tests for LiberoMemRolloutRunner."""

import numpy as np

from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.simulator.libero_mem_env import LiberoMemEnv


class MockPolicy:
    """Mock policy adapter outputting zero action chunks."""

    def predict_action_chunk(self, observation, prompt):
        chunk = np.zeros((50, 7), dtype=np.float32)
        chunk[:, -1] = -1.0  # open gripper
        return chunk


def test_rollout_runner_closed_loop():
    env = LiberoMemEnv(
        task_name="KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
        target_cycles=1,
        horizon=20,
    )
    policy = MockPolicy()

    # Condition F0 (no memory)
    runner_f0 = LiberoMemRolloutRunner(
        env=env,
        policy_adapter=policy,
        condition="F0",
        execution_horizon=5,
        max_steps=20,
    )
    res_f0 = runner_f0.run_episode("ep_f0", "pick up bowl", initial_state_id=0)
    assert res_f0.condition == "F0"
    assert res_f0.total_steps == 20
    assert not res_f0.success  # zero actions do not solve task
    assert all("Memory:" not in p for p in res_f0.prompt_history)

    # Condition F1 (predicted memory)
    runner_f1 = LiberoMemRolloutRunner(
        env=env,
        policy_adapter=policy,
        condition="F1",
        execution_horizon=5,
        max_steps=20,
    )
    res_f1 = runner_f1.run_episode("ep_f1", "pick up bowl", initial_state_id=0)
    assert res_f1.condition == "F1"
    assert res_f1.total_steps == 20
    assert len(res_f1.prompt_history) > 0
