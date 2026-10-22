"""Unit tests for LiberoMemEvaluator and LiberoMemEnv."""

import numpy as np
import pytest

from src.evaluation.libero_mem_metrics import LiberoMemEvaluator
from src.simulator.libero_mem_env import LiberoMemEnv


def test_evaluator_subgoal_completion_rate():
    evaluator = LiberoMemEvaluator(
        task_name="T1",
        target_cycles=1,
    )
    assert len(evaluator.subgoals) == 2  # lift then return

    # Step 0: at rest
    r0 = evaluator.evaluate_step(0, is_on_support=True, is_lifted=False)
    assert r0.scr == 0.0
    assert not r0.full_success

    # Step 5: lifted
    r1 = evaluator.evaluate_step(5, is_on_support=False, is_lifted=True)
    assert r1.scr == 0.5
    assert not r1.full_success
    assert r1.completed_subgoals == 1

    # Step 10: placed back
    r2 = evaluator.evaluate_step(10, is_on_support=True, is_lifted=False)
    assert r2.scr == 1.0
    assert r2.full_success
    assert r2.completed_subgoals == 2
    assert not r2.overshot


def test_evaluator_multi_cycle_and_overshoot():
    evaluator = LiberoMemEvaluator(
        task_name="T3",
        target_cycles=3,
    )
    assert len(evaluator.subgoals) == 6

    # 3 cycles: lift & return x3
    evaluator.evaluate_step(1, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(2, is_on_support=True, is_lifted=False)
    evaluator.evaluate_step(3, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(4, is_on_support=True, is_lifted=False)
    evaluator.evaluate_step(5, is_on_support=False, is_lifted=True)
    r3 = evaluator.evaluate_step(6, is_on_support=True, is_lifted=False)

    assert r3.full_success
    assert r3.scr == 1.0
    assert not r3.overshot
    assert r3.confirmed_cycles == 3

    # Extra lift causes overshoot!
    r_over = evaluator.evaluate_step(7, is_on_support=False, is_lifted=True)
    assert r_over.overshot
    assert not r_over.full_success


def test_libero_mem_env_step_and_observation():
    env = LiberoMemEnv(
        task_name="KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
        target_cycles=1,
        privileged_oracle=True,
    )
    obs = env.reset()
    assert "agentview_image" in obs
    assert "robot0_eye_in_hand_image" in obs
    assert obs["robot0_eef_pos"].shape == (3,)
    assert obs["robot0_gripper_qpos"].shape == (2,)

    action = np.zeros(7, dtype=np.float32)
    next_obs, reward, done, info = env.step(action)
    assert "eval_result" in info
    assert "scr" in info
    assert "oracle_signals" in info
