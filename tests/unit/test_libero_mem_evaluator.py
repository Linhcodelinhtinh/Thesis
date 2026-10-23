"""Unit tests for LiberoMemEvaluator and LiberoMemEnv."""

import numpy as np
import pytest

from src.evaluation.libero_mem_metrics import LiberoMemEvaluator
from src.simulator.libero_mem_env import LiberoMemEnv


def test_evaluator_subgoal_completion_rate():
    evaluator = LiberoMemEvaluator(
        task_name="T1",
        target_cycles=1,
        dwell_steps=2,
        live_time_threshold=1,
        nonlive_time_threshold=1,
    )
    assert len(evaluator.subgoals) == 1  # 1 sequential On goal for T1

    # Step 0: at rest
    r0 = evaluator.evaluate_step(0, is_on_support=True, is_lifted=False)
    assert r0.scr == 0.0
    assert not r0.full_success

    # Step 5-6: lift with 2 dwell steps
    evaluator.evaluate_step(5, is_on_support=False, is_lifted=True)
    r1 = evaluator.evaluate_step(6, is_on_support=False, is_lifted=True)
    assert r1.scr == 0.0  # Lifted is departure phase, not yet completed cycle
    assert not r1.full_success
    assert r1.completed_subgoals == 0

    # Step 10: single return frame (dwell not yet met)
    r2 = evaluator.evaluate_step(10, is_on_support=True, is_lifted=False)
    assert not r2.full_success
    assert r2.completed_subgoals == 0

    # Step 11: 2nd return frame (dwell satisfied -> completed!)
    r3 = evaluator.evaluate_step(11, is_on_support=True, is_lifted=False)
    assert r3.scr == 1.0
    assert r3.full_success
    assert r3.completed_subgoals == 1
    assert not r3.overshot


def test_evaluator_multi_cycle_and_overshoot():
    evaluator = LiberoMemEvaluator(
        task_name="T3",
        target_cycles=3,
        dwell_steps=2,
        live_time_threshold=0,
        nonlive_time_threshold=0,
        overshoot_threshold=0,
    )
    assert len(evaluator.subgoals) == 3  # 3 sequential On goals for T3

    # Cycle 1: 2 steps lift, 2 steps return
    evaluator.evaluate_step(1, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(2, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(3, is_on_support=True, is_lifted=False)
    r1 = evaluator.evaluate_step(4, is_on_support=True, is_lifted=False)
    assert r1.confirmed_cycles == 1
    assert r1.completed_subgoals == 1
    assert abs(r1.scr - (1.0 / 3.0)) < 1e-4

    # Cycle 2: 2 steps lift, 2 steps return
    evaluator.evaluate_step(5, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(6, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(7, is_on_support=True, is_lifted=False)
    r2 = evaluator.evaluate_step(8, is_on_support=True, is_lifted=False)
    assert r2.confirmed_cycles == 2
    assert r2.completed_subgoals == 2
    assert abs(r2.scr - (2.0 / 3.0)) < 1e-4

    # Cycle 3: 2 steps lift, 2 steps return -> Full success!
    evaluator.evaluate_step(9, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(10, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(11, is_on_support=True, is_lifted=False)
    r3 = evaluator.evaluate_step(12, is_on_support=True, is_lifted=False)
    assert r3.full_success
    assert r3.scr == 1.0
    assert not r3.overshot
    assert r3.confirmed_cycles == 3

    # Extra lift with dwell causes overshoot!
    evaluator.evaluate_step(13, is_on_support=False, is_lifted=True)
    r_over = evaluator.evaluate_step(14, is_on_support=False, is_lifted=True)
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
