"""Smoke tests for closed-loop VLA rollout with simulator and GPU (Phase 6).

Marked with @pytest.mark.gpu and @pytest.mark.libero.
Only runs when explicitly targeting live hardware/simulation.
"""

import pytest
import torch

from src.evaluation.rollout import rollout_episode
from src.models.smolvla.adapter import SmolVLAAdapter
from src.simulator.libero_env import LiberoEnv


@pytest.mark.gpu
@pytest.mark.libero
def test_smolvla_closed_loop_smoke():
    """Execute a 10-step closed loop rollout in simulation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for live GPU smoke test.")

    # Initialize environment on task 0
    env = LiberoEnv(
        task_suite_name="libero_object",
        task_id=0,
        initial_state_id=0,
    )

    try:
        # Initialize SmolVLA adapter
        adapter = SmolVLAAdapter()
        # Verify 10-step receding horizon execution
        result = rollout_episode(
            env=env,
            policy=adapter,
            instruction=env.task_description,
            initial_state_id=0,
            execution_horizon=5,
            max_steps=10,
            task_name="smoke_test",
        )

        assert result.num_steps == 10
        assert result.actions.shape == (10, 7)
        assert len(result.simulation_latencies_ms) == 10
    finally:
        env.close()
