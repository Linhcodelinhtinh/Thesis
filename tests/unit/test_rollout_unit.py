"""Unit tests for closed-loop rollout engine (Phase 6).

Fast, offline unit tests (< 1s) verifying receding-horizon chunking logic
across s in {1, 5, 10, 25, 50}, telemetry recording, video export, and
model_output.jsonl serialization per SRS.md Section 8.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from src.evaluation.rollout import rollout_episode, EpisodeResult, save_video
from src.models.base import VLAPolicy


class MockEnv:
    """Mock simulation environment for fast offline testing."""

    def __init__(self, success_at_step: int = 15, max_steps: int = 100) -> None:
        self.step_count = 0
        self.success_at_step = success_at_step
        self.max_steps = max_steps

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        }

    def reset(self, initial_state_id: int = 0) -> Dict[str, Any]:
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.step_count += 1
        obs = self._get_obs()
        reward = 1.0 if self.check_success() else 0.0
        done = self.step_count >= self.max_steps
        return obs, reward, done, {}

    def check_success(self) -> bool:
        return self.step_count >= self.success_at_step

    def render(self, camera_name: str = "agentview") -> np.ndarray:
        return np.full((128, 128, 3), fill_value=self.step_count % 256, dtype=np.uint8)


class MockPolicy(VLAPolicy):
    """Mock policy that tracks predict_action_chunk calls."""

    def __init__(self, chunk_size: int = 50, action_dim: int = 7) -> None:
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self.prediction_count = 0

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        pass

    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        return obs

    def predict_action_chunk(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        self.prediction_count += 1
        return np.ones((self.chunk_size, self.action_dim), dtype=np.float32) * self.prediction_count

    def postprocess(self, output: Any) -> np.ndarray:
        return np.asarray(output, dtype=np.float32)

    def validate_interface(self) -> Dict[str, Any]:
        return {"mock": True}

    @property
    def observation_spec(self) -> Dict[str, Any]:
        return {}


@pytest.mark.parametrize("s,expected_predictions", [
    (1, 15),   # replans every step: 15 steps -> 15 predictions
    (5, 3),    # replans every 5 steps: 15 steps -> 3 predictions
    (10, 2),   # replans at step 0 and 10: 15 steps -> 2 predictions
    (25, 1),   # s=25 > 15: exactly 1 prediction
    (50, 1),   # s=50 > 15: exactly 1 prediction
])
def test_receding_horizon_sweep(s, expected_predictions):
    """Verify that receding horizon s in {1, 5, 10, 25, 50} triggers exact replanning counts."""
    env = MockEnv(success_at_step=15, max_steps=50)
    policy = MockPolicy(chunk_size=50, action_dim=7)

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction="test instruction",
        execution_horizon=s,
        max_steps=50,
        task_name="mock_task",
    )

    assert result.success is True
    assert result.num_steps == 15
    assert policy.prediction_count == expected_predictions
    assert result.actions.shape == (15, 7)
    assert len(result.inference_latencies_ms) == expected_predictions
    assert len(result.simulation_latencies_ms) == 15
    assert len(result.model_outputs) == 15


def test_rollout_result_serialization(tmp_path: Path):
    """Verify EpisodeResult serialization, dictionary format, and JSONL export."""
    env = MockEnv(success_at_step=5, max_steps=10)
    policy = MockPolicy(chunk_size=50, action_dim=7)

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction="pick up the object",
        execution_horizon=10,
        max_steps=10,
        task_name="test_task",
    )

    d = result.to_dict()
    assert d["task_name"] == "test_task"
    assert d["instruction"] == "pick up the object"
    assert d["execution_horizon_s"] == 10
    assert d["num_steps"] == 5
    assert d["success"] is True
    assert "mean_inference_ms" in d
    assert "mean_simulation_ms" in d
    assert d["action_shape"] == [5, 7]

    # Verify JSONL model output export
    jsonl_path = tmp_path / "model_output.jsonl"
    result.save_model_outputs(jsonl_path)
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    first_step = eval(lines[0].replace("true", "True").replace("false", "False"))
    assert first_step["step"] == 0
    assert first_step["chunk_step"] == 0
    assert len(first_step["action"]) == 7


def test_rollout_video_recording(tmp_path: Path):
    """Verify that video frames are captured and saved when record_video is True."""
    env = MockEnv(success_at_step=5, max_steps=10)
    policy = MockPolicy(chunk_size=50, action_dim=7)
    video_target = tmp_path / "test_rollout.mp4"

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction="pick up the object",
        execution_horizon=5,
        max_steps=10,
        task_name="test_task",
        record_video=True,
        video_path=video_target,
    )

    assert len(result.video_frames) > 0
    assert video_target.exists()
    assert video_target.stat().st_size > 0


def test_rollout_invalid_horizon_fails_fast():
    """Verify execution_horizon < 1 raises ValueError."""
    env = MockEnv()
    policy = MockPolicy()
    with pytest.raises(ValueError, match="execution_horizon must be >= 1"):
        rollout_episode(env=env, policy=policy, instruction="test", execution_horizon=0)
