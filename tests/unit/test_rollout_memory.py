"""Unit tests for V2 Memory Rollout Integration (P4, P5).

Verifies:
- memory_condition="off" runs without memory overhead and preserves exact instructions.
- memory_condition="text_shadow" logs memory trace while passing unmodified instruction to policy.
- memory_condition="text_only" injects rendered memory into policy prompt when allowed.
- Shadow vs Off invariance: Mock policy receives identical prompts in "off" and "text_shadow".
"""

from typing import Any, Dict, List
import numpy as np
import pytest

from src.evaluation.rollout import rollout_episode, EpisodeResult
from src.models.base import VLAPolicy


class PromptTrackingMockPolicy(VLAPolicy):
    """Mock policy that tracks all instruction prompts passed to predict_action_chunk."""

    def __init__(self, chunk_size: int = 10, action_dim: int = 7) -> None:
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self.prompts_received: List[str] = []

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        pass

    def preprocess(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        return obs

    def predict_action_chunk(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        self.prompts_received.append(instruction)
        return np.ones((self.chunk_size, self.action_dim), dtype=np.float32) * 0.1

    def postprocess(self, output: Any) -> np.ndarray:
        return np.asarray(output, dtype=np.float32)

    def validate_interface(self) -> Dict[str, Any]:
        return {"mock": True}

    @property
    def observation_spec(self) -> Dict[str, Any]:
        return {}


class MockMemoryEnv:
    """Mock environment with object state metadata for memory extraction."""

    def __init__(self, max_steps: int = 25, completed_subtask: bool = False) -> None:
        self.step_count = 0
        self.max_steps = max_steps
        self.completed_subtask = completed_subtask
        self.language_instruction = "pick up the black bowl and place it on the plate"

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
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
        return self.step_count >= 20


def test_rollout_memory_off():
    env = MockMemoryEnv(max_steps=15)
    policy = PromptTrackingMockPolicy(chunk_size=5)
    instruction = "pick up the black bowl and place it on the plate"

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction=instruction,
        execution_horizon=5,
        max_steps=15,
        task_name="pick_up_black_bowl",
        memory_condition="off",
    )

    assert result.memory is None
    # All replan queries received exact instruction
    assert len(policy.prompts_received) == 3
    for p in policy.prompts_received:
        assert p == instruction


def test_rollout_memory_shadow_invariance():
    """Verify that in text_shadow mode, policy receives identical prompt to OFF."""
    env = MockMemoryEnv(max_steps=15)
    policy_off = PromptTrackingMockPolicy(chunk_size=5)
    policy_shadow = PromptTrackingMockPolicy(chunk_size=5)
    instruction = "pick up the black bowl and place it on the plate"

    res_off = rollout_episode(
        env=env,
        policy=policy_off,
        instruction=instruction,
        execution_horizon=5,
        max_steps=15,
        task_name="pick_up_black_bowl",
        memory_condition="off",
    )

    res_shadow = rollout_episode(
        env=env,
        policy=policy_shadow,
        instruction=instruction,
        execution_horizon=5,
        max_steps=15,
        task_name="pick_up_black_bowl",
        memory_condition="text_shadow",
        allow_oracle_memory=True,
    )

    # Invariance check: Policy inputs MUST be bitwise identical
    assert policy_off.prompts_received == policy_shadow.prompts_received
    np.testing.assert_array_equal(res_off.actions, res_shadow.actions)

    # But shadow mode must have recorded memory telemetry
    assert res_shadow.memory is not None
    assert res_shadow.memory["condition"] == "text_shadow"
    assert len(res_shadow.memory["trace"]) == 3
    assert res_shadow.memory["total_objects"] > 0
    # Under grounded contract: initial decision has no historical progress,
    # so rendered_context is None and policy prompt matches original instruction
    first_trace = res_shadow.memory["trace"][0]
    assert first_trace["rendered_context"] is None
    assert first_trace["tokenizer_truncation_detected"] is False


def test_rollout_memory_text_only_prompt_injection():
    """Verify that text_only mode injects rendered memory into policy prompt when subtask is completed."""
    env = MockMemoryEnv(max_steps=15, completed_subtask=True)
    policy = PromptTrackingMockPolicy(chunk_size=5)
    instruction = "pick up the black bowl and place it on the plate"

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction=instruction,
        execution_horizon=5,
        max_steps=15,
        task_name="pick_up_black_bowl",
        memory_condition="text_only",
        allow_oracle_memory=True,
    )

    assert result.memory is not None
    assert result.memory["condition"] == "text_only"
    assert len(policy.prompts_received) == 3

    # Policy prompt contains concise memory block without leaking verbose metadata
    for p in policy.prompts_received:
        assert p.startswith(instruction)
        assert "Memory:" in p
        assert "black bowl" in p.lower()
        assert "source=oracle_simulator" not in p
        assert "[status=" not in p

    # Verify audit telemetry fields in trace
    first_trace = result.memory["trace"][0]
    assert first_trace["tokenizer_truncation_detected"] is False
    assert len(first_trace["facts_included"]) > 0


def test_rollout_memory_text_static_suffix():
    """Verify that text_static_suffix injects static initial detections at every step."""
    env = MockMemoryEnv(max_steps=15)
    policy = PromptTrackingMockPolicy(chunk_size=5)
    instruction = "put both the alphabet soup and the tomato sauce in the basket"

    result = rollout_episode(
        env=env,
        policy=policy,
        instruction=instruction,
        execution_horizon=5,
        max_steps=15,
        task_name="put_both_alphabet_soup_and_tomato_sauce_in_basket",
        memory_condition="text_static_suffix",
        target_object_name=None,
        goal_container_name="basket",
    )

    assert result.memory is not None
    assert result.memory["condition"] == "text_static_suffix"
    assert len(policy.prompts_received) == 3

    expected_suffix = "Memory: Alphabet soup is detected. Basket is target destination. Tomato sauce is detected."
    for p in policy.prompts_received:
        assert p.startswith(instruction)
        assert expected_suffix in p

    first_trace = result.memory["trace"][0]
    assert first_trace["memory_condition"] == "text_static_suffix"
    assert "Alphabet soup is detected." in first_trace["facts_included"]
    assert "Basket is target destination." in first_trace["facts_included"]
    assert "Tomato sauce is detected." in first_trace["facts_included"]


def test_rollout_memory_invalid_condition_fails_fast():
    env = MockMemoryEnv(max_steps=5)
    policy = PromptTrackingMockPolicy()

    with pytest.raises(ValueError, match="Invalid memory_condition"):
        rollout_episode(
            env=env,
            policy=policy,
            instruction="task",
            memory_condition="invalid_mode",
        )


def test_rollout_memory_fails_when_tokenizer_missing_on_non_mock_policy():
    """A real policy without an official tokenizer must fail loudly per AGENTS.md."""
    class FakeRealPolicy(PromptTrackingMockPolicy):
        def validate_interface(self) -> Dict[str, Any]:
            return {"mock": False}

    env = MockMemoryEnv(max_steps=5)
    policy = FakeRealPolicy()

    with pytest.raises(RuntimeError, match="Could not resolve official tokenizer"):
        rollout_episode(
            env=env,
            policy=policy,
            instruction="task",
            memory_condition="text_only",
        )


def test_rollout_memory_resolves_input_tokenizer_from_pipeline():
    """Verify rollout successfully resolves input_tokenizer and max_length from preprocessor step."""
    class MockTokenizerStep:
        def __init__(self):
            class DummyTokenizer:
                def encode(self, text, add_special_tokens=True):
                    return [1] * len(text.split())
            self.input_tokenizer = DummyTokenizer()
            self.tokenizer = None  # None like LeRobot TokenizerProcessorStep
            self.max_length = 48

    class MockPreprocessor:
        def __init__(self):
            self.steps = [MockTokenizerStep()]

    class MockPolicyWithPreprocessor(PromptTrackingMockPolicy):
        def __init__(self):
            super().__init__()
            self.preprocessor = MockPreprocessor()
        def validate_interface(self) -> Dict[str, Any]:
            return {"mock": False}

    env = MockMemoryEnv(max_steps=10, completed_subtask=True)
    policy = MockPolicyWithPreprocessor()

    res = rollout_episode(
        env=env,
        policy=policy,
        instruction="pick up bowl",
        execution_horizon=5,
        max_steps=10,
        task_name="pick_up_bowl",
        memory_condition="text_only",
        allow_oracle_memory=True,
    )
    assert res.memory is not None
    assert len(policy.prompts_received) > 0
    assert "Memory:" in policy.prompts_received[0]

