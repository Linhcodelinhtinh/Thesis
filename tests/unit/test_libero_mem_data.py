"""Unit tests for LiberoMemDataProcessor."""

import numpy as np

from src.training.libero_mem_data import LiberoMemDataProcessor, MemoryStream


def test_data_processor_offset_and_chunking():
    processor = LiberoMemDataProcessor(
        chunk_size=10,
        action_dim=7,
        obs_action_offset=0,
    )

    n_steps = 15
    observations = [
        {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 0.85], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        for _ in range(n_steps)
    ]
    actions = np.ones((n_steps, 7), dtype=np.float32)

    samples = processor.process_episode(
        episode_id="ep_01",
        instruction="lift bowl",
        observations=observations,
        actions=actions,
        stream=MemoryStream.NO_MEMORY,
    )

    assert len(samples) == n_steps
    for s in samples:
        assert s.action_chunk.shape == (10, 7)
        assert s.action_mask.shape == (10,)
        assert not s.has_memory
        assert s.rendered_prompt == "lift bowl"

    # Terminal samples have padded action masks
    last_sample = samples[-1]
    assert last_sample.action_mask[0] is True or last_sample.action_mask[0] == 1
    assert last_sample.action_mask[1:].sum() == 0  # Only 1 action remaining


def test_data_processor_predicted_memory_stream():
    processor = LiberoMemDataProcessor(
        chunk_size=5,
        action_dim=7,
        obs_action_offset=0,
    )

    observations = []
    actions = np.zeros((10, 7), dtype=np.float32)
    for step in range(10):
        z = 1.2 if 2 <= step <= 6 else 0.85
        obs = {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([-0.05, 0.20, z], dtype=np.float32),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.01, 0.01] if 2 <= step <= 6 else [0.04, 0.04], dtype=np.float32),
            "detected_objects": {"bowl_A": np.array([-0.05, 0.20, z], dtype=np.float32)},
        }
        observations.append(obs)

    samples = processor.process_episode(
        episode_id="ep_02",
        instruction="lift bowl",
        observations=observations,
        actions=actions,
        stream=MemoryStream.PREDICTED_MEMORY,
    )

    assert len(samples) == 10
    # Steps after return should have confirmed cycle memory
    late_sample = samples[-1]
    assert "Memory:" in late_sample.rendered_prompt
    assert late_sample.has_memory
