"""Matched memory/no-memory samples and causal observation/action alignment."""
import numpy as np
import pytest
from src.training.libero_mem_data import LiberoMemDataProcessor, MemoryStream
from test_rgb_cycle_writer import frame


def observations(n=16):
    return [frame(step, lift=.15 if 4 <= step < 8 else 0., gap=.005 if 4 <= step < 8 else .08) for step in range(n)]


def test_data_processor_offset_and_chunking():
    processor = LiberoMemDataProcessor(chunk_size=10)
    obs = observations()
    actions = np.arange(16*7, dtype=np.float32).reshape(16, 7) / 200
    samples = processor.process_episode("ep_01", "lift bowl", obs, actions, MemoryStream.NO_MEMORY)
    assert len(samples) == 16
    for t, sample in enumerate(samples):
        assert sample.rendered_prompt == "lift bowl" and not sample.has_memory
        assert sample.source_action_start_idx == t
        np.testing.assert_equal(sample.action_chunk[0], actions[t])
        assert sample.agentview_image is obs[t]["agentview_image"]
    assert samples[-1].action_mask.sum() == 1


def test_data_processor_predicted_memory_stream_matches_control():
    processor = LiberoMemDataProcessor(chunk_size=5)
    obs = observations()
    actions = np.zeros((len(obs), 7), dtype=np.float32)
    predicted = processor.process_episode("ep", "lift bowl", obs, actions)
    control = processor.process_episode("ep", "lift bowl", obs, actions, MemoryStream.NO_MEMORY)
    assert "1 returns" in predicted[-1].rendered_prompt
    for p, c in zip(predicted, control):
        np.testing.assert_equal(p.robot_states, c.robot_states)
        np.testing.assert_equal(p.action_chunk, c.action_chunk)
        assert p.agentview_image is c.agentview_image
        assert p.wrist_image is c.wrist_image


def test_missing_camera_is_not_fabricated():
    obs = observations()
    obs[0].pop("robot0_eye_in_hand_image")
    with pytest.raises(ValueError, match="missing required"):
        LiberoMemDataProcessor().process_episode("ep", "lift bowl", obs, np.zeros((len(obs), 7)), MemoryStream.NO_MEMORY)


def test_future_action_targets_do_not_change_memory():
    obs = observations()
    actions = np.zeros((len(obs), 7), dtype=np.float32)
    changed = actions.copy()
    changed[8:] = .75
    processor = LiberoMemDataProcessor()
    a = processor.process_episode("ep", "lift bowl", obs, actions)
    b = processor.process_episode("ep", "lift bowl", obs, changed)
    assert [s.rendered_prompt for s in a[:9]] == [s.rendered_prompt for s in b[:9]]
    assert not np.array_equal(a[8].action_chunk, b[8].action_chunk)
