"""Synthetic unit fixtures only; these are not benchmark demonstrations."""
import numpy as np
import pytest

from src.training.memory_data import action_window, annotate_episode, assign_splits, drop_memory, validate_episode


class TokenizerFixture:
    def encode(self, text, add_special_tokens=True):
        return text.split()


def trajectory():
    robot = np.zeros((9, 9), dtype=np.float32)
    robot[:, :2] = [0.025, -0.025]
    robot[:, 4] = 0.8
    robot[:, 8] = 1
    actions = np.zeros((9, 7), dtype=np.float32)
    actions[:, -1] = 1
    return robot, actions


def test_causal_labels_do_not_see_future_observations_or_actions():
    robot, actions = trajectory()
    first = annotate_episode("put the butter in the basket", robot, actions, TokenizerFixture(), "a")
    robot[5:, :2] = 0
    actions[5:, -1] = -1
    second = annotate_episode("put the butter in the basket", robot, actions, TokenizerFixture(), "a")
    assert first[:5] == second[:5]
    assert first[0]["has_memory"] is False
    assert any(r["has_memory"] for r in first)
    assert all(r["action_start_index"] == r["observation_index"] + 1 for r in first)
    assert all(r["max_observed_source_index"] == r["observation_index"] for r in first)


def test_multi_object_writer_does_not_fabricate_identity_for_training():
    robot, actions = trajectory()
    records = annotate_episode("put both the cream cheese and butter in the basket", robot, actions, TokenizerFixture(), "a")
    assert not any(r["has_memory"] for r in records)


def test_episode_tail_is_masked_and_never_reaches_next_episode():
    actions = np.arange(28, dtype=np.float32).reshape(4, 7)
    chunk, padding = action_window(actions, 2, 5)
    np.testing.assert_array_equal(chunk[:2], actions[2:])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[-1:], 3, axis=0))
    assert padding.tolist() == [False, False, True, True, True]


def test_grouped_split_reproducible_disjoint_and_has_all_three_sets():
    groups = {str(i): str(i // 2) for i in range(20)}
    result = assign_splits(groups)
    assert result == assign_splits(dict(reversed(list(groups.items()))))
    assert set(result.values()) == {"train", "val", "test"}
    assert all(result[str(i)] == result[str(i + 1)] for i in range(0, 20, 2))
    with pytest.raises(ValueError):
        assign_splits({"a": "one", "b": "one"})


def test_dropout_is_reproducible_and_endpoints_work():
    assert not drop_memory(42, 1, "episode/frame", 0)
    assert drop_memory(42, 1, "episode/frame", 1)
    a = [drop_memory(42, 1, str(i), 0.2) for i in range(100)]
    assert a == [drop_memory(42, 1, str(i), 0.2) for i in range(100)]
    assert 5 < sum(a) < 40


def test_contract_rejects_wrong_robot_state_order():
    robot, actions = trajectory()
    source = {"actions": actions, "robot_states": robot,
              "obs/gripper_states": robot[:, :2], "obs/ee_pos": robot[:, 2:5],
              "obs/agentview_rgb": np.zeros((9, 8, 8, 3), dtype=np.uint8),
              "obs/eye_in_hand_rgb": np.zeros((9, 8, 8, 3), dtype=np.uint8)}
    validate_episode(source)
    source["obs/ee_pos"] = np.ones((9, 3))
    with pytest.raises(AssertionError):
        validate_episode(source)


def test_over_budget_instruction_fails_instead_of_truncating():
    robot, actions = trajectory()
    with pytest.raises(ValueError, match="instruction exceeds"):
        annotate_episode("word " * 60, robot, actions, TokenizerFixture(), "a")


def test_export_relocation_changes_paths_only_and_detects_corruption(tmp_path):
    import json
    from scripts.memory_sft_assets import relocate
    from src.training.memory_data import sha256, write_json
    # File-integrity fixture only, never loaded as a policy.
    immutable = tmp_path / "normalizer.safetensors"
    immutable.write_bytes(b"unit-test-normalizer-bytes")
    (tmp_path / "backbone").mkdir()
    write_json(tmp_path / "config.json", {"vlm_model_name": "/old/location", "load_vlm_weights": False, "chunk_size": 50})
    write_json(tmp_path / "policy_preprocessor.json", {"steps": [
        {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "/old/location", "max_length": 48}}]})
    write_json(tmp_path / "export_manifest.json", {"immutable_files": {immutable.name: sha256(immutable)}})
    relocate(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    processor = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    assert config["vlm_model_name"] == str(tmp_path / "backbone")
    assert config["chunk_size"] == 50
    assert processor["steps"][0]["config"]["max_length"] == 48
    assert immutable.read_bytes() == b"unit-test-normalizer-bytes"
    immutable.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        relocate(tmp_path)


def test_export_keeps_lerobot_choice_discriminator(tmp_path):
    import json
    from src.training.export_utils import preserve_policy_type
    from src.training.memory_data import write_json
    write_json(tmp_path / "config.json", {"chunk_size": 50})
    preserve_policy_type(tmp_path, "smolvla")
    assert json.loads((tmp_path / "config.json").read_text()) == {"chunk_size": 50, "type": "smolvla"}
    with pytest.raises(ValueError, match="type does not match"):
        preserve_policy_type(tmp_path, "another_policy")
