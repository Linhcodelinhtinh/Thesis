"""Opt-in tests against real author demonstrations (never synthetic benchmark runs).

Set MEMORY_SFT_DATA to a prepared libero_object smoke dataset. Requires the
audited model environment; no simulation or model weights are needed here.
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not os.environ.get("MEMORY_SFT_DATA"), reason="Set MEMORY_SFT_DATA for real-data checks")


def test_real_data_alignment_and_adapter_parity():
    import h5py
    import torch
    from src.training.memory_dataset import MemoryDataset, collate_memory
    from src.models.smolvla.adapter import SmolVLAAdapter
    data = MemoryDataset(os.environ["MEMORY_SFT_DATA"], "train")
    item = data[0]
    episode = data.episodes[0]
    source = ROOT / "resources/demonstrations/libero_object" / (episode["task"] + "_demo.hdf5")
    with h5py.File(source) as f:
        demo = f["data"][episode["demo"]]
        np.testing.assert_array_equal(item["agentview"], demo["obs/agentview_rgb"][0])
        np.testing.assert_allclose(item["action"][:50], demo["actions"][1:51].astype(np.float32))
    features = collate_memory([item])
    robot = item["robot"]
    reference = SmolVLAAdapter().build_raw_features(
        {"agentview_image": item["agentview"], "robot0_eye_in_hand_image": item["wrist"],
         "robot0_gripper_qpos": robot[:2], "robot0_eef_pos": robot[2:5], "robot0_eef_quat": robot[5:9]}, item["task"])
    for key in ("observation.state", "observation.images.image", "observation.images.image2"):
        torch.testing.assert_close(features[key][0], reference[key], rtol=0, atol=0)
    assert features["observation.state"].shape == (1, 8)
    assert features["action"].shape == (1, 50, 7)
    assert features["actions_id_pad"].dtype == torch.bool


def test_real_train_validation_test_groups_disjoint():
    from src.training.memory_dataset import MemoryDataset
    sets = {}
    for split in ("train", "val", "test"):
        data = MemoryDataset(os.environ["MEMORY_SFT_DATA"], split)
        sets[split] = {(e["task"], e["initial_state_sha256"]) for e in data.episodes}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
