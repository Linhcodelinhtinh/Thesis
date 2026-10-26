"""Lazy HDF5 loading: keep two cameras, causal action chunks and episode splits."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.memory_data import action_window, drop_memory


class MemoryDataset(Dataset):
    def __init__(self, root, split, horizon=50, condition="memory", dropout=0.0, seed=42):
        if split not in {"train", "val", "test"} or condition not in {"memory", "no_memory"}:
            raise ValueError("Invalid split/condition")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        self.index = [(j, t) for j, e in enumerate(self.episodes) for t in range(e["length"])]
        if not self.index:
            raise ValueError(f"Empty {split} split")
        self.horizon, self.condition, self.dropout, self.seed = horizon, condition, dropout, seed
        self.epoch = 0
        self._files = {}

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}  # no HDF5 handles shared with spawned workers
        return state

    def __getitem__(self, index):
        episode_index, t = self.index[index]
        episode = self.episodes[episode_index]
        filename = episode["shard"]
        if filename not in self._files:
            self._files[filename] = h5py.File(self.root / filename, "r")
        data = self._files[filename][episode["demo"]]
        prompt = data["prompts"].asstr()[t]
        sample_id = f'{episode["task"]}/{episode["demo"]}/{t}'
        if self.condition == "no_memory" or drop_memory(self.seed, self.epoch, sample_id, self.dropout):
            prompt = data.attrs["instruction"]
        # Slice first, avoiding loading the complete trajectory for every sample.
        segment = data["actions"][t:min(t + self.horizon, episode["length"])]
        actions, padding = action_window(segment, 0, self.horizon)
        return {"agentview": data["agentview_rgb"][t], "wrist": data["eye_in_hand_rgb"][t],
                "robot": data["robot_states"][t], "task": prompt,
                "action": torch.from_numpy(actions), "actions_id_pad": torch.from_numpy(padding),
                "action_is_pad": torch.from_numpy(padding)}


def collate_memory(samples):
    # Use the SAME resize implementation as the evaluated adapter, then the
    # official LiberoProcessorStep. Never rotate a previously rotated frame.
    from src.models.smolvla.adapter import _resize_rgb_image
    from lerobot.processor.env_processor import LiberoProcessorStep
    robot = torch.from_numpy(np.stack([s["robot"] for s in samples])).float()
    raw = {"observation.robot_state": {"gripper": {"qpos": robot[:, :2]},
                                        "eef": {"pos": robot[:, 2:5], "quat": robot[:, 5:9]}}}
    for source, key in (("agentview", "observation.images.image"), ("wrist", "observation.images.image2")):
        raw[key] = torch.from_numpy(np.stack([_resize_rgb_image(s[source]) for s in samples])).permute(0, 3, 1, 2)
    batch = LiberoProcessorStep()._process_observation(raw)
    pad_tensor = torch.stack([s["actions_id_pad"] for s in samples])
    batch.update(task=[s["task"] for s in samples], action=torch.stack([s["action"] for s in samples]),
                 actions_id_pad=pad_tensor, action_is_pad=pad_tensor)
    return batch
