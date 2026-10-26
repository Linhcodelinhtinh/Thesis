"""Strict, bounded-memory loading of author-refined LIBERO-Mem demonstrations.

Metadata supplies instruction/initial-state grouping only. Segmentation, depth,
exo_boxes and subgoal labels never enter predicted-memory observations. No
quaternions, camera frames or oracle predicates are fabricated.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional
import h5py
import numpy as np

from src.training.memory_data import validate_episode

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "libero_mem"


@dataclass
class DemonstrationEpisode:
    episode_id: str
    task_name: str
    instruction: str
    actions: np.ndarray
    observations: list
    initial_state: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None
    success: bool = False
    split: str = "train"


def read_episode_metadata(path, task_name, demo_keys):
    """Stream one metadata episode at a time; never json.load the 713 MB file."""
    import ijson
    wanted, result = set(demo_keys), {}
    with Path(path).open("rb") as stream:
        for demo, value in ijson.kvitems(stream, task_name, use_float=True):
            if demo in wanted:
                result[demo] = {k: value[k] for k in ("initial_state", "task_description", "success") if k in value}
                if len(result) == len(wanted):
                    break
    if set(result) != wanted:
        raise ValueError(f"Missing episode metadata for {task_name}: {wanted - set(result)}")
    return result


class LiberoMemHDF5Parser:
    def __init__(self, data_dir=None, val_ratio=.1, seed=42):
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must lie between zero and one")
        self.val_ratio, self.seed = val_ratio, seed

    def resolve_hdf5_path(self, task_name):
        clean = task_name.removesuffix(".hdf5").removesuffix(".bddl").removesuffix("_demo")
        candidate = self.data_dir / f"{clean}_demo.hdf5"
        if not candidate.is_file():
            raise FileNotFoundError(f"Exact author demonstration file not found: {candidate}")
        return candidate

    def is_val_episode(self, episode_id, initial_state=None):
        if initial_state is None:
            raise ValueError("Initial state required for a split; episode-name splitting leaks T1/T3 states")
        # Canonical dtype/endian, independent of task ID so T1/T3 repeats stay together.
        state = np.asarray(initial_state, dtype="<f8")
        if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
            raise ValueError("Invalid initial state")
        key = hashlib.sha256(state.tobytes()).hexdigest()
        value = hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()
        return int(value[:16], 16) / 2**64 < self.val_ratio

    def iter_task(self, task_name, max_episodes=None, split_filter=None, demo_ids=None):
        path = self.resolve_hdf5_path(task_name)
        task = path.name.removesuffix("_demo.hdf5")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        if split_filter not in (None, "train", "val"):
            raise ValueError("Unknown split")
        with h5py.File(path, "r") as file:
            demos = sorted(file["data"], key=lambda key: int(key.removeprefix("demo_")))
            if demo_ids is not None:
                requested = {f"demo_{int(i)}" for i in demo_ids}
                if not requested <= set(demos):
                    raise ValueError(f"Missing requested demonstrations: {requested - set(demos)}")
                demos = [key for key in demos if key in requested]
            if max_episodes is not None:
                demos = demos[:max_episodes]
            metadata = read_episode_metadata(self.data_dir / "metainfo.json", task, demos)
            for demo in demos:
                meta = metadata[demo]
                if not meta.get("task_description") or "initial_state" not in meta:
                    raise ValueError(f"{task}/{demo} missing author instruction or initial state")
                initial = np.asarray(meta["initial_state"], dtype=np.float64)
                episode_id = f"{task}_{demo}"
                split = "val" if self.is_val_episode(episode_id, initial) else "train"
                if split_filter is not None and split_filter != split:
                    continue
                group = file[f"data/{demo}"]
                actions, robot, _ = validate_episode(group)
                rgb = group["obs/agentview_rgb"][:]
                wrist = group["obs/eye_in_hand_rgb"][:]
                observations = [dict(agentview_image=rgb[t], robot0_eye_in_hand_image=wrist[t],
                                     robot0_gripper_qpos=robot[t, :2], robot0_eef_pos=robot[t, 2:5],
                                     robot0_eef_quat=robot[t, 5:9]) for t in range(len(actions))]
                yield DemonstrationEpisode(episode_id, task, meta["task_description"], actions, observations,
                                           initial, group["rewards"][:] if "rewards" in group else None,
                                           group["dones"][:] if "dones" in group else None,
                                           bool(meta.get("success", False)), split)

    def parse_task(self, task_name, max_episodes=None, split_filter=None):
        """Compatibility list API; production preparation should use iter_task."""
        return list(self.iter_task(task_name, max_episodes, split_filter))
