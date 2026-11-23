"""X-VLA domain handler that reads the SmolVLA V2 shards directly.

Same sampling as upstream LiberoHandler (X-VLA 6c4ecfb, datasets/domain_handler/simulations.py):
  * abs_action_6d [T,10], gripper binarised with > 0 (1 = close)
  * images dropped by one frame (image row idx+1 is paired with proprio abs[idx]
    and targets abs[idx+1 : idx+31]); valid because shard row i is the
    observation right BEFORE shard action i
  * freq 30, qdur 1.0, index_candidates = range(0, T-10), static-segment skip

Differences, all deliberate:
  * images come from the V2 shard (canonical raw OpenGL); agentview is flipped on
    both axes here, wrist stays raw, matching the X-VLA LIBERO client
    (_flip_agentview on agentview only)
  * prompt is per frame: prompt_mode="memory" uses shard prompts[idx+1]
    (the causal prompt rendered at the observation that is shown); "instruction"
    uses the episode instruction (F0, and base in both arms)
  * extra keys `is_mem` and `has_memory` (popped by train_xvla_mem.py before the forward)
"""
from __future__ import annotations

import os
import random
from typing import Iterable

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d

HANDLER_NAME = "libero_v2_shard"
LIBERO_DOMAIN_ID = 3


def _decode(x):
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def make_handler_cls(DomainHandler):
    class LiberoV2ShardHandler(DomainHandler):
        dataset_name = HANDLER_NAME
        freq, qdur = 30.0, 1.0

        def __init__(self, meta: dict, num_views: int) -> None:
            super().__init__(meta, num_views)
            self.mode = meta.get("prompt_mode", "instruction")
            if self.mode not in ("instruction", "memory"):
                raise ValueError(f"prompt_mode must be instruction|memory, got {self.mode}")

        def _abs(self, ep):
            with h5py.File(self.meta["actions_h5"], "r") as fa:
                return fa[ep["shard"]][ep["demo"]]["abs_action_6d"][()]

        def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool, image_aug,
                         lang_aug_map=None, **kwargs) -> Iterable[dict]:
            ep = self.meta["datalist"][traj_idx]
            a = self._abs(ep)
            n = a.shape[0]
            left = np.concatenate([a[:, :9], (a[:, 9:] > 0.0)], axis=-1).astype(np.float64)
            right = np.zeros_like(left)
            t = np.arange(n, dtype=np.float64) / self.freq
            L = interp1d(t, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
            idxs = [i for i in range(0, max(0, n - 10)) if i + 1 < n]
            if training:
                random.shuffle(idxs)
            is_mem = torch.tensor(ep["domain"] == "mem")
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:2] = True
            path = os.path.join(self.meta["shards_root"], ep["shard"])
            with h5py.File(path, "r") as f:
                g = f[ep["demo"]]
                av, wr = g["agentview_rgb"], g["eye_in_hand_rgb"]
                if av.shape[0] != n or wr.shape[0] != n:
                    raise ValueError(f"{path}:{ep['demo']} images {av.shape[0]} != actions {n}")
                instruction = _decode(g.attrs["instruction"])
                prompts = [_decode(p) for p in g["prompts"][()]] if self.mode == "memory" else None
                if self.mode == "memory" and (overlay_path := self.meta.get("memory_overlay")):
                    with h5py.File(overlay_path, "r") as overlay:
                        labels = overlay[ep["shard"]][ep["demo"]]
                        prompts = labels["prompts"].asstr()[()]
                        eligible = labels["eligible"][()]
                    if len(prompts) != n or eligible.shape != (n,):
                        raise ValueError(f"Memory overlay length mismatch: {path}:{ep['demo']}")
                    idxs = [idx for idx in idxs if eligible[idx + 1]]
                for idx in idxs:
                    cur = t[idx]
                    q = np.linspace(cur, min(cur + self.qdur, float(t.max())), num_actions + 1)
                    lseq = torch.tensor(L(q))
                    rseq = torch.zeros_like(lseq)
                    if (lseq[1] - lseq[0]).abs().max() < 1e-5:
                        continue
                    ins = prompts[idx + 1] if prompts is not None else instruction
                    if training and lang_aug_map and ins in lang_aug_map:
                        ins = random.choice(lang_aug_map[ins])
                    main = np.ascontiguousarray(av[idx + 1][::-1, ::-1])  # 180 deg, = _flip_agentview
                    wrist = np.ascontiguousarray(wr[idx + 1])
                    imgs = [image_aug(Image.fromarray(main)), image_aug(Image.fromarray(wrist))]
                    while len(imgs) < self.num_views:
                        imgs.append(torch.zeros_like(imgs[0]))
                    yield {
                        "language_instruction": ins,
                        "image_input": torch.stack(imgs, 0),
                        "image_mask": image_mask,
                        "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
                        "is_mem": is_mem,
                        "has_memory": torch.tensor(ins != instruction),
                    }

    return LiberoV2ShardHandler


def register(xvla_root: str | None = None):
    """Register the handler + domain id inside the upstream X-VLA package (no file edits)."""
    import sys
    if xvla_root and xvla_root not in sys.path:
        sys.path.insert(0, xvla_root)
    from datasets.domain_handler import registry
    from datasets.domain_handler.base import DomainHandler
    from datasets import domain_config
    registry._REGISTRY[HANDLER_NAME] = make_handler_cls(DomainHandler)
    domain_config.DATA_DOMAIN_ID[HANDLER_NAME] = LIBERO_DOMAIN_ID
    return registry._REGISTRY[HANDLER_NAME]
