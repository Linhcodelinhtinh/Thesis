"""Frame index, balanced sampler and shard reader for a termination ("task complete") head.

Kept separate from the X-VLA action pipeline on purpose: the action handler
(scripts/xvla_v2/shard_handler.py) is unchanged and still skips the last 10 frames and
static segments. The head is trained on its own sampler, which does include the idle tail,
so the action head never sees extra "stand still" targets.

Example kinds (per frame t of a Mem demo, from termination_labels.jsonl)
  positive        t >= done_from
  hard_negative   bowl resting on the plate but the task not done yet: frames after an
                  earlier milestone (T3 cycles 1 and 2) and the frames right before done_from
  negative        everything else (approach, grasp, carry)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

POSITIVE, HARD_NEGATIVE, NEGATIVE = "positive", "hard_negative", "negative"


@dataclass(frozen=True)
class FrameExample:
    task: str
    demo: str
    frame: int
    label: int
    kind: str
    prompt_returns: Optional[int]
    target: int


def load_labels(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def frame_examples(rows, hard_window=15, observable_only=False) -> List[FrameExample]:
    """Expand demo rows into labelled frames.

    observable_only: label a frame positive only if the causal prompt already shows the
    target count (done_observable_from). Frames that are done but not yet visible in memory
    are dropped rather than labelled negative, so the head is never trained to contradict
    the ground truth.
    """
    examples = []
    for row in rows:
        done_from = row["done_from"]
        returns = row.get("prompt_returns")
        if observable_only and returns is None:
            raise ValueError("observable_only requires labels built with shard prompts")
        observable_from = row.get("done_observable_from")
        hard = set()
        for milestone in row["milestones"]:
            if done_from is None or milestone < done_from:
                hard.update(range(milestone, milestone + hard_window))
        if done_from is not None:
            hard.update(range(max(0, done_from - hard_window), done_from))
        for t in range(row["frames"]):
            done = done_from is not None and t >= done_from
            if done and observable_only and (observable_from is None or t < observable_from):
                continue
            kind = POSITIVE if done else HARD_NEGATIVE if t in hard else NEGATIVE
            examples.append(FrameExample(row["task"], row["demo"], t, int(done), kind,
                                         None if returns is None else int(returns[t]), row["target"]))
    return examples


def balanced_indices(examples: Sequence[FrameExample], size, rng, positive_fraction=0.4,
                     hard_negative_fraction=0.3):
    """Sample with replacement: fixed shares of positives and hard negatives, rest negatives."""
    if not 0 < positive_fraction < 1 or not 0 <= hard_negative_fraction < 1 or \
            positive_fraction + hard_negative_fraction >= 1:
        raise ValueError("Invalid class fractions")
    pools = {kind: np.array([i for i, e in enumerate(examples) if e.kind == kind], dtype=np.int64)
             for kind in (POSITIVE, HARD_NEGATIVE, NEGATIVE)}
    if len(pools[POSITIVE]) == 0 or len(pools[NEGATIVE]) == 0:
        raise ValueError("Need both positive and negative frames")
    counts = {POSITIVE: int(round(size * positive_fraction)),
              HARD_NEGATIVE: int(round(size * hard_negative_fraction)) if len(pools[HARD_NEGATIVE]) else 0}
    counts[NEGATIVE] = size - counts[POSITIVE] - counts[HARD_NEGATIVE]
    chosen = np.concatenate([rng.choice(pools[k], counts[k], replace=True) for k in counts if counts[k]])
    rng.shuffle(chosen)
    return chosen


def split_by_demo(examples, val_fraction, rng):
    """Hold out whole demos (never frames of a training demo) for head validation."""
    demos = sorted({(e.task, e.demo) for e in examples})
    order = rng.permutation(len(demos))
    held = {demos[i] for i in order[:max(1, int(round(len(demos) * val_fraction)))]}
    train = [e for e in examples if (e.task, e.demo) not in held]
    val = [e for e in examples if (e.task, e.demo) in held]
    return train, val


class TerminationShardReader:
    """Read (agentview, wrist, prompt) at frame t with the same image convention as shard_handler.

    Agentview is rotated 180 degrees (= the X-VLA client _flip_agentview), wrist stays raw.
    Unlike the action handler there is no +1 frame offset: the label belongs to frame t, and
    so do the image and the causal prompt.
    """

    def __init__(self, shards_root):
        self.root = Path(shards_root)
        self._files = {}

    def _file(self, task):
        if task not in self._files:
            import h5py
            self._files[task] = h5py.File(self.root / f"mem__{task}.hdf5", "r")
        return self._files[task]

    def read(self, example: FrameExample):
        group = self._file(example.task)[example.demo]
        prompt = group["prompts"][example.frame]
        prompt = prompt.decode() if isinstance(prompt, bytes) else str(prompt)
        main = np.ascontiguousarray(group["agentview_rgb"][example.frame][::-1, ::-1])
        wrist = np.ascontiguousarray(group["eye_in_hand_rgb"][example.frame])
        return dict(agentview=main, wrist=wrist, prompt=prompt, label=example.label, kind=example.kind)

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files.clear()
