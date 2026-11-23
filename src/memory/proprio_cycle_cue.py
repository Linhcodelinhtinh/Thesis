"""Proprioceptive cue for a completed bowl lift-and-return cycle (gripper aperture + end-effector position).

The cue needs no detector. It watches one closed-gripper phase: the grasp pose is the lowest end-effector height
in the first frames after the gripper closes, and a *touchdown* fires once, in that phase, when the end effector has
risen at least `lift_meters` above the grasp height and then comes back within `touchdown_height_meters` of it and
within `touchdown_xy_meters` of the grasp position. That is the proprioceptive signature of putting the bowl back
where it was picked up. It is a separate event stream: it never edits a writer, so a writer's prompt text is unchanged.

Limits: it cannot tell a held bowl from a closed, empty gripper, so a lift-and-lower of nothing also fires.
The default thresholds were set from the 198 development demonstrations (lift rise 0.082-0.199 m), not held out.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "memory" / "proprio_cycle_cue_v1.json"


@dataclass(frozen=True)
class ProprioCueConfig:
    closed_aperture_meters: float = .045
    released_aperture_meters: float = .055
    lift_meters: float = .08
    touchdown_height_meters: float = .03
    touchdown_xy_meters: float = .06
    grasp_height_frames: int = 8
    confirmation_frames: int = 2

    def __post_init__(self):
        if self.closed_aperture_meters >= self.released_aperture_meters:
            raise ValueError("The closed aperture must be below the released aperture")
        for name in ("lift_meters", "touchdown_height_meters", "touchdown_xy_meters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.touchdown_height_meters >= self.lift_meters:
            raise ValueError("The touchdown height must be smaller than the lift")
        if self.grasp_height_frames < 1 or self.confirmation_frames < 0:
            raise ValueError("Frame counts must be non-negative and the grasp window at least one frame")

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class ProprioCycleCue:
    """Streaming detector: call `step` once per frame; it returns the touchdown event of that frame or None."""

    def __init__(self, config=None):
        self.config = config or ProprioCueConfig.load()
        self.reset()

    def reset(self):
        self.closed_since = None
        self.grasp_height = self.grasp_xy = self.peak_height = None
        self.fired = False
        self.touchdowns = []

    def step(self, step, aperture, eef_pos):
        config, position = self.config, np.asarray(eef_pos, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(aperture):
            raise ValueError("Expected a finite end-effector position (3,) and a finite aperture")
        if self.closed_since is None:
            if aperture <= config.closed_aperture_meters:
                self.closed_since, self.fired = step, False
                self.grasp_height = self.peak_height = float(position[2])
                self.grasp_xy = position[:2].copy()
            return None
        if aperture >= config.released_aperture_meters:
            self.closed_since = None
            return None
        if step - self.closed_since < config.grasp_height_frames:
            self.grasp_height = min(self.grasp_height, float(position[2]))
        self.peak_height = max(self.peak_height, float(position[2]))
        if (not self.fired and self.peak_height - self.grasp_height >= config.lift_meters
                and position[2] <= self.grasp_height + config.touchdown_height_meters
                and np.linalg.norm(position[:2] - self.grasp_xy) <= config.touchdown_xy_meters):
            self.fired = True
            event = dict(type="touchdown", occurred_at=step, available_at=step + config.confirmation_frames,
                         rise=self.peak_height - self.grasp_height, grasp_started=self.closed_since)
            self.touchdowns.append(event)
            return event
        return None


def touchdown_steps(apertures, positions, config=None):
    """Batch form for replays: the `available_at` step of every touchdown on one episode."""
    cue = ProprioCycleCue(config)
    return [event["available_at"] for event in (cue.step(step, aperture, position)
                                                for step, (aperture, position) in enumerate(zip(apertures, positions)))
            if event is not None]


def fuse_returns(vision_returns, proprio_returns, gap=20):
    """Add a proprioceptive return only when the writer recorded none within `gap` frames of it."""
    fused = list(vision_returns)
    fused += [step for step in proprio_returns if all(abs(step - other) > gap for other in vision_returns)]
    return sorted(fused)
