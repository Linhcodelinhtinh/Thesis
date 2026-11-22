"""Schedule and probe thresholds for the sparse-detector relation writer.

The values here decide only *when* the detector runs. Commit thresholds,
association rules and prompt rendering stay with the wrapped dense writer.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.utils.file_hash import sha256_file

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/memory/relation_sparse_v1.json"


@dataclass(frozen=True)
class SparseWriterConfig:
    # Detector runs on every K-th observation while the writer is at rest.
    keyframe_interval: int = 8
    # Keyframes sit at step % K == 0, matching the policy's query boundaries.
    align_keyframes: bool = True
    # The dense writer needs the first steps to ground the origin anchors.
    dense_prefix_steps: int = 20
    # Consecutive dense frames in the rest state before sparse mode resumes.
    rest_confirm_frames: int = 2
    # Process every frame while the gripper is not open (contact is where events occur).
    dense_while_engaged: bool = True
    engaged_aperture_meters: float = .065
    # Keyframe-versus-last-processed change probe (fractions of the earlier box size).
    probe_score: float = .15
    probe_motion_ratio: float = .08
    probe_size_ratio: float = .15
    probe_plate_motion_ratio: float = .08
    probe_wrist: bool = True
    # Unprocessed frames kept for dense replay. Must cover one keyframe gap.
    ring_buffer_frames: int = 16

    def __post_init__(self):
        if (type(self.keyframe_interval) is not int or self.keyframe_interval < 1
                or type(self.dense_prefix_steps) is not int or self.dense_prefix_steps < 0
                or type(self.rest_confirm_frames) is not int or self.rest_confirm_frames < 1
                or type(self.ring_buffer_frames) is not int
                or self.ring_buffer_frames < self.keyframe_interval):
            raise ValueError("Invalid sparse writer schedule")
        if any(type(getattr(self, name)) is not bool for name in
               ("align_keyframes", "dense_while_engaged", "probe_wrist")):
            raise ValueError("Sparse writer flags must be booleans")
        ratios = ("engaged_aperture_meters", "probe_score", "probe_motion_ratio",
                  "probe_size_ratio", "probe_plate_motion_ratio")
        if any(not getattr(self, name) > 0 for name in ratios) or self.probe_score > 1:
            raise ValueError("Sparse writer probe thresholds must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        config = cls(**values)
        if config.to_dict() != values:
            raise ValueError("Sparse writer config does not match its validated schema")
        return config

    @staticmethod
    def file_sha256(path=CONFIG_PATH):
        return sha256_file(path)
