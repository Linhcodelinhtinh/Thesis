"""Versioned geometric and temporal thresholds for the bowl-cycle writer."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/memory/relation_cycle_v3.json"


@dataclass(frozen=True)
class BowlCycleConfig:
    track_score: float = .15
    measure_score: float = .18
    max_missing_steps: int = 22
    pending_expiry_steps: int = 12
    return_offset_ratio: float = .35
    lift_center_rise_ratio: float = .30
    lift_bottom_rise_ratio: float = .40
    closed_aperture_meters: float = .045
    released_aperture_meters: float = .055
    stable_motion_ratio: float = .20
    confirmation_frames: int = 3
    association_max_distance: float = .12
    association_initial_distance: float = .07
    association_distance_growth: float = .025
    association_low_score_distance: float = .04
    association_tie_margin: float = .02
    association_min_size_ratio: float = .5
    association_max_size_ratio: float = 2.0
    detector_seed_score_margin: float = .12
    detector_seed_score_ratio: float = 1.5
    plate_duplicate_overlap_ratio: float = .90
    plate_association_distance: float = .035
    object_association_distance: float = .07
    minimum_instance_separation: float = .025
    support_horizontal_ratio: float = .45
    support_vertical_tolerance_ratio: float = .10
    wrist_center_tolerance: float = .25
    wrist_min_size: float = .25
    coherent_rise_ratio: float = .04
    return_descent_ratio: float = .08
    settled_offset_ratio: float = .35

    def __post_init__(self):
        numeric = ("track_score", "measure_score", "return_offset_ratio", "lift_center_rise_ratio",
                   "lift_bottom_rise_ratio", "closed_aperture_meters", "released_aperture_meters",
                   "stable_motion_ratio", "association_max_distance", "association_initial_distance",
                   "association_distance_growth", "association_low_score_distance",
                   "association_tie_margin", "association_min_size_ratio", "association_max_size_ratio",
                   "detector_seed_score_margin", "detector_seed_score_ratio",
                   "plate_duplicate_overlap_ratio", "plate_association_distance",
                   "object_association_distance", "minimum_instance_separation",
                   "support_horizontal_ratio", "support_vertical_tolerance_ratio",
                   "wrist_center_tolerance", "wrist_min_size", "coherent_rise_ratio",
                   "return_descent_ratio", "settled_offset_ratio")
        if any(getattr(self, name) < 0 for name in numeric):
            raise ValueError("Bowl-cycle thresholds must be nonnegative")
        if (self.max_missing_steps < 0 or self.pending_expiry_steps < 1
                or self.confirmation_frames < 2
                or self.track_score > 1 or self.measure_score > 1
                or self.detector_seed_score_margin > 1
                or self.plate_duplicate_overlap_ratio > 1
                or self.detector_seed_score_ratio < 1
                or self.association_min_size_ratio == 0
                or self.association_min_size_ratio > self.association_max_size_ratio
                or self.closed_aperture_meters >= self.released_aperture_meters):
            raise ValueError("Invalid bowl-cycle temporal, identity, or gripper thresholds")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        config = cls(**values)
        if any(config.to_dict()[key] != value for key, value in values.items()):
            raise ValueError("Bowl-cycle config does not match its validated schema")
        return config
