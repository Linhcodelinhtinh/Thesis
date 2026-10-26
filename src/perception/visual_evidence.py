"""Typed, frame-bound RGB evidence. No simulator predicates or annotations here."""
from dataclasses import dataclass
import hashlib
from typing import Mapping, Tuple

import numpy as np

from src.memory.models import EvidenceSource

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def unambiguous_bowl(detections, margin=.12, ratio=1.5):
    """Single-bowl profile: reject close competing hypotheses, not every weak proposal.

    Scores are uncalibrated detector scores. This rule does not convert a score
    into a probability or establish identity in a multi-bowl scene.
    """
    ordered = sorted(detections, key=lambda d: -d.score)
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) > 1 and ordered[0].score - ordered[1].score >= margin and ordered[0].score >= ratio * ordered[1].score:
        return ordered[0]
    return None


def rgb_views(observation, convention):
    if convention not in ("libero_mem_refined", "libero_raw"):
        raise ValueError("Explicit image convention required: libero_mem_refined or libero_raw")
    views = {}
    for camera in CAMERAS:
        if camera not in observation:
            raise ValueError(f"Missing RGB camera: {camera}")
        image = np.asarray(observation[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{camera} must be H x W x 3 RGB uint8")
        # Author get_visual: horizontal flip, then rotate 180 = vertical flip.
        views[camera] = np.ascontiguousarray(image[::-1] if convention == "libero_raw" else image)
    return views


def image_digest(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


@dataclass(frozen=True)
class Detection2D:
    label: str
    xyxy: Tuple[float, float, float, float]  # normalized image coordinates, NOT XYZ
    score: float
    evidence_id: str

    def __post_init__(self):
        box = tuple(float(x) for x in self.xyxy)
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError("Detection box must contain four finite coordinates")
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("Detection box must be normalized xyxy with positive area")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Invalid detector score")
        if not self.evidence_id or not self.label:
            raise ValueError("Detection requires label and evidence_id")
        object.__setattr__(self, "xyxy", box)

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def size(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([x2 - x1, y2 - y1])

    def to_dict(self):
        return dict(label=self.label, xyxy=list(self.xyxy), score=self.score, evidence_id=self.evidence_id)


@dataclass(frozen=True)
class VisualEvidence:
    step: int
    views: Mapping[str, Tuple[Detection2D, ...]]
    image_hashes: Mapping[str, str]
    model_id: str
    revision: str
    source: EvidenceSource = EvidenceSource.PREDICTED_RGB_TRACKER
    metadata: Mapping = None

    def __post_init__(self):
        from types import MappingProxyType
        if self.step < 0 or not self.model_id or not self.revision:
            raise ValueError("Evidence requires non-negative step and model provenance")
        object.__setattr__(self, "views", MappingProxyType({k: tuple(v) for k, v in self.views.items()}))
        object.__setattr__(self, "image_hashes", MappingProxyType(dict(self.image_hashes)))
        from src.memory.models import _freeze_value
        object.__setattr__(self, "metadata", _freeze_value(self.metadata or {}))

    def validate(self, step, images):
        if self.step != step:
            raise ValueError(f"Stale/future visual evidence: {self.step} != {step}")
        if self.source != EvidenceSource.PREDICTED_RGB_TRACKER:
            raise ValueError("Predicted writer rejects privileged or untyped evidence")
        if set(self.views) != set(images) or set(self.image_hashes) != set(images):
            raise ValueError("Visual evidence camera set does not match observation")
        for camera, image in images.items():
            if self.image_hashes[camera] != image_digest(image):
                raise ValueError(f"Visual evidence is from a different RGB frame: {camera}")
        ids = [d.evidence_id for view in self.views.values() for d in view]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate detection evidence IDs")

    def to_dict(self):
        from src.memory.models import _json_value
        return dict(step=self.step, model_id=self.model_id, revision=self.revision,
                    source=self.source.value, image_hashes=dict(self.image_hashes),
                    metadata=_json_value(self.metadata),
                    views={k: [d.to_dict() for d in v] for k, v in self.views.items()})

    @classmethod
    def from_dict(cls, value):
        return cls(step=value["step"], model_id=value["model_id"], revision=value["revision"],
                   source=EvidenceSource(value["source"]), image_hashes=value["image_hashes"],
                   metadata=value.get("metadata", {}),
                   views={k: tuple(Detection2D(**d) for d in v) for k, v in value["views"].items()})
