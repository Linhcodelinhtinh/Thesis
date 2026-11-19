"""Bowl/plate cycle recognition with bounded association through occlusion."""
import numpy as np

from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.models import IdentityStatus
from src.memory.relation_memory import RelationMemoryWriter, RelationReducer
from src.perception.grounding_dino import box_iou
from src.perception.relation_cycle_tracking import MAX_MISSING, BowlCycleTracker
from src.perception.relation_tracking import support_overlap


RETURN_OFFSET = .35
LIFT_CENTER_RISE = .30
LIFT_BOTTOM_RISE = .40


def _relative_position(bowl, plate):
    # Horizontal center plus the vertical center and lower edge of the bowl,
    # expressed relative to the plate on the same camera.
    x, y = bowl.center - plate.center
    return np.array([x, y, bowl.xyxy[3] - plate.xyxy[3]])


class BowlCycleReducer(RelationReducer):
    """Keep tentative manipulation separate from the existing committed events."""

    def __init__(self, config=None):
        self.config = config or BowlCycleConfig()
        super().__init__(count_cycles=True, confirmation_frames=self.config.confirmation_frames)
        self.pending = False
        self.missing = 0

    def measure(self, step, bowl, plate, aperture, wrist_near, allow_pending=True):
        closed = aperture <= self.config.closed_aperture_meters
        if allow_pending and self.ready and closed and wrist_near:
            self.pending = True
        if any(track.identity.identity_status == IdentityStatus.AMBIGUOUS for track in (bowl, plate)):
            return "conflicted", None, ()
        if any(track.box is None or track.anchor is None for track in (bowl, plate)):
            return "unknown", None, ()

        reference = plate.identity.object_id
        scale = bowl.anchor.size[[0, 1, 1]]
        pose = _relative_position(bowl.box, plate.box)
        offset = (pose - _relative_position(bowl.anchor, plate.anchor)) / scale
        near_support = (support_overlap(bowl.box, plate.box, self.config.support_horizontal_ratio,
                                        self.config.support_vertical_tolerance_ratio)
                        and np.linalg.norm(offset[[0, 2]]) <= self.config.return_offset_ratio)
        if self.departure is not None and near_support and box_iou(bowl.box.xyxy, plate.box.xyxy) >= .90:
            # A near-duplicate bowl/plate prediction cannot establish two
            # distinct objects at the point where a count would be committed.
            return "conflicted", None, ()
        gripper_id = f"gripper:{step}"
        previous = bowl.previous, plate.previous
        if near_support and aperture >= self.config.released_aperture_meters and all(previous):
            relative_motion = (pose - _relative_position(*previous)) / scale
            stable_plate = np.linalg.norm(
                (plate.box.center - plate.previous.center) / plate.anchor.size
            ) <= self.config.stable_motion_ratio
            if (stable_plate and
                    np.linalg.norm(relative_motion) <= self.config.stable_motion_ratio):
                return "relation", reference, RelationMemoryWriter._ids(bowl, plate) + (gripper_id,)

        if self.ready and self.origin == reference and closed:
            # Gross displacement starts an attempt even without the wrist cue.
            # Require the center and lower edge to rise together for a lift.
            if (self.pending and -offset[1] >= self.config.lift_center_rise_ratio
                    and -offset[2] >= self.config.lift_bottom_rise_ratio):
                ids = (bowl.box.evidence_id, plate.box.evidence_id,
                       bowl.anchor.evidence_id, plate.anchor.evidence_id, gripper_id)
                return "departure", reference, ids
            if allow_pending:
                self.pending |= -offset[1] >= self.config.stable_motion_ratio
        return "unsettled", None, ()

    def update(self, step, status, reference, evidence, emit):
        resting_origin = self.ready
        if status in ("unknown", "conflicted"):
            self.missing += 1
            if self.missing <= self.config.max_missing_steps:
                self.status, self.reference = status, None
                self.complete = False
                return
            status = "gap"
        else:
            self.missing = 0
        if status == "gap":
            self.pending = False
            # Detection loss expires the attempt. A previously observed rest
            # can still ground a fresh lift after reacquisition. A skipped
            # input step invalidates even that continuity assumption.
        super().update(step, status, reference, evidence, emit)
        if status == "gap" and emit is not None:
            self.ready = resting_origin
        if status == "relation" and self.status == "confirmed":
            self.pending = False


class BowlCycleWriter(RelationMemoryWriter):
    """Opt-in successor for the active single-bowl T1/T3 profiles."""

    VERSION = "rgb_relations_v2"

    def __init__(self, episode_id, profile, detector, image_convention, config=None):
        if (profile.subject, profile.subject_count, profile.reference, profile.predicate, profile.count_cycles) != (
                "bowl", 1, "plate", "on", True):
            raise ValueError(f"{self.VERSION} requires the single-bowl return profile")
        super().__init__(episode_id, profile, detector, image_convention)
        self.config = config or BowlCycleConfig()
        self.tracker = BowlCycleTracker(profile, config=self.config)
        self.reducers["bowl_1"] = BowlCycleReducer(self.config)

    def _measure(self, subject, references, reducer, released):
        step = self.tracker.step
        return reducer.measure(step, subject, references["plate_1"],
                               self.records[f"gripper:{step}"]["aperture"], self.tracker.wrist_near)
