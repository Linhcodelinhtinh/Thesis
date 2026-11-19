"""Versioned successor for single-bowl return tracking on T1/T3."""

import numpy as np

from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer import BowlCycleReducer, BowlCycleWriter


class BowlCycleReducerV3(BowlCycleReducer):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.origin_before_step = None
        self.allow_late_origin = False
        self.pending_started_step = None
        self.pending_expired = False
        self.current_step = -1
        self.current_aperture = 0.0
        self.gap_active = False
        self.gap_origin = None
        self.gap_resolvable = False
        self.rest_pose = None
        self.measured_pose = None
        self.peak_bottom = None

    def observe_sample(self, step, aperture):
        self.current_step = step
        self.current_aperture = aperture
        if self.gap_active and aperture < self.config.released_aperture_meters:
            self.gap_resolvable = False
        if (self.pending and self.pending_started_step is not None
                and step - self.pending_started_step > self.config.pending_expiry_steps):
            self.pending = False
            self.pending_started_step = None
            self.pending_expired = True
            self.complete = False

    def measure(self, step, bowl, plate, aperture, wrist_near):
        self.observe_sample(step, aperture)
        pending_before = self.pending
        visible_and_unambiguous = all(
            track.box is not None and track.anchor is not None
            and track.identity.identity_status.value != "ambiguous"
            for track in (bowl, plate)
        )
        allow_pending = (
            visible_and_unambiguous
            and not (self.pending_expired and aperture < self.config.released_aperture_meters)
        )
        result = super().measure(step, bowl, plate, aperture, wrist_near,
                                 allow_pending=allow_pending)
        self.measured_pose = None
        if visible_and_unambiguous and (self.rest_pose is not None or result[0] == "relation"):
            x1, top, x2, bottom = bowl.box.xyxy
            px1, _, px2, plate_bottom = plate.box.xyxy
            width, height = bowl.anchor.size
            self.measured_pose = (
                (x1 + x2 - px1 - px2) / (2 * width),
                (top - plate_bottom) / height,
                (bottom - plate_bottom) / height,
                (bowl.box.evidence_id, plate.box.evidence_id, f"gripper:{step}"),
            )
            result = self._check_motion(result, aperture)
        if self.pending and not pending_before:
            self.pending_started_step = step
        elif not self.pending:
            self.pending_started_step = None
        return result

    def _check_motion(self, result, aperture):
        if self.rest_pose is None:
            return result
        status, reference, ids = result
        x, top, bottom, pose_ids = self.measured_pose
        rest_x, rest_top, rest_bottom, rest_ids = self.rest_pose
        if status == "departure":
            if min(rest_top - top, rest_bottom - bottom) < self.config.coherent_rise_ratio:
                return "unsettled", None, ()
            return status, reference, ids + rest_ids
        if self.departure is not None and aperture <= self.config.closed_aperture_meters:
            if self.peak_bottom is None or bottom < self.peak_bottom[0]:
                self.peak_bottom = bottom, pose_ids
        if status == "relation" and self.departure is not None:
            if (self.peak_bottom is None
                    or bottom - self.peak_bottom[0] < self.config.return_descent_ratio
                    or np.hypot(x - rest_x, bottom - rest_bottom) > self.config.settled_offset_ratio):
                return "unsettled", None, ()
            return status, reference, ids + rest_ids + self.peak_bottom[1]
        return result

    def update(self, step, status, reference, evidence, emit):
        skipped_step = status == "gap" and emit is None
        if status in ("unknown", "conflicted") and not self.gap_active:
            self.gap_active = True
            self.gap_origin = self.origin
            self.gap_resolvable = (
                self.origin is not None
                and self.ready
                and self.departure is None
                and not self.pending
                and self.current_aperture >= self.config.released_aperture_meters
            )
        if skipped_step:
            self.gap_active = True
            self.gap_resolvable = False
            self.pending = False
            self.pending_started_step = None
            self.pending_expired = True

        origin_was_missing = self.origin is None
        super().update(step, status, reference, evidence, emit)

        if status in ("unknown", "conflicted") and self.missing > self.config.max_missing_steps:
            self.gap_resolvable = False
            self.pending = False
            self.pending_started_step = None
            self.pending_expired = True

        if status == "relation" and self.status == "confirmed":
            if self.origin is None and self.allow_late_origin:
                self.origin = reference
                self.ready = True
                self.initial_support = self.last_event
                self.historical_relation_valid = True
            if origin_was_missing and self.allow_late_origin and self.origin == reference:
                self.complete = True
            if (self.gap_active and self.gap_resolvable and self.origin == self.gap_origin
                    and reference == self.origin
                    and self.current_aperture >= self.config.released_aperture_meters
                    and self.departure is None and not self.pending):
                self.complete = True
            if self.current_aperture >= self.config.released_aperture_meters:
                self.pending_expired = False
            self.gap_active = False
            self.gap_resolvable = False
            self.gap_origin = None
            if self.ready and self.measured_pose is not None:
                self.rest_pose = self.measured_pose

        if self.departure is None:
            self.peak_bottom = None
        elif self.peak_bottom is None and self.measured_pose is not None:
            self.peak_bottom = self.measured_pose[2:]
        self.retained_evidence = (
            (() if self.rest_pose is None else self.rest_pose[3])
            + (() if self.peak_bottom is None else self.peak_bottom[1])
        )

        if self.pending and self.pending_started_step is None:
            self.pending_started_step = step
        elif not self.pending:
            self.pending_started_step = None


class BowlCycleWriterV3(BowlCycleWriter):
    """Resolve late origins conservatively and record a versioned config."""

    VERSION = "rgb_relations_v3"

    def __init__(self, episode_id, profile, detector, image_convention, config=None):
        config = config or BowlCycleConfig.load()
        super().__init__(episode_id, profile, detector, image_convention, config=config)
        self.tracker.allow_late_anchors = False
        self.reducers["bowl_1"] = BowlCycleReducerV3(config)
        self._origin_acquisition_open = True

    def step(self, step, observation, last_action=None):
        if step == self._step:
            return super().step(step, observation, last_action)
        aperture = float(np.abs(np.asarray(observation["robot0_gripper_qpos"], dtype=float)).sum())
        if step != self._step + 1 or aperture < self.config.released_aperture_meters:
            self._origin_acquisition_open = False
        self.tracker.allow_late_anchors = self._origin_acquisition_open
        reducer = self.reducers["bowl_1"]
        reducer.allow_late_origin = self._origin_acquisition_open
        snapshot = super().step(step, observation, last_action)
        if (self.tracker.anchors.get("bowl_1") is None
                or self.tracker.previous.get("bowl_1") is None):
            self._origin_acquisition_open = False
            self.tracker.allow_late_anchors = False
            reducer.allow_late_origin = False
        return snapshot

    def _measure(self, subject, references, reducer, released):
        reducer.allow_late_origin = self._origin_acquisition_open
        return super()._measure(subject, references, reducer, released)
