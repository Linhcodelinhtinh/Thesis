"""Single-bowl association for cycle recognition from Grounding DINO boxes."""
from dataclasses import replace

import numpy as np

from src.memory.models import IdentityStatus
from src.memory.relation_cycle_config import BowlCycleConfig
from src.perception.relation_tracking import RelationTracker
from src.perception.visual_evidence import unambiguous_bowl


TRACK_SCORE = .15
MEASURE_SCORE = .18
# Number of missing or contested samples that can be bridged before a lift
# attempt expires. Measured on the development demonstrations.
MAX_MISSING = 22


class BowlCycleTracker(RelationTracker):
    """Pass only unambiguous, fresh measurements on to the relation tracker."""

    def __init__(self, profile, *, config=None, allow_late_anchors=False):
        config = config or BowlCycleConfig()
        super().__init__(profile, detection_score=config.track_score,
                         plate_duplicate_overlap=config.plate_duplicate_overlap_ratio,
                         plate_association_distance=config.plate_association_distance,
                         object_association_distance=config.object_association_distance,
                         association_margin=config.association_tie_margin,
                         minimum_instance_separation=config.minimum_instance_separation,
                         support_horizontal_ratio=config.support_horizontal_ratio,
                         support_vertical_tolerance_ratio=config.support_vertical_tolerance_ratio)
        self.step = -1
        self._link = None
        self._linked_at = -1
        self.wrist_near = False
        self.config = config
        self.track_score = config.track_score
        self.measure_score = config.measure_score
        self.max_missing = config.max_missing_steps
        self.allow_late_anchors = allow_late_anchors

    def observe(self, evidence):
        if evidence.step != self.step + 1:
            self._link = None
        self.step = evidence.step
        agent = "agentview_image"
        boxes = evidence.views[agent]
        bowls = [box for box in boxes if box.label == "bowl" and box.score >= self.track_score]
        selected, contested = self._associate(bowls)
        measured = selected if selected is not None and selected.score >= self.measure_score else None
        # A weak hit can update the association, but supplies no geometric measurement.
        if selected is not None:
            self._link, self._linked_at = selected, self.step
        filtered = tuple(box for box in boxes if box.label != "bowl" or box is measured)
        tracks = super().observe(replace(evidence, views={**evidence.views, agent: filtered}),
                                 allow_late_anchors=self.allow_late_anchors)
        if contested:
            track = tracks["bowl_1"]
            tracks["bowl_1"] = replace(track, identity=replace(
                track.identity, identity_status=IdentityStatus.AMBIGUOUS))
        # The wrist image supplies an optional proximity cue, in its own image
        # coordinates. It does not measure contact or provide a motion vector.
        wrist = unambiguous_bowl([box for box in evidence.views["robot0_eye_in_hand_image"]
                                  if box.label == "bowl" and box.score >= self.measure_score],
                                 margin=self.config.detector_seed_score_margin,
                                 ratio=self.config.detector_seed_score_ratio)
        self.wrist_near = (
            wrist is not None
            and np.all(np.abs(wrist.center - .5) <= self.config.wrist_center_tolerance)
            and np.min(wrist.size) >= self.config.wrist_min_size
        )
        return tracks

    def _associate(self, boxes):
        anchor = self.anchors.get("bowl_1")
        recent = self._link is not None and self.step - self._linked_at <= self.max_missing + 1
        if anchor is None or not recent:
            seed = unambiguous_bowl(boxes, margin=self.config.detector_seed_score_margin,
                                    ratio=self.config.detector_seed_score_ratio)
            return (seed, False) if seed is not None and seed.score >= self.measure_score else (None, len(boxes) > 1)

        ranked = []
        elapsed = self.step - self._linked_at
        for box in boxes:
            distance = float(np.linalg.norm(box.center - self._link.center))
            size_ratio = box.size / anchor.size
            limit = min(self.config.association_max_distance,
                        self.config.association_initial_distance +
                        self.config.association_distance_growth * (elapsed - 1))
            if box.score < self.measure_score:
                limit = self.config.association_low_score_distance
            if (distance <= limit and
                    np.all((self.config.association_min_size_ratio <= size_ratio) &
                           (size_ratio <= self.config.association_max_size_ratio))):
                ranked.append((distance, box))
        ranked.sort(key=lambda row: row[0])
        if (len(ranked) > 1 and
                ranked[1][0] - ranked[0][0] < self.config.association_tie_margin):
            return None, True
        return (ranked[0][1], False) if ranked else (None, False)
