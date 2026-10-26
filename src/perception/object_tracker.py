"""Causal RGB relation estimates for the single-bowl T1/T3 pilot.

Image-space geometry is an estimate, never a simulator contact predicate. A
miss is UNKNOWN, not departure. EEF/gripper cannot supply object pose.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.memory.models import IdentityStatus, VisibilityStatus
from src.perception.visual_evidence import VisualEvidence, rgb_views, unambiguous_bowl


@dataclass(frozen=True)
class TrackerConfig:
    dwell_steps: int = 3
    detection_score: float = 0.15
    lift_height_ratio: float = 0.40
    return_distance_ratio: float = 0.30
    stable_motion_ratio: float = 0.15
    support_match_ratio: float = 0.45
    release_gap: float = 0.055  # measured Panda aperture, not action command
    origin_acquisition_steps: int = 20
    max_stability_gap: int = 3  # current evidence still mandatory; gaps remain incomplete
    ambiguity_score_margin: float = .12
    ambiguity_score_ratio: float = 1.5
    association_max_distance: float = .25
    association_distance_margin: float = .20
    identity_jump_distance: float = .75
    reacquisition_score: float = .35

    def __post_init__(self):
        if self.dwell_steps < 2:
            raise ValueError("At least two consecutive observations are required")
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid tracker setting {name}")


@dataclass
class TrackedInstance:
    object_id: str
    semantic_name: str
    origin_anchor: str
    identity_status: IdentityStatus = IdentityStatus.UNKNOWN
    visibility: VisibilityStatus = VisibilityStatus.UNKNOWN
    last_observed_step: int = -1
    container_id: Optional[str] = None
    is_on_support: Optional[bool] = None


class CausalObjectTracker:
    """Track the single bowl and original plate, grounded in current RGB.

    Multiple plausible bowls/plates are ambiguous. A new plate never replaces
    the original. Fixed agent-view geometry estimates lift/return. Wrist
    detections are auxiliary evidence, not calibrated multi-view 3D contact.
    """
    def __init__(self, target_object_id="bowl_A", target_semantic_name="bowl",
                 origin_anchor="bowl on its initially observed plate", support_id="plate_A",
                 support_semantic_name="plate", detector=None, image_convention="libero_mem_refined",
                 dwell_steps=3, config=None):
        self.target_object_id = target_object_id
        self.target_semantic_name = target_semantic_name
        self.origin_anchor = origin_anchor
        self.support_id = support_id
        self.support_semantic_name = support_semantic_name
        self.detector = detector
        self.image_convention = image_convention
        self.config = config or TrackerConfig(dwell_steps=dwell_steps)
        self.reset()

    def reset(self, initial_observation=None):
        self.target = TrackedInstance(self.target_object_id, self.target_semantic_name, self.origin_anchor)
        self._last_step = -1
        self._anchor = None
        self._baseline = None
        self._previous = None
        self._previous_step = -1
        self.last_evidence = None
        self.last_signals = None
        if initial_observation is not None:
            self.update(0, initial_observation, None)

    def update(self, step, observation, last_action=None):
        if step < 0 or step <= self._last_step:
            raise ValueError(f"Step out of order: {step} <= {self._last_step}")
        images = rgb_views(observation, self.image_convention)
        gap_state = np.asarray(observation.get("robot0_gripper_qpos"), dtype=float)
        eef = np.asarray(observation.get("robot0_eef_pos"), dtype=float)
        if gap_state.shape != (2,) or eef.shape != (3,) or not np.isfinite(gap_state).all() or not np.isfinite(eef).all():
            raise ValueError("Writer requires finite Panda gripper(2) and EEF position(3)")
        # detected_objects, oracle_signals, reward and subgoals are ignored.
        evidence = observation.get("visual_evidence")
        if evidence is None:
            if self.detector is None:
                raise RuntimeError("Predicted memory requires an RGB detector or validated VisualEvidence; no proprioception fallback")
            evidence = self.detector.detect(step, images)
        if not isinstance(evidence, VisualEvidence):
            raise TypeError("visual_evidence must be a VisualEvidence instance")
        evidence.validate(step, images)
        self.last_evidence = evidence
        self._last_step = step
        cfg = self.config
        gap = float(np.abs(gap_state).sum())
        detections = evidence.views["agentview_image"]
        bowls = [d for d in detections if d.label == self.target_semantic_name and d.score >= cfg.detection_score]
        plates = [d for d in detections if d.label == self.support_semantic_name and d.score >= cfg.detection_score]
        signals = dict(step=step, has_grounding_evidence=False, raw_on_support=False,
                       raw_lifted=False, gripper_released=gap >= cfg.release_gap,
                       stable=False, evidence_ids=(), reason="missing_target", gripper_gap=gap,
                       identity_association="none",
                       wrist_target_visible=any(d.label == self.target_semantic_name and d.score >= cfg.detection_score
                                                for d in evidence.views["robot0_eye_in_hand_image"]))
        self.target.visibility = VisibilityStatus.UNKNOWN
        self.target.identity_status = IdentityStatus.UNKNOWN
        self.target.is_on_support = None
        bowl = unambiguous_bowl(bowls, cfg.ambiguity_score_margin, cfg.ambiguity_score_ratio)
        identity_reference = ()
        signals["identity_association"] = "detector_score" if bowl is not None else "none"
        # Associate current detections with the last measured RGB instance.
        # The historical location is a matching aid, never a substitute for a
        # detection. Close competing assignments remain unknown.
        if bowl is None and bowls and self._previous is not None and step - self._previous_step <= cfg.max_stability_gap:
            def association_cost(candidate):
                delta = np.array([candidate.center[0] - self._previous.center[0],
                                  candidate.xyxy[3] - self._previous.xyxy[3]])
                return float(np.linalg.norm(delta / self._baseline.size) +
                             .25 * abs(np.log(candidate.size[0] / self._previous.size[0])))
            ranked = sorted((association_cost(candidate), index, candidate) for index, candidate in enumerate(bowls))
            if ranked[0][0] <= cfg.association_max_distance and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= cfg.association_distance_margin):
                bowl = ranked[0][2]
                signals["identity_association"] = "measured_rgb_continuity"
                identity_reference = (self._previous.evidence_id,)
        if bowl is not None and self._previous is not None:
            delta = np.array([bowl.center[0] - self._previous.center[0], bowl.xyxy[3] - self._previous.xyxy[3]])
            if np.linalg.norm(delta / self._baseline.size) > cfg.identity_jump_distance and bowl.score < cfg.reacquisition_score:
                signals["reason"] = "target_identity_jump"
                return self._finish(signals, None)
        if bowl is None:
            if len(bowls) > 1:
                self.target.identity_status = IdentityStatus.AMBIGUOUS
                signals["reason"] = "ambiguous_target"
            return self._finish(signals, None)
        # Same-class nested proposals can describe the full plate and its
        # visible inner disk. Keep the enclosing proposal, not multiple IDs.
        # This is limited to the T1/T3 profile of separate, unstacked plates.
        unique_plates = []
        for candidate in sorted(plates, key=lambda p: -float(np.prod(p.size))):
            area = float(np.prod(candidate.size))
            if any(float(np.prod(np.maximum(0, np.minimum(candidate.xyxy[2:], p.xyxy[2:]) -
                                              np.maximum(candidate.xyxy[:2], p.xyxy[:2])))) / area >= .9
                   for p in unique_plates):
                continue
            unique_plates.append(candidate)
        plates = unique_plates
        self.target.visibility = VisibilityStatus.VISIBLE
        self.target.identity_status = IdentityStatus.TRACKED
        self.target.last_observed_step = step
        if self._anchor is None:
            if step >= cfg.origin_acquisition_steps:
                signals["reason"] = "original_support_not_observed_at_start"
                return self._finish(signals, None)
            candidates = [p for p in plates
                          if abs(bowl.center[0] - p.center[0]) <= .45 * p.size[0]
                          and p.xyxy[1] <= bowl.xyxy[3] <= p.xyxy[3] + .1 * p.size[1]
                          and bowl.center[1] <= p.center[1] + .1 * p.size[1]]
            if len(candidates) != 1 or not signals["gripper_released"]:
                signals["reason"] = "unobserved_or_ambiguous_origin"
                return self._finish(signals, None)
            self._anchor = candidates[0]
            self._baseline = bowl
        # Match the original plate anchor, never the bowl's NEW location.
        # A plate detection contained almost entirely in the bowl is a confused
        # class proposal, not a second visible support.
        plates = [p for p in plates if not (p.xyxy[0] >= bowl.xyxy[0] - .01 and
                  p.xyxy[2] <= bowl.xyxy[2] + .01 and p.xyxy[3] <= bowl.xyxy[3] + .01)]
        def bottom_center(detection):
            return np.array([detection.center[0], detection.xyxy[3]])
        matches = [p for p in plates if np.linalg.norm((bottom_center(p) - bottom_center(self._anchor)) / self._anchor.size) <= cfg.support_match_ratio]
        if len(matches) != 1:
            signals["reason"] = "missing_original_support" if not matches else "ambiguous_original_support"
            return self._finish(signals, None)
        plate = matches[0]
        baseline_offset = bottom_center(self._baseline) - bottom_center(self._anchor)
        delta = (bottom_center(bowl) - bottom_center(plate)) - baseline_offset
        distance = float(np.linalg.norm(delta / self._baseline.size))
        lift = float(-delta[1] / self._baseline.size[1])
        scale = bowl.size / self._baseline.size
        near_origin = distance <= cfg.return_distance_ratio and .6 <= scale[0] <= 1.5 and .3 <= scale[1] <= 1.5
        stable = (self._previous is not None and step - self._previous_step <= cfg.max_stability_gap
                  and np.linalg.norm((bottom_center(bowl) - bottom_center(self._previous)) / self._baseline.size) <= cfg.stable_motion_ratio)
        signals.update(has_grounding_evidence=True, raw_on_support=bool(near_origin),
                       raw_lifted=bool(lift >= cfg.lift_height_ratio and not near_origin),
                       stable=bool(stable), evidence_ids=tuple(dict.fromkeys((bowl.evidence_id, plate.evidence_id,
                                                                           self._baseline.evidence_id, self._anchor.evidence_id) + identity_reference)),
                       lift_ratio=lift, origin_distance_ratio=distance, reason="observed")
        self.target.is_on_support = bool(near_origin)
        return self._finish(signals, bowl)

    def _finish(self, signals, current):
        # Retain only the last measured image location for a bounded stability
        # comparison. Never expose it as a current detection when evidence is missing.
        if current is not None:
            self._previous = current
            self._previous_step = signals["step"]
        self.last_signals = signals
        return signals
