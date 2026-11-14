"""Conservative one-to-one association and relation estimates from RGB boxes."""
from dataclasses import dataclass, replace

import numpy as np

from src.memory.models import IdentityStatus, ObjectIdentity, VisibilityStatus


@dataclass(frozen=True)
class Track:
    identity: ObjectIdentity
    label: str
    box: object
    previous: object
    anchor: object


class RelationTracker:
    def __init__(self, profile):
        self.profile = profile
        self.anchors = {}
        self.previous = {}
        self.last_measured = {}
        self.last_seen = {}

    def observe(self, evidence):
        tracks = {}
        for label, count in ((self.profile.subject, self.profile.subject_count),
                             (self.profile.reference, self.profile.reference_count)):
            boxes = [box for box in evidence.views["agentview_image"]
                     if box.label == label and box.score >= .15]
            if label == "plate":
                distinct = []
                for box in sorted(boxes, key=lambda item: -float(np.prod(item.size))):
                    low, high = np.array(box.xyxy[:2]), np.array(box.xyxy[2:])
                    if any(np.prod(np.maximum(0, np.minimum(high, other.xyxy[2:]) -
                                              np.maximum(low, other.xyxy[:2]))) >= .9 * np.prod(box.size)
                           for other in distinct):
                        continue
                    distinct.append(box)
                boxes = distinct
            if self.profile.count_cycles and label == self.profile.reference:
                count = 1
                if "plate_1" not in self.anchors:
                    subject = tracks[f"{self.profile.subject}_1"]
                    boxes = [box for box in boxes if subject.box is not None
                             and support_overlap(subject.box, box)]
            ids = [f"{label}_{i + 1}" for i in range(count)]
            if not any(key in self.anchors for key in ids) and len(boxes) == count and evidence.step < 20:
                ordered = sorted(boxes, key=lambda b: b.center[0])
                if all(b.center[0] - a.center[0] >= .025 for a, b in zip(ordered, ordered[1:])):
                    self.anchors.update(zip(ids, ordered))
                    self.previous.update(zip(ids, ordered))
                    self.last_measured.update(zip(ids, ordered))
                    self.last_seen.update((key, evidence.step) for key in ids)
            previous = {key: self.previous.get(key) for key in ids}
            proposals = {}
            for key in ids:
                anchor = self.anchors.get(key)
                # Plates are stationary in these profiles. Moving instances may
                # continue across adjacent measured frames only. A unique class
                # can be observed again after an interruption.
                ref = anchor if label == "plate" else previous[key]
                if (ref is None and len(boxes) == count and
                        sum(previous[other] is None for other in ids) == 1 and
                        evidence.step - self.last_seen.get(key, -100) <= 3):
                    ref = self.last_measured[key]
                if not anchor:
                    continue
                if count == 1 and len(boxes) == 1 and label != "plate":
                    proposals[key] = boxes[0]
                elif ref is not None:
                    limit = .035 if label == "plate" else .07
                    ranked = sorted(((float(np.linalg.norm(b.center - ref.center)), b) for b in boxes),
                                    key=lambda row: row[0])
                    if ranked and ranked[0][0] <= limit and (
                            len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= .02):
                        distance, best = ranked[0]
                        # A missed/ambiguous same-class track could have
                        # generated this detection. Reject a reassignment.
                        competitors = [self.last_measured.get(other) for other in ids
                                       if other != key and other in self.last_measured]
                        if label == "plate" or all(
                            np.linalg.norm(best.center - p.center) >= distance + .02
                            for p in competitors
                        ):
                            proposals[key] = best
            for key in ids:
                box = proposals.get(key)
                if box is not None and sum(box is b for b in proposals.values()) != 1:
                    box = None
                anchor = self.anchors.get(key)
                status = (IdentityStatus.TRACKED if box is not None else IdentityStatus.AMBIGUOUS
                          if boxes and anchor else IdentityStatus.UNKNOWN)
                identity = ObjectIdentity(key, f"initial {label} {key.rsplit('_', 1)[1]} from left",
                    status, VisibilityStatus.VISIBLE if box else VisibilityStatus.UNKNOWN,
                    metadata={"origin_evidence": anchor.evidence_id if anchor else None})
                tracks[key] = Track(identity, label, box, previous[key], anchor)
                self.previous[key] = box
            self.last_measured.update((key, tracks[key].box) for key in ids if tracks[key].box is not None)
            self.last_seen.update((key, evidence.step) for key in ids if tracks[key].box is not None)
        for label in (self.profile.subject, self.profile.reference):
            group = [track for track in tracks.values() if track.label == label]
            visible = sorted((track for track in group if track.box), key=lambda t: t.box.center[0])
            if len(visible) != len(group):
                continue
            if any(b.box.center[0] - a.box.center[0] < .025 for a, b in zip(visible, visible[1:])):
                continue
            for i, track in enumerate(visible, 1):
                desc = label if len(group) == 1 else (
                    f"left {label}" if i == 1 else f"right {label}" if i == len(group) else
                    f"{label} {i} from left")
                identity = replace(track.identity, current_location_desc=desc, metadata={
                    **track.identity.metadata, "location_evidence": tuple(t.box.evidence_id for t in visible)})
                tracks[identity.object_id] = replace(track, identity=identity)
        return tracks


def relation_candidates(subject, references, predicate):
    """Return geometric candidates. A box overlap is only a visual estimate."""
    box = subject.box
    if box is None:
        return []
    candidates = []
    for key, ref in references.items():
        if ref.box is None:
            continue
        x, y = box.center
        width, height = ref.box.size
        bottom = box.xyxy[3]
        if predicate == "on":
            plausible = support_overlap(box, ref.box)
        elif predicate == "in":
            x1, y1, x2, y2 = ref.box.xyxy
            plausible = x1 + .12 * width < x < x2 - .12 * width and y1 < y < y2 and bottom <= y2 + .08 * height
        else:
            raise ValueError(f"Unsupported visual relation: {predicate}")
        if plausible:
            candidates.append(key)
    return candidates


def support_overlap(subject, plate):
    x, y = subject.center
    _, top, _, bottom = plate.xyxy
    width, height = plate.size
    return (abs(x - plate.center[0]) <= .45 * width and top <= subject.xyxy[3] <= bottom + .10 * height
            and y <= plate.center[1] + .10 * height)


def visible_departure(subject, origin):
    if subject.box is None or subject.anchor is None or origin.box is None or origin.anchor is None:
        return False
    initial = subject.anchor.xyxy[3] - origin.anchor.xyxy[3]
    current = subject.box.xyxy[3] - origin.box.xyxy[3]
    return (initial - current) >= .40 * subject.anchor.size[1]
