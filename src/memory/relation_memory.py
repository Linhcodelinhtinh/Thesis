"""Online reduction of frame-bound observations into facts and event history."""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from src.memory.models import EvidenceSource, V2MemoryEvent, _freeze_value, _json_value
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.relation_tracking import RelationTracker, relation_candidates, visible_departure
from src.perception.visual_evidence import VisualEvidence, image_digest, rgb_views
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.models import ObjectIdentity
from typing import Mapping


@dataclass(frozen=True)
class RelationState:
    status: str
    reference: str | None
    last_event: V2MemoryEvent | None
    last_return: V2MemoryEvent | None
    departure: V2MemoryEvent | None
    candidate_frames: int
    returns: int
    history_complete: bool
    armed: bool
    historical_relation_valid: bool
    current_evidence_ids: tuple


@dataclass(frozen=True)
class RelationMemorySnapshot:
    episode_id: str
    as_of_observation: int
    last_consumed_action: int
    profile: MemoryProfile
    objects: Mapping[str, ObjectIdentity]
    relations: Mapping[str, RelationState]
    evidence: Mapping
    recent_events: tuple[V2MemoryEvent, ...]

    def to_dict(self):
        return dict(episode_id=self.episode_id, as_of_observation=self.as_of_observation,
                    last_consumed_action=self.last_consumed_action,
                    profile=vars(self.profile),
                    objects={key: value.to_dict() for key, value in self.objects.items()},
                    relations={key: {name: field.to_dict() if isinstance(field, V2MemoryEvent) else field
                                     for name, field in vars(value).items()}
                               for key, value in self.relations.items()},
                    evidence=_json_value(self.evidence),
                    recent_events=[event.to_dict() for event in self.recent_events])


class RelationReducer:
    """Confirm repeated observations and count only observed departure/return pairs."""
    def __init__(self, count_cycles):
        self.count_cycles = count_cycles
        self.origin = None
        self.departure = None
        self.last_event = None
        self.last_return = None
        self.candidate = None
        self.frames = []
        self.returns = 0
        self.complete = True
        self.status = "unknown"
        self.reference = None
        self.ready = False
        self.historical_relation_valid = False
        self.initial_support = None

    def update(self, step, status, reference, evidence, emit):
        self.status, self.reference = status, None
        if status in ("unknown", "conflicted", "gap"):
            self.complete = False
            if status in ("conflicted", "gap"):
                self.ready = False
                self.departure = None
            if status == "conflicted":
                self.historical_relation_valid = False
            self.status = "conflicted" if status == "conflicted" else "unknown"
            self.candidate, self.frames = None, []
            return
        if status == "unsettled":
            self.status = "unknown"
            self.candidate, self.frames = None, []
            return
        candidate = (status, reference)
        if status == "relation" and self.last_event is not None and self.last_event.details["reference"] != reference:
            self.historical_relation_valid = False
        if candidate != self.candidate:
            self.candidate, self.frames = candidate, []
        self.frames.append((step, evidence))
        self.frames = self.frames[-3:]
        self.status = "candidate"
        if len(self.frames) < 3:
            return
        self.status = "confirmed"
        self.reference = reference if status == "relation" else None
        first = self.frames[0][0]
        ids = [eid for _, frame_ids in self.frames for eid in frame_ids]
        if status == "departure":
            if self.ready and self.origin is not None and self.departure is None:
                ids.extend(self.initial_support.evidence_ids)
                self.departure = emit("departure", self.origin, first, ids)
                self.ready = False
            return
        if self.origin is None and first < 20:
            self.origin = reference
        returning = self.departure is not None and reference == self.origin
        changed = self.last_event is None or self.last_event.details["reference"] != reference
        if returning or changed or not self.historical_relation_valid:
            if returning and self.count_cycles:
                self.returns += 1
                ids.extend(self.departure.evidence_ids)
            kind = ("return" if returning and self.count_cycles else
                    "initial_relation" if self.last_event is None else "relation")
            self.last_event = emit(kind, reference, first, ids)
            if kind == "return":
                self.last_return = self.last_event
        self.departure = None
        self.ready = reference == self.origin
        if self.ready and self.initial_support is None:
            self.initial_support = self.last_event
        self.historical_relation_valid = True

    def view(self):
        return RelationState(self.status, self.reference, self.last_event, self.last_return, self.departure,
                             len(self.frames), self.returns, self.complete, self.departure is not None,
                             self.historical_relation_valid,
                             self.frames[-1][1] if self.reference and self.frames else ())


class RelationMemoryWriter:
    VERSION = "rgb_relations_v1"

    def __init__(self, episode_id, profile, detector, image_convention):
        self.episode_id, self.profile = episode_id, profile
        self.detector, self.image_convention = detector, image_convention
        self.tracker = RelationTracker(profile)
        self.reducers = {}
        self.outside = {}
        self.records = {}
        self.recent_events = []
        self.last_events = ()
        self.last_evidence = None
        self.last_signals = None
        self._next_event = 0
        self._step = -1
        self._signature = None
        self._snapshot = None

    def step(self, step, observation, last_action=None):
        if type(step) is not int or step < 0 or step < self._step:
            raise ValueError("Out-of-order observation")
        grip = np.asarray(observation["robot0_gripper_qpos"], dtype=float)
        action = None if last_action is None else np.asarray(last_action)
        if grip.shape != (2,) or not np.isfinite(grip).all():
            raise ValueError("Expected measured gripper aperture inputs")
        if action is not None and (step not in (self._step, self._step + 1)
                                   or step == 0 or action.shape != (7,) or not np.isfinite(action).all()):
            raise ValueError("Expected only the action preceding this observation")
        images = rgb_views(observation, self.image_convention)
        hashes = {key: image_digest(value) for key, value in images.items()}
        evidence = observation.get("visual_evidence")
        if evidence is not None and not isinstance(evidence, VisualEvidence):
            raise TypeError("Expected typed visual evidence")
        payload = json.dumps([hashes, grip.tolist(), None if action is None else action.tolist(),
                              None if evidence is None else evidence.to_dict()], sort_keys=True)
        signature = hashlib.sha256(payload.encode()).hexdigest()
        if step == self._step:
            if signature != self._signature:
                raise ValueError("Conflicting duplicate observation")
            return self._snapshot
        if evidence is None:
            if self.detector is None:
                raise ValueError("RGB evidence required")
            evidence = self.detector.detect(step, images, self.profile.queries,
                                            pilot_crop=self.profile.count_cycles and self.profile.subject == "bowl")
        evidence.validate(step, images)
        if (evidence.model_id, evidence.revision) != (MODEL_ID, REVISION):
            raise ValueError("Unexpected visual evidence model or revision")
        records = {}
        aperture_id = f"gripper:{step}"
        records[aperture_id] = dict(step=step, source="measured_proprioception",
                                    aperture=float(np.abs(grip).sum()))
        for camera, boxes in evidence.views.items():
            for box in boxes:
                if box.label not in self.profile.queries or box.evidence_id in self.records:
                    raise ValueError("Unexpected or reused detection evidence")
                records[box.evidence_id] = dict(step=step, camera=camera, image_sha256=hashes[camera],
                    model=evidence.model_id, revision=evidence.revision, detection=box.to_dict())
        self.records.update(records)

        if step != self._step + 1:
            self.tracker.previous.clear()
            self.tracker.last_seen.clear()
            for reducer in self.reducers.values():
                reducer.update(step, "gap", None, (), None)
        tracks = self.tracker.observe(evidence)
        references = {key: track for key, track in tracks.items() if track.label == self.profile.reference}
        new_events = []
        released = np.abs(grip).sum() >= .055
        for key, subject in tracks.items():
            if subject.label != self.profile.subject:
                continue
            reducer = self.reducers.setdefault(key, RelationReducer(self.profile.count_cycles))
            status, reference, ids = self._measure(subject, references, reducer, released)
            if status == "relation":
                ids += (aperture_id,)

            def emit(kind, ref, first, ids):
                ids = tuple(dict.fromkeys(ids))
                if not ids or any(eid not in self.records or self.records[eid]["step"] > step for eid in ids):
                    raise ValueError("Incomplete or future event evidence")
                event = V2MemoryEvent(f"{self.episode_id}:{self._next_event}", kind, first, step,
                    EvidenceSource.PREDICTED_RGB_TRACKER, evidence_ids=ids,
                    details={"subject": key, "reference": ref, "predicate": self.profile.predicate,
                             "observed_returns": reducer.returns,
                             "paired_departure": reducer.departure.id if kind == "return" else None,
                             "previous_return": (reducer.last_return.id
                                                 if kind == "return" and reducer.last_return else None)})
                self._next_event += 1
                new_events.append(event)
                return event

            reducer.update(step, status, reference, ids, emit)
        self._step, self._signature = step, signature
        self.last_evidence, self.last_events = evidence, tuple(new_events)
        self.last_signals = {"new_events": [event.to_dict() for event in new_events],
                             "visual_evidence": evidence.to_dict()}
        self.recent_events = (self.recent_events + new_events)[-8:]
        needed = {d.evidence_id for t in tracks.values() for d in (t.box, t.anchor) if d is not None}
        needed.update(eid for ids in self.outside.values() for eid in ids)
        events = list(self.recent_events)
        for reducer in self.reducers.values():
            for _, ids in reducer.frames:
                needed.update(ids)
            events.extend(e for e in (reducer.last_event, reducer.last_return,
                                      reducer.departure, reducer.initial_support) if e is not None)
        needed.update(eid for event in events for eid in event.evidence_ids)
        self.records = {key: self.records[key] for key in sorted(needed)}
        self._snapshot = RelationMemorySnapshot(
            self.episode_id, step, step - 1 if action is not None else -1, self.profile,
            MappingProxyType({key: track.identity for key, track in tracks.items()}),
            MappingProxyType({key: reducer.view() for key, reducer in self.reducers.items()}),
            _freeze_value(self.records), tuple(self.recent_events))
        return self._snapshot

    def get_snapshot(self):
        return self._snapshot

    def _measure(self, subject, references, reducer, released):
        if subject.box is None or any(ref.box is None for ref in references.values()):
            return "unknown", None, ()
        candidates = relation_candidates(subject, references, self.profile.predicate)
        if self.profile.predicate == "in":
            key = subject.identity.object_id
            if not candidates:
                self.outside[key] = tuple(d.evidence_id for d in
                                          (subject.box, *(ref.box for ref in references.values())))
                if reducer.last_event is not None:
                    return "conflicted", None, ()
            elif key not in self.outside:
                return "unsettled", None, ()
        if self.profile.count_cycles and candidates and subject.anchor is not None:
            ref = references[candidates[0]]
            if ref.anchor is not None:
                def bottom_center(box):
                    return np.array([box.center[0], box.xyxy[3]])

                delta = (bottom_center(subject.box) - bottom_center(ref.box)) - (
                    bottom_center(subject.anchor) - bottom_center(ref.anchor))
                if np.linalg.norm(delta / subject.anchor.size) > .35:
                    candidates = []
        if len(candidates) > 1:
            return "conflicted", None, ()
        previous = subject.previous
        stable = previous is not None and np.linalg.norm(
            (subject.box.center - previous.center) / subject.box.size) <= .20
        if candidates and stable and released:
            ref = references[candidates[0]]
            ref_stable = ref.previous is not None and np.linalg.norm(
                (ref.box.center - ref.previous.center) / ref.box.size) <= .20
            if ref_stable:
                origin = self.outside.get(subject.identity.object_id, ())
                return "relation", candidates[0], self._ids(subject, ref) + origin
        if not candidates and reducer.origin and self.profile.count_cycles:
            ref = references[reducer.origin]
            if visible_departure(subject, ref):
                return "departure", reducer.origin, self._ids(subject, ref)
        return "unsettled", None, ()

    @staticmethod
    def _ids(subject, reference):
        return tuple(d.evidence_id for track in (subject, reference)
                     for d in (track.box, track.anchor, track.previous) if d is not None)
