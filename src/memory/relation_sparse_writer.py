"""Relation writers that call the RGB detector on sparse keyframes.

A sparse writer wraps one of the dense writers (V1 or V3) and keeps the prompt
contract of that writer. It never changes how a frame is turned into a fact; it
only chooses which frames reach the detector.

* While the dense writer is at rest, only keyframes run the detector. Frames in
  between are published with the last snapshot and an advanced
  ``as_of_observation``. They are *unobserved*, not "unchanged".
* A keyframe is compared with the last processed frame. If the probe sees any
  change, the checkpointed dense writer replays every buffered frame with the
  detector, so events keep their true ``occurred_at``.
* A keyframe that looks quiet is fed to the dense writer as a declared skip.
  If the dense writer then leaves its rest state, the checkpoint is restored and
  the gap is replayed densely.
* Frames with an engaged gripper, and all frames until the writer is back at
  rest, are processed densely.

The wrapper never reads detection evidence for an unprocessed frame, even when
a replay harness attaches it to the observation.
"""
from contextlib import suppress
import copy
import copyreg
from dataclasses import dataclass, replace
import hashlib
import json
import time
from types import MappingProxyType

import numpy as np

from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_sparse_config import SparseWriterConfig
from src.perception.visual_evidence import (CAMERAS, VisualEvidence, image_digest, rgb_views,
                                            unambiguous_bowl)

# The frozen snapshots hold read-only mapping views. Register them once so a
# writer checkpoint can be taken with copy.deepcopy.
with suppress(Exception):
    copyreg.pickle(MappingProxyType, lambda view: (MappingProxyType, (dict(view),)))

AGENT = CAMERAS[0]
WRIST = CAMERAS[1]


@dataclass
class _Frame:
    step: int
    observation: dict
    action: object
    aperture: float
    signature: str


def _slim(observation):
    """Keep only what the dense writer reads, with private copies of arrays."""
    slim = {camera: np.array(observation[camera], copy=True) for camera in CAMERAS}
    slim["robot0_gripper_qpos"] = np.array(observation["robot0_gripper_qpos"], dtype=float, copy=True)
    return slim


def _top(evidence, label, score):
    boxes = [box for box in evidence.views[AGENT] if box.label == label and box.score >= score]
    return len(boxes), (max(boxes, key=lambda box: box.score) if boxes else None)


def wrist_near(evidence, config):
    """The bowl-tracker's optional wrist proximity cue, recomputed from evidence."""
    wrist = unambiguous_bowl([box for box in evidence.views[WRIST]
                              if box.label == "bowl" and box.score >= config.measure_score],
                             margin=config.detector_seed_score_margin,
                             ratio=config.detector_seed_score_ratio)
    return bool(wrist is not None
                and np.all(np.abs(wrist.center - .5) <= config.wrist_center_tolerance)
                and np.min(wrist.size) >= config.wrist_min_size)


def probe_changes(previous, current, profile, sparse, cycle_config=None):
    """Reasons why *current* may differ from *previous*; empty means quiet."""
    reasons = []
    for label, limit in ((profile.subject, sparse.probe_motion_ratio),
                         (profile.reference, sparse.probe_plate_motion_ratio)):
        count_before, before = _top(previous, label, sparse.probe_score)
        count_now, now = _top(current, label, sparse.probe_score)
        if count_before != count_now:
            reasons.append(f"count:{label}")
            continue
        if before is None:
            continue
        if np.linalg.norm((now.center - before.center) / before.size) > limit:
            reasons.append(f"motion:{label}")
        if np.max(np.abs(now.size - before.size) / before.size) > sparse.probe_size_ratio:
            reasons.append(f"size:{label}")
    if sparse.probe_wrist and cycle_config is not None:
        if wrist_near(previous, cycle_config) != wrist_near(current, cycle_config):
            reasons.append("wrist")
    return reasons


def detector_calls_in_trace(records):
    """Detector calls of a sparse run, from its memory-trace records.

    The rollout's own ``detector_calls`` counter assumes one call per step, which
    over-counts a sparse run, and the rollout is shared, hash-pinned source. Each
    record's ``signals["sparse"]["detector_calls"]`` is exact, including replays.
    """
    calls = 0
    for record in records:
        sparse = (record.get("signals") or {}).get("sparse")
        if sparse is None:
            raise ValueError("Trace record was not written by a sparse writer")
        calls += sparse["detector_calls"]
    return calls


class SparseWriterMixin:
    """Shared control flow. Concrete classes set ``CORE``, ``CORE_VERSION`` and ``VERSION``."""

    CORE = None
    CORE_VERSION = None

    def __init__(self, episode_id, profile, detector, image_convention, config=None, sparse=None):
        self.sparse = sparse or SparseWriterConfig.load()
        self.episode_id, self.profile = episode_id, profile
        self.detector, self.image_convention = detector, image_convention
        # The core has no detector: it must be handed the evidence we choose.
        self.core = self._make_core(episode_id, profile, image_convention, config)
        self._buffer = []
        self._step = -1
        self._core_step = -1
        self._signature = None
        self._snapshot = None
        self._rest_run = 0
        self.last_events = ()
        self.last_evidence = None
        self.last_signals = None
        self.stats = dict(frames=0, processed=0, skipped=0, detector_calls=0, replays=0,
                          replayed_frames=0, rollbacks=0, quiet_keyframes=0)
        self.detected_steps = []

    # -- delegation -----------------------------------------------------------------
    def _make_core(self, episode_id, profile, image_convention, config):
        raise NotImplementedError

    def __getattr__(self, name):
        # Only reached for attributes this object does not define itself.
        if name in ("core", "__setstate__") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.core, name)

    @property
    def detector_calls(self):
        return self.stats["detector_calls"]

    def get_snapshot(self):
        return self._snapshot

    # -- input handling -------------------------------------------------------------
    def _ingest(self, step, observation, last_action):
        grip = np.asarray(observation["robot0_gripper_qpos"], dtype=float)
        action = None if last_action is None else np.asarray(last_action)
        if grip.shape != (2,) or not np.isfinite(grip).all():
            raise ValueError("Expected measured gripper aperture inputs")
        if action is not None and (step not in (self._step, self._step + 1)
                                   or step == 0 or action.shape != (7,) or not np.isfinite(action).all()):
            raise ValueError("Expected only the action preceding this observation")
        images = rgb_views(observation, self.image_convention)
        hashes = {key: image_digest(value) for key, value in images.items()}
        # Attached evidence is deliberately not inspected here: only frames the
        # schedule selects may use it (see _detect).
        payload = json.dumps([hashes, grip.tolist(), None if action is None else action.tolist()],
                             sort_keys=True)
        return _Frame(step, observation, action, float(np.abs(grip).sum()),
                      hashlib.sha256(payload.encode()).hexdigest())

    def _detect(self, frame):
        """Evidence for a frame that has been selected for processing."""
        self.stats["detector_calls"] += 1
        self.detected_steps.append(frame.step)
        evidence = frame.observation.get("visual_evidence")
        if evidence is not None:
            if not isinstance(evidence, VisualEvidence):
                raise TypeError("Expected typed visual evidence")
            return evidence
        if self.detector is None:
            raise ValueError("RGB evidence required")
        images = rgb_views(frame.observation, self.image_convention)
        hashes = {key: image_digest(value) for key, value in images.items()}
        return self.detector.detect(frame.step, images, self.profile.queries,
                                    pilot_crop=self.profile.count_cycles and self.profile.subject == "bowl",
                                    image_digests=hashes)

    # -- scheduling -----------------------------------------------------------------
    def _reason(self, frame):
        sparse = self.sparse
        if frame.step < sparse.dense_prefix_steps:
            return "prefix"
        if self._rest_run < sparse.rest_confirm_frames:
            return "not_at_rest"
        if sparse.dense_while_engaged and frame.aperture < sparse.engaged_aperture_meters:
            return "engaged"
        interval = sparse.keyframe_interval
        if sparse.align_keyframes and frame.step % interval == 0:
            return "keyframe"
        if frame.step - self._core_step >= interval:
            return "keyframe"
        return None

    def _at_rest(self):
        reducers = self.core.reducers
        if not reducers:
            return False
        return all(reducer.status == "confirmed" and reducer.reference is not None
                   and reducer.ready and reducer.departure is None
                   and not getattr(reducer, "pending", False)
                   and not getattr(reducer, "gap_active", False)
                   for reducer in reducers.values())

    def _fingerprint(self, core):
        return tuple((key, reducer.returns, getattr(reducer.departure, "id", None),
                      getattr(reducer.last_event, "id", None), reducer.complete, reducer.ready,
                      reducer.origin, bool(getattr(reducer, "pending", False)))
                     for key, reducer in sorted(core.reducers.items()))

    # -- core stepping --------------------------------------------------------------
    def _core_step_frame(self, frame, evidence):
        snapshot = self.core.step(frame.step, dict(frame.observation, visual_evidence=evidence), frame.action)
        self._core_step = frame.step
        return snapshot, list(self.core.last_events)

    def _declared_skip(self, frame, evidence):
        """Feed a keyframe as if it followed the last processed frame directly."""
        core = self.core
        core._step = frame.step - 1
        if hasattr(core.tracker, "step"):
            core.tracker.step = frame.step - 1
        return self._core_step_frame(frame, evidence)

    def _replay(self, frames, frame, evidence):
        events = []
        for earlier in frames:
            _, new = self._core_step_frame(earlier, self._detect(earlier))
            events.extend(new)
        self.stats["replays"] += 1
        self.stats["replayed_frames"] += len(frames)
        snapshot, new = self._core_step_frame(frame, evidence)
        return snapshot, events + new

    def _advance(self, frame):
        base = self.core.get_snapshot()
        consumed = frame.step - 1 if frame.action is not None else -1
        return replace(base, as_of_observation=frame.step, last_consumed_action=consumed)

    # -- public interface -----------------------------------------------------------
    def step(self, step, observation, last_action=None):
        if type(step) is not int or step < 0 or step < self._step:
            raise ValueError("Out-of-order observation")
        frame = self._ingest(step, observation, last_action)
        if step == self._step:
            if frame.signature != self._signature:
                raise ValueError("Conflicting duplicate observation")
            return self._snapshot
        started = time.perf_counter()
        calls_before = self.stats["detector_calls"]
        self.stats["frames"] += 1
        external_gap = self._step >= 0 and step != self._step + 1
        self._step, self._signature = step, frame.signature
        reason = "gap" if external_gap else self._reason(frame)
        if external_gap:
            self._buffer.clear()
        if reason is None:
            self.stats["skipped"] += 1
            self._buffer.append(replace(frame, observation=_slim(frame.observation)))
            if len(self._buffer) > self.sparse.ring_buffer_frames:
                raise ValueError("Sparse writer buffer overflow: keyframe gap exceeds the ring buffer")
            self._snapshot = self._advance(frame)
            self.last_events, self.last_evidence = (), None
            self.last_signals = dict(new_events=[], visual_evidence=None,
                                     sparse=dict(mode="skip", processed_through=self._core_step,
                                                 detector_calls=0))
            return self._snapshot

        self.stats["processed"] += 1
        evidence = self._detect(frame)
        gap = self._buffer
        self._buffer = []
        mode, reasons = "dense", []
        if not gap:
            snapshot, events = self._core_step_frame(frame, evidence)
        else:
            previous = self.core.last_evidence
            cycle_config = getattr(self.core, "config", None)
            reasons = (["no_reference"] if previous is None
                       else probe_changes(previous, evidence, self.profile, self.sparse, cycle_config))
            if reasons:
                mode = "replay"
                snapshot, events = self._replay(gap, frame, evidence)
            else:
                checkpoint = copy.deepcopy(self.core)
                snapshot, events = self._declared_skip(frame, evidence)
                if events or not self._at_rest() or self._fingerprint(self.core) != self._fingerprint(checkpoint):
                    self.core = checkpoint
                    self.stats["rollbacks"] += 1
                    mode, reasons = "replay", ["post_check"]
                    snapshot, events = self._replay(gap, frame, evidence)
                else:
                    mode = "quiet_keyframe"
                    self.stats["quiet_keyframes"] += 1
        self._rest_run = self._rest_run + 1 if self._at_rest() else 0
        self._snapshot = snapshot
        self.last_events, self.last_evidence = tuple(events), evidence
        self.last_signals = {
            "new_events": [event.to_dict() for event in events],
            "visual_evidence": evidence.to_dict(),
            "sparse": dict(mode=mode, reason=reason, probe=reasons, gap_frames=len(gap),
                           processed_through=frame.step,
                           detector_calls=self.stats["detector_calls"] - calls_before,
                           step_ms=(time.perf_counter() - started) * 1000.0),
        }
        return snapshot


class SparseRelationWriterV1(SparseWriterMixin, RelationMemoryWriter):
    CORE = RelationMemoryWriter
    CORE_VERSION = RelationMemoryWriter.VERSION
    VERSION = "rgb_relations_sparse_v1"

    def _make_core(self, episode_id, profile, image_convention, config):
        return RelationMemoryWriter(episode_id, profile, None, image_convention)


class SparseBowlCycleWriterV3(SparseWriterMixin, BowlCycleWriterV3):
    CORE = BowlCycleWriterV3
    CORE_VERSION = BowlCycleWriterV3.VERSION
    VERSION = "rgb_relations_sparse_v3"

    def _make_core(self, episode_id, profile, image_convention, config):
        return BowlCycleWriterV3(episode_id, profile, None, image_convention, config=config)
