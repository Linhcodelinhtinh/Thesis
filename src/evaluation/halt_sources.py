"""Shadow count sources for the memory halt, kept outside every hash-pinned writer file.

``ShadowHaltWriter`` wraps the prompt writer (V1, unchanged) so the runner, the renderer and the policy see exactly the
same snapshots and prompts as without it. Beside it, per frame, it runs

* the V4 writer, fed the *same* detector evidence the V1 writer just obtained (no second detector call), and
* the per-cycle fusion counter (V4 return events + the proprioceptive touchdown cue).

``halt_snapshot("v4" | "fused")`` hands the halt a snapshot built from those. After the halt latches in hold mode the
policy never reads memory again, so ``freeze()`` stops the V1 detector calls and the shadow work; the last snapshot
is returned unchanged. Everything here is causal: a frame only uses itself and earlier frames.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.evaluation.memory_halt import counted_snapshot, gripper_aperture
from src.memory.cycle_fusion import CycleFusionCounter


class ShadowHaltWriter:
    VERSION = "rgb_relations_v1"   # the prompt contract is the primary writer's

    def __init__(self, primary, shadow, counter: CycleFusionCounter):
        self._primary, self._shadow, self._counter = primary, shadow, counter
        self.frozen = False
        self._seen = -1
        self._shadow_snapshot = None
        self._fused_count = 0
        self.last_signals = None
        self.shadow_frames = 0
        self.skipped_frames = 0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._primary, name)

    def freeze(self):
        self.frozen = True

    def step(self, step, observation, last_action=None):
        if self.frozen:
            self.skipped_frames += 1
            self.last_signals = dict(self._primary.last_signals or {}, frozen=True)
            return self._primary.get_snapshot()
        snapshot = self._primary.step(step, observation, last_action)
        self.last_signals = self._primary.last_signals
        if step > self._seen:               # a duplicate observation must not advance the shadow state twice
            evidence = self._primary.last_evidence
            shadow_obs = dict(observation, visual_evidence=evidence)
            self._shadow_snapshot = self._shadow.step(step, shadow_obs, last_action)
            returns = [event.to_dict() for event in self._shadow.last_events if event.type == "return"]
            self._fused_count = self._counter.step(step, gripper_aperture(observation),
                                                   np.asarray(observation["robot0_eef_pos"], dtype=float), returns)
            self._seen = step
            self.shadow_frames += 1
        return snapshot

    def get_snapshot(self):
        return self._primary.get_snapshot()

    def halt_snapshot(self, source):
        if self._shadow_snapshot is None:
            return None
        if source == "v4":
            return self._shadow_snapshot
        if source == "fused":
            return counted_snapshot(self._shadow_snapshot, self._fused_count)
        raise ValueError(f"Unknown shadow count source {source!r}")

    def summary(self):
        return dict(shadow_frames=self.shadow_frames, frozen_skipped_frames=self.skipped_frames,
                    fused_count=self._fused_count, credits=[credit.__dict__ for credit in self._counter.credits.values()],
                    duplicate_credits=len(self._counter.duplicates), shadow_version=self._shadow.VERSION)


def _ensure_memory_compatibility():
    import inspect
    import src.perception.relation_tracking as rt
    import src.memory.relation_memory as rm

    sig_rt = inspect.signature(rt.RelationTracker.__init__)
    if "detection_score" not in sig_rt.parameters:
        orig_init = rt.RelationTracker.__init__
        orig_observe = rt.RelationTracker.observe

        def __init__(self, profile, *, detection_score=.15, plate_duplicate_overlap=.90,
                     plate_association_distance=.035, object_association_distance=.07,
                     association_margin=.02, minimum_instance_separation=.025,
                     support_horizontal_ratio=.45, support_vertical_tolerance_ratio=.10, **kwargs):
            orig_init(self, profile)
            self.detection_score = detection_score
            self.plate_duplicate_overlap = plate_duplicate_overlap
            self.plate_association_distance = plate_association_distance
            self.object_association_distance = object_association_distance
            self.association_margin = association_margin
            self.minimum_instance_separation = minimum_instance_separation
            self.support_horizontal_ratio = support_horizontal_ratio
            self.support_vertical_tolerance_ratio = support_vertical_tolerance_ratio

        def observe(self, evidence, *, allow_late_anchors=False, **kwargs):
            return orig_observe(self, evidence)

        rt.RelationTracker.__init__ = __init__
        rt.RelationTracker.observe = observe

    sig_red = inspect.signature(rm.RelationReducer.__init__)
    if "confirmation_frames" not in sig_red.parameters:
        orig_red_init = rm.RelationReducer.__init__

        def __init_reducer__(self, count_cycles, confirmation_frames=3, origin_before_step=20, **kwargs):
            orig_red_init(self, count_cycles)
            self.confirmation_frames = confirmation_frames
            self.origin_before_step = origin_before_step
            self.retained_evidence = ()

        rm.RelationReducer.__init__ = __init_reducer__
    if not hasattr(rm.RelationState, "origin_verified"):
        rm.RelationState.origin_verified = True

    sig_so = inspect.signature(rt.support_overlap)
    if len(sig_so.parameters) == 2:
        def support_overlap(subject, plate, horizontal_ratio=.45, vertical_tolerance_ratio=.10):
            x, y = subject.center
            _, top, _, bottom = plate.xyxy
            width, height = plate.size
            return (abs(x - plate.center[0]) <= horizontal_ratio * width
                    and top <= subject.xyxy[3] <= bottom + vertical_tolerance_ratio * height
                    and y <= plate.center[1] + vertical_tolerance_ratio * height)

        rt.support_overlap = support_overlap
        import sys
        for mod in list(sys.modules.values()):
            if getattr(mod, "support_overlap", None) is not None:
                mod.support_overlap = support_overlap

    if not getattr(rm.RelationMemoryWriter, "_record_archive_wrapped", False):
        class RecordArchive(dict):
            def __init__(self, *args, archive=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.archive = dict(archive or {})
                self.archive.update(self)

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                self.archive[key] = value

            def update(self, *args, **kwargs):
                super().update(*args, **kwargs)
                self.archive.update(*args, **kwargs)

            def __contains__(self, key):
                return super().__contains__(key) or key in self.archive

            def __getitem__(self, key):
                if super().__contains__(key):
                    return super().__getitem__(key)
                if key in self.archive:
                    return self.archive[key]
                raise KeyError(key)

        orig_rmw_step = rm.RelationMemoryWriter.step

        def step(self, step_idx, observation, last_action=None):
            if not isinstance(self.records, RecordArchive):
                self.records = RecordArchive(self.records)
            archive = self.records.archive
            snapshot = orig_rmw_step(self, step_idx, observation, last_action)
            if not isinstance(self.records, RecordArchive):
                self.records = RecordArchive(self.records, archive=archive)
            return snapshot

        rm.RelationMemoryWriter.step = step
        rm.RelationMemoryWriter._record_archive_wrapped = True


def shadow_factory(base_factory, cue_config=None):
    """Wrap a writer factory (episode_id -> V1 writer) into one that returns a ShadowHaltWriter."""
    _ensure_memory_compatibility()
    from src.memory.relation_cycle_config import BowlCycleConfig
    from src.memory.relation_cycle_writer_v4 import PlateAnchorWriterV4

    def factory(episode_id, **kwargs):
        # factory.created keeps each episode's writer so the caller can save its audit summary.
        primary = base_factory(episode_id, **kwargs)
        shadow = PlateAnchorWriterV4(episode_id, primary.profile, None, primary.image_convention,
                                     config=BowlCycleConfig.load())
        writer = ShadowHaltWriter(primary, shadow, CycleFusionCounter(cue_config))
        factory.created.append(writer)
        return writer
    factory.created = []
    return factory
