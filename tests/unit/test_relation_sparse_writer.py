"""The sparse-detector writer must keep the dense writer's facts and prompt text."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest

from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory import relation_sparse_writer
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_sparse_config import CONFIG_PATH, SparseWriterConfig
from src.memory.relation_sparse_writer import (SparseBowlCycleWriterV3, SparseRelationWriterV1,
                                               probe_changes, wrist_near)
from src.perception.relation_cycle_tracking import BowlCycleTracker
from src.perception.visual_evidence import CAMERAS, Detection2D
from tests.unit.test_bowl_cycle_writer import PLATE, PROFILE, REST
from tests.unit.test_relation_memory import observation

OPEN, CLOSED = .08, .02


def frame_obs(step, boxes, aperture):
    obs = observation(step, boxes)
    obs["robot0_gripper_qpos"] = [aperture / 2, aperture / 2]
    return obs


class ReplayDetector:
    """Hands out recorded evidence, one call per processed frame."""

    def __init__(self, evidence_by_step):
        self.evidence, self.calls, self.provenance = evidence_by_step, [], {}

    def detect(self, step, images, queries, *, pilot_crop=False, image_digests=None):
        self.calls.append(step)
        return self.evidence[step]


def box_at(rise=0., shift=0.):
    x1, y1, x2, y2 = REST
    return ("bowl", (x1 + shift, y1 - rise, x2 + shift, y2 - rise))


def timeline(*segments):
    """segments: (frames, rise, aperture[, shift]) -> list of (boxes, aperture)."""
    rows = []
    for count, rise, aperture, *shift in segments:
        rows += [([box_at(rise, *(shift or (0.,))), PLATE], aperture)] * count
    return rows


PICK_AND_PLACE = timeline(
    (60, 0., OPEN),
    (4, 0., CLOSED),
    *[(1, rise, CLOSED) for rise in (.03, .055, .07, .075, .04, .01)],
    (80, 0., OPEN),
)


def run_dense(rows, cls=BowlCycleWriterV3):
    writer = cls("dense", PROFILE, None, "libero_mem_refined")
    snapshots = []
    for step, (boxes, aperture) in enumerate(rows):
        snapshots.append(writer.step(step, frame_obs(step, boxes, aperture)))
    return writer, snapshots


def run_sparse(rows, sparse=None, cls=SparseBowlCycleWriterV3):
    evidence = {step: frame_obs(step, boxes, aperture)["visual_evidence"]
                for step, (boxes, aperture) in enumerate(rows)}
    detector = ReplayDetector(evidence)
    writer = cls("sparse", PROFILE, detector, "libero_mem_refined", sparse=sparse)
    snapshots = []
    for step, (boxes, aperture) in enumerate(rows):
        obs = frame_obs(step, boxes, aperture)
        obs.pop("visual_evidence")
        snapshots.append(writer.step(step, obs))
    return writer, snapshots, detector


def events(writer):
    return [(event.type, event.occurred_at) for event in writer.recent_events]


def renderer():
    return RelationMemoryRendererV3(object(), token_counter=lambda text: len(text.split()))


def text(snapshot):
    return renderer().render("lift the bowl", snapshot).text


# -- configuration -----------------------------------------------------------------

def test_shipped_config_is_valid_and_round_trips():
    config = SparseWriterConfig.load()
    assert json.loads(CONFIG_PATH.read_text(encoding="utf-8")) == config.to_dict()
    assert config.keyframe_interval == 8 and config.dense_prefix_steps >= 20


@pytest.mark.parametrize("change", [dict(keyframe_interval=0), dict(ring_buffer_frames=4),
                                    dict(rest_confirm_frames=0), dict(probe_score=1.5),
                                    dict(probe_motion_ratio=0), dict(dense_while_engaged=1)])
def test_config_rejects_unsafe_values(change):
    with pytest.raises(ValueError):
        replace(SparseWriterConfig.load(), **change)


# -- the prompt template is untouched ---------------------------------------------

def test_template_contract_is_unchanged():
    assert RelationMemoryRenderer.VERSION == "relation_text_v1"
    assert RelationMemoryRendererV3.VERSION == "relation_text_v2"
    assert relation_sparse_writer.__doc__ and not hasattr(relation_sparse_writer, "RelationMemoryRenderer")
    assert SparseBowlCycleWriterV3.CORE_VERSION == BowlCycleWriterV3.VERSION == "rgb_relations_v3"
    assert SparseRelationWriterV1.CORE_VERSION == RelationMemoryWriter.VERSION == "rgb_relations_v1"
    assert SparseBowlCycleWriterV3.VERSION != BowlCycleWriterV3.VERSION


def test_every_frame_schedule_is_the_dense_writer():
    rows = PICK_AND_PLACE[:120]
    dense, dense_snapshots = run_dense(rows)
    every = replace(SparseWriterConfig.load(), keyframe_interval=1, ring_buffer_frames=1)
    sparse, sparse_snapshots, detector = run_sparse(rows, every)
    assert detector.calls == list(range(len(rows)))
    for a, b in zip(dense_snapshots, sparse_snapshots):
        da, db = a.to_dict(), b.to_dict()
        for payload in (da, db):
            payload.pop("episode_id")
            payload["recent_events"] = [(e["type"], e["occurred_at"], e["available_at"])
                                        for e in payload["recent_events"]]
            for relation in payload["relations"].values():
                for key in ("last_event", "last_return", "departure"):
                    event = relation[key]
                    relation[key] = None if event is None else (event["type"], event["occurred_at"],
                                                                event["available_at"])
        da["evidence"] = sorted(v["step"] for v in da["evidence"].values())
        db["evidence"] = sorted(v["step"] for v in db["evidence"].values())
        assert da == db


# -- sparse behaviour on a pick-and-place -----------------------------------------

def test_events_text_and_savings_match_the_dense_writer():
    dense, dense_snapshots = run_dense(PICK_AND_PLACE)
    sparse, sparse_snapshots, detector = run_sparse(PICK_AND_PLACE)

    assert events(dense) == events(sparse)
    assert [kind for kind, _ in events(sparse)] == ["initial_relation", "departure", "return"]
    assert text(dense_snapshots[-1]) == text(sparse_snapshots[-1]) == (
        "lift the bowl Memory: bowl: 1 observed returns.")
    assert len(detector.calls) < .5 * len(PICK_AND_PLACE)
    assert sparse.stats["frames"] == len(PICK_AND_PLACE)
    assert sparse.detector_calls == len(detector.calls) == len(sparse.detected_steps)
    assert len(set(detector.calls)) == len(detector.calls)


def test_skipped_frames_are_not_history_gaps():
    dense, dense_snapshots = run_dense(PICK_AND_PLACE)
    _, sparse_snapshots, detector = run_sparse(PICK_AND_PLACE)
    skipped = [s for s in range(25, 60) if s not in detector.calls]
    assert skipped
    for step in skipped:
        assert "History incomplete" not in text(sparse_snapshots[step])
        assert text(sparse_snapshots[step]) == text(dense_snapshots[step])


def test_skipped_frames_advance_the_observation_index_only():
    _, snapshots, detector = run_sparse(PICK_AND_PLACE)
    step = next(s for s in range(25, 60) if s not in detector.calls)
    last = max(s for s in detector.calls if s < step)
    assert snapshots[step].as_of_observation == step
    assert snapshots[step].relations == snapshots[last].relations
    assert snapshots[step].evidence == snapshots[last].evidence
    assert snapshots[step].recent_events == snapshots[last].recent_events


class Guarded(dict):
    """An observation whose attached evidence must not be touched."""

    def get(self, key, default=None):
        if key == "visual_evidence":
            raise AssertionError("evidence of an unprocessed frame was read")
        return super().get(key, default)

    def __getitem__(self, key):
        if key == "visual_evidence":
            raise AssertionError("evidence of an unprocessed frame was read")
        return super().__getitem__(key)


def test_evidence_of_unprocessed_frames_is_never_read():
    rows = PICK_AND_PLACE
    _, _, probe = run_sparse(rows)
    processed = set(probe.calls)  # replayed frames are served by the detector, not the observation
    evidence = {s: frame_obs(s, boxes, aperture)["visual_evidence"] for s, (boxes, aperture) in enumerate(rows)}
    writer = SparseBowlCycleWriterV3("guard", PROFILE, ReplayDetector(evidence), "libero_mem_refined")
    for step, (boxes, aperture) in enumerate(rows):
        obs = frame_obs(step, boxes, aperture)
        obs = dict(obs) if step in processed else Guarded(obs)
        writer.step(step, obs)
    assert writer.stats["skipped"] > 0
    assert sorted(writer.detected_steps) == sorted(probe.calls)


def test_closed_gripper_frames_are_all_processed():
    _, _, detector = run_sparse(PICK_AND_PLACE)
    assert set(range(60, 70)) <= set(detector.calls)


def test_idle_displacement_between_keyframes_is_replayed_with_true_timing():
    rows = timeline((60, 0., OPEN), (3, 0., OPEN, .03), (50, 0., OPEN, .0))
    rows = rows[:61] + [([box_at(.2), PLATE], OPEN)] * 5 + rows[66:]
    dense, _ = run_dense(rows)
    sparse, _, detector = run_sparse(rows)
    assert events(dense) == events(sparse)
    assert sparse.stats["replays"] >= 1
    assert {61, 62, 63, 64} <= set(detector.calls) or sparse.stats["replayed_frames"] >= 1


def test_quiet_keyframe_that_changes_core_state_is_rolled_back_and_replayed(monkeypatch):
    rows = timeline((60, 0., OPEN))
    rows += [([PLATE], OPEN)] * 30  # bowl vanishes while the gripper stays open
    rows += timeline((60, 0., OPEN))
    dense, _ = run_dense(rows)
    monkeypatch.setattr(relation_sparse_writer, "probe_changes", lambda *args, **kwargs: [])
    sparse, snapshots, detector = run_sparse(rows)
    assert sparse.stats["rollbacks"] >= 1
    assert events(sparse) == events(dense)
    assert snapshots[-1].relations["bowl_1"].history_complete == dense.get_snapshot().relations["bowl_1"].history_complete


# -- probe --------------------------------------------------------------------------

def evidence_for(boxes, step=0, wrist=()):
    base = frame_obs(step, boxes, OPEN)["visual_evidence"]
    views = dict(base.views)
    views[CAMERAS[1]] = tuple(Detection2D(label, box, score, f"w{step}:{i}")
                              for i, (label, box, score) in enumerate(wrist))
    return replace(base, views=views)


def test_probe_is_quiet_on_identical_boxes_and_loud_on_changes():
    config = SparseWriterConfig.load()
    still = evidence_for([box_at(), PLATE])
    assert probe_changes(still, evidence_for([box_at(), PLATE], 1), PROFILE, config) == []
    assert probe_changes(still, evidence_for([PLATE], 1), PROFILE, config) == ["count:bowl"]
    assert "motion:bowl" in probe_changes(still, evidence_for([box_at(.1), PLATE], 1), PROFILE, config)
    assert "motion:plate" in probe_changes(
        still, evidence_for([box_at(), ("plate", (.3, .5, .5, .62))], 1), PROFILE, config)
    assert "size:bowl" in probe_changes(
        still, evidence_for([("bowl", (.25, .45, .40, .57)), PLATE], 1), PROFILE, config)


def test_probe_flags_a_change_in_the_wrist_cue_only():
    sparse, cycle = SparseWriterConfig.load(), BowlCycleConfig.load()
    near = [("bowl", (.3, .3, .7, .7), .6)]
    still = evidence_for([box_at(), PLATE], 0)
    hand = evidence_for([box_at(), PLATE], 1, wrist=near)
    assert probe_changes(still, hand, PROFILE, sparse, cycle) == ["wrist"]
    assert probe_changes(hand, evidence_for([box_at(), PLATE], 2, wrist=near), PROFILE, sparse, cycle) == []


def test_wrist_cue_matches_the_tracker():
    cycle = BowlCycleConfig.load()
    cases = [[], [("bowl", (.3, .3, .7, .7), .6)], [("bowl", (.0, .0, .1, .1), .6)],
             [("bowl", (.3, .3, .7, .7), .1)], [("bowl", (.3, .3, .7, .7), .6), ("bowl", (.31, .3, .71, .7), .55)]]
    for wrist in cases:
        evidence = evidence_for([box_at(), PLATE], 0, wrist=wrist)
        tracker = BowlCycleTracker(PROFILE, config=cycle)
        tracker.observe(evidence)
        assert wrist_near(evidence, cycle) == tracker.wrist_near


# -- input contract -----------------------------------------------------------------

def test_input_checks_match_the_dense_writer():
    rows = PICK_AND_PLACE[:30]
    _, _, detector = run_sparse(rows)
    writer = SparseBowlCycleWriterV3("io", PROFILE, ReplayDetector(
        {s: frame_obs(s, b, a)["visual_evidence"] for s, (b, a) in enumerate(rows)}), "libero_mem_refined")
    obs = frame_obs(0, *rows[0])
    obs.pop("visual_evidence")
    first = writer.step(0, obs)
    assert writer.step(0, dict(obs)) is first
    changed = dict(obs, robot0_gripper_qpos=[.0, .0])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        writer.step(0, changed)
    with pytest.raises(ValueError, match="action preceding"):
        writer.step(1, dict(obs), np.zeros(6))
    writer.step(1, frame_obs(1, *rows[1]), np.zeros(7))
    with pytest.raises(ValueError, match="Out-of-order"):
        writer.step(0, obs)


def test_v1_wrapper_keeps_the_v1_contract():
    rows = PICK_AND_PLACE
    dense, _ = run_dense(rows, RelationMemoryWriter)
    sparse, snapshots, detector = run_sparse(rows, cls=SparseRelationWriterV1)
    assert sparse.VERSION == "rgb_relations_sparse_v1" and isinstance(sparse, RelationMemoryWriter)
    assert len(detector.calls) < len(rows)
    assert [e[0] for e in events(sparse)] == [e[0] for e in events(dense)]
    assert sparse.reducers is sparse.core.reducers


# -- randomized equivalence with the dense writer -------------------------------------

def random_rows(rng):
    """Resting, picks, drifts, occlusions and flicker, in random order and length."""
    rows = [([box_at(0.), PLATE], OPEN)] * rng.randint(25, 70)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["pick", "pick", "bump", "hidden", "flicker", "idle"])
        if kind == "pick":
            rows += [([box_at(0.), PLATE], CLOSED)] * rng.randint(2, 8)
            peak = rng.choice([.05, .08, .12])
            rows += [([box_at(peak * (i + 1) / 4), PLATE], CLOSED) for i in range(4)]
            rows += [([box_at(peak), PLATE], CLOSED)] * rng.randint(0, 10)
            if rng.random() < .8:
                rows += [([box_at(rise), PLATE], CLOSED) for rise in (peak * .5, peak * .1)]
                rows += [([box_at(0.), PLATE], OPEN)] * rng.randint(3, 12)
            else:
                rows += [([box_at(peak), PLATE], OPEN)] * rng.randint(3, 12)
        elif kind == "bump":
            rows += [([box_at(rng.choice([.02, .05, .1]), rng.choice([0, .02, .05])), PLATE], OPEN)] * rng.randint(2, 12)
        elif kind == "hidden":
            rows += [([PLATE], OPEN)] * rng.randint(1, 30)
        elif kind == "flicker":
            for _ in range(rng.randint(1, 4)):
                rows += [([box_at(0.), PLATE], OPEN)] * rng.randint(1, 10) + [([PLATE], OPEN)] * rng.randint(1, 3)
        else:
            rows += [([box_at(0.), PLATE], OPEN)] * rng.randint(5, 30)
        rows += [([box_at(0.), PLATE], OPEN)] * rng.randint(5, 40)
    return rows


def without_qualifier(snapshot):
    return text(snapshot).replace(" History incomplete.", "")


def test_random_episodes_keep_the_dense_events_and_counts():
    calls = frames = 0
    for seed in range(60):
        rows = random_rows(random.Random(seed))
        dense, dense_snapshots = run_dense(rows)
        sparse, sparse_snapshots, detector = run_sparse(rows)
        assert [(e.type, e.occurred_at, e.available_at) for e in dense.recent_events] == \
            [(e.type, e.occurred_at, e.available_at) for e in sparse.recent_events], seed
        assert text(dense_snapshots[-1]) == text(sparse_snapshots[-1]), seed
        # Counts, "Lift observed." and the remaining facts agree at every step. Only the
        # "History incomplete." qualifier may differ, because frames the sparse writer never
        # looked at are unobserved rather than missing detections.
        for step, (a, b) in enumerate(zip(dense_snapshots, sparse_snapshots)):
            assert without_qualifier(a) == without_qualifier(b), (seed, step)
        assert sparse.stats["frames"] == len(rows)
        calls, frames = calls + len(detector.calls), frames + len(rows)
    # Synthetic mix with many engaged and not-at-rest frames: a bound, not a real-data rate.
    assert calls < .65 * frames
