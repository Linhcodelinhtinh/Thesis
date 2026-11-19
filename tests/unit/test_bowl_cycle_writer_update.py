"""Regressions and acceptance cases for the planned T1/T3 writer update."""

from dataclasses import replace

import numpy as np
import pytest

from src.memory import relation_memory
from src.memory.relation_cycle_config import BowlCycleConfig
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.perception import visual_evidence
from tests.unit.test_bowl_cycle_writer import PLATE, PROFILE, REST, lift, settle
from tests.unit.test_relation_memory import observation


CONFIGURED_MAX_MISSING = BowlCycleConfig.load().max_missing_steps


def new_writer():
    return BowlCycleWriterV3("update-test", PROFILE, None, "libero_mem_refined")


def v3_renderer():
    return RelationMemoryRendererV3(object(), token_counter=lambda text: len(text.split()))


def render_facts(snapshot):
    return v3_renderer().render("lift the bowl", snapshot).facts_included


@pytest.fixture
def v3_scenario(request):
    writer = new_writer()
    plate = ("plate", (.24, .46, .35, .58)) if getattr(request, "param", False) else PLATE

    def frame(*, rise=0., released=True, hidden=False, extra=None, alias=False):
        previous = writer.get_snapshot()
        step = previous.as_of_observation + 1 if previous else 0
        x1, y1, x2, y2 = REST
        bowl = ("bowl", (x1, y1 - rise, x2, y2 - rise))
        if alias:
            bowl = ("bowl", plate[1])
        boxes = [plate] if hidden else [bowl, plate]
        if extra:
            boxes.append(("bowl", extra))
        snapshot = writer.step(step, observation(step, boxes, released))
        return snapshot, snapshot.relations["bowl_1"]

    for _ in range(4):
        frame()
    return frame


def observe_cycle(writer, steps, shift=0.):
    rises = (.03, .055, .07, .075, .04, .01, 0., 0., 0., 0., 0.)
    x1, y1, x2, y2 = REST
    left, top, right, bottom = PLATE[1]
    plate = ("plate", (left + shift, top, right + shift, bottom))
    for sample, (step, rise) in enumerate(zip(steps, rises, strict=True)):
        bowl = ("bowl", (x1 + shift, y1 - rise, x2 + shift, y2 - rise))
        snapshot = writer.step(step, observation(step, [bowl, plate], released=sample >= 6))
    return snapshot.relations["bowl_1"]


def test_idle_occlusion_can_be_resolved_before_next_lift(v3_scenario):
    frame = v3_scenario
    frame(hidden=True)  # The measured gripper stays open through the flicker.
    for _ in range(4):
        frame()
    lift(frame)
    snapshot, state = settle(frame)

    assert state.returns == 1
    assert render_facts(snapshot) == ("bowl: 1 observed returns.",)


@pytest.mark.parametrize("missing_frames", [1, CONFIGURED_MAX_MISSING],
                         ids=["one-frame", "at-configured-limit"])
def test_short_occlusion_during_lift_retains_observed_departure(v3_scenario, missing_frames):
    frame = v3_scenario
    _, lifted = lift(frame)
    assert lifted.armed
    for _ in range(missing_frames):
        _, hidden = frame(released=False, hidden=True)
        assert hidden.departure is lifted.departure

    snapshot, returned = settle(frame)
    assert returned.returns == 1
    assert returned.last_return.details["paired_departure"] == lifted.departure.id
    assert render_facts(snapshot) == ("bowl: 1 observed returns. History incomplete.",)


def test_tentative_lift_expires_across_repeated_occlusions(v3_scenario):
    frame = v3_scenario
    frame(rise=.03, released=False)
    frame(rise=.055, released=False)
    long_gap = CONFIGURED_MAX_MISSING + 1
    for rise in (.07, .075):
        for _ in range(long_gap):
            frame(released=False, hidden=True)
        _, state = frame(rise=rise, released=False)

    assert not state.armed
    assert settle(frame)[1].returns == 0


def test_first_plate_after_warmup_can_ground_idle_origin():
    writer = new_writer()
    for step in range(40):
        boxes = [("bowl", REST)] if step < 35 else [("bowl", REST), PLATE]
        snapshot = writer.step(step, observation(step, boxes))

    state = snapshot.relations["bowl_1"]
    assert state.status == "confirmed"
    assert state.reference == "plate_1"
    assert state.last_event.type == "initial_relation"
    returned = observe_cycle(writer, range(40, 51))
    assert returned.returns == 1
    assert returned.last_return.details["reference"] == "plate_1"

    unseen = new_writer()
    for step in range(40):
        boxes = [] if step < 35 else [("bowl", REST), PLATE]
        snapshot = unseen.step(step, observation(step, boxes))

    assert not snapshot.relations["bowl_1"].origin_verified
    assert render_facts(snapshot) == ()


def test_late_rest_after_gripper_closure_cannot_present_origin_count():
    writer = new_writer()
    for step in range(30):
        snapshot = writer.step(step, observation(step, [("bowl", REST), PLATE], released=step >= 20))

    assert all("0 observed returns" not in fact for fact in render_facts(snapshot))


@pytest.mark.parametrize("shift", [0., .012], ids=["unshifted", "common-horizontal-shift"])
def test_small_common_box_shift_preserves_lift_and_return(shift):
    writer = new_writer()
    for step in range(4):
        writer.step(step, observation(step, [("bowl", REST), PLATE]))
    state = observe_cycle(writer, range(4, 15), shift)
    assert state.returns == 1
    assert state.last_return.details["paired_departure"] is not None


def test_shrinking_bowl_box_does_not_start_a_lift():
    config = replace(BowlCycleConfig(), lift_center_rise_ratio=.10,
                     lift_bottom_rise_ratio=.20, confirmation_frames=2)
    writer = BowlCycleWriterV3("shrink", PROFILE, None, "libero_mem_refined", config=config)
    x1, top, x2, bottom = REST
    shrunk = (x1, top, x2, bottom - .028)
    for step in range(15):
        box = REST if step < 5 else shrunk
        snapshot = writer.step(step, observation(step, [("bowl", box), PLATE],
                                                 released=not 5 <= step < 10))
        assert not snapshot.relations["bowl_1"].armed
        assert snapshot.relations["bowl_1"].returns == 0


def test_open_gripper_at_lift_pose_cannot_commit_return():
    config = replace(BowlCycleConfig(), lift_center_rise_ratio=.10,
                     lift_bottom_rise_ratio=.20, return_offset_ratio=.55,
                     confirmation_frames=2)
    writer = BowlCycleWriterV3("hover", PROFILE, None, "libero_mem_refined", config=config)
    x1, top, x2, bottom = REST
    lifted_pose = (x1, top - .03, x2, bottom - .03)
    for step in range(13):
        box = REST if step < 5 else lifted_pose
        snapshot = writer.step(step, observation(step, [("bowl", box), PLATE],
                                                 released=not 5 <= step < 9))
        state = snapshot.relations["bowl_1"]
        assert state.returns == 0
    lifted = state.departure
    assert lifted is not None
    for step in range(13, 18):
        snapshot = writer.step(step, observation(step, [("bowl", REST), PLATE]))
    returned = snapshot.relations["bowl_1"]
    assert returned.returns == 1
    assert returned.last_return.details["paired_departure"] == lifted.id
    assert all(eid in snapshot.evidence for eid in returned.last_return.evidence_ids)


def test_weak_but_coherent_lift_counts_and_can_repeat():
    config = replace(BowlCycleConfig(), lift_center_rise_ratio=.10,
                     lift_bottom_rise_ratio=.20, confirmation_frames=2)
    writer = BowlCycleWriterV3("shallow", PROFILE, None, "libero_mem_refined", config=config)
    rises = (0.,) * 5 + (0., .03, .03, .03, .015, 0., 0., 0.) * 2
    x1, top, x2, bottom = REST
    for step, rise in enumerate(rises):
        box = (x1, top - rise, x2, bottom - rise)
        released = step < 5 or (step - 5) % 8 >= 4
        snapshot = writer.step(step, observation(step, [("bowl", box), PLATE], released))
    assert snapshot.relations["bowl_1"].returns == 2


@pytest.mark.parametrize("skipped", [False, True], ids=["contiguous", "one-missing-step"])
def test_second_cycle_requires_observation_continuity(skipped):
    writer = new_writer()
    for step in range(4):
        writer.step(step, observation(step, [("bowl", REST), PLATE]))
    assert observe_cycle(writer, range(4, 15)).returns == 1

    # The second lift has three measured samples at steps 16 through 18.
    # When testing a skipped observation, omit step 19 before the descent.
    descent_step = 20 if skipped else 19
    state = observe_cycle(writer, (*range(15, 19), *range(descent_step, descent_step + 7)))
    assert state.returns == (1 if skipped else 2)
    assert not state.armed


@pytest.mark.parametrize("changed", ["gripper", "action", "detection"])
def test_conflicting_duplicate_cannot_replace_the_committed_snapshot(changed):
    writer = new_writer()
    boxes = [("bowl", REST), PLATE]
    writer.step(0, observation(0, boxes))
    obs = observation(1, boxes)
    action = np.zeros(7)
    snapshot = writer.step(1, obs, action)
    duplicate = dict(obs)
    duplicate_action = action.copy()
    if changed == "gripper":
        duplicate["robot0_gripper_qpos"] = [.01, .01]
    elif changed == "action":
        duplicate_action[0] = .1
    else:
        evidence = obs["visual_evidence"]
        camera = visual_evidence.CAMERAS[0]
        bowl, plate = evidence.views[camera]
        duplicate["visual_evidence"] = replace(
            evidence, views={**evidence.views, camera: (replace(bowl, score=.7), plate)},
        )

    with pytest.raises(ValueError, match="Conflicting duplicate observation"):
        writer.step(1, duplicate, duplicate_action)
    assert writer.get_snapshot() is snapshot
    assert writer.step(1, obs, action) is snapshot


def test_cached_and_detector_steps_digest_each_rgb_camera_once(monkeypatch):
    obs = observation(0, [("bowl", REST), PLATE])
    image_digest = visual_evidence.image_digest
    digests = []

    def counted_digest(image):
        digest = image_digest(image)
        digests.append(digest)
        return digest

    monkeypatch.setattr(relation_memory, "image_digest", counted_digest)
    monkeypatch.setattr(visual_evidence, "image_digest", counted_digest)
    cached_writer = new_writer()
    cached_writer.step(0, obs)
    assert len(digests) == len(visual_evidence.CAMERAS)

    digests.clear()
    evidence = obs.pop("visual_evidence")

    class CachedDetector:
        def detect(self, step, images, queries, *, pilot_crop, image_digests):
            assert step == evidence.step
            assert set(image_digests) == set(images) == set(visual_evidence.CAMERAS)
            assert queries == PROFILE.queries and pilot_crop
            assert image_digests == evidence.image_hashes
            return evidence

    detector_writer = BowlCycleWriterV3("detector-test", PROFILE, CachedDetector(), "libero_mem_refined")
    detector_writer.step(0, obs)

    assert len(digests) == len(visual_evidence.CAMERAS)
