"""Regression coverage for causal return recognition and the existing text schema."""
from dataclasses import replace

import pytest

from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.relation_cycle_writer import BowlCycleWriter
from src.perception.relation_cycle_tracking import MAX_MISSING
from tests.unit.test_relation_memory import observation, renderer


PLATE = ("plate", (.2, .50, .4, .62))
REST = (.25, .45, .35, .57)
PROFILE = MemoryProfile("bowl", 1, "plate", 3, "on", True)


@pytest.fixture
def scenario(request):
    writer = BowlCycleWriter("test", PROFILE, None, "libero_mem_refined")
    plate = ("plate", (.24, .46, .35, .58)) if getattr(request, "param", False) else PLATE

    def frame(*, rise=0., bottom_shift=None, released=True, hidden=False, low_score=False, extra=None, alias=False):
        step = writer.get_snapshot().as_of_observation + 1 if writer.get_snapshot() else 0
        x1, y1, x2, y2 = REST
        bowl = ("bowl", (x1, y1 - rise, x2, y2 - (rise if bottom_shift is None else bottom_shift)))
        if alias:
            bowl = ("bowl", plate[1])
        boxes = [plate] if hidden else [bowl, plate]
        if extra:
            boxes.append(("bowl", extra))
        obs = observation(step, boxes, released)
        if low_score:
            evidence = obs["visual_evidence"]
            agent = "agentview_image"
            weak, plate_detection = evidence.views[agent]
            obs["visual_evidence"] = replace(evidence, views={**evidence.views, agent: (
                replace(weak, score=.16), plate_detection,
            )})
        snapshot = writer.step(step, obs)
        return snapshot, snapshot.relations["bowl_1"]

    for _ in range(4):
        frame()
    return frame


def lift(frame):
    for rise in (.03, .055, .07, .075):
        result = frame(rise=rise, released=False)
    return result


def settle(frame):
    for rise in (.04, .01):
        frame(rise=rise, released=False)
    for _ in range(5):
        result = frame()
    return result


def test_lift_candidate_survives_short_loss_but_weak_boxes_do_not_confirm(scenario):
    frame = scenario
    frame(rise=.03, released=False)
    frame(rise=.06, released=False)
    frame(rise=.07, released=False, low_score=True)
    snapshot, state = frame(hidden=True, released=False)
    assert state.candidate_frames == 1 and not state.armed
    assert renderer().render("lift the bowl", snapshot).text == "lift the bowl Memory: bowl: 0 observed returns. History incomplete."
    assert not frame(rise=.075, released=False)[1].armed
    _, state = frame(rise=.08, released=False)
    assert state.armed
    assert state.departure.available_at == 9
    assert {"6:0", "7:0"}.isdisjoint(state.departure.evidence_ids)

    snapshot, state = settle(frame)
    assert state.returns == 1 and not state.armed
    assert renderer().render("lift the bowl", snapshot).text == "lift the bowl Memory: bowl: 1 observed returns. History incomplete."
    assert all(eid in snapshot.evidence for eid in state.last_return.evidence_ids)
    assert state.last_return.details["paired_departure"] is not None
    assert settle(frame)[1].returns == 1
    assert lift(frame)[1].armed
    assert settle(frame)[1].returns == 2


@pytest.mark.parametrize("open_gripper", [False, True])
def test_box_flicker_cannot_create_return(scenario, open_gripper):
    frame = scenario
    for _ in range(4):
        # The lower edge jumps up while the top stays fixed.
        for _ in range(4):
            frame(bottom_shift=.052, released=open_gripper)
        assert settle(frame)[1].returns == 0
    for rise in (.03, .055, .07, .075):
        _, state = frame(rise=rise, released=True)
    assert not state.armed
    assert settle(frame)[1].returns == 0


@pytest.mark.parametrize("scenario", [True], indirect=True)
def test_near_duplicate_plate_detection_cannot_commit_return(scenario):
    frame = scenario
    assert lift(frame)[1].armed
    for rise in (.04, .01):
        frame(rise=rise, released=False)
    for _ in range(4):
        _, state = frame(alias=True)
        assert state.returns == 0 and state.status == "conflicted"
    for _ in range(4):
        _, state = frame()
    assert state.returns == 1


@pytest.mark.parametrize("interruption", ["ambiguity", "long_loss"])
def test_reacquisition_cannot_commit_a_stale_departure(scenario, interruption):
    frame = scenario
    assert lift(frame)[1].armed
    for i in range(MAX_MISSING + 1):
        if interruption == "ambiguity":
            _, state = frame(rise=.075, released=False, extra=(.26, .375, .36, .495))
            if i < MAX_MISSING:
                assert state.status == "conflicted" and state.armed
        else:
            _, state = frame(hidden=True, released=False)
    assert not state.armed
    assert settle(frame)[1].returns == 0
