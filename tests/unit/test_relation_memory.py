"""Check the relation memory contract with controlled input traces."""
from dataclasses import replace

import numpy as np
import pytest

from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.models import EvidenceSource
from src.memory.relation_memory import RelationMemoryWriter
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.visual_evidence import CAMERAS, Detection2D, VisualEvidence, image_digest


def observation(step, boxes, released=True):
    rgb = {camera: np.full((4, 4, 3), step % 256, dtype=np.uint8) for camera in CAMERAS}
    evidence = VisualEvidence(
        step, {CAMERAS[0]: tuple(Detection2D(label, box, .8, f"{step}:{i}")
                               for i, (label, box) in enumerate(boxes)), CAMERAS[1]: ()},
        {camera: image_digest(frame) for camera, frame in rgb.items()}, MODEL_ID, REVISION)
    return dict(rgb, visual_evidence=evidence, robot0_gripper_qpos=[.03, .03] if released else [.01, .01])


def add_frames(writer, step, boxes, count=4, released=True):
    for i in range(step, step + count):
        obs = observation(i, boxes, released)
        action = None if i == 0 else np.zeros(7)
        snapshot = writer.step(i, obs, action)
        assert writer.step(i, obs, action) is snapshot
    return snapshot


def renderer(budget=48):
    return RelationMemoryRenderer(object(), budget, token_counter=lambda text: len(text.split()))


@pytest.mark.parametrize("label", ["bowl", "bottle"])
def test_cycle_pairs_measured_transitions_across_an_occluded_interval(label):
    writer = RelationMemoryWriter("ep", MemoryProfile(label, 1, "plate", 3, "on", True), None, "libero_mem_refined")
    plate = ("plate", (.2, .50, .4, .62))
    rest = [(label, (.25, .45, .35, .57)), plate]
    lift = [(label, (.25, .28, .35, .4)), plate]
    before = add_frames(writer, 0, rest)
    add_frames(writer, 4, lift, released=False)
    after = add_frames(writer, 8, rest)
    assert after.relations[f"{label}_1"].returns == 1
    add_frames(writer, 12, rest)
    add_frames(writer, 16, lift, released=False)
    add_frames(writer, 20, [plate], count=1)
    uncertain = add_frames(writer, 21, rest, count=5)
    assert uncertain.relations[f"{label}_1"].returns == 2
    assert before.relations[f"{label}_1"].returns == 0
    text = renderer().render(f"lift the {label}", uncertain).text
    assert "2 observed returns" in text and "History incomplete" in text
    assert renderer(5).render(f"lift the {label}", uncertain).text == f"lift the {label}"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        writer.step(25, observation(25, [plate]), np.zeros(7))


def test_containment_remains_a_qualified_observation_when_item_is_missing():
    profile = MemoryProfile("cream cheese", 1, "basket", 2, "in")
    writer = RelationMemoryWriter("ep", profile, None, "libero_mem_refined")
    baskets = [("basket", (.1, .4, .3, .63)), ("basket", (.6, .4, .8, .63))]
    full = baskets + [("cream cheese", (.15, .48, .22, .56))]
    for t, x in enumerate((.32, .29, .26, .23, .20, .17)):
        writer.step(t, observation(t, baskets + [("cream cheese", (x, .48, x + .07, .56))]),
                    None if t == 0 else np.zeros(7))
    seen = add_frames(writer, 6, full)
    instruction = "put the cream cheese in the nearest basket and move the empty basket"
    assert "visually in left basket" in renderer().render(instruction, seen).text
    hidden = add_frames(writer, 10, baskets)
    result = renderer().render(instruction, hidden)
    assert "last seen in left basket, now uncertain" in result.text
    assert "empty" not in result.facts_included[0]
    add_frames(writer, 14, baskets + [("cream cheese", (.42, .48, .49, .56))])
    hidden_again = add_frames(writer, 18, baskets)
    assert renderer().render(instruction, hidden_again).dependency_failures
    lost_identity = add_frames(writer, 22, [baskets[1]])
    assert renderer().render(instruction, lost_identity).dependency_failures


def test_conflicting_supports_cannot_produce_a_relation():
    writer = RelationMemoryWriter("ep", MemoryProfile("bowl", 2, "plate", 3, "on"), None, "libero_mem_refined")
    plates = [("plate", (.15, .5, .35, .65)), ("plate", (.2, .5, .4, .65)), ("plate", (.6, .5, .8, .65))]
    bowls = [("bowl", (.23, .43, .32, .59)), ("bowl", (.65, .43, .75, .59))]
    snap = add_frames(writer, 0, plates + bowls)
    assert snap.relations["bowl_1"].status == "conflicted"
    assert snap.relations["bowl_1"].last_event is None
    assert snap.relations["bowl_2"].reference == "plate_3"


def test_dropped_frame_cannot_complete_a_departure_pair():
    writer = RelationMemoryWriter("ep", MemoryProfile("bottle", 1, "plate", 3, "on", True), None, "libero_mem_refined")
    plate = ("plate", (.2, .50, .4, .62))
    rest = [("bottle", (.25, .45, .35, .57)), plate]
    lift = [("bottle", (.25, .28, .35, .4)), plate]
    add_frames(writer, 0, rest)
    add_frames(writer, 4, lift, released=False)
    writer.step(9, observation(9, rest))
    end = add_frames(writer, 10, rest)
    assert end.relations["bottle_1"].returns == 0
    assert not end.relations["bottle_1"].history_complete


@pytest.mark.parametrize("violation", ["future", "hash", "camera", "oracle"])
def test_writer_rejects_noncausal_evidence(violation):
    writer = RelationMemoryWriter("ep", MemoryProfile("bowl", 1, "plate", 3, "on", True), None, "libero_mem_refined")
    obs = observation(0, [])
    evidence = obs["visual_evidence"]
    change = {
        "future": dict(step=1), "hash": dict(image_hashes={key: "wrong" for key in CAMERAS}),
        "camera": dict(views={CAMERAS[0]: ()}), "oracle": dict(source=EvidenceSource.ORACLE_SIMULATOR),
    }[violation]
    obs["visual_evidence"] = replace(evidence, **change)
    with pytest.raises(ValueError):
        writer.step(0, obs)
