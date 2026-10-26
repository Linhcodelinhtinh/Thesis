"""Behavioral/adversarial fixtures, not benchmark demonstrations or scores."""
from dataclasses import replace
import numpy as np
import pytest

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.event_writer import LiberoMemEventWriter
from src.memory.models import EvidenceSource
from src.perception.object_tracker import CausalObjectTracker
from src.perception.visual_evidence import CAMERAS, Detection2D, VisualEvidence, image_digest, rgb_views


def frame(step, lift=0., gap=.08, missing=False, support=True, ambiguous=False, x=0.):
    images = {key: np.full((16, 16, 3), step % 255, dtype=np.uint8) for key in CAMERAS}
    bowl = Detection2D("bowl", (.40+x, .50-lift, .55+x, .64-lift), .9, f"{step}:bowl")
    plate = Detection2D("plate", (.38, .60, .57, .69), .9, f"{step}:plate")
    detections = ([] if missing else [bowl]) + ([plate] if support else [])
    if ambiguous:
        detections.append(Detection2D("bowl", (.415+x, .50-lift, .565+x, .64-lift), .9, f"{step}:other"))
    evidence = VisualEvidence(step, {CAMERAS[0]: tuple(detections), CAMERAS[1]: ()},
                              {key: image_digest(im) for key, im in images.items()}, "unit-fixture", "unit-v1")
    return dict(images, robot0_eef_pos=np.array([0., 0., 1.20]),
                robot0_gripper_qpos=np.array([gap/2, -gap/2]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                visual_evidence=evidence)


def rest(writer, start, n=4, **kwargs):
    for step in range(start, start+n):
        writer.step(step, frame(step, **kwargs))
    return start+n


def cycle(writer, start):
    end = rest(writer, start, n=4, lift=.15, gap=.005)
    return rest(writer, end)


def test_three_returns_with_provenance_and_causal_confirmation():
    writer = LiberoMemEventWriter("T3")
    assert not writer.get_snapshot().relations
    assert not writer.get_snapshot().cycles
    step = rest(writer, 0)
    assert writer.confirmed_cycles == 0
    for expected in range(1, 4):
        step = cycle(writer, step)
        assert writer.confirmed_cycles == expected
    frozen = writer.get_snapshot()
    step = rest(writer, step, n=20)
    assert writer.confirmed_cycles == 3  # no extra count while resting
    events = [ev for ev in frozen.events if ev.type == "returned_to_support"]
    assert len(events) == 3
    assert all(ev.available_at - ev.occurred_at == 2 for ev in events)
    assert all(len(ev.evidence_ids) >= 6 for ev in events)
    assert all(ev.source == EvidenceSource.PREDICTED_RGB_TRACKER for ev in events)
    with pytest.raises(TypeError):
        events[0].details["cycle_count"] = 99
    assert frozen.as_of_observation < writer.get_snapshot().as_of_observation


def test_high_eef_and_tiny_gap_do_not_prevent_visually_observed_cycle():
    writer = LiberoMemEventWriter("height")
    step = rest(writer, 0)
    cycle(writer, step)
    assert writer.confirmed_cycles == 1  # EEF z=1.2 throughout, closed gap=.005


def test_gripper_commands_and_eef_motion_without_bowl_motion_do_not_count():
    writer = LiberoMemEventWriter("no_fake_grasp")
    for step in range(40):
        obs = frame(step, gap=.005 if step % 10 < 5 else .08)
        obs["robot0_eef_pos"][2] = 1.1 + step * .03
        writer.step(step, obs, np.array([0., 0., 0., 0., 0., 0., (-1.)**step]) if step else None)
    assert writer.confirmed_cycles == 0
    assert not any(ev.type == "departed_support" for ev in writer.get_snapshot().events)


def test_release_command_without_measured_opening_never_confirms_return():
    writer = LiberoMemEventWriter("command")
    step = rest(writer, 0)
    step = rest(writer, step, lift=.15, gap=.005)
    for t in range(step, step+10):
        writer.step(t, frame(t, gap=.005), np.array([0., 0., 0., 0., 0., 0., -1.]))
    assert writer.confirmed_cycles == 0
    assert writer.in_progress


@pytest.mark.parametrize("bad", [{"missing": True}, {"support": False}, {"ambiguous": True}])
def test_unknown_never_becomes_return_and_lower_bound_survives(bad):
    writer = LiberoMemEventWriter("gap")
    step = rest(writer, 0)
    step = cycle(writer, step)
    step = rest(writer, step, lift=.15, gap=.005)
    step = rest(writer, step, n=5, **bad)
    snapshot = writer.get_snapshot()
    assert writer.confirmed_cycles == 1
    assert not snapshot.history_complete
    assert snapshot.cycles["bowl_A_plate_A"].uncertain_count == 1
    text = LiberoMemTextRenderer().render("lift the bowl", snapshot).text
    assert "at least 1" in text and "history incomplete" in text
    step = rest(writer, step)
    assert writer.confirmed_cycles == 2
    assert not writer.history_complete  # reacquisition cannot restore missing history


def test_jitter_and_interrupted_dwell_cannot_count():
    writer = LiberoMemEventWriter("jitter")
    step = rest(writer, 0)
    for t in range(step, step+20):
        writer.step(t, frame(t, lift=.15 if t % 2 else 0.))
    assert writer.confirmed_cycles == 0
    assert not writer.in_progress


def test_wrong_plate_return_and_late_origin_are_rejected():
    writer = LiberoMemEventWriter("wrong_plate")
    step = rest(writer, 0)
    step = rest(writer, step, lift=.15)
    rest(writer, step, n=6, x=.25)
    assert writer.confirmed_cycles == 0
    late = LiberoMemEventWriter("late")
    rest(late, 30)
    assert not late.get_snapshot().cycles
    assert late.last_signals["reason"] == "original_support_not_observed_at_start"


def test_gap_resets_dwell_and_repeat_steps_raise_without_mutation():
    writer = LiberoMemEventWriter("time")
    rest(writer, 0)
    writer.step(4, frame(4, lift=.15))
    writer.step(10, frame(10, lift=.15))
    assert not writer.in_progress
    assert not writer.history_complete
    snap = writer.get_snapshot().to_dict()
    with pytest.raises(ValueError, match="out of order"):
        writer.step(10, frame(10))
    assert writer.get_snapshot().to_dict() == snap


@pytest.mark.parametrize("change", ["step", "image", "source", "raw_annotation"])
def test_untrusted_or_misaligned_evidence_fails(change):
    obs = frame(0)
    if change == "step":
        obs["visual_evidence"] = replace(obs["visual_evidence"], step=1)
    elif change == "image":
        obs[CAMERAS[0]][0, 0, 0] = 99
    elif change == "source":
        obs["visual_evidence"] = replace(obs["visual_evidence"], source=EvidenceSource.HUMAN_ANNOTATION)
    else:
        obs.pop("visual_evidence")
        obs["detected_objects"] = {"bowl_A": [0, 0, .85]}
        obs["oracle_signals"] = {"is_on_support": True}
    with pytest.raises((ValueError, RuntimeError)):
        LiberoMemEventWriter("bad").step(0, obs)


def test_offline_and_online_orientation_produce_identical_memory():
    offline = LiberoMemEventWriter("same")
    online = LiberoMemEventWriter("same", tracker=CausalObjectTracker(image_convention="libero_raw"))
    for step in range(16):
        obs = frame(step, lift=.15 if 4 <= step < 8 else 0.)
        # Use asymmetric pixels so horizontal/vertical confusion cannot pass.
        for camera in CAMERAS:
            obs[camera][0, 1, 0] = 91
        obs["visual_evidence"] = replace(obs["visual_evidence"], image_hashes={c: image_digest(obs[c]) for c in CAMERAS})
        raw = dict(obs, **{c: obs[c][::-1].copy() for c in CAMERAS})
        assert offline.step(step, obs).to_dict() == online.step(step, raw).to_dict()


def test_renderer_exact_off_and_rejects_counter_without_events():
    writer = LiberoMemEventWriter("render")
    step = rest(writer, 0)
    cycle(writer, step)
    snapshot = writer.get_snapshot()
    renderer = LiberoMemTextRenderer()
    original = "  lift the bowl  \n"
    assert renderer.render(original, snapshot, enabled=False).text == original
    assert "1 returns" in renderer.render(original, snapshot).text
    forged = replace(snapshot, events=())
    assert renderer.render(original, forged).text == original
    oracle = replace(snapshot, events=tuple(replace(ev, source=EvidenceSource.ORACLE_SIMULATOR) for ev in snapshot.events))
    assert renderer.render(original, oracle).text == original


def test_actual_tokenizer_contract_and_budget():
    class RecordingTokenizer:
        def encode(self, text, add_special_tokens):
            assert text.endswith("\n") and add_special_tokens
            return list(range(len(text.split()) + 2))
    renderer = LiberoMemTextRenderer(max_total_tokens=3, tokenizer=RecordingTokenizer(), strict_tokenizer=True)
    original = "a long original instruction must remain unchanged"
    result = renderer.render(original, None, False)
    assert result.text == original and result.tokenizer_truncated


def test_nested_plate_proposals_do_not_create_a_second_support_identity():
    writer = LiberoMemEventWriter("nested")
    for step in range(16):
        obs = frame(step, lift=.15 if 4 <= step < 8 else 0.)
        ev = obs["visual_evidence"]
        nested = Detection2D("plate", (.42, .65, .52, .68), .6, f"{step}:inner_disk")
        obs["visual_evidence"] = replace(ev, views=dict(ev.views, **{CAMERAS[0]: ev.views[CAMERAS[0]] + (nested,)}))
        writer.step(step, obs)
    assert writer.confirmed_cycles == 1
    assert writer.history_complete


def test_short_detection_gap_can_compare_last_measurement_but_cannot_supply_current_evidence():
    tracker = CausalObjectTracker()
    tracker.update(0, frame(0))
    lost = tracker.update(1, frame(1, missing=True))
    assert not lost["has_grounding_evidence"] and not lost["stable"]
    reacquired = tracker.update(2, frame(2))
    assert reacquired["stable"] and reacquired["has_grounding_evidence"]
    assert not tracker.update(8, frame(8))["stable"]


def test_unknown_initial_departure_does_not_become_exact_zero_later():
    writer = LiberoMemEventWriter("initial_gap")
    writer.step(0, frame(0))
    rest(writer, 1, lift=.15)
    rest(writer, 5)
    snapshot = writer.get_snapshot()
    assert not snapshot.history_complete
    assert "count unknown" in LiberoMemTextRenderer().render("lift bowl", snapshot).text


def test_weak_spurious_bowl_proposal_does_not_hide_strong_observed_target():
    writer = LiberoMemEventWriter("weak_proposal")
    for step in range(16):
        obs = frame(step, lift=.15 if 4 <= step < 8 else 0.)
        ev = obs["visual_evidence"]
        weak = Detection2D("bowl", (.7, .5, .8, .64), .2, f"{step}:weak")
        obs["visual_evidence"] = replace(ev, views=dict(ev.views, **{CAMERAS[0]: ev.views[CAMERAS[0]] + (weak,)}))
        writer.step(step, obs)
    assert writer.confirmed_cycles == 1


def test_rgb_identity_continuity_resolves_separated_hypotheses_with_close_scores():
    tracker = CausalObjectTracker()
    tracker.update(0, frame(0))
    obs = frame(1)
    ev = obs["visual_evidence"]
    same = replace(ev.views[CAMERAS[0]][0], score=.3)
    composite = Detection2D("bowl", (.38, .50, .57, .69), .25, "1:composite")
    obs["visual_evidence"] = replace(ev, views=dict(ev.views, **{CAMERAS[0]: (same, composite, ev.views[CAMERAS[0]][1])}))
    signals = tracker.update(1, obs)
    assert signals["has_grounding_evidence"]
    assert signals["identity_association"] == "measured_rgb_continuity"
    assert "0:bowl" in signals["evidence_ids"] and "1:bowl" in signals["evidence_ids"]
    # A stale track cannot disambiguate observations much later.
    later = frame(8)
    ev = later["visual_evidence"]
    later["visual_evidence"] = replace(ev, views=dict(ev.views, **{CAMERAS[0]: (
        replace(same, evidence_id="8:bowl"), replace(composite, evidence_id="8:composite"), ev.views[CAMERAS[0]][1])}))
    assert not tracker.update(8, later)["has_grounding_evidence"]


def test_low_score_identity_jump_does_not_reassign_the_track():
    tracker = CausalObjectTracker()
    tracker.update(0, frame(0))
    obs = frame(1, x=.3)
    ev = obs["visual_evidence"]
    obs["visual_evidence"] = replace(ev, views=dict(ev.views, **{CAMERAS[0]: (
        replace(ev.views[CAMERAS[0]][0], score=.2), ev.views[CAMERAS[0]][1])}))
    signals = tracker.update(1, obs)
    assert not signals["has_grounding_evidence"]
    assert signals["reason"] == "target_identity_jump"
