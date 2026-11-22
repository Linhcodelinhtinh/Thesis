"""The writer comparison harness, on tiny synthetic timelines (unit fixtures, not benchmark data)."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.compare_memory_writers import (HEADLINE_TOLERANCE, RecordedDetector, SYNTHETIC, agreement, bootstrap,
                                            compare_hits, exact_mcnemar, hit_sets, parse_evidence, prf,
                                            returns_of, run_writer, score, summarize, venn, writer_specs)
from tests.unit.test_relation_sparse_writer import PICK_AND_PLACE, box_at, frame_obs
from tests.unit.test_bowl_cycle_writer import PLATE


def timeline_inputs(rows=PICK_AND_PLACE, altered=None):
    evidence, grip = [], []
    for step, (boxes, aperture) in enumerate(rows):
        if altered is not None and step == altered:
            boxes = [box_at(shift=.3), PLATE]
        observation = frame_obs(step, boxes, aperture)
        evidence.append(observation["visual_evidence"].to_dict())
        grip.append(observation["robot0_gripper_qpos"])
    return parse_evidence(evidence), grip


def run(name, evidence, grip, task="T1"):
    return run_writer(writer_specs()[name], name, "ep", task, evidence, grip)


def test_statistics_helpers():
    assert exact_mcnemar(0, 0) == 1.0 and exact_mcnemar(3, 3) == 1.0
    assert exact_mcnemar(5, 0) == pytest.approx(2 / 32)
    assert prf(3, 1, 1) == (.75, .75, .75) and prf(0, 0, 0) == (None, None, None)
    counts = [[1, 0, 0]] * 30 + [[0, 0, 1]] * 10
    interval = bootstrap(counts)
    assert interval["recall"][0] <= .75 <= interval["recall"][1] and interval["precision"] == [1.0, 1.0]
    assert bootstrap(counts) == interval  # fixed seed
    difference = bootstrap(counts, counts)
    assert difference["recall"] == [0.0, 0.0]


def fake_run(returns, queries=(), calls=10, frames=100, final=0):
    events = [dict(type="return", occurred_at=step - 3, available_at=step, published=step) for step in returns]
    return dict(events=events, queries=[list(query) for query in queries], calls=calls, frames=frames,
                final_returns=final, millis=[1.0, 2.0], schedule={}, stats=None)


def test_scoring_counts_matches_false_positives_and_query_counts():
    references = {"a": dict(task="T1", frames=100, milestones=[40, 80]), "b": dict(task="T1", frames=100, milestones=[])}
    runs = {"a": fake_run([48, 105], queries=[(0, 0, True, False, "x"), (48, 1, True, False, "y"), (120, 1, False, False, "y")],
                          final=1),
            "b": fake_run([10], queries=[(0, 0, True, False, "x")])}
    scored = score(runs, references, HEADLINE_TOLERANCE)
    assert scored["a"]["hits"] == [40] and scored["a"]["false_positive"] == 1 and scored["a"]["false_negative"] == 1
    summary = summarize(runs, references, scored, HEADLINE_TOLERANCE)
    assert (summary["true_positive"], summary["false_positive"], summary["false_negative"]) == (1, 2, 1)
    assert summary["confirm_delay"]["median"] == 8 and summary["recall_by_ordinal"]["1"]["captured"] == 1
    assert summary["recall_by_ordinal"]["2"]["captured"] == 0
    # Query 48 is within the tolerance of milestone 40 and is skipped; the others are scored.
    assert summary["query_frames_outside_tolerance"] == 3 and summary["query_count_correct_rate"] == pytest.approx(2 / 3)
    assert summary["history_incomplete_query_rate"] == pytest.approx(1 / 4)
    assert summary["final_count_exact_episodes"] == 1 and summary["detector_calls_per_frame"] == .1


def test_hit_sets_venn_and_discordance():
    references = {"a": dict(task="T1", frames=10, milestones=[3, 7])}
    first = score({"a": fake_run([3, 7])}, references, 1)
    second = score({"a": fake_run([7])}, references, 1)
    assert compare_hits(hit_sets(first, references), hit_sets(second, references), references) == dict(
        both=1, only_first=1, only_second=0, neither=0, exact_p=1.0)
    assert venn(dict(x=first, y=second), ["x", "y"], references) == {"x+y": 1, "x": 1}


def test_agreement_counts_identical_events_and_prompts():
    queries = [(0, 0, True, False, "a"), (8, 1, True, False, "b")]
    dense = {"e": fake_run([5], queries)}
    other = {"e": fake_run([5], [queries[0], (8, 1, True, False, "c")])}
    result = agreement(other, dense)
    assert result["identical_return_steps"] == 1 and result["identical_prompt_text"] == 1
    assert result["identical_count"] == 2 and result["query_frames"] == 2


def test_recorded_detector_refuses_evidence_of_another_frame():
    evidence, _ = timeline_inputs()
    fed = {step: replace(item, image_hashes=SYNTHETIC[step % 256]) for step, item in enumerate(evidence[:3])}
    detector = RecordedDetector(fed)
    assert detector.detect(1, None, None, image_digests=SYNTHETIC[1]) is fed[1] and detector.steps == [1]
    with pytest.raises(ValueError, match="does not belong"):
        detector.detect(1, None, None, image_digests=SYNTHETIC[2])


def test_dense_and_sparse_writers_capture_the_same_return_on_a_clean_pick_and_place():
    evidence, grip = timeline_inputs()
    dense = run("v3", evidence, grip)
    sparse = run("sparse_v3_k8", evidence, grip)
    assert len(returns_of(dense)) == 1 and returns_of(sparse) == returns_of(dense)
    assert dense["calls"] == len(evidence) and sparse["calls"] < dense["calls"]
    agree = agreement({"ep": sparse}, {"ep": dense})
    assert agree["identical_event_sequences"] == 1 and agree["identical_prompt_text"] == agree["query_frames"]
    assert sparse["final_returns"] == dense["final_returns"] == 1


def test_a_sparse_writer_never_reads_the_evidence_of_a_frame_it_skips():
    clean, grip = timeline_inputs()
    # Frame 30 is neither in the dense prefix nor a keyframe, so a sparse writer at rest skips it.
    altered, _ = timeline_inputs(altered=30)
    assert run("sparse_v3_k8", clean, grip)["events"] == run("sparse_v3_k8", altered, grip)["events"]
    assert run("sparse_v3_k8", altered, grip)["calls"] == run("sparse_v3_k8", clean, grip)["calls"]


def test_a_subsampled_run_renumbers_frames_and_maps_events_back():
    evidence, grip = timeline_inputs()
    result = run("subsampled_v3_k4", evidence, grip)
    assert result["calls"] == len(range(0, len(evidence), 4))
    assert all(event["available_at"] % 4 == 0 and event["occurred_at"] % 4 == 0 for event in result["events"])
