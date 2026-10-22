"""Unit tests for Schema V2 episodic memory data models."""

import pytest

from src.memory.models import (
    CycleCounter,
    EvidenceSource,
    IdentityStatus,
    ObjectIdentity,
    RelationFact,
    RelationStatus,
    SchemaV2EpisodeMemory,
    V2MemoryEvent,
    VisibilityStatus,
)


def test_object_identity_invariants():
    obj = ObjectIdentity(
        object_id="bowl_A",
        origin_anchor="bowl initially on the left plate",
        identity_status=IdentityStatus.TRACKED,
        visibility=VisibilityStatus.VISIBLE,
    )
    assert obj.object_id == "bowl_A"
    assert obj.identity_status == IdentityStatus.TRACKED
    data = obj.to_dict()
    assert data["origin_anchor"] == "bowl initially on the left plate"
    assert data["visibility"] == "visible"

    with pytest.raises(ValueError):
        ObjectIdentity(object_id="", origin_anchor="plate")


def test_relation_fact_causal_invariants():
    fact = RelationFact(
        subject="bowl_A",
        predicate="on",
        object="plate_A",
        status=RelationStatus.CONFIRMED_BY_WRITER,
        occurred_at=10,
        available_at=12,
        evidence_ids=("obs_10", "obs_12"),
    )
    assert fact.available_at >= fact.occurred_at
    d = fact.to_dict()
    assert d["status"] == "confirmed_by_writer"
    assert d["occurred_at"] == 10
    assert d["available_at"] == 12

    # available_at cannot precede occurred_at (anti-causal error)
    with pytest.raises(ValueError, match="available_at cannot precede occurred_at"):
        RelationFact(
            subject="bowl_A",
            predicate="on",
            object="plate_A",
            status=RelationStatus.CONFIRMED_BY_WRITER,
            occurred_at=15,
            available_at=10,
        )


def test_cycle_counter_and_event():
    counter = CycleCounter(
        cycle_key="bowl_A_plate_A",
        object_id="bowl_A",
        support_id="plate_A",
        confirmed=2,
        in_progress=False,
        history_complete=True,
    )
    assert counter.confirmed == 2
    assert not counter.in_progress
    assert counter.to_dict()["confirmed"] == 2

    event = V2MemoryEvent(
        id="evt_1",
        type="returned_to_support",
        occurred_at=50,
        available_at=52,
        source=EvidenceSource.PREDICTED_RGB_TRACKER,
        probability=0.85,
    )
    assert event.source == EvidenceSource.PREDICTED_RGB_TRACKER
    assert event.to_dict()["source"] == "predicted_rgb_tracker"


def test_schema_v2_episode_memory_serialization():
    obj = ObjectIdentity(
        object_id="bowl_A",
        origin_anchor="bowl initially on left plate",
    )
    counter = CycleCounter(
        cycle_key="bowl_A_plate_A",
        object_id="bowl_A",
        support_id="plate_A",
        confirmed=1,
        in_progress=False,
    )
    mem = SchemaV2EpisodeMemory(
        episode_id="ep_001",
        as_of_observation=100,
        last_consumed_action=99,
        history_complete=True,
        objects={"bowl_A": obj},
        relations=(),
        cycles={"bowl_A_plate_A": counter},
        events=(),
    )
    data = mem.to_dict()
    assert data["schema_version"] == 2
    assert data["episode_id"] == "ep_001"
    assert data["as_of_observation"] == 100
    assert data["last_consumed_action"] == 99
    assert data["cycles"]["bowl_A_plate_A"]["confirmed"] == 1
