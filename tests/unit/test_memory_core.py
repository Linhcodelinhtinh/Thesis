"""Unit tests for V2 Memory Core (P1-P3).

Covers:
- models.py: schema validation, immutability, serialization
- store.py: episode isolation, time-ordered append, snapshots, reset
- updater.py: evidence-driven state transitions, invalidation
- retriever.py: task-conditioned retrieval, status filtering, bounded ranking
- text_memory.py: byte-for-byte pass-through, context formatting, oracle filtering
"""

import pytest

from src.memory.models import (
    EvidenceSource,
    MemoryEvent,
    MemoryQuery,
    MemoryStatus,
    ObjectMemory,
    RetrievedMemory,
    WorldMemorySnapshot,
)
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater
from src.memory.retriever import DeterministicMemoryRetriever
from src.interfaces.text_memory import TextMemoryRenderer


# -------------------------------------------------------------------------
# P1: Memory Data Model Tests
# -------------------------------------------------------------------------

class TestMemoryModels:
    def test_memory_event_valid(self):
        event = MemoryEvent(
            event_id="evt_01",
            episode_id="ep_01",
            step=5,
            timestamp=0.25,
            event_type="grasp",
            object_id="akita_black_bowl_1",
            semantic_label="black bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER,
            confidence=0.95,
            object_state_before="on_table",
            object_state_after="grasped",
        )
        assert event.event_id == "evt_01"
        assert event.validity == MemoryStatus.CONFIRMED
        d = event.to_dict()
        assert d["event_id"] == "evt_01"
        assert d["confidence"] == 0.95
        assert d["evidence_source"] == "observation_tracker"

    def test_memory_event_immutability(self):
        event = MemoryEvent(
            event_id="evt_01",
            episode_id="ep_01",
            step=5,
            timestamp=0.25,
            event_type="grasp",
            object_id="obj_1",
            semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER,
            confidence=0.9,
        )
        with pytest.raises(Exception):
            event.confidence = 0.5  # type: ignore

    def test_memory_event_invalid_inputs(self):
        # Empty IDs
        with pytest.raises(ValueError):
            MemoryEvent(
                event_id="",
                episode_id="ep_01",
                step=0,
                timestamp=0.0,
                event_type="grasp",
                object_id="obj",
                semantic_label="lbl",
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=1.0,
            )
        # Negative step
        with pytest.raises(ValueError):
            MemoryEvent(
                event_id="e1",
                episode_id="ep_01",
                step=-1,
                timestamp=0.0,
                event_type="grasp",
                object_id="obj",
                semantic_label="lbl",
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=1.0,
            )
        # Invalid confidence
        with pytest.raises(ValueError):
            MemoryEvent(
                event_id="e1",
                episode_id="ep_01",
                step=0,
                timestamp=0.0,
                event_type="grasp",
                object_id="obj",
                semantic_label="lbl",
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
                confidence=1.5,
            )
        # Invalid evidence source type
        with pytest.raises(TypeError):
            MemoryEvent(
                event_id="e1",
                episode_id="ep_01",
                step=0,
                timestamp=0.0,
                event_type="grasp",
                object_id="obj",
                semantic_label="lbl",
                evidence_source="raw_string",  # type: ignore
                confidence=0.8,
            )

    def test_object_memory_step_ordering(self):
        # first_seen_step cannot exceed last_updated_step
        with pytest.raises(ValueError):
            ObjectMemory(
                episode_id="ep_01",
                object_id="obj_1",
                semantic_label="bowl",
                state="on_table",
                first_seen_step=10,
                last_updated_step=5,
                confidence=0.9,
                evidence_source=EvidenceSource.OBSERVATION_TRACKER,
            )


# -------------------------------------------------------------------------
# P2: Store and Updater Tests
# -------------------------------------------------------------------------

class TestStoreAndUpdater:
    def test_store_isolation_and_reset(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        event = MemoryEvent(
            event_id="evt_01",
            episode_id="ep_01",
            step=1,
            timestamp=0.05,
            event_type="detected",
            object_id="obj_1",
            semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER,
            confidence=0.9,
            object_state_after="on_table",
        )
        store.append_event(event)
        obj = ObjectMemory(
            episode_id="ep_01",
            object_id="obj_1",
            semantic_label="bowl",
            state="on_table",
            first_seen_step=1,
            last_updated_step=1,
            confidence=0.9,
            evidence_source=EvidenceSource.OBSERVATION_TRACKER,
        )
        store.upsert_object(obj)

        snap = store.snapshot(as_of_step=1)
        assert len(snap.events) == 1
        assert len(snap.objects) == 1

        # Reset clears everything
        store.reset(episode_id="ep_02")
        assert store.episode_id == "ep_02"
        snap2 = store.snapshot(as_of_step=1)
        assert len(snap2.events) == 0
        assert len(snap2.objects) == 0

    def test_store_cross_episode_rejection(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        event_other = MemoryEvent(
            event_id="evt_other",
            episode_id="ep_OTHER",
            step=1,
            timestamp=0.05,
            event_type="detected",
            object_id="obj_1",
            semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER,
            confidence=0.9,
        )
        with pytest.raises(ValueError, match="does not match this store"):
            store.append_event(event_other)

    def test_store_snapshot_time_horizon(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        e1 = MemoryEvent(
            event_id="e1", episode_id="ep_01", step=10, timestamp=0.5,
            event_type="detect", object_id="o1", semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9
        )
        e2 = MemoryEvent(
            event_id="e2", episode_id="ep_01", step=50, timestamp=2.5,
            event_type="grasp", object_id="o1", semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9
        )
        store.append_event(e1)
        store.append_event(e2)

        # Snapshot at step 20 must NOT contain future event e2 (step 50)
        snap = store.snapshot(as_of_step=20)
        assert len(snap.events) == 1
        assert snap.events[0].event_id == "e1"

    def test_updater_state_transitions(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        updater = MemoryUpdater(store)

        # Step 0: CREATE
        e1 = MemoryEvent(
            event_id="e1", episode_id="ep_01", step=0, timestamp=0.0,
            event_type="initial_detect", object_id="bowl_1", semantic_label="black bowl",
            evidence_source=EvidenceSource.ORACLE_SIMULATOR, confidence=1.0,
            object_state_after="on_table"
        )
        updater.record(e1)
        obj = store.get_object("bowl_1")
        assert obj is not None
        assert obj.state == "on_table"
        assert obj.validity == MemoryStatus.CONFIRMED

        # Step 20: UPDATE (Grasp)
        e2 = MemoryEvent(
            event_id="e2", episode_id="ep_01", step=20, timestamp=1.0,
            event_type="grasp", object_id="bowl_1", semantic_label="black bowl",
            evidence_source=EvidenceSource.ORACLE_SIMULATOR, confidence=1.0,
            object_state_before="on_table", object_state_after="grasped"
        )
        updater.record(e2)
        obj2 = store.get_object("bowl_1")
        assert obj2 is not None
        assert obj2.state == "grasped"
        assert obj2.last_updated_step == 20

        # Step 30: INVALIDATE
        e3 = MemoryEvent(
            event_id="e3", episode_id="ep_01", step=30, timestamp=1.5,
            event_type="sight_lost", object_id="bowl_1", semantic_label="black bowl",
            evidence_source=EvidenceSource.ORACLE_SIMULATOR, confidence=0.8,
            validity=MemoryStatus.INVALIDATED
        )
        updater.record(e3)
        obj3 = store.get_object("bowl_1")
        assert obj3 is not None
        assert obj3.validity == MemoryStatus.INVALIDATED
        assert obj3.state == "grasped"  # retains historical state but invalidated


# -------------------------------------------------------------------------
# P3: Retrieval and Text Interface Tests
# -------------------------------------------------------------------------

class TestRetrieverAndRenderer:
    def test_retriever_entity_filtering(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        updater = MemoryUpdater(store)

        # Record two objects
        updater.record(MemoryEvent(
            event_id="e1", episode_id="ep_01", step=0, timestamp=0.0,
            event_type="init", object_id="bowl", semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
            object_state_after="on_table"
        ))
        updater.record(MemoryEvent(
            event_id="e2", episode_id="ep_01", step=0, timestamp=0.0,
            event_type="init", object_id="plate", semantic_label="plate",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
            object_state_after="on_table"
        ))
        updater.record(MemoryEvent(
            event_id="e3", episode_id="ep_01", step=0, timestamp=0.0,
            event_type="init", object_id="unrelated_item", semantic_label="item",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
            object_state_after="on_table"
        ))

        retriever = DeterministicMemoryRetriever()
        snap = store.snapshot(as_of_step=5)

        # Query only for bowl and plate
        query = MemoryQuery(
            episode_id="ep_01",
            as_of_step=5,
            entity_ids=("bowl", "plate"),
        )
        result = retriever.retrieve(snap, query)
        retrieved_ids = {o.object_id for o in result.objects}
        assert retrieved_ids == {"bowl", "plate"}
        assert "unrelated_item" not in retrieved_ids

    def test_retriever_hides_invalidated_and_stale(self):
        store = EpisodeMemoryStore(episode_id="ep_01")
        updater = MemoryUpdater(store)

        updater.record(MemoryEvent(
            event_id="e1", episode_id="ep_01", step=0, timestamp=0.0,
            event_type="init", object_id="bowl", semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
            object_state_after="on_table", validity=MemoryStatus.CONFIRMED
        ))
        # Mark invalidated
        updater.record(MemoryEvent(
            event_id="e2", episode_id="ep_01", step=10, timestamp=0.5,
            event_type="lost", object_id="bowl", semantic_label="bowl",
            evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
            validity=MemoryStatus.INVALIDATED
        ))

        retriever = DeterministicMemoryRetriever()
        snap = store.snapshot(as_of_step=15)
        query = MemoryQuery(episode_id="ep_01", as_of_step=15, entity_ids=("bowl",))
        result = retriever.retrieve(snap, query)
        assert len(result.objects) == 0

    def test_renderer_empty_pass_through(self):
        renderer = TextMemoryRenderer()
        instruction = "pick up the black bowl and place it on the plate"
        empty_retrieved = RetrievedMemory(episode_id="ep_01", as_of_step=0)

        rendered = renderer.render(instruction, empty_retrieved)
        # Strict byte-for-byte invariance
        assert rendered == instruction
        assert len(rendered) == len(instruction)

    def test_renderer_with_facts_and_oracle_filter(self):
        # Default renderer excludes oracle evidence
        renderer = TextMemoryRenderer()
        instruction = "pick up the soup"

        obj_oracle = ObjectMemory(
            episode_id="ep_01", object_id="soup", semantic_label="soup",
            state="grasped", first_seen_step=0, last_updated_step=10,
            confidence=1.0, evidence_source=EvidenceSource.ORACLE_SIMULATOR
        )
        mem = RetrievedMemory(episode_id="ep_01", as_of_step=10, objects=(obj_oracle,))

        # With default renderer, oracle is excluded -> returns unmodified instruction
        assert renderer.render(instruction, mem) == instruction

        # When opting in to oracle evidence:
        oracle_renderer = TextMemoryRenderer(allowed_sources=frozenset(EvidenceSource))
        rendered_oracle = oracle_renderer.render(instruction, mem)
        assert "[Episode memory]" in rendered_oracle
        assert "soup (soup) is grasped" in rendered_oracle
        assert "source=oracle_simulator" in rendered_oracle

    def test_renderer_context_limit(self):
        renderer = TextMemoryRenderer(
            max_context_chars=120,
            allowed_sources=frozenset(EvidenceSource)
        )
        instruction = "task instruction"
        objects = tuple(
            ObjectMemory(
                episode_id="ep_01", object_id=f"obj_{i}", semantic_label=f"object_{i}",
                state="on_table", first_seen_step=0, last_updated_step=1,
                confidence=0.9, evidence_source=EvidenceSource.OBSERVATION_TRACKER
            )
            for i in range(10)
        )
        mem = RetrievedMemory(episode_id="ep_01", as_of_step=1, objects=objects)
        rendered = renderer.render(instruction, mem)
        # Context block is strictly within max_context_chars
        context_block = rendered[len(instruction):].strip()
        assert len(context_block) <= 120

    def test_renderer_audit_truncation(self):
        renderer = TextMemoryRenderer(
            max_context_chars=140,
            allowed_sources=frozenset(EvidenceSource),
        )
        instruction = "task instruction"
        objects = tuple(
            ObjectMemory(
                episode_id="ep_01", object_id=f"obj_{i}", semantic_label=f"object_{i}",
                state="on_table", first_seen_step=0, last_updated_step=1,
                confidence=0.9, evidence_source=EvidenceSource.OBSERVATION_TRACKER
            )
            for i in range(10)
        )
        mem = RetrievedMemory(episode_id="ep_01", as_of_step=1, objects=objects)
        res = renderer.render_with_audit(instruction, mem)

        assert res.truncated is True
        assert res.total_facts_considered == 10
        assert res.included_facts_count < 10
        assert res.dropped_facts_count > 0
        assert len(res.dropped_facts) == res.dropped_facts_count
        assert res.rendered_context_chars <= 140
        assert res.text.startswith(instruction)

