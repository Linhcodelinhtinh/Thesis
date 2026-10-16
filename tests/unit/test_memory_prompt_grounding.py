"""Targeted verification for Step 1 Memory Prompt Logic & Grounded Contract.

Verifies:
1. Task-declared entities are NOT fabricated into visual detections at episode start.
2. Grasp is not blindly attributed to target_entity_names[0] in multi-target tasks.
3. Empty gripper closing in free air does NOT fire false grasp or false slip.
4. Superseded recovery events (slip followed by newer release or placement) are never rendered.
5. No memory suffix is added when there is no useful episode history (prompt remains unmodified).
"""

import numpy as np
import pytest

from src.interfaces.text_memory import TextMemoryRenderer
from src.memory.models import (
    EvidenceSource,
    MemoryEvent,
    MemoryStatus,
    ObjectMemory,
    RetrievedMemory,
)
from src.memory.observation_writer import ObservationMemoryWriter
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater


def test_no_synthetic_visual_detection_at_episode_start():
    """Requirement 1: Task-declared entities must not become visual detections."""
    store = EpisodeMemoryStore(episode_id="ep_test_start")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("alphabet_soup", "tomato_sauce"),
        goal_container_name="basket",
    )
    writer.on_episode_start(episode_id="ep_test_start", env=None, obs={}, step=0, timestamp=0.0)

    # Memory store must have zero fabricated visual detections
    snap = store.snapshot(as_of_step=0)
    assert len(snap.objects) == 0
    assert len(snap.events) == 0
    assert store.get_object("alphabet_soup") is None
    assert store.get_object("tomato_sauce") is None
    assert store.get_object("basket") is None


def test_no_blind_first_entity_grasp_in_multitarget():
    """Requirement 2: Do NOT attribute grasp to target_entity_names[0] without identity grounding."""
    store = EpisodeMemoryStore(episode_id="ep_multi")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("alphabet_soup", "tomato_sauce"),
        goal_container_name="basket",
    )
    writer.on_episode_start(episode_id="ep_multi", env=None, obs={}, step=0, timestamp=0.0)

    # Simulate persistent obstruction near table
    obs_blocked = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.85]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),  # gap = 0.04m
    }
    writer.on_step(step=10, timestamp=0.5, action=np.array([0, 0, 0, 0, 0, 0, 1.0]), next_obs=obs_blocked)
    events = writer.on_step(step=11, timestamp=0.55, action=np.array([0, 0, 0, 0, 0, 0, 1.0]), next_obs=obs_blocked)

    assert len(events) == 1
    # Must NOT assign to alphabet_soup!
    assert events[0].object_id != "alphabet_soup"
    assert events[0].object_id == "unidentified_object"
    assert events[0].validity == MemoryStatus.UNCERTAIN


def test_free_air_closure_does_not_trigger_grasp_or_slip():
    """Requirement 3: An empty closing gripper sweeps through gap [0.015, 0.065] without triggering grasp."""
    store = EpisodeMemoryStore(episode_id="ep_free_air")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl",),
        goal_container_name="plate",
    )
    writer.on_episode_start(episode_id="ep_free_air", env=None, obs={}, step=0, timestamp=0.0)

    # Empty gripper closing sweeps fast: 0.08 -> 0.06 -> 0.04 -> 0.02 -> 0.00
    trajectory = [0.08, 0.06, 0.04, 0.02, 0.005]
    all_events = []
    for step_i, gap in enumerate(trajectory):
        obs = {
            "robot0_eef_pos": np.array([0.0, 0.0, 0.85]),
            "robot0_gripper_qpos": np.array([gap / 2.0, gap / 2.0]),
        }
        evs = writer.on_step(
            step=step_i + 1,
            timestamp=(step_i + 1) * 0.05,
            action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
            next_obs=obs,
        )
        all_events.extend(evs)

    # Zero grasps and zero slips should be emitted during free-air closure
    assert len(all_events) == 0
    assert store.get_object("black_bowl") is None


def test_superseded_recovery_events_not_rendered():
    """Requirement 4: A slip event must NOT be resurrected if a newer release or placement occurred."""
    renderer = TextMemoryRenderer(compact_mode=True)
    instruction = "put both the cream cheese and the butter in the basket"

    # Event 1: Slip at step 74
    ev_slip = MemoryEvent(
        event_id="e1", episode_id="ep1", step=74, timestamp=3.7,
        event_type="observed_slip", object_id="cream_cheese", semantic_label="cream cheese",
        evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.75,
        object_state_before="grasped", object_state_after="contact_lost",
        validity=MemoryStatus.CONFIRMED,
    )
    # Event 2: Placement at step 187
    ev_place = MemoryEvent(
        event_id="e2", episode_id="ep1", step=187, timestamp=9.35,
        event_type="place", object_id="cream_cheese", semantic_label="cream cheese",
        evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=1.0,
        object_state_before="grasped", object_state_after="in_basket",
        validity=MemoryStatus.CONFIRMED,
    )
    obj_cream_cheese = ObjectMemory(
        episode_id="ep1", object_id="cream_cheese", semantic_label="cream cheese",
        state="in_basket", first_seen_step=74, last_updated_step=187,
        confidence=1.0, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
    )
    mem = RetrievedMemory(
        episode_id="ep1",
        as_of_step=190,
        objects=(obj_cream_cheese,),
        events=(ev_place, ev_slip),
    )

    res = renderer.render_with_audit(instruction, mem)
    # Must NOT contain "Lost contact"
    assert "Lost contact" not in res.text
    # Must contain completed subtask
    assert "Cream cheese is already in the basket" in res.text


def test_no_suffix_when_no_useful_history():
    """Requirement 5: Initial states (on_table, detected, target_destination) must NOT add suffix."""
    renderer = TextMemoryRenderer(compact_mode=True)
    instruction = "put both the alphabet soup and the tomato sauce in the basket"

    # Memory only has unmanipulated resting states
    obj1 = ObjectMemory(
        episode_id="ep1", object_id="alphabet_soup", semantic_label="alphabet soup",
        state="on_table", first_seen_step=0, last_updated_step=0,
        confidence=0.8, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
    )
    obj2 = ObjectMemory(
        episode_id="ep1", object_id="basket", semantic_label="basket",
        state="target_destination", first_seen_step=0, last_updated_step=0,
        confidence=0.8, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
    )
    mem = RetrievedMemory(
        episode_id="ep1",
        as_of_step=0,
        objects=(obj1, obj2),
        events=(),
    )

    res = renderer.render_with_audit(instruction, mem)
    # Output must be the exact raw instruction without any "Memory: ..."
    assert res.text == instruction
    assert "Memory:" not in res.text
    assert res.included_facts_count == 0
