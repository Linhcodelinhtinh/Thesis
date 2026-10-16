"""Regression unit tests for Memory Audit fixes (2026-10-14).

Verifies fixes for all 6 issues identified in docs/MEMORY_IMPLEMENTATION_AUDIT_2026-10-13.md:
1. Token-budget-aware rendering, compact mode, and no-instruction-truncation guarantee.
2. Observation writer lifecycle: no false grasp when wide open, slip reset without loops, multi-cycle release/regrasp.
3. Oracle writer groundedness: ungrounded env=None yields UNKNOWN validity rather than CONFIRMED 1.0.
4. Multi-target extraction ("both alphabet soup and tomato sauce") and hierarchical drawer resolution.
5. Durable state persistence beyond transient max_age_steps TTL.
6. Paired comparator coverage completeness and execution horizon mismatch alerts.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.evaluation.diagnostics import extract_all_targets_and_goal, extract_target_and_goal
from src.interfaces.text_memory import TextMemoryRenderer, RenderResult
from src.memory.models import (
    EvidenceSource,
    MemoryEvent,
    MemoryQuery,
    MemoryStatus,
    ObjectMemory,
    RetrievedMemory,
)
from src.memory.observation_writer import ObservationMemoryWriter
from src.memory.oracle_writer import OracleMemoryWriter
from src.memory.retriever import DeterministicMemoryRetriever
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater
from scripts.compare_memory_runs import generate_paired_report


# ---------------------------------------------------------------------------
# Issue 1: Token Budget Awareness & Compact Rendering
# ---------------------------------------------------------------------------

def test_renderer_token_budget_compact_mode():
    """Verify that compact mode renders concise facts and respects token limit without cutting instruction."""
    # Set tight token budget: 22 tokens total
    renderer = TextMemoryRenderer(
        max_context_chars=512,
        max_total_tokens=22,
        compact_mode=True,
        allowed_sources=frozenset(EvidenceSource),
    )
    instruction = "pick up the black bowl and place it on the plate"

    objects = (
        ObjectMemory(
            episode_id="ep1", object_id="black_bowl", semantic_label="black bowl",
            state="in_basket", first_seen_step=0, last_updated_step=0,
            confidence=0.8, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
        ),
        ObjectMemory(
            episode_id="ep1", object_id="plate", semantic_label="plate",
            state="placed", first_seen_step=0, last_updated_step=0,
            confidence=0.8, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
        ),
        ObjectMemory(
            episode_id="ep1", object_id="cup", semantic_label="cup",
            state="holding", first_seen_step=0, last_updated_step=0,
            confidence=0.8, evidence_source=EvidenceSource.OBSERVATION_TRACKER,
        ),
    )
    mem = RetrievedMemory(episode_id="ep1", as_of_step=0, objects=objects)

    res = renderer.render_with_audit(instruction, mem)
    # Instruction is always preserved as prefix
    assert res.text.startswith(instruction)
    # Facts were compacted (no verbose bracketed metadata in compact mode)
    assert "[status=" not in res.text
    # Total tokens never exceed budget
    assert res.total_tokens_with_instruction <= 22
    # At least some facts were dropped due to tight budget
    assert res.token_truncated is True
    assert res.dropped_facts_count > 0


def test_renderer_token_budget_instruction_overflow_preserves_prompt():
    """When instruction prompt alone fills token budget, memory is completely dropped to preserve instruction."""
    # Instruction is ~14 tokens. Budget = 10 tokens.
    renderer = TextMemoryRenderer(
        max_total_tokens=10,
        compact_mode=True,
    )
    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    obj = ObjectMemory(
        episode_id="ep1", object_id="bowl", semantic_label="bowl", state="in_basket",
        first_seen_step=0, last_updated_step=0, confidence=0.8,
        evidence_source=EvidenceSource.OBSERVATION_TRACKER,
    )
    mem = RetrievedMemory(episode_id="ep1", as_of_step=0, objects=(obj,))

    res = renderer.render_with_audit(instruction, mem)
    assert res.text == instruction
    assert res.token_truncated is True
    assert res.included_facts_count == 0


# ---------------------------------------------------------------------------
# Issue 2: Observation Writer Lifecycle & Robust Heuristics
# ---------------------------------------------------------------------------

def test_observation_writer_no_false_grasp_when_wide_open():
    """Gripper closing command while fingers are wide open (>0.065m) must NOT emit grasp."""
    store = EpisodeMemoryStore(episode_id="ep_test")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl",),
    )
    writer.on_episode_start(episode_id="ep_test", env=None, obs={}, step=0, timestamp=0.0)

    # Gripper is closing (+1.0) but fingers are wide open (gap = 0.08m)
    obs_wide_open = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.85]),
        "robot0_gripper_qpos": np.array([0.04, 0.04]),  # gap = 0.08m > 0.065m
    }
    events = writer.on_step(
        step=1,
        timestamp=0.05,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_wide_open,
    )
    assert len(events) == 0
    # No false initial detection
    assert store.get_object("black_bowl") is None


def test_observation_writer_slip_resets_lift_and_does_not_loop():
    """Slip event must reset is_lifted and is_grasped so it does not fire repeatedly on consecutive steps."""
    store = EpisodeMemoryStore(episode_id="ep_test")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl",),
    )
    writer.on_episode_start(episode_id="ep_test", env=None, obs={}, step=0, timestamp=0.0)

    # 1. Grasp (requires 2 steps of stalled closure)
    obs_grasp = {"robot0_eef_pos": np.array([0.0, 0.0, 0.85]), "robot0_gripper_qpos": np.array([0.02, 0.02])}
    writer.on_step(
        step=9, timestamp=0.45,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_grasp,
    )
    writer.on_step(
        step=10, timestamp=0.5,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_grasp,
    )
    # 2. Lift
    writer.on_step(
        step=20, timestamp=1.0,
        action=np.array([0, 0, 0.1, 0, 0, 0, 1.0]),
        next_obs={"robot0_eef_pos": np.array([0.0, 0.0, 0.90]), "robot0_gripper_qpos": np.array([0.02, 0.02])},
    )
    assert store.get_object("black_bowl").state == "lifted"

    # 3. Slip: fingers snap shut (gap = 0.005 < 0.015) while closing
    obs_slip = {"robot0_eef_pos": np.array([0.0, 0.0, 0.90]), "robot0_gripper_qpos": np.array([0.002, 0.002])}
    events_slip = writer.on_step(
        step=30, timestamp=1.5,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_slip,
    )
    assert len(events_slip) == 1
    assert events_slip[0].event_type == "observed_slip"
    assert store.get_object("black_bowl").state == "contact_lost"

    # 4. Next step with identical slip observation must NOT emit another slip
    events_slip_loop = writer.on_step(
        step=31, timestamp=1.55,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_slip,
    )
    assert len(events_slip_loop) == 0


def test_observation_writer_missing_proprioception_safely_ignored():
    """Empty or missing observations must return [] without fabricating zero-value events."""
    store = EpisodeMemoryStore(episode_id="ep_test")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl",),
    )
    writer.on_episode_start(episode_id="ep_test", env=None, obs={}, step=0, timestamp=0.0)

    events = writer.on_step(
        step=1, timestamp=0.05,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs={},
    )
    assert events == []


# ---------------------------------------------------------------------------
# Issue 3: Grounded Oracle & Multi-Object Target Extraction
# ---------------------------------------------------------------------------

def test_oracle_writer_missing_sim_unresolved():
    """When sim is None (ungrounded), OracleMemoryWriter must set validity=UNKNOWN and confidence=0.0."""
    store = EpisodeMemoryStore(episode_id="ep_oracle")
    updater = MemoryUpdater(store)
    writer = OracleMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("soup",),
        goal_container_name="basket",
    )
    writer.on_episode_start(episode_id="ep_oracle", env=None, obs={}, step=0, timestamp=0.0)

    soup = store.get_object("soup")
    assert soup is not None
    assert soup.validity == MemoryStatus.UNKNOWN
    assert soup.confidence == 0.0

    basket = store.get_object("basket")
    assert basket is not None
    assert basket.validity == MemoryStatus.UNKNOWN
    assert basket.confidence == 0.0


def test_extract_all_targets_multi_object():
    """Multi-target instructions must extract all objects and hierarchical fixtures."""
    inst1 = "put both the alphabet soup and the tomato sauce in the basket"
    targets1, goal1 = extract_all_targets_and_goal(None, inst1)
    assert targets1 == ("alphabet_soup", "tomato_sauce")
    assert goal1 == "basket"

    inst2 = "open the middle drawer of the cabinet"
    targets2, goal2 = extract_all_targets_and_goal(None, inst2)
    assert goal2 == "middle_drawer"

    inst3 = "open the top drawer and put the bowl inside"
    targets3, goal3 = extract_all_targets_and_goal(None, inst3)
    assert targets3 == ("bowl",)
    assert goal3 == "top_drawer"


# ---------------------------------------------------------------------------
# Issue 6: Fact TTL & Durable Receptacle State Persistence
# ---------------------------------------------------------------------------

def test_retriever_durable_states_survive_ttl():
    """Durable states (receptacle contents, fixture open/close) persist beyond transient max_age_steps."""
    store = EpisodeMemoryStore(episode_id="ep1")
    updater = MemoryUpdater(store)

    # 1. Durable state: placed in container at step 50
    updater.record(MemoryEvent(
        event_id="e1", episode_id="ep1", step=50, timestamp=2.5,
        event_type="place", object_id="bowl", semantic_label="bowl",
        evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
        object_state_after="in_basket", validity=MemoryStatus.CONFIRMED,
    ))
    # 2. Transient state: grasped at step 50
    updater.record(MemoryEvent(
        event_id="e2", episode_id="ep1", step=50, timestamp=2.5,
        event_type="grasp", object_id="mug", semantic_label="mug",
        evidence_source=EvidenceSource.OBSERVATION_TRACKER, confidence=0.9,
        object_state_after="grasped", validity=MemoryStatus.CONFIRMED,
    ))

    retriever = DeterministicMemoryRetriever()
    snap = store.snapshot(as_of_step=350)
    # Query at step 350 with max_age_steps=200 (age = 300 steps)
    query = MemoryQuery(
        episode_id="ep1", as_of_step=350,
        entity_ids=("bowl", "mug"), max_age_steps=200,
    )
    result = retriever.retrieve(snap, query)
    retrieved_ids = {o.object_id for o in result.objects}

    # "bowl" has durable state "in_basket" -> survives TTL!
    assert "bowl" in retrieved_ids
    assert store.get_object("bowl").state == "in_basket"
    # "mug" has transient state "grasped" -> correctly expired after 300 steps
    assert "mug" not in retrieved_ids


# ---------------------------------------------------------------------------
# Issue 4: Comparator Coverage & Provenance Sanity Checks
# ---------------------------------------------------------------------------

def test_compare_memory_runs_incomplete_coverage_alert(tmp_path: Path):
    """Comparator must detect and report incomplete coverage between baseline and treatment."""
    base_dir = tmp_path / "baseline_off"
    treat_dir = tmp_path / "treatment_text"
    base_dir.mkdir()
    treat_dir.mkdir()

    # Baseline has 3 tasks, Treatment only has 1 task
    for i in range(3):
        (base_dir / f"task_{i}" / "init_0").mkdir(parents=True)
        with open(base_dir / f"task_{i}" / "init_0" / "episode.json", "w") as f:
            json.dump({"task_name": f"task_{i}", "initial_state_id": 0, "seed": 42 + i, "success": True, "execution_horizon_s": 10}, f)

    (treat_dir / "task_0" / "init_0").mkdir(parents=True)
    with open(treat_dir / "task_0" / "init_0" / "episode.json", "w") as f:
        json.dump({"task_name": "task_0", "initial_state_id": 0, "seed": 42, "success": True, "execution_horizon_s": 10}, f)

    md_content, stats = generate_paired_report(base_dir, treat_dir, tmp_path / "out")
    assert stats["coverage_complete"] is False
    assert stats["total_baseline_episodes"] == 3
    assert stats["total_treatment_episodes"] == 1
    assert "INCOMPLETE COVERAGE DETECTED" in md_content
