"""Unit tests for Stage A (Oracle) and Stage B (Observation Tracker) Memory Writers.

Verifies:
- Entity separation: Goal container is not duplicated as a manipulable target.
- Grounded initialization: Initial object states use physical elevation / joints.
- ObservationMemoryWriter: Proprioceptive events emit OBSERVATION_TRACKER with realistic confidence.
- OracleMemoryWriter: Transitions require physical contact, lift, and proximity evidence.
"""

from typing import Any, Dict
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.memory.models import EvidenceSource, MemoryStatus
from src.memory.observation_writer import ObservationMemoryWriter
from src.memory.oracle_writer import OracleMemoryWriter
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater


def test_observation_writer_initialization_and_entity_separation():
    store = EpisodeMemoryStore(episode_id="ep_test")
    updater = MemoryUpdater(store)

    # Plate is both target and goal in uncleaned input
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl", "plate"),
        goal_container_name="plate",
    )

    # Goal container should be filtered out of target entities
    assert writer.target_entity_names == ("black_bowl",)
    assert writer.goal_container_name == "plate"

    writer.on_episode_start(episode_id="ep_test", env=None, obs={}, step=0, timestamp=0.0)

    snap = store.snapshot(as_of_step=0)
    assert len(snap.objects) == 2
    bowl = store.get_object("black_bowl")
    plate = store.get_object("plate")

    assert bowl is not None
    assert bowl.state == "detected"
    assert bowl.evidence_source == EvidenceSource.OBSERVATION_TRACKER
    assert bowl.confidence == 0.80

    assert plate is not None
    assert plate.state == "target_destination"
    assert plate.evidence_source == EvidenceSource.OBSERVATION_TRACKER


def test_observation_writer_step_heuristics():
    store = EpisodeMemoryStore(episode_id="ep_test")
    updater = MemoryUpdater(store)
    writer = ObservationMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl",),
        goal_container_name="plate",
    )
    writer.on_episode_start(episode_id="ep_test", env=None, obs={}, step=0, timestamp=0.0)

    # 1. Grasp: action closing (+1.0) and fingers blocked (gap = 0.04m > 0.02m)
    obs_grasp = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.85]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),
    }
    events = writer.on_step(
        step=10,
        timestamp=0.5,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs=obs_grasp,
    )
    assert len(events) == 1
    assert events[0].event_type == "observed_grasp"
    assert events[0].evidence_source == EvidenceSource.OBSERVATION_TRACKER
    assert store.get_object("black_bowl").state == "grasped"

    # 2. Lift: EEF rises by >= 0.04m (z = 0.90m, delta = +0.05m)
    obs_lift = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.90]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),
    }
    events_lift = writer.on_step(
        step=20,
        timestamp=1.0,
        action=np.array([0, 0, 0.1, 0, 0, 0, 1.0]),
        next_obs=obs_lift,
    )
    assert len(events_lift) == 1
    assert events_lift[0].event_type == "observed_lift"
    assert store.get_object("black_bowl").state == "lifted"

    # 3. Release: opening action (-1.0) while lifted
    obs_release = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.90]),
        "robot0_gripper_qpos": np.array([0.04, 0.04]),
    }
    events_release = writer.on_step(
        step=30,
        timestamp=1.5,
        action=np.array([0, 0, 0, 0, 0, 0, -1.0]),
        next_obs=obs_release,
    )
    assert len(events_release) == 1
    assert events_release[0].event_type == "observed_release"
    assert store.get_object("black_bowl").state == "released"


def test_oracle_writer_grounded_initialization_and_transitions():
    store = EpisodeMemoryStore(episode_id="ep_oracle")
    updater = MemoryUpdater(store)

    # Mock simulator
    mock_sim = MagicMock()
    mock_sim.model.nbody = 2
    mock_sim.model.body_id2name.side_effect = lambda i: "black_bowl" if i == 0 else "plate"
    mock_sim.data.body_xpos = [
        np.array([0.1, 0.2, 0.84]),  # table height ~0.84m -> on_table
        np.array([0.3, 0.2, 0.84]),  # plate
    ]

    mock_env = MagicMock()
    mock_env.sim = mock_sim

    writer = OracleMemoryWriter(
        store=store,
        updater=updater,
        target_entity_names=("black_bowl", "plate"),
        goal_container_name="plate",
    )
    assert writer.target_entity_names == ("black_bowl",)

    writer.on_episode_start(
        episode_id="ep_oracle",
        env=mock_env,
        obs={},
        step=0,
        timestamp=0.0,
    )

    bowl = store.get_object("black_bowl")
    assert bowl is not None
    assert bowl.state == "on_table"
    assert bowl.evidence_source == EvidenceSource.ORACLE_SIMULATOR
    assert bowl.confidence == 1.0

    plate = store.get_object("plate")
    assert plate is not None
    assert plate.state == "empty"

    # Transition 1: Grasp
    events_grasp = writer.on_step(
        step=5,
        timestamp=0.25,
        action=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        next_obs={},
        contact_info={"both_fingers_contact": True},
    )
    assert len(events_grasp) == 1
    assert events_grasp[0].event_type == "grasp"
    assert store.get_object("black_bowl").state == "grasped"

    # Transition 2: Lift
    events_lift = writer.on_step(
        step=10,
        timestamp=0.5,
        action=np.array([0, 0, 0.1, 0, 0, 0, 1.0]),
        next_obs={},
        contact_info={"both_fingers_contact": True},
        lift_info={"is_lifted": True},
    )
    assert len(events_lift) == 1
    assert events_lift[0].event_type == "lift"
    assert store.get_object("black_bowl").state == "lifted_in_air"

    # Transition 3: Placement (verified via proximity <= 0.08m and gripper open)
    events_place = writer.on_step(
        step=20,
        timestamp=1.0,
        action=np.array([0, 0, 0, 0, 0, 0, -1.0]),
        next_obs={},
        contact_info={"both_fingers_contact": False},
        dist_to_goal=0.04,  # Within 0.08m
    )
    assert len(events_place) == 2  # object place + goal receive_object
    assert store.get_object("black_bowl").state == "in_plate"
    assert store.get_object("plate").state == "contains_black_bowl"
