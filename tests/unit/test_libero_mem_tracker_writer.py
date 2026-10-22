"""Unit tests for CausalObjectTracker and LiberoMemEventWriter."""

import numpy as np
import pytest

from src.memory.event_writer import FSMState, LiberoMemEventWriter
from src.perception.object_tracker import CausalObjectTracker


def test_tracker_hysteresis_and_causality():
    tracker = CausalObjectTracker(
        target_object_id="bowl_A",
        dwell_steps=3,
        support_origin_pos=np.array([0.0, 0.0, 0.85], dtype=np.float32),
    )

    # Initial resting observation
    obs_rest = {
        "robot0_eef_pos": np.array([0.0, 0.0, 0.85], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([0.0, 0.0, 0.85], dtype=np.float32)},
    }
    s0 = tracker.update(0, obs_rest, None)
    assert s0["raw_on_support"]
    assert not s0["is_departed_candidate"]

    # Step out of order raises error
    with pytest.raises(ValueError, match="Step out of order"):
        tracker.update(0, obs_rest, None)

    # 1 transient departure step (should not trigger candidate yet)
    obs_away = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.2], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, 0.01], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([0.0, 0.0, 1.2], dtype=np.float32)},
    }
    s1 = tracker.update(1, obs_away, None)
    assert not s1["is_departed_candidate"]

    # Step 2 away
    s2 = tracker.update(2, obs_away, None)
    assert not s2["is_departed_candidate"]

    # Step 3 away -> 3 dwell steps met!
    s3 = tracker.update(3, obs_away, None)
    assert s3["is_departed_candidate"]


def test_writer_fsm_complete_cycle():
    writer = LiberoMemEventWriter(
        episode_id="test_ep_01",
        target_object_id="bowl_A",
        dwell_steps=2,
    )

    assert writer.fsm_state == FSMState.SUPPORT_OBSERVED
    assert writer.confirmed_cycles == 0
    assert not writer.in_progress

    # Lift away
    obs_lift = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.2], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, 0.01], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([0.0, 0.0, 1.2], dtype=np.float32)},
    }
    writer.step(1, obs_lift, None)
    assert writer.fsm_state == FSMState.DEPARTURE_CANDIDATE

    writer.step(2, obs_lift, None)
    assert writer.fsm_state == FSMState.DEPARTURE_CONFIRMED
    assert writer.in_progress

    # Return
    obs_return = {
        "robot0_eef_pos": np.array([-0.05, 0.20, 0.85], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([-0.05, 0.20, 0.85], dtype=np.float32)},
    }
    writer.step(3, obs_return, None)
    assert writer.fsm_state == FSMState.RETURN_CANDIDATE

    writer.step(4, obs_return, None)
    # Cycle 1 confirmed!
    assert writer.confirmed_cycles == 1
    assert not writer.in_progress
    assert writer.fsm_state == FSMState.SUPPORT_OBSERVED

    snap = writer.get_snapshot()
    assert snap.cycles["bowl_A_plate_A"].confirmed == 1
    assert not snap.cycles["bowl_A_plate_A"].in_progress
    assert len(snap.events) == 2  # departed_support and returned_to_support
