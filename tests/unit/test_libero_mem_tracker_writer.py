"""Regression tests for typed RGB tracking, not oracle XYZ fixtures."""
import pytest
from src.memory.event_writer import FSMState, LiberoMemEventWriter
from src.perception.object_tracker import CausalObjectTracker
from test_rgb_cycle_writer import frame, rest, cycle


def test_tracker_rejects_out_of_order_and_does_not_impute_lost_object():
    tracker = CausalObjectTracker()
    assert tracker.update(0, frame(0))["raw_on_support"]
    with pytest.raises(ValueError, match="out of order"):
        tracker.update(0, frame(0))
    lost = tracker.update(1, frame(1, missing=True, gap=.005))
    assert not lost["has_grounding_evidence"]
    assert not lost["raw_lifted"]
    assert not lost["raw_on_support"]


def test_writer_requires_observed_origin_and_rearms_after_return():
    writer = LiberoMemEventWriter("test_ep_01")
    assert writer.fsm_state == FSMState.UNINITIALIZED
    step = rest(writer, 0)
    assert writer.fsm_state == FSMState.SUPPORT_OBSERVED
    step = cycle(writer, step)
    assert writer.confirmed_cycles == 1
    assert not writer.in_progress
    assert writer.fsm_state == FSMState.SUPPORT_OBSERVED
    assert [ev.type for ev in writer.get_snapshot().events] == [
        "initial_support_observed", "departed_support", "returned_to_support"]
