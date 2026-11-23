"""Per-cycle fusion of vision returns and the proprioceptive touchdown cue (synthetic trajectories)."""
import pytest

from src.memory.cycle_fusion import CycleFusionCounter
from src.memory.proprio_cycle_cue import ProprioCueConfig
from tests.unit.test_proprio_cycle_cue import cycle


def run(rows, vision=None, config=None):
    """`vision` maps a frame to the list of return events the writer publishes on it."""
    counter = CycleFusionCounter(config or ProprioCueConfig())
    counts = [counter.step(step, aperture, position, (vision or {}).get(step, ()))
              for step, (aperture, position) in enumerate(rows)]
    return counter, counts


def test_touchdown_and_a_late_vision_return_of_the_same_cycle_count_once():
    rows = cycle(drop_to=.0) + cycle(drop_to=.0)[:5]  # then a second approach with the gripper still open
    touchdown_free, _ = run(rows)
    assert touchdown_free.count_at(len(rows)) == 1
    # Vision confirms 30 frames after the cycle began, after the gripper opened: still the same cycle.
    late = {len(rows) - 1: [dict(occurred_at=len(cycle(drop_to=.0)) - 2)]}
    counter, counts = run(rows, late)
    assert counts[-1] == 1 and [c.source for c in counter.credits.values()] == ["proprio"]
    assert len(counter.duplicates) == 1 and counter.duplicates[0].source == "vision"


def test_vision_alone_credits_a_cycle_the_cue_missed():
    rows = cycle(rise=.05, drop_to=.0)  # too shallow for the cue
    end = len(rows) - 1
    counter, counts = run(rows, {end: [dict(occurred_at=end - 1)]})
    assert counts[-1] == 1 and counter.sources() == ["vision"]


def test_two_cycles_count_twice_and_events_go_to_their_own_cycle():
    first = cycle(drop_to=.0)
    rows = first + cycle(rise=.05, drop_to=.0)
    end = len(rows) - 1
    # Vision reports the second (shallow) cycle; its occurred_at is inside the second closed phase.
    counter, counts = run(rows, {end: [dict(occurred_at=end - 3)]})
    assert counts[-1] == 2 and sorted(counter.credits) == [0, 1]


def test_a_vision_return_before_any_grasp_has_its_own_slot():
    rows = cycle(drop_to=.0)
    counter, _ = run(rows, {3: [dict(occurred_at=2)]})
    assert counter.count_at(len(rows)) == 2 and -1 in counter.credits


def test_a_touchdown_counts_only_after_its_confirmation_frames():
    rows = cycle(drop_to=.0)
    counter, counts = run(rows)
    credit = counter.credits[0]
    assert counts[credit.step - 1] == 0 and counts[credit.step] == 1


def test_frames_must_be_consecutive():
    counter = CycleFusionCounter(ProprioCueConfig())
    counter.step(0, .078, (0, 0, 1))
    with pytest.raises(ValueError):
        counter.step(2, .078, (0, 0, 1))
