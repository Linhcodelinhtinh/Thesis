"""The proprioceptive touchdown cue, on small synthetic trajectories (unit fixtures, not benchmark data)."""
import pytest

from src.memory.proprio_cycle_cue import ProprioCueConfig, ProprioCycleCue, fuse_returns, touchdown_steps

OPEN, CLOSED = .078, .005


def cycle(rise=.14, drop_to=.0, lateral=.0, release=True, start=10, hold=12, back=True):
    """Open and approach, grasp at z=0.95, lift, move sideways, lower to `drop_to` above the grasp height, release."""
    rows = [(OPEN, (0, 0, 1.1 - .015 * step)) for step in range(start)]
    rows += [(CLOSED, (0, 0, .95))] * 4
    rows += [(CLOSED, (lateral * k / 5, 0, .95 + rise * k / 5)) for k in range(1, 6)]
    rows += [(CLOSED, (lateral, 0, .95 + rise))] * hold
    rows += [(CLOSED, (lateral * (1 - k / 5 if back else 1), 0, .95 + rise + (drop_to - rise) * k / 5)) for k in range(1, 6)]
    if release:
        rows += [(OPEN, (0, 0, .95 + drop_to))] * 4
    return rows


def steps(rows, config=None):
    return touchdown_steps([a for a, _ in rows], [p for _, p in rows], config or ProprioCueConfig())


def test_config_round_trip_and_rejection():
    config = ProprioCueConfig.load()
    assert ProprioCueConfig(**config.to_dict()) == config
    with pytest.raises(ValueError):
        ProprioCueConfig(closed_aperture_meters=.06)
    with pytest.raises(ValueError):
        ProprioCueConfig(touchdown_height_meters=.1, lift_meters=.08)
    with pytest.raises(ValueError):
        ProprioCueConfig(lift_meters=0)


def test_a_clean_lift_and_return_fires_once_when_the_end_effector_is_back_down():
    rows = cycle(drop_to=.01)
    found = steps(rows)
    assert len(found) == 1
    # It fires on the frame the height comes within tolerance, then waits the confirmation frames.
    assert rows[found[0] - 2][1][2] <= .95 + .03 and rows[found[0] - 3][1][2] > .95 + .03


def test_it_waits_for_the_descent_and_ignores_a_release_in_the_air():
    assert steps(cycle(drop_to=.1)) == []          # released while still raised
    assert steps(cycle(drop_to=.01, release=False)) != []   # touchdown does not need the release


def test_a_shallow_lift_or_a_far_return_do_not_fire():
    assert steps(cycle(rise=.05, drop_to=.0)) == []
    assert steps(cycle(drop_to=.0, lateral=.2, back=False)) == []


def test_one_event_per_closed_phase_and_a_second_phase_fires_again():
    rows = cycle(drop_to=.0) + cycle(drop_to=.0)
    found = steps(rows)
    assert len(found) == 2 and found[1] - found[0] > 20


def test_streaming_reports_the_grasp_and_rise():
    cue = ProprioCycleCue(ProprioCueConfig())
    events = [e for step, (a, p) in enumerate(cycle(rise=.12, drop_to=.0)) if (e := cue.step(step, a, p))]
    assert len(events) == 1 and events[0]["rise"] == pytest.approx(.12) and events[0]["type"] == "touchdown"
    assert cue.touchdowns == events


def test_bad_inputs_are_rejected():
    cue = ProprioCycleCue(ProprioCueConfig())
    with pytest.raises(ValueError):
        cue.step(0, .05, (0, 0))
    with pytest.raises(ValueError):
        cue.step(0, float("nan"), (0, 0, 1))


def test_fusion_adds_only_returns_the_writer_missed():
    assert fuse_returns([100], [105, 300]) == [100, 300]
    assert fuse_returns([], []) == [] and fuse_returns([100], []) == [100]
    assert fuse_returns([100], [121]) == [100, 121]   # beyond the gap
