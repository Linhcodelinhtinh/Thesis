"""Memory-gated halt: decision rule, latching, hold chunks and delegation."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.memory_halt import (HaltConfig, MemoryHaltPolicy, evaluate_halt,
                                        xvla_hold_chunk_factory)

OPEN = np.array([0.03, -0.03])     # aperture 0.06 >= 0.055
CLOSED = np.array([0.01, -0.01])   # aperture 0.02


def state(returns, status="confirmed", reference="plate_1", armed=False):
    return SimpleNamespace(returns=returns, status=status, reference=reference, armed=armed)


def snapshot(step, *states, count_cycles=True):
    return SimpleNamespace(as_of_observation=step, profile=SimpleNamespace(count_cycles=count_cycles),
                           relations={f"bowl_{i}": s for i, s in enumerate(states)})


def obs(grip=OPEN):
    return {"robot0_gripper_qpos": grip}


class FakeWriter:
    def __init__(self):
        self.snap = None

    def get_snapshot(self):
        return self.snap


class FakePolicy:
    chunk_size, action_dim = 30, 7
    mem_adapter_manifest = {"arm": "F1"}

    def __init__(self):
        self.calls, self.raw_chunks, self.bound = 0, [], None

    def reset(self):
        self.raw_chunks = []

    def bind_memory_runtime(self, writer, renderer, **kwargs):
        self.bound = (writer, renderer)

    def predict_action_chunk(self, obs, prompt):
        self.calls += 1
        chunk = np.zeros((30, 7), np.float32)
        chunk[:, -1] = 1.0  # close gripper: a re-grasp attempt
        return chunk


def make(config, pose=(np.array([0.1, 0.2, 1.0]), np.array([1., 0, 0, 0, 1, 0]))):
    poses = {"p": pose}
    made = []

    def current_pose():
        return poses["p"][0].copy(), poses["p"][1].copy()

    def make_hold(anchor, rot6d, start, ramp):
        made.append((anchor.copy(), rot6d.copy(), start.copy(), ramp))
        chunk = np.zeros((30, 7), np.float32)
        chunk[:, :3] = anchor
        chunk[:, -1] = -1.0
        return chunk

    inner, writer = FakePolicy(), FakeWriter()
    wrapped = MemoryHaltPolicy(inner, config, current_pose, make_hold)
    wrapped.reset()
    wrapped.bind_memory_runtime(writer, renderer="r")
    return wrapped, inner, writer, made, poses


def test_config_validation():
    with pytest.raises(ValueError):
        HaltConfig(0)
    with pytest.raises(ValueError):
        HaltConfig(1, mode="stop")
    with pytest.raises(ValueError):
        HaltConfig(1, retract_m=0.5)
    assert HaltConfig(3).to_dict()["version"] == "memory_halt_v1"


@pytest.mark.parametrize("snap,grip,reason", [
    (None, OPEN, "no_snapshot"),
    (snapshot(5), OPEN, "no_tracked_subject"),
    (snapshot(5, state(2)), OPEN, "count_below_target"),
    (snapshot(5, state(3, status="candidate")), OPEN, "subject_not_confirmed_on_support"),
    (snapshot(5, state(3, reference=None)), OPEN, "subject_not_confirmed_on_support"),
    (snapshot(5, state(3, armed=True)), OPEN, "subject_not_confirmed_on_support"),
    (snapshot(5, state(3)), CLOSED, "gripper_not_open"),
    (snapshot(5, state(3)), OPEN, "target_count_observed"),
    (snapshot(5, state(4)), OPEN, "target_count_observed"),
])
def test_rule(snap, grip, reason):
    record = evaluate_halt(snap, obs(grip), HaltConfig(3))
    assert record["reason"] == reason
    assert record["halt_condition"] is (reason == "target_count_observed")


def test_ablation_flags_relax_rule():
    cfg = HaltConfig(1, require_on_support=False, require_gripper_open=False)
    assert evaluate_halt(snapshot(1, state(1, status="unknown", reference=None)), obs(CLOSED), cfg)["halt_condition"]


def test_rule_uses_best_tracked_subject_and_rejects_non_counting_profiles():
    rec = evaluate_halt(snapshot(9, state(0), state(1)), obs(), HaltConfig(1))
    assert rec["subject"] == "bowl_1" and rec["halt_condition"]
    with pytest.raises(ValueError):
        evaluate_halt(snapshot(9, state(1), count_cycles=False), obs(), HaltConfig(1))


def test_transparent_until_halt_then_latched_hold():
    wrapped, inner, writer, made, poses = make(HaltConfig(1))
    writer.snap = snapshot(0, state(0))
    assert wrapped.predict_action_chunk(obs(), "p")[0, -1] == 1.0
    assert inner.calls == 1 and not wrapped.halted
    writer.snap = snapshot(10, state(1))
    chunk = wrapped.predict_action_chunk(obs(), "p")
    assert wrapped.halted and wrapped.halt_step == 10 and wrapped.halt_query == 1
    assert inner.calls == 1  # policy never queried again in hold mode
    assert (chunk[:, -1] == -1).all() and np.allclose(chunk[:, :3], [0.1, 0.2, 1.0])
    # Latched: even if memory later regresses, the hold stays and the anchor does not move.
    writer.snap = snapshot(20, state(0, status="unknown"))
    poses["p"] = (np.array([0.1, 0.2, 0.98]), poses["p"][1])
    wrapped.predict_action_chunk(obs(CLOSED), "p")
    assert inner.calls == 1 and np.allclose(made[-1][0], [0.1, 0.2, 1.0])
    assert np.allclose(made[-1][2], [0.1, 0.2, 0.98])
    assert [r["decision"] for r in wrapped.halt_trace] == ["continue", "halt"]
    summary = wrapped.summary()
    assert summary["halted"] and summary["halt_step"] == 10 and summary["queries"] == 3


def test_confirm_queries_requires_consecutive_boundaries():
    wrapped, inner, writer, _, _ = make(HaltConfig(1, confirm_queries=2))
    for step, s in ((10, state(1)), (20, state(1, status="unknown")), (30, state(1)), (40, state(1))):
        writer.snap = snapshot(step, s)
        wrapped.predict_action_chunk(obs(), "p")
    assert wrapped.halt_step == 40 and inner.calls == 3


def test_retract_raises_anchor_once():
    wrapped, _, writer, made, _ = make(HaltConfig(1, retract_m=0.05))
    writer.snap = snapshot(10, state(1))
    wrapped.predict_action_chunk(obs(), "p")
    wrapped.predict_action_chunk(obs(), "p")
    assert np.allclose(made[0][0], [0.1, 0.2, 1.05]) and np.allclose(made[1][0], [0.1, 0.2, 1.05])


def test_gripper_lock_keeps_policy_but_forces_open():
    wrapped, inner, writer, made, _ = make(HaltConfig(1, mode="gripper_lock"))
    writer.snap = snapshot(10, state(1))
    chunk = wrapped.predict_action_chunk(obs(), "p")
    assert inner.calls == 1 and (chunk[:, -1] == -1).all() and not made


def test_reset_clears_latch_and_delegates():
    wrapped, inner, writer, _, _ = make(HaltConfig(1))
    writer.snap = snapshot(10, state(1))
    wrapped.predict_action_chunk(obs(), "p")
    inner.raw_chunks = ["x"]
    assert wrapped.raw_chunks == ["x"]  # delegated attribute
    wrapped.reset()
    assert not wrapped.halted and wrapped.halt_trace == [] and inner.raw_chunks == []
    with pytest.raises(RuntimeError):
        wrapped.predict_action_chunk(obs(), "p")  # writer must be re-bound per episode


def test_xvla_hold_chunk_factory_builds_author_wire_chunk():
    seen = {}

    class Env:
        last_info = {"xvla_controller": dict(position=[0.1, 0.2, 1.0], rotation6d=[1, 0, 0, 0, 1, 0],
                                             use_delta=False)}

        def decode_xvla(self, raw):
            seen["raw"] = raw
            out = np.zeros((30, 7), np.float32)
            out[:, :3] = raw[:, :3]
            out[:, -1] = np.where(raw[:, 9] > .5, 1., -1.)
            return out

    current_pose, make_hold = xvla_hold_chunk_factory(Env())
    pos, rot = current_pose()
    chunk = make_hold(pos + [0, 0, 0.05], rot, pos, 10)
    raw = seen["raw"]
    assert raw.shape == (30, 20) and (raw[:, 9] == 0).all() and np.allclose(raw[:, 3:9], rot)
    assert np.isclose(raw[0, 2], 1.005) and np.allclose(raw[9:, 2], 1.05)
    assert (chunk[:, -1] == -1).all()
    Env.last_info["xvla_controller"]["use_delta"] = True
    with pytest.raises(ValueError):
        current_pose()


def _with_head(prob, combine):
    wrapped, inner, writer, made, _ = make(HaltConfig(1))
    wrapped._termination_fn = lambda o, p: prob
    wrapped.combine = combine
    return wrapped, inner, writer


@pytest.mark.parametrize("combine,returns,prob,halts", [
    ("shadow", 0, 0.9, False), ("shadow", 1, 0.1, True),
    ("or", 0, 0.9, True), ("or", 0, 0.1, False),
    ("and", 1, 0.1, False), ("and", 1, 0.9, True),
])
def test_termination_head_combination(combine, returns, prob, halts):
    wrapped, inner, writer = _with_head(prob, combine)
    writer.snap = snapshot(10, state(returns))
    wrapped.predict_action_chunk(obs(), "p")
    assert wrapped.halted is halts
    record = wrapped.halt_trace[-1]
    assert record["head_probability"] == prob and record["rule_condition"] is (returns >= 1)


def test_termination_head_arguments_validated():
    with pytest.raises(ValueError):
        MemoryHaltPolicy(FakePolicy(), HaltConfig(1), None, None, combine="xor")
    with pytest.raises(ValueError):
        MemoryHaltPolicy(FakePolicy(), HaltConfig(1), None, None, termination_threshold=1.0)
