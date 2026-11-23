"""Halt shadow sources, count-source halt and the fork replay core (fake env, no simulator)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.xvla_v2.fork_halt_replay import fork_episode, parse_jobs, work_items
from scripts.xvla_v2.plan_halt_forks import latched
from src.evaluation.halt_sources import ShadowHaltWriter
from src.evaluation.memory_halt import (HaltConfig, MemoryHaltPolicy, counted_snapshot, oracle_snapshot)

OPEN = np.array([0.03, -0.03])
CLOSED = np.array([0.01, -0.01])


def rel(returns, status="confirmed", armed=False):
    return SimpleNamespace(returns=returns, status=status, reference="plate_1", armed=armed)


def snap(step, *states):
    return SimpleNamespace(as_of_observation=step, profile=SimpleNamespace(count_cycles=True),
                           relations={f"bowl_{i + 1}": s for i, s in enumerate(states)})


class FakeEnv:
    """Counts steps; reports a controller pose that follows the last action's x for a simple hold check."""
    def __init__(self, done_at=1000):
        self.steps, self.done_at, self.executed, self.last_info = 0, done_at, [], {}
        self.pos = np.zeros(3)

    def _info(self):
        ev = SimpleNamespace(full_success=self.steps >= 5, scr=1.0, overshot=False, confirmed_cycles=1)
        self.last_info = dict(eval_result=ev, final_predicate=True, xvla_controller=dict(
            use_delta=False, position=self.pos.tolist(), rotation6d=[1, 0, 0, 0, 1, 0]))
        return dict(self.last_info)

    def reset(self, index):
        self.steps, self.executed, self.pos = 0, [], np.zeros(3)
        self._info()
        return dict(robot0_eef_pos=self.pos.copy())

    def step(self, action):
        self.steps += 1
        self.executed.append(np.asarray(action, float))
        self.pos = self.pos + 0.001
        return dict(robot0_eef_pos=self.pos.copy()), 0.0, self.steps >= self.done_at, self._info()


def hold_chunk_for(env):
    def current_pose():
        c = env.last_info["xvla_controller"]
        return np.asarray(c["position"], float), np.asarray(c["rotation6d"], float)

    def make(anchor, rot, start, ramp):
        chunk = np.zeros((30, 7), np.float32)
        chunk[:, :3] = anchor
        return chunk
    return current_pose, make


def test_fork_replays_prefix_then_holds_to_done():
    env = FakeEnv(done_at=60)
    actions = [np.full(7, i, float) for i in range(100)]
    eef = np.array([[0.001 * i] * 3 for i in range(101)])
    out = fork_episode(env, 0, actions, eef, 30, retract_m=0.03, hold_chunk=hold_chunk_for(env))
    assert out["forked"] and out["steps"] == 60 and out["success"]
    assert [a[0] for a in env.executed[:30]] == list(range(30))            # recorded prefix, in order
    held = np.stack(env.executed[30:])
    assert np.allclose(held[:, 2], 0.001 * 30 + 0.03)                     # anchored at halt pose + retract
    assert out["max_eef_deviation_m"] < 1e-9


def test_fork_reports_infidelity_and_verify_mode():
    env = FakeEnv(done_at=20)
    actions = [np.zeros(7) for _ in range(20)]
    eef = np.array([[0.001 * i + 0.01] * 3 for i in range(21)])           # recording disagrees by 1 cm
    out = fork_episode(env, 0, actions, eef, None, hold_chunk=hold_chunk_for(env))
    assert not out["forked"] and out["steps"] == 20 and out["max_eef_deviation_m"] == pytest.approx(0.01)


def test_fork_ended_before_halt():
    env = FakeEnv(done_at=10)
    actions = [np.zeros(7) for _ in range(50)]
    out = fork_episode(env, 0, actions, np.zeros((51, 3)) + np.arange(51)[:, None] * 0.001, 30,
                       hold_chunk=hold_chunk_for(env))
    assert out["ended_before_halt"] and out["steps"] == 10


def test_work_items_dedupe_and_shard():
    plan = dict(episodes=[
        dict(reference=dict(episode="a", task="T1"), conditions={
            "X": dict(halt_step=40), "Y": dict(halt_step=40), "Z": dict(halt_step=None)}),
        dict(reference=dict(episode="b", task="T3"), conditions={
            "X": dict(halt_step=60), "Y": dict(halt_step=70), "Z": dict(halt_step=10)})])
    jobs = parse_jobs(["X", "Y", "Z", "Y:0.03"])
    everything = work_items(plan, jobs, ["T1", "T3"], 0, 1)
    assert everything[("a", 40, 0.0)] == ["X", "Y"] and ("a", None, 0.0) not in everything
    assert len(everything) == 6                                            # a40, a40@.03, b60, b70, b70@.03, b10
    halves = [work_items(plan, jobs, ["T1", "T3"], i, 2) for i in range(2)]
    assert set(halves[0]) | set(halves[1]) == set(everything) and not set(halves[0]) & set(halves[1])
    assert work_items(plan, jobs, ["T1"], 0, 1).keys() == {("a", 40, 0.0), ("a", 40, 0.03)}


def test_latched_requires_consecutive_confirmations():
    rows = [dict(halt_condition=v) for v in (False, True, False, True, True, True)]
    assert [r["halt_condition"] for r in latched(rows, 1)] == [False, True, False, False, False, False]
    assert [r["halt_condition"] for r in latched(rows, 2)] == [False, False, False, False, True, False]


class Primary:
    VERSION = "rgb_relations_v1"
    profile = SimpleNamespace(count_cycles=True)

    def __init__(self):
        self.last_signals, self.last_evidence, self.calls, self._snap = None, None, 0, None

    def step(self, step, obs, last_action=None):
        self.calls += 1
        self.last_evidence = ("evidence", step)
        self.last_signals = dict(new_events=[], visual_evidence=self.last_evidence)
        self._snap = snap(step, rel(0))
        return self._snap

    def get_snapshot(self):
        return self._snap


class Shadow:
    VERSION = "rgb_relations_v4"

    def __init__(self):
        self.seen, self.last_events = [], ()

    def step(self, step, obs, last_action=None):
        self.seen.append((step, obs["visual_evidence"]))
        self.last_events = ()
        return snap(step, rel(0))


class Counter:
    def __init__(self):
        self.calls, self.credits, self.duplicates = [], {}, []

    def step(self, step, aperture, eef, returns):
        self.calls.append((step, round(aperture, 3), tuple(eef)))
        return len(self.calls) // 3

    def count_at(self, step):
        return 0


def frame():
    return dict(robot0_gripper_qpos=OPEN, robot0_eef_pos=[0.1, 0.2, 0.3])


def test_shadow_writer_shares_evidence_and_does_not_double_step():
    primary, shadow, counter = Primary(), Shadow(), Counter()
    writer = ShadowHaltWriter(primary, shadow, counter)
    assert writer.VERSION == "rgb_relations_v1"
    for step in range(3):
        returned = writer.step(step, frame(), None)
        assert returned is primary.get_snapshot()
    writer.step(2, frame(), None)                                          # duplicate observation
    assert [s for s, _ in shadow.seen] == [0, 1, 2] and len(counter.calls) == 3
    assert shadow.seen[1][1] == ("evidence", 1)                            # V4 got V1's detector evidence
    assert writer.halt_snapshot("fused").relations["bowl_1"].returns == 1
    assert writer.halt_snapshot("v4").relations["bowl_1"].returns == 0


def test_freeze_stops_all_work_and_keeps_last_snapshot():
    primary, shadow, counter = Primary(), Shadow(), Counter()
    writer = ShadowHaltWriter(primary, shadow, counter)
    writer.step(0, frame(), None)
    writer.freeze()
    last = primary.get_snapshot()
    assert writer.step(1, frame(), None) is last and primary.calls == 1 and len(shadow.seen) == 1
    assert writer.last_signals["frozen"] is True and writer.summary()["frozen_skipped_frames"] == 1


class StubPolicy:
    chunk_size, action_dim = 30, 7
    mem_adapter_manifest = {}

    def __init__(self):
        self.calls = 0

    def bind_memory_runtime(self, writer, renderer, **_):
        return None

    def predict_action_chunk(self, obs, prompt):
        self.calls += 1
        return np.zeros((30, 7), np.float32)


class ShadowLike(Primary):
    def __init__(self, fused, v4):
        super().__init__()
        self.fused, self.v4, self.frozen = fused, v4, False

    def halt_snapshot(self, source):
        return self.fused if source == "fused" else self.v4

    def freeze(self):
        self.frozen = True


def halt_policy(source, writer, oracle=None, freeze=False, target=1):
    pose = (lambda: (np.zeros(3), np.zeros(6)))
    hold = (lambda a, r, s, n: np.zeros((30, 7), np.float32))
    policy = MemoryHaltPolicy(StubPolicy(), HaltConfig(target, require_on_support=False), pose, hold,
                              count_source=source, oracle_count=oracle, freeze_writer=freeze)
    policy.bind_memory_runtime(writer, None)
    return policy


def test_fused_source_halts_on_fused_count_and_freezes_writer():
    writer = ShadowLike(fused=snap(10, rel(1)), v4=snap(10, rel(0)))
    writer.step(10, frame())
    policy = halt_policy("fused", writer, freeze=True)
    policy.predict_action_chunk(frame(), "p")
    assert policy.halted and policy.halt_step == 10 and writer.frozen and policy._policy.calls == 0
    v4 = halt_policy("v4", ShadowLike(fused=snap(10, rel(1)), v4=snap(10, rel(0))))
    v4._writer.step(10, frame())
    v4.predict_action_chunk(frame(), "p")
    assert not v4.halted and v4._policy.calls == 1                         # same frame, V4 alone had no return


def test_oracle_source_uses_simulator_count_and_gripper_gate():
    writer = ShadowLike(fused=None, v4=None)
    writer.step(20, frame())
    counts = [0]
    policy = halt_policy("oracle", writer, oracle=lambda: counts[0])
    policy.predict_action_chunk(frame(), "p")
    assert not policy.halted
    counts[0] = 1
    policy.predict_action_chunk(dict(robot0_gripper_qpos=CLOSED), "p")
    assert not policy.halted                                               # closed gripper blocks the halt
    policy.predict_action_chunk(frame(), "p")
    assert policy.halted and policy.summary()["count_source"] == "oracle"


def test_count_source_validation_and_helpers():
    with pytest.raises(ValueError):
        halt_policy("oracle", ShadowLike(None, None))
    with pytest.raises(ValueError):
        MemoryHaltPolicy(StubPolicy(), HaltConfig(1), None, None, count_source="nope")
    base = snap(5, rel(0, status="confirmed"))
    assert counted_snapshot(base, 3).relations["bowl_1"].returns == 3
    assert oracle_snapshot(7, base.profile, 2).relations["bowl_1"].returns == 2
    default = MemoryHaltPolicy(StubPolicy(), HaltConfig(1), None, None)
    assert default.count_source == "writer"
