"""Memory-gated halt: stop executing the policy once the writer has observed the target count.

A separate execution treatment, never a change to the policy weights, the prompt
template or the shared rollout runner (libero_mem_rollout.py stays byte-identical,
so its hash in cross-backbone audits is unchanged). The wrapper sits between the
runner and the policy adapter:

* the runner still renders the same prompt and calls ``predict_action_chunk``
  at every query boundary (every ``execution_horizon`` steps);
* before forwarding, the wrapper reads the causal writer snapshot that the runner
  bound through ``bind_memory_runtime`` and the measured gripper aperture;
* once the halt condition holds for ``confirm_queries`` consecutive query
  boundaries, the halt latches for the rest of the episode.

Halt modes
  ``hold``          the policy is no longer queried; every later chunk holds the
                    end-effector pose measured at halt time (optionally raised by
                    ``retract_m``) with the gripper open.
  ``gripper_lock``  the policy keeps running, but the gripper command is forced
                    open, so the bowl cannot be lifted again.

The halt uses only predicted memory (writer output) and measured proprioception.
No simulator success flag, upstream counter or privileged state is read.

Optional learned termination head (prepared, not trained yet): ``termination_fn(obs, prompt)``
returns P(task complete). With ``combine="or"`` it can trigger a halt the rule missed,
with ``combine="and"`` it vetoes rule halts it disagrees with. Its probability is logged
on every query boundary either way, so a head can be audited in shadow mode
(``combine="shadow"``) before it is allowed to act.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

HALT_VERSION = "memory_halt_v1"
HALT_MODES = ("hold", "gripper_lock")
# Where the halt reads its return count. "writer" is the original behaviour (the prompt writer's snapshot).
COUNT_SOURCES = ("writer", "v4", "fused", "oracle")
COMBINE_MODES = ("shadow", "or", "and")
# Same released-gripper threshold as RelationMemoryWriter._measure (aperture >= .055).
DEFAULT_OPEN_APERTURE = 0.055


@dataclass(frozen=True)
class HaltConfig:
    target_cycles: int
    mode: str = "hold"
    require_gripper_open: bool = True
    open_aperture: float = DEFAULT_OPEN_APERTURE
    require_on_support: bool = True
    confirm_queries: int = 1
    retract_m: float = 0.0
    ramp_steps: int = 10

    def __post_init__(self):
        if self.target_cycles < 1:
            raise ValueError("target_cycles must be >= 1")
        if self.mode not in HALT_MODES:
            raise ValueError(f"Unknown halt mode {self.mode!r}; expected one of {HALT_MODES}")
        if self.confirm_queries < 1 or self.ramp_steps < 1:
            raise ValueError("confirm_queries and ramp_steps must be >= 1")
        if not 0.0 <= self.retract_m <= 0.15:
            raise ValueError("retract_m must be within [0, 0.15] m")
        if self.open_aperture <= 0:
            raise ValueError("open_aperture must be positive")

    def to_dict(self):
        return dict(version=HALT_VERSION, **asdict(self))


def gripper_aperture(obs) -> float:
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    if grip.shape != (2,) or not np.isfinite(grip).all():
        raise ValueError("Halt requires the measured 2-finger gripper aperture")
    return float(np.abs(grip).sum())


def evaluate_halt(snapshot, obs, config: HaltConfig) -> Dict[str, Any]:
    """Pure decision on one query boundary. Returns a JSON-safe record."""
    record = dict(halt_condition=False, reason=None, observed_returns=None, subject=None,
                  aperture=None, snapshot_step=None)
    if snapshot is None:
        record["reason"] = "no_snapshot"
        return record
    record["snapshot_step"] = int(snapshot.as_of_observation)
    if not snapshot.profile.count_cycles:
        raise ValueError("Memory halt supports return-counting profiles only (T1/T3)")
    best = None
    for key, state in snapshot.relations.items():
        if best is None or state.returns > best[1].returns:
            best = (key, state)
    if best is None:
        record["reason"] = "no_tracked_subject"
        return record
    key, state = best
    record.update(subject=key, observed_returns=int(state.returns),
                  relation_status=state.status, relation_reference=state.reference,
                  departure_pending=bool(state.armed))
    if state.returns < config.target_cycles:
        record["reason"] = "count_below_target"
        return record
    if config.require_on_support and (state.status != "confirmed" or state.reference is None or state.armed):
        record["reason"] = "subject_not_confirmed_on_support"
        return record
    aperture = gripper_aperture(obs)
    record["aperture"] = aperture
    if config.require_gripper_open and aperture < config.open_aperture:
        record["reason"] = "gripper_not_open"
        return record
    record.update(halt_condition=True, reason="target_count_observed")
    return record


def counted_snapshot(base, returns, subject="bowl_1"):
    """Copy of a relation snapshot whose subject return count is replaced (status/reference/armed are kept)."""
    from types import SimpleNamespace
    states = {}
    key_to_replace = subject if subject in base.relations else next(iter(base.relations), None)
    for key, state in base.relations.items():
        states[key] = (SimpleNamespace(returns=int(returns), status=state.status, reference=state.reference,
                                       armed=state.armed) if key == key_to_replace else state)
    return SimpleNamespace(as_of_observation=base.as_of_observation, profile=base.profile, relations=states)


def oracle_snapshot(step, target_profile, confirmed_cycles, subject="bowl_1"):
    """Privileged count from the simulator evaluator (upper bound only; never a deployable source)."""
    from types import SimpleNamespace
    state = SimpleNamespace(returns=int(confirmed_cycles), status="confirmed", reference="plate_1", armed=False)
    return SimpleNamespace(as_of_observation=int(step), profile=target_profile, relations={subject: state})


def xvla_hold_chunk_factory(env, chunk_size=30):
    """Build hold chunks with the same author decoder the X-VLA policy uses.

    The raw (30, 20) wire chunk is filled with the measured EEF position and
    6D rotation, gripper probability 0 (open), so the decoded absolute OSC
    command is exactly the pose the controller reports.
    """
    def current_pose():
        controller = env.last_info["xvla_controller"]
        if controller["use_delta"]:
            raise ValueError("XVLA hold requires absolute OSC")
        return np.asarray(controller["position"], float), np.asarray(controller["rotation6d"], float)

    def make(anchor_pos, anchor_rot6d, start_pos, ramp_steps):
        raw = np.zeros((chunk_size, 20), dtype=np.float64)
        alpha = np.minimum(1.0, np.arange(1, chunk_size + 1) / float(ramp_steps))[:, None]
        raw[:, :3] = start_pos[None] + alpha * (anchor_pos - start_pos)[None]
        raw[:, 3:9] = anchor_rot6d[None]
        raw[:, 9] = 0.0
        return env.decode_xvla(raw)

    return current_pose, make


class MemoryHaltPolicy:
    """Wrap a policy adapter; transparent until the memory halt latches."""

    def __init__(self, policy, config: HaltConfig, current_pose: Callable, make_hold_chunk: Callable,
                 observer: Optional[Callable] = None, termination_fn: Optional[Callable] = None,
                 termination_threshold: float = 0.5, combine: str = "shadow",
                 count_source: str = "writer", oracle_count: Optional[Callable] = None,
                 freeze_writer: bool = False):
        if count_source not in COUNT_SOURCES:
            raise ValueError(f"count_source must be one of {COUNT_SOURCES}")
        if count_source == "oracle" and oracle_count is None:
            raise ValueError("count_source 'oracle' needs oracle_count() -> confirmed cycles")
        self.count_source, self._oracle_count, self._freeze_writer = count_source, oracle_count, freeze_writer
        if combine not in COMBINE_MODES:
            raise ValueError(f"combine must be one of {COMBINE_MODES}")
        if not 0.0 < termination_threshold < 1.0:
            raise ValueError("termination_threshold must be in (0, 1)")
        self._termination_fn = termination_fn
        self.termination_threshold, self.combine = termination_threshold, combine
        self._policy = policy
        self.config = config
        self._current_pose = current_pose
        self._make_hold_chunk = make_hold_chunk
        self._observer = observer
        self.chunk_size, self.action_dim = policy.chunk_size, policy.action_dim
        self.mem_adapter_manifest = getattr(policy, "mem_adapter_manifest", None)
        self.reset_halt()

    # -- delegation ---------------------------------------------------------
    def __getattr__(self, name):
        # Only called for attributes not found on the wrapper (raw_chunks, export, ...).
        if name == "_policy":
            raise AttributeError(name)
        return getattr(self._policy, name)

    def reset_halt(self):
        self._writer = None
        self.halted = False
        self.halt_step = None
        self.halt_query = None
        self._streak = 0
        self._anchor = None
        self._queries = 0
        self.halt_trace: List[Dict[str, Any]] = []

    def reset(self):
        self.reset_halt()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def bind_instruction_only_runtime(self, *args, **kwargs):
        return self._policy.bind_instruction_only_runtime(*args, **kwargs)

    def bind_memory_runtime(self, writer, renderer, **kwargs):
        result = self._policy.bind_memory_runtime(writer, renderer, **kwargs)
        self._writer = writer
        return result

    # -- control ------------------------------------------------------------
    def _log(self, record):
        self.halt_trace.append(record)
        if self._observer is not None:
            self._observer("decision", **record)

    def predict_action_chunk(self, obs, prompt):
        query = self._queries
        self._queries += 1
        started = time.perf_counter()
        if self._writer is None:
            raise RuntimeError("Memory halt requires a bound memory writer (memory-on condition)")
        snapshot = self._writer.get_snapshot()
        if self.count_source in ("v4", "fused"):
            snapshot = self._writer.halt_snapshot(self.count_source)
        elif self.count_source == "oracle" and snapshot is not None:
            snapshot = oracle_snapshot(snapshot.as_of_observation, snapshot.profile, self._oracle_count())
        step = None if snapshot is None else int(snapshot.as_of_observation)
        if not self.halted:
            record = evaluate_halt(snapshot, obs, self.config)
            rule = record["halt_condition"]
            if self._termination_fn is not None:
                probability = float(self._termination_fn(obs, prompt))
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("termination_fn must return a probability")
                head = probability >= self.termination_threshold
                record.update(rule_condition=rule, head_probability=probability, head_condition=head)
                if self.combine == "or":
                    record["halt_condition"] = rule or head
                elif self.combine == "and":
                    record["halt_condition"] = rule and head
            self._streak = self._streak + 1 if record["halt_condition"] else 0
            latch = self._streak >= self.config.confirm_queries
            if latch:
                self.halted, self.halt_step, self.halt_query = True, step, query
                position, rotation6d = self._current_pose()
                anchor = position.copy()
                anchor[2] += self.config.retract_m
                self._anchor = (anchor, rotation6d)
                if self._freeze_writer and hasattr(self._writer, "freeze"):
                    # Hold never reads memory again, so the detector can stop. Actions are unaffected.
                    self._writer.freeze()
            decision = "halt" if latch else "continue"
            self._log(dict(step=step, query_id=query, decision=decision, treatment=HALT_VERSION,
                           mode=self.config.mode, streak=self._streak,
                           decision_ms=(time.perf_counter() - started) * 1000.0, **record))
        if not self.halted:
            return self._policy.predict_action_chunk(obs, prompt)
        if self.config.mode == "gripper_lock":
            chunk = np.array(self._policy.predict_action_chunk(obs, prompt), copy=True)
            chunk[:, -1] = -1.0
            return chunk
        position, _ = self._current_pose()
        anchor, rotation6d = self._anchor
        chunk = self._make_hold_chunk(anchor, rotation6d, position, self.config.ramp_steps)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError("Hold chunk differs from the policy action contract")
        return chunk

    def summary(self):
        return dict(treatment=HALT_VERSION, config=self.config.to_dict(), halted=self.halted,
                    count_source=self.count_source, writer_frozen=self._freeze_writer and self.halted,
                    termination_head=(None if self._termination_fn is None else dict(
                        threshold=self.termination_threshold, combine=self.combine)),
                    halt_step=self.halt_step, halt_query=self.halt_query, queries=self._queries,
                    policy_queries_after_halt=(0 if self.config.mode == "hold" else None))
