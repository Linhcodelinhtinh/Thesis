"""Closed-loop evaluation runner for LIBERO-Mem tasks across experimental conditions.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 10 & 11:
- Evaluates policy in closed loop with execution horizon s (default 10).
- Supports B0/F0 (memory off), F1 (predicted memory), and O1 (privileged oracle).
- Memory writer updates at every observation step t using obs[t] and executed a[t-1].
- Requeries policy every s steps in the parent protocol.
- Supports a separate, opt-in decision execution treatment for relation memory.
- Records comprehensive latency, token usage, SCR progress, and overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.evaluation.mem_conditions import condition_spec
from src.evaluation.memory_supervisor import MemorySupervisor

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.event_writer import LiberoMemEventWriter
from src.memory.libero_mem_oracle import LiberoMemOracleWriter

if TYPE_CHECKING:
    from src.simulator.libero_mem_env import LiberoMemEnv


def action_shape(policy, env):
    chunk_size, action_dim = policy.chunk_size, policy.action_dim
    if min(chunk_size, action_dim) < 1:
        raise ValueError("Invalid policy action contract")
    if hasattr(env, "protocol"):
        protocol = env.protocol
        if (chunk_size != protocol["rollout"]["action_chunk"] or
                action_dim != len(protocol["reset"]["warmup_action"])):
            raise ValueError("Policy action contract differs from the locked simulator protocol")
    return chunk_size, action_dim


def bind_instruction_only(policy, condition, instruction, renderer):
    adapted = getattr(policy, "mem_adapter_manifest", None) is not None
    if condition == "F1-off" or (condition == "F0" and adapted):
        policy.bind_instruction_only_runtime(condition, instruction, renderer)
    elif adapted and condition != "F1":
        raise ValueError(f"{condition} cannot use an adapted checkpoint")


@dataclass
class RolloutResult:
    """Detailed summary of a completed closed-loop evaluation episode."""

    episode_id: str
    task_name: str
    condition: str  # B0, F0, F1, O1
    success: bool
    scr: float
    completed_subgoals: int
    total_subgoals: int
    overshot: bool
    total_steps: int
    confirmed_cycles: int
    target_cycles: int
    avg_latency_ms: float
    total_wall_time_s: float
    termination_reason: str
    prompt_history: List[str] = field(default_factory=list)
    memory_trace: List[Dict[str, Any]] = field(default_factory=list)
    evaluator_provenance: Dict[str, Any] = field(default_factory=dict)
    final_memory_snapshot: Optional[Dict[str, Any]] = None
    decision_metrics: Optional[Dict[str, Any]] = None


class LiberoMemRolloutRunner:
    """Executes closed-loop evaluation episodes for LIBERO-Mem."""

    def __init__(
        self,
        env: LiberoMemEnv,
        policy_adapter: Any,
        condition: str = "F1",  # "B0", "F0", "F1", "O1"
        execution_horizon: int = 10,
        max_steps: int = 1000,
        renderer: Optional[LiberoMemTextRenderer] = None,
        writer_factory=None,
        observer=None,
        stream_memory_trace: bool = False,
        execution_treatment=None,
    ) -> None:
        self.env = env
        self.policy = policy_adapter
        self.condition = condition
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps
        self.writer_factory = writer_factory
        self.observer = observer
        if stream_memory_trace and observer is None:
            raise ValueError("Streaming memory trace requires an episode recorder")
        self.stream_memory_trace = stream_memory_trace
        self.execution_treatment = execution_treatment
        self.treatment = condition_spec(condition)
        self.action_shape = action_shape(policy_adapter, env)
        if not 1 <= execution_horizon <= self.action_shape[0] or max_steps < 1:
            raise ValueError("Invalid execution horizon/step budget")
        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = LiberoMemTextRenderer(
                max_total_tokens=self.treatment.text_budget,
                allow_oracle=self.treatment.allow_oracle,
                tokenizer=getattr(policy_adapter, "tokenizer", None),
                strict_tokenizer=condition in ("F1", "O1"),
            )

    def run_episode(
        self,
        episode_id: str,
        instruction: str,
        initial_state_id: int = 0,
    ) -> RolloutResult:
        """Run a single closed-loop episode."""
        predicted_memory = self.treatment.memory_source == "predicted_rgb_v3"
        if predicted_memory and self.writer_factory is None:
            raise ValueError("F1 requires an explicit RGB writer_factory with the online camera convention")
        if getattr(self.env, "use_real_sim", None) is False:
            raise RuntimeError("Benchmark runner refuses the simulator's kinematic fixture/fallback mode")
        if self.condition == "F0" and hasattr(self.env, "protocol") and not getattr(
                self.policy, "mem_adapter_manifest", None):
            raise ValueError("F0 requires a trained no-memory adapter")
        if not self.treatment.allow_oracle and self.renderer.allow_oracle:
            raise ValueError("This treatment forbids privileged memory")
        if hasattr(self.env, "protocol"):
            protocol = self.env.protocol["rollout"]
            if self.max_steps != protocol["horizon_policy_actions"] or self.execution_horizon != protocol["execution_horizon"]:
                raise ValueError("Rollout horizon differs from the locked protocol; create a separate protocol for an ablation")
        start_time = time.perf_counter()
        if hasattr(self.policy, "reset") and callable(self.policy.reset):
            self.policy.reset()

        bind_instruction_only(self.policy, self.condition, instruction, self.renderer)
        obs = self.env.reset(initial_state_id=initial_state_id)
        if self.observer is not None:
            self.observer("observation", step=0, observation=obs, info=getattr(self.env, "last_info", {}))

        # Initialize writers based on condition
        writer: Optional[LiberoMemEventWriter] = None
        oracle_writer: Optional[LiberoMemOracleWriter] = None

        if predicted_memory:
            writer = self.writer_factory(
                episode_id=episode_id,
                target_object_id="bowl_A",
                support_id="plate_A",
            )
            if getattr(self.policy, "mem_adapter_manifest", None) is not None:
                self.policy.bind_memory_runtime(writer, self.renderer)
        elif self.condition == "O1":
            oracle_writer = LiberoMemOracleWriter(
                episode_id=episode_id,
                target_object_id="bowl_A",
                support_id="plate_A",
            )

        step = 0
        done = False
        last_action: Optional[np.ndarray] = None
        info: Dict[str, Any] = {}
        prompt_history: List[str] = []
        latencies_ms: List[float] = []
        memory_trace: List[Dict[str, Any]] = []
        treatment = self.execution_treatment
        supervisor = None
        prefix_limit = self.execution_horizon
        abstained = False
        requery_count = 0
        triggered_checks = 0
        supervisor_ms = 0.0
        writer_total_ms = 0.0
        detector_calls = 0
        if treatment is not None:
            from src.interfaces.relation_memory_text import RelationMemoryRenderer
            from src.memory.relation_memory import RelationMemoryWriter
            if not isinstance(writer, RelationMemoryWriter) or not isinstance(self.renderer, RelationMemoryRenderer):
                raise ValueError("Decision treatment requires the relation writer and renderer")
            if self.observer is None:
                raise ValueError("Decision treatment requires a trace recorder")
            if treatment.mode in ("shadow", "event"):
                supervisor = MemorySupervisor(treatment)
            if treatment.mode == "periodic":
                prefix_limit = treatment.periodic_prefix

        def update_writer(index, frame, executed):
            nonlocal writer_total_ms, detector_calls
            started = time.perf_counter()
            snapshot = writer.step(index, frame, executed)
            writer_ms = (time.perf_counter() - started) * 1000.0
            if treatment is not None:
                writer_total_ms += writer_ms
                detector_calls += int("visual_evidence" not in frame)
            record = {"step": index, "snapshot": snapshot.to_dict(),
                      "signals": writer.last_signals, "writer_ms": writer_ms}
            if not self.stream_memory_trace:
                memory_trace.append(record)
            if self.observer is not None:
                self.observer("memory", **record)
            return snapshot, record["snapshot"]

        # Initial step 0 observation ingestion
        if predicted_memory:
            snapshot, raw_snapshot = update_writer(0, obs, None)
        elif self.condition == "O1" and oracle_writer is not None:
            if "oracle_signals" not in obs:
                raise ValueError("O1 requires measured reset oracle_signals; initial support cannot be fabricated")
            oracle_writer.step(0, obs["oracle_signals"])

        while step < self.max_steps and not done:
            if treatment is not None and len(latencies_ms) >= treatment.max_queries:
                abstained = True
                self.observer("decision", step=step, decision="abstain", reason="query_budget_exhausted",
                              queries=len(latencies_ms))
                break
            # 1. Retrieve current memory snapshot
            mem_snapshot = None
            if predicted_memory:
                mem_snapshot = writer.get_snapshot()
            elif self.condition == "O1" and oracle_writer is not None:
                mem_snapshot = oracle_writer.get_snapshot()

            # 2. Render prompt
            mem_enabled = self.treatment.memory_source != "none"
            render_res = self.renderer.render(instruction, mem_snapshot, enabled=mem_enabled)
            if getattr(render_res, "tokenizer_truncated", False):
                raise ValueError("Instruction exceeds the pinned tokenizer budget")
            active_prompt = render_res.text
            prompt_history.append(active_prompt)
            query_id = len(latencies_ms)
            query_metadata = {}
            if supervisor is not None:
                started = time.perf_counter()
                context = supervisor.bind(query_id, instruction, snapshot, raw_snapshot, self.renderer, render_res)
                supervisor_ms += (time.perf_counter() - started) * 1000
                query_metadata["decision_context"] = context.trace()

            # 3. Query policy (fail-fast: mock/fallback execution strictly forbidden)
            t_policy_start = time.perf_counter()
            if not hasattr(self.policy, "predict_action_chunk") or not callable(self.policy.predict_action_chunk):
                raise TypeError(
                    f"Policy adapter {type(self.policy).__name__} does not implement callable 'predict_action_chunk'. "
                    "Mock or fallback execution is strictly forbidden in benchmark runs per AGENTS.md Rule 2."
                )
            chunk = self.policy.predict_action_chunk(obs, active_prompt)
            if not isinstance(chunk, np.ndarray) or chunk.shape != self.action_shape or not np.isfinite(chunk).all():
                raise ValueError(
                    f"Invalid action chunk returned by policy: shape={getattr(chunk, 'shape', None)}, "
                    f"finite={np.isfinite(chunk).all() if isinstance(chunk, np.ndarray) else False}"
                )
            t_policy_end = time.perf_counter()
            latencies_ms.append((t_policy_end - t_policy_start) * 1000.0)
            if treatment is not None:
                query_metadata.update(query_id=query_id, prefix_limit=prefix_limit,
                                      chunk_sha256=hashlib.sha256(chunk.tobytes()).hexdigest())
            if self.observer is not None:
                self.observer("query", step=step, prompt=active_prompt, chunk=chunk,
                              render=render_res, policy_ms=latencies_ms[-1], **query_metadata)

            # 4. Execute action chunk for execution_horizon steps
            for action_idx in range(prefix_limit):
                if step >= self.max_steps or done:
                    break

                action = chunk[action_idx]
                obs, reward, done, info = self.env.step(action)
                if self.observer is not None:
                    self.observer("action", step=step, action=action, chunk_index=action_idx,
                                  reward=reward, done=done, **({"query_id": query_id} if treatment else {}))
                last_action = action
                step += 1
                if self.observer is not None:
                    self.observer("observation", step=step, observation=obs, info=info)

                # Update writer between chunk steps
                if predicted_memory:
                    snapshot, raw_snapshot = update_writer(step, obs, last_action)
                elif self.condition == "O1" and oracle_writer is not None:
                    oracle_signals = info.get("oracle_signals", {})
                    oracle_writer.step(step, oracle_signals)

                if supervisor is not None and not done and step < self.max_steps:
                    started = time.perf_counter()
                    changes = supervisor.evaluate(instruction, snapshot, raw_snapshot, self.renderer)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    supervisor_ms += elapsed_ms
                    if changes:
                        triggered_checks += 1
                        early = action_idx + 1 < prefix_limit
                        requery = early and treatment.mode == "event"
                        self.observer("decision", step=step, query_id=query_id,
                                      decision="requery" if requery else "continue",
                                      reason="semantic_change" if early else "normal_query_boundary",
                                      prefix_actions=action_idx + 1, changes=changes, supervisor_ms=elapsed_ms)
                        if requery:
                            requery_count += 1
                            break

        total_wall_time = time.perf_counter() - start_time
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        eval_res = info.get("eval_result")
        if eval_res is None:
            raise RuntimeError(
                f"Missing required 'eval_result' in environment step info for episode {episode_id}. "
                "Synthesizing dummy timeout or falling back is strictly forbidden per AGENTS.md Rule 10."
            )

        success = eval_res.full_success and not abstained
        scr = eval_res.scr
        completed_sg = eval_res.completed_subgoals
        total_sg = eval_res.total_subgoals
        overshot = eval_res.overshot
        confirmed_cycles = eval_res.confirmed_cycles
        target_cycles = eval_res.target_cycles
        term_reason = "decision_abstention" if abstained else eval_res.termination_reason

        return RolloutResult(
            episode_id=episode_id,
            task_name=self.env.task_name,
            condition=self.condition,
            success=success,
            scr=scr,
            completed_subgoals=completed_sg,
            total_subgoals=total_sg,
            overshot=overshot,
            total_steps=step,
            confirmed_cycles=confirmed_cycles,
            target_cycles=target_cycles,
            avg_latency_ms=avg_latency,
            total_wall_time_s=total_wall_time,
            termination_reason=term_reason,
            prompt_history=prompt_history,
            memory_trace=memory_trace,
            final_memory_snapshot=writer.get_snapshot().to_dict() if writer is not None else None,
            decision_metrics=(dict(queries=len(latencies_ms), early_requeries=requery_count,
                                   triggered_checks=triggered_checks, supervisor_ms=supervisor_ms,
                                   abstained=abstained, policy_total_ms=sum(latencies_ms),
                                   writer_total_ms=writer_total_ms, detector_calls=detector_calls)
                              if treatment is not None else None),
            evaluator_provenance={key: info[key] for key in
                ("protocol_id", "split_lock_sha256", "source_demo", "initial_state_sha256") if key in info},
        )
