"""Closed-loop evaluation runner for LIBERO-Mem tasks across experimental conditions.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 10 & 11:
- Evaluates policy in closed loop with execution horizon s (default 10).
- Supports B0/F0 (memory off), F1 (predicted memory), and O1 (privileged oracle).
- Memory writer updates at every observation step t using obs[t] and executed a[t-1].
- Requeries policy only at query boundaries (every s steps).
- Records comprehensive latency, token usage, SCR progress, and overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.event_writer import LiberoMemEventWriter
from src.memory.libero_mem_oracle import LiberoMemOracleWriter

if TYPE_CHECKING:
    from src.simulator.libero_mem_env import LiberoMemEnv


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
    ) -> None:
        self.env = env
        self.policy = policy_adapter
        self.condition = condition
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps
        self.writer_factory = writer_factory
        if condition not in ("B0", "F0", "F1", "O1"):
            raise ValueError("Unknown memory condition")
        if not 1 <= execution_horizon <= 50 or max_steps < 1:
            raise ValueError("Invalid execution horizon/step budget")
        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = LiberoMemTextRenderer(
                max_total_tokens=48,
                allow_oracle=(condition == "O1"),
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
        if self.condition == "F1" and self.writer_factory is None:
            raise ValueError("F1 requires an explicit RGB writer_factory with the online camera convention")
        if getattr(self.env, "use_real_sim", None) is False:
            raise RuntimeError("Benchmark runner refuses the simulator's kinematic fixture/fallback mode")
        if hasattr(self.policy, "reset") and callable(self.policy.reset):
            self.policy.reset()

        obs = self.env.reset(initial_state_id=initial_state_id)
        start_time = time.perf_counter()

        # Initialize writers based on condition
        writer: Optional[LiberoMemEventWriter] = None
        oracle_writer: Optional[LiberoMemOracleWriter] = None

        if self.condition == "F1":
            writer = self.writer_factory(
                episode_id=episode_id,
                target_object_id="bowl_A",
                support_id="plate_A",
            )
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

        def update_writer(index, frame, executed):
            started = time.perf_counter()
            snapshot = writer.step(index, frame, executed)
            memory_trace.append({"step": index, "snapshot": snapshot.to_dict(),
                                 "signals": writer.last_signals,
                                 "writer_ms": (time.perf_counter() - started) * 1000.0})

        # Initial step 0 observation ingestion
        if self.condition == "F1" and writer is not None:
            update_writer(0, obs, None)
        elif self.condition == "O1" and oracle_writer is not None:
            if "oracle_signals" not in obs:
                raise ValueError("O1 requires measured reset oracle_signals; initial support cannot be fabricated")
            oracle_writer.step(0, obs["oracle_signals"])

        while step < self.max_steps and not done:
            # 1. Retrieve current memory snapshot
            mem_snapshot = None
            if self.condition == "F1" and writer is not None:
                mem_snapshot = writer.get_snapshot()
            elif self.condition == "O1" and oracle_writer is not None:
                mem_snapshot = oracle_writer.get_snapshot()

            # 2. Render prompt
            mem_enabled = self.condition in ("F1", "O1")
            render_res = self.renderer.render(instruction, mem_snapshot, enabled=mem_enabled)
            active_prompt = render_res.text
            prompt_history.append(active_prompt)

            # 3. Query policy (fail-fast: mock/fallback execution strictly forbidden)
            t_policy_start = time.perf_counter()
            if not hasattr(self.policy, "predict_action_chunk") or not callable(self.policy.predict_action_chunk):
                raise TypeError(
                    f"Policy adapter {type(self.policy).__name__} does not implement callable 'predict_action_chunk'. "
                    "Mock or fallback execution is strictly forbidden in benchmark runs per AGENTS.md Rule 2."
                )
            chunk = self.policy.predict_action_chunk(obs, active_prompt)
            if not isinstance(chunk, np.ndarray) or chunk.shape != (50, 7) or not np.isfinite(chunk).all():
                raise ValueError(
                    f"Invalid action chunk returned by policy: shape={getattr(chunk, 'shape', None)}, "
                    f"finite={np.isfinite(chunk).all() if isinstance(chunk, np.ndarray) else False}"
                )
            t_policy_end = time.perf_counter()
            latencies_ms.append((t_policy_end - t_policy_start) * 1000.0)

            # 4. Execute action chunk for execution_horizon steps
            for action_idx in range(self.execution_horizon):
                if step >= self.max_steps or done:
                    break

                action = chunk[action_idx]
                obs, reward, done, info = self.env.step(action)
                last_action = action
                step += 1

                # Update writer between chunk steps
                if self.condition == "F1" and writer is not None:
                    update_writer(step, obs, last_action)
                elif self.condition == "O1" and oracle_writer is not None:
                    oracle_signals = info.get("oracle_signals", {})
                    oracle_writer.step(step, oracle_signals)

        total_wall_time = time.perf_counter() - start_time
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        eval_res = info.get("eval_result")
        if eval_res is None:
            raise RuntimeError(
                f"Missing required 'eval_result' in environment step info for episode {episode_id}. "
                "Synthesizing dummy timeout or falling back is strictly forbidden per AGENTS.md Rule 10."
            )

        success = eval_res.full_success
        scr = eval_res.scr
        completed_sg = eval_res.completed_subgoals
        total_sg = eval_res.total_subgoals
        overshot = eval_res.overshot
        confirmed_cycles = eval_res.confirmed_cycles
        target_cycles = eval_res.target_cycles
        term_reason = eval_res.termination_reason

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
        )
