"""Closed-Loop Policy Rollout Engine with Receding-Horizon Execution (Phase 6).

Implements the standard closed-loop interaction loop between a VLA policy and the
LIBERO simulator environment per SRS.md Section 8 and AGENTS.md.
Supports variable receding-horizon chunk execution:
  - Total chunk size H (typically 50)
  - Execution horizon s in {1, 5, 10, 25, 50}
  - At step t, if internal executed count >= s: re-query policy for a fresh chunk.
  - Step simulator with chunk[step_in_chunk].
  - Check success predicate and record telemetry (latencies, states, actions).
  - Record model outputs (step, chunk index, action vector, latency) to model_output.jsonl.
  - Record video frames and save to MP4 video.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.models.base import VLAPolicy
from src.evaluation.diagnostics import (
    ContactDiagnostics,
    EpisodeDiagnosticsCollector,
    DiagnosticsConfig,
    DiagnosticReport,
    TerminationReason,
    FailurePhase,
)


def save_video(frames: List[np.ndarray], output_path: Union[str, Path], fps: int = 20) -> bool:
    """Save a list of RGB image frames as an MP4 video.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays in RGB format.
        output_path: Target path for the MP4 video.
        fps: Playback frame rate (default 20 Hz matching LIBERO control frequency).

    Returns:
        True if video was successfully written, False otherwise.
    """
    if not frames:
        return False

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try imageio
    try:
        import imageio
        with imageio.get_writer(str(path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return True
    except Exception:
        pass

    # 2. Try OpenCV cv2
    try:
        import cv2
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return True
    except Exception:
        pass

    return False


def apply_overlay_to_frame(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw diagnostic text overlay on a copy of a video frame (never mutating observation)."""
    try:
        import cv2
        canvas = frame.copy()
        y = 20
        for line in lines:
            # Subtle black outline, white text
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
        return canvas
    except Exception:
        return frame


@dataclass
class EpisodeResult:
    """Telemetry and outcome of a closed-loop evaluation episode."""
    task_name: str
    instruction: str
    execution_horizon: int  # s in {1, 5, 10, 25, 50}
    num_steps: int
    success: bool
    total_reward: float
    actions: np.ndarray  # (T, action_dim)
    states: Optional[np.ndarray] = None  # (T, state_dim) proprioceptive states
    inference_latencies_ms: List[float] = field(default_factory=list)
    simulation_latencies_ms: List[float] = field(default_factory=list)
    mean_inference_ms: float = 0.0
    mean_simulation_ms: float = 0.0
    model_outputs: List[Dict[str, Any]] = field(default_factory=list)
    video_frames: List[np.ndarray] = field(default_factory=list)
    video_path: Optional[str] = None
    diagnostics_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        return {
            "task_name": self.task_name,
            "instruction": self.instruction,
            "execution_horizon_s": self.execution_horizon,
            "num_steps": self.num_steps,
            "success": self.success,
            "total_reward": self.total_reward,
            "mean_inference_ms": self.mean_inference_ms,
            "mean_simulation_ms": self.mean_simulation_ms,
            "inference_latencies_ms": self.inference_latencies_ms,
            "simulation_latencies_ms": self.simulation_latencies_ms,
            "action_shape": list(self.actions.shape),
            "state_shape": list(self.states.shape) if self.states is not None else None,
            "video_path": self.video_path,
            "diagnostics": self.diagnostics_report,
        }

    def save_model_outputs(self, path: Union[str, Path]) -> None:
        """Save model output stream to JSONL file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            for item in self.model_outputs:
                f.write(json.dumps(item) + "\n")

    def save_diagnostics(self, path: Union[str, Path]) -> None:
        """Save diagnostics report to JSON file."""
        if not self.diagnostics_report:
            return
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.diagnostics_report, f, indent=2)


def rollout_episode(
    env: Any,
    policy: VLAPolicy,
    instruction: str,
    initial_state_id: Optional[int] = None,
    execution_horizon: int = 50,
    max_steps: int = 1000,
    task_name: str = "unknown_task",
    record_video: bool = False,
    video_path: Optional[Union[str, Path]] = None,
    enable_diagnostics: bool = True,
    target_object_name: Optional[str] = None,
    goal_container_name: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    record_video_policy: str = "all",  # choices: "all", "failed_and_first", "none"
    render_overlay: bool = False,
) -> EpisodeResult:
    """Execute a single closed-loop episode with receding-horizon action chunking.

    Strictly complies with official LIBERO benchmark evaluation protocol:
    - Default max_steps is 1000 per SRS.md Section 8.4 and benchmark standard.
    - Fails fast on invalid initial_state_id or action dimension.
    - Records model outputs, latencies, video frames, and diagnostics telemetry.

    Args:
        env: LiberoEnv or compliant gym/robosuite environment.
        policy: VLAPolicy adapter instance.
        instruction: Task language instruction text.
        initial_state_id: Optional initial state index (0 to 49).
        execution_horizon: s steps executed per chunk before replanning (s in {1, 5, 10, 25, 50}).
        max_steps: Maximum allowable environment steps before timeout (default 1000).
        task_name: Human-readable name of the evaluated task.
        record_video: Whether to capture RGB camera frames for video recording.
        video_path: Target path to save MP4 video if record_video is True.
        enable_diagnostics: Whether to collect physical contact, lift, and phase telemetry.
        target_object_name: Name/substring of target object for diagnostics.
        goal_container_name: Name/substring of goal fixture/container for diagnostics.
        diagnostics_config: Optional auxiliary thresholds configuration.
        record_video_policy: "all", "failed_and_first" (save if init 0 or failed), or "none".
        render_overlay: Whether to draw text diagnostics overlay on saved video frames.

    Returns:
        EpisodeResult containing outcome, telemetry, executed trajectory, model outputs, and diagnostics.
    """
    if execution_horizon < 1:
        raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")

    collector = None
    if enable_diagnostics:
        collector = EpisodeDiagnosticsCollector(
            env=env,
            target_object_name=target_object_name,
            goal_container_name=goal_container_name,
            config=diagnostics_config,
            instruction=instruction,
            task_name=task_name,
        )

    # Reset environment & policy without silent fallback
    if initial_state_id is not None:
        obs = env.reset(initial_state_id=initial_state_id)
    else:
        obs = env.reset()

    policy.reset()

    # Capture pristine ground-truth object baseline at t=0 before first action
    if collector is not None:
        collector.capture_baseline(obs)

    recorded_actions: List[np.ndarray] = []
    recorded_states: List[np.ndarray] = []
    inference_times: List[float] = []
    step_times: List[float] = []
    model_outputs: List[Dict[str, Any]] = []
    video_frames: List[np.ndarray] = []
    total_reward: float = 0.0
    success: bool = False
    termination_reason = TerminationReason.MAX_STEPS
    caught_exception: Optional[Exception] = None

    current_chunk: Optional[np.ndarray] = None
    chunk_step_idx: int = 0
    last_infer_latency: float = 0.0

    # Initial frame capture if recording video
    if record_video:
        raw_frame = None
        if hasattr(env, "render"):
            try:
                raw_frame = env.render(camera_name="agentview")
            except Exception:
                pass
        elif "agentview_image" in obs:
            raw_frame = obs["agentview_image"]

        if raw_frame is not None:
            if render_overlay:
                overlay_lines = [
                    f"Task: {task_name}",
                    f"Init State: {initial_state_id if initial_state_id is not None else 0}",
                    "Phase: RESET (t=0)",
                ]
                video_frames.append(apply_overlay_to_frame(raw_frame, overlay_lines))
            else:
                video_frames.append(raw_frame)

    for step_num in range(max_steps):
        # Determine if we need to predict a new action chunk
        is_replanned = False
        effective_horizon = min(execution_horizon, getattr(policy, "chunk_size", execution_horizon))
        if current_chunk is not None:
            effective_horizon = min(effective_horizon, len(current_chunk))

        if current_chunk is None or chunk_step_idx >= effective_horizon:
            t0_infer = time.perf_counter()
            try:
                current_chunk = policy.predict_action_chunk(obs, instruction)
            except Exception as e:
                caught_exception = e
                termination_reason = TerminationReason.POLICY_ERROR
                break
            t1_infer = time.perf_counter()

            last_infer_latency = (t1_infer - t0_infer) * 1000.0
            inference_times.append(last_infer_latency)
            chunk_step_idx = 0
            is_replanned = True

            if current_chunk.ndim != 2 or current_chunk.shape[1] != policy.action_dim:
                raise ValueError(
                    f"Predicted chunk shape mismatch: expected (*, {policy.action_dim}), "
                    f"got {current_chunk.shape}"
                )

        # Retrieve dual-action telemetry from policy if supported
        telemetry_raw_chunk = None
        telemetry_unnorm_chunk = None
        if hasattr(policy, "get_last_telemetry"):
            tel = policy.get_last_telemetry()
            telemetry_raw_chunk = tel.get("raw_normalized_chunk")
            telemetry_unnorm_chunk = tel.get("unnormalized_chunk")

        # Select action for the current step in the chunk (with safe bounds protection)
        current_step_in_chunk = min(chunk_step_idx, len(current_chunk) - 1)
        action = current_chunk[current_step_in_chunk]

        # Check for invalid action (NaN or Inf)
        if not np.all(np.isfinite(action)):
            termination_reason = TerminationReason.INVALID_ACTION
            break

        raw_norm_act = (
            [float(x) for x in telemetry_raw_chunk[current_step_in_chunk]]
            if telemetry_raw_chunk is not None and current_step_in_chunk < len(telemetry_raw_chunk)
            else None
        )
        unnorm_act = (
            [float(x) for x in telemetry_unnorm_chunk[current_step_in_chunk]]
            if telemetry_unnorm_chunk is not None and current_step_in_chunk < len(telemetry_unnorm_chunk)
            else None
        )

        chunk_step_idx += 1
        recorded_actions.append(action)

        # Step simulation environment
        t0_step = time.perf_counter()
        try:
            step_res = env.step(action)
        except Exception as e:
            caught_exception = e
            termination_reason = TerminationReason.SIMULATOR_ERROR
            break
        t1_step = time.perf_counter()
        step_times.append((t1_step - t0_step) * 1000.0)

        # Unpack gym / robosuite step return
        if len(step_res) == 4:
            next_obs, reward, done, info = step_res
        elif len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected env.step return length: {len(step_res)}")

        total_reward += float(reward)

        # Check success predicate
        if hasattr(env, "check_success"):
            success = bool(env.check_success())
        elif hasattr(env, "_check_success"):
            success = bool(env._check_success())

        # Record physical state from observation post-action (exact state transitions)
        eef_pos_actual = (
            [float(x) for x in next_obs["robot0_eef_pos"]]
            if "robot0_eef_pos" in next_obs
            else None
        )
        gripper_qpos_actual = (
            [float(x) for x in next_obs["robot0_gripper_qpos"]]
            if "robot0_gripper_qpos" in next_obs
            else None
        )

        state_vec: List[float] = []
        if "robot0_eef_pos" in next_obs:
            state_vec.extend([float(x) for x in next_obs["robot0_eef_pos"]])
        if "robot0_eef_quat" in next_obs:
            state_vec.extend([float(x) for x in next_obs["robot0_eef_quat"]])
        if "robot0_gripper_qpos" in next_obs:
            state_vec.extend([float(x) for x in next_obs["robot0_gripper_qpos"]])
        if state_vec:
            recorded_states.append(np.asarray(state_vec, dtype=np.float32))

        # Step-level telemetry evaluated on resulting physical state (zero-lag observation)
        diag_step_info = None
        if collector is not None:
            diag_step_info = collector.record_step(
                step_num=step_num,
                obs=next_obs,
                action=action,
                eef_pos=np.asarray(eef_pos_actual) if eef_pos_actual else None,
                is_replanned=is_replanned,
                infer_latency_ms=last_infer_latency if is_replanned else 0.0,
            )

        # Record model output telemetry (Dual-Action Observability - Phase 2)
        model_output_entry = {
            "step": step_num,
            "chunk_step": current_step_in_chunk,
            "raw_normalized_action": raw_norm_act,
            "unnormalized_action": unnorm_act,
            "action": [float(x) for x in action],  # Executed action in simulator
            "is_replanned": is_replanned,
            "inference_time_ms": last_infer_latency if is_replanned else None,
            "eef_pos_actual": eef_pos_actual,
            "gripper_qpos_actual": gripper_qpos_actual,
            "contact_diagnostics": diag_step_info,
            "success": success,
        }
        model_outputs.append(model_output_entry)
        obs = next_obs

        # Record video frame with optional telemetry overlay
        if record_video:
            raw_frame = None
            if hasattr(env, "render"):
                try:
                    raw_frame = env.render(camera_name="agentview")
                except Exception:
                    pass
            elif "agentview_image" in obs:
                raw_frame = obs["agentview_image"]

            if raw_frame is not None:
                if render_overlay and diag_step_info:
                    d_obj = diag_step_info.get("dist_eef_to_object")
                    delta_z = diag_step_info.get("object_delta_z", 0.0)
                    both_c = diag_step_info.get("both_fingers_contact", False)
                    overlay_lines = [
                        f"Step: {step_num} | s={execution_horizon}",
                        f"Dist to Obj: {d_obj:.3f}m" if d_obj is not None else "Dist: N/A",
                        f"Lift Delta Z: {delta_z:.3f}m | Contact: {both_c}",
                    ]
                    video_frames.append(apply_overlay_to_frame(raw_frame, overlay_lines))
                else:
                    video_frames.append(raw_frame)

        if success:
            termination_reason = TerminationReason.SUCCESS
            break
        if done:
            break

    actions_arr = (
        np.array(recorded_actions, dtype=np.float32)
        if recorded_actions
        else np.empty((0, policy.action_dim), dtype=np.float32)
    )
    states_arr = (
        np.array(recorded_states, dtype=np.float32)
        if recorded_states
        else np.empty((0,), dtype=np.float32)
    )

    mean_infer = float(np.mean(inference_times)) if inference_times else 0.0
    mean_sim = float(np.mean(step_times)) if step_times else 0.0

    # Finalize diagnostics
    diag_report_dict = None
    if collector is not None:
        report = collector.finalize(
            final_success=success,
            termination_reason=termination_reason,
            exception=caught_exception,
        )
        diag_report_dict = report.to_dict()

    # Video saving based on record_video_policy
    should_save_video = False
    if record_video and video_path and video_frames:
        if record_video_policy == "all":
            should_save_video = True
        elif record_video_policy == "failed_and_first":
            # Save only if initial_state_id == 0 or episode failed
            is_first = (initial_state_id == 0) if initial_state_id is not None else True
            should_save_video = is_first or (not success)
        elif record_video_policy == "none":
            should_save_video = False

    saved_video_path = None
    if should_save_video:
        success_save = save_video(video_frames, video_path)
        if success_save:
            saved_video_path = str(video_path)

    return EpisodeResult(
        task_name=task_name,
        instruction=instruction,
        execution_horizon=execution_horizon,
        num_steps=len(recorded_actions),
        success=success,
        total_reward=total_reward,
        actions=actions_arr,
        states=states_arr,
        inference_latencies_ms=inference_times,
        simulation_latencies_ms=step_times,
        mean_inference_ms=mean_infer,
        mean_simulation_ms=mean_sim,
        model_outputs=model_outputs,
        video_frames=video_frames,
        video_path=saved_video_path,
        diagnostics_report=diag_report_dict,
    )
