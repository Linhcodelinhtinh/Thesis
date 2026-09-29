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
from src.evaluation.diagnostics import ContactDiagnostics


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
    inference_latencies_ms: List[float] = field(default_factory=list)
    simulation_latencies_ms: List[float] = field(default_factory=list)
    mean_inference_ms: float = 0.0
    mean_simulation_ms: float = 0.0
    model_outputs: List[Dict[str, Any]] = field(default_factory=list)
    video_frames: List[np.ndarray] = field(default_factory=list)
    video_path: Optional[str] = None

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
            "action_shape": list(self.actions.shape),
            "video_path": self.video_path,
        }

    def save_model_outputs(self, path: Union[str, Path]) -> None:
        """Save model output stream to JSONL file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            for item in self.model_outputs:
                f.write(json.dumps(item) + "\n")


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
    enable_diagnostics: bool = False,
    target_object_name: Optional[str] = None,
) -> EpisodeResult:
    """Execute a single closed-loop episode with receding-horizon action chunking.

    Strictly complies with official LIBERO benchmark evaluation protocol:
    - Default max_steps is 1000 per SRS.md Section 8.4 and benchmark standard.
    - Fails fast on invalid initial_state_id or action dimension.
    - Records model outputs, latencies, and video frames.

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
        enable_diagnostics: Whether to collect physical contact and lift telemetry.
        target_object_name: Name/substring of target object for diagnostics.

    Returns:
        EpisodeResult containing outcome, telemetry, executed trajectory, and model outputs.
    """
    if execution_horizon < 1:
        raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")

    diag = ContactDiagnostics(env=env, target_object_name=target_object_name) if enable_diagnostics else None

    # Reset environment & policy without silent fallback
    if initial_state_id is not None:
        obs = env.reset(initial_state_id=initial_state_id)
    else:
        obs = env.reset()

    policy.reset()

    recorded_actions: List[np.ndarray] = []
    inference_times: List[float] = []
    step_times: List[float] = []
    model_outputs: List[Dict[str, Any]] = []
    video_frames: List[np.ndarray] = []
    total_reward: float = 0.0
    success: bool = False

    current_chunk: Optional[np.ndarray] = None
    chunk_step_idx: int = 0
    last_infer_latency: float = 0.0

    # Initial frame capture if recording video
    if record_video:
        if hasattr(env, "render"):
            try:
                frame = env.render(camera_name="agentview")
                video_frames.append(frame)
            except Exception:
                pass
        elif "agentview_image" in obs:
            video_frames.append(obs["agentview_image"])

    for step_num in range(max_steps):
        # Determine if we need to predict a new action chunk
        is_replanned = False
        if current_chunk is None or chunk_step_idx >= execution_horizon:
            t0_infer = time.perf_counter()
            current_chunk = policy.predict_action_chunk(obs, instruction)
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

        # Select action for the current step in the chunk
        action = current_chunk[chunk_step_idx]
        raw_norm_act = (
            [float(x) for x in telemetry_raw_chunk[chunk_step_idx]]
            if telemetry_raw_chunk is not None and chunk_step_idx < len(telemetry_raw_chunk)
            else None
        )
        unnorm_act = (
            [float(x) for x in telemetry_unnorm_chunk[chunk_step_idx]]
            if telemetry_unnorm_chunk is not None and chunk_step_idx < len(telemetry_unnorm_chunk)
            else None
        )

        chunk_step_idx += 1
        recorded_actions.append(action)

        # Record physical state from observation before action
        eef_pos_actual = (
            [float(x) for x in obs["robot0_eef_pos"]]
            if "robot0_eef_pos" in obs
            else None
        )
        gripper_qpos_actual = (
            [float(x) for x in obs["robot0_gripper_qpos"]]
            if "robot0_gripper_qpos" in obs
            else None
        )

        # Record model output telemetry (Dual-Action Observability - Phase 2)
        model_output_entry = {
            "step": step_num,
            "chunk_step": chunk_step_idx - 1,
            "raw_normalized_action": raw_norm_act,
            "unnormalized_action": unnorm_act,
            "action": [float(x) for x in action],  # Executed action in simulator
            "is_replanned": is_replanned,
            "inference_time_ms": last_infer_latency if is_replanned else None,
            "eef_pos_actual": eef_pos_actual,
            "gripper_qpos_actual": gripper_qpos_actual,
        }

        # Step simulation environment
        t0_step = time.perf_counter()
        step_res = env.step(action)
        t1_step = time.perf_counter()
        step_times.append((t1_step - t0_step) * 1000.0)

        # Unpack gym / robosuite step return
        if len(step_res) == 4:
            obs, reward, done, info = step_res
        elif len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected env.step return length: {len(step_res)}")

        total_reward += float(reward)

        # Check success predicate
        if hasattr(env, "check_success"):
            success = bool(env.check_success())
        elif hasattr(env, "_check_success"):
            success = bool(env._check_success())

        # Physical contact & lift diagnostics (Phase 4)
        if diag is not None:
            c_info = diag.inspect_gripper_contacts()
            l_info = diag.inspect_object_lift()
            model_output_entry["contact_diagnostics"] = {
                "any_contact": c_info["any_contact"],
                "both_fingers_contact": c_info["both_fingers_contact"],
                "left_finger_contact": c_info["left_finger_contact"],
                "right_finger_contact": c_info["right_finger_contact"],
                "is_lifted": l_info["is_lifted"],
                "delta_z": l_info["delta_z"],
            }

        model_output_entry["success"] = success
        model_outputs.append(model_output_entry)

        # Record video frame
        if record_video:
            if hasattr(env, "render"):
                try:
                    frame = env.render(camera_name="agentview")
                    video_frames.append(frame)
                except Exception:
                    pass
            elif "agentview_image" in obs:
                video_frames.append(obs["agentview_image"])

        if success or done:
            break

    actions_arr = (
        np.array(recorded_actions, dtype=np.float32)
        if recorded_actions
        else np.empty((0, policy.action_dim), dtype=np.float32)
    )

    mean_infer = float(np.mean(inference_times)) if inference_times else 0.0
    mean_sim = float(np.mean(step_times)) if step_times else 0.0

    # Save video if requested
    saved_video_path = None
    if record_video and video_path and video_frames:
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
        inference_latencies_ms=inference_times,
        simulation_latencies_ms=step_times,
        mean_inference_ms=mean_infer,
        mean_simulation_ms=mean_sim,
        model_outputs=model_outputs,
        video_frames=video_frames,
        video_path=saved_video_path,
    )
