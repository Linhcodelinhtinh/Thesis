"""Command-line runner for closed-loop VLA policy evaluation episodes (Phase 6).

Executes a policy on a specified LIBERO benchmark task under a receding-horizon
execution parameter s in {1, 5, 10, 25, 50}, recording evaluation telemetry and
saving standardized artifacts per SRS.md Section 8 and AGENTS.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.evaluation.rollout import rollout_episode
from src.models.registry import get_model_class
import src.models.smolvla.adapter  # Explicitly register SmolVLA policy class
from src.simulator.libero_env import LiberoEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run closed-loop VLA policy rollout on LIBERO.")
    parser.add_argument(
        "--task-suite",
        "--benchmark-name",
        dest="task_suite",
        type=str,
        default="libero_object",
        help="LIBERO task suite name (e.g., libero_object, libero_spatial).",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index within suite (0 to N-1).")
    parser.add_argument("--model-name", type=str, default="smolvla_libero", help="Registered model adapter name.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="resources/checkpoints/smolvla_libero",
        help="Path to local official checkpoint directory.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=50,
        choices=[1, 5, 10, 25, 50],
        help="Receding horizon execution steps s per predicted chunk.",
    )
    parser.add_argument("--initial-state-id", type=int, default=0, help="Initial state index (0 to 49).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum allowable steps (official benchmark standard is 1000).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run policy on (cpu, cuda).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/raw_baseline",
        help="Directory to save episode metrics and trajectories.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Capture camera frames and save an MP4 video of the rollout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("VLA Policy Closed-Loop Rollout Runner (Phase 6)")
    print("=" * 60)
    print(f"Task Suite: {args.task_suite}, Task ID: {args.task_id}")
    print(f"Model: {args.model_name} ({args.checkpoint})")
    print(f"Execution Horizon (s): {args.execution_horizon}")
    print(f"Initial State ID: {args.initial_state_id}")
    print(f"Max Horizon: {args.max_steps} steps")

    # Fail fast per AGENTS.md Rule 2: never run uninitialized mock policy for benchmarks
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(
            f"ERROR: Checkpoint path '{args.checkpoint}' does not exist!\n"
            "Per AGENTS.md Rule 2 & 10, mock/fallback policy execution is strictly forbidden in benchmark evaluation.\n"
            "Please download or verify the official checkpoint before running evaluation.",
            file=sys.stderr,
        )
        return 1

    # 1. Instantiate and load policy
    print(f"\nInstantiating policy '{args.model_name}'...")
    policy_cls = get_model_class(args.model_name)
    policy = policy_cls()

    print(f"Loading checkpoint from: {args.checkpoint}...")
    policy.load(args.checkpoint, device=args.device)
    if not policy.loaded:
        print(f"ERROR: Policy '{args.model_name}' failed to load from '{args.checkpoint}'.", file=sys.stderr)
        return 1

    # 2. Initialize simulation environment
    print("\nInitializing LiberoEnv...")
    env = LiberoEnv(
        benchmark_name=args.task_suite,
        task_id=args.task_id,
        horizon=args.max_steps,
    )

    task_desc = env.language_instruction
    print(f"Task Language Instruction: \"{task_desc}\"")

    provenance = env.get_provenance_metadata()
    print(f"Execution Tier: {provenance.get('execution_tier')}")

    out_dir = (
        Path(args.output_dir)
        / f"{args.task_suite}_task{args.task_id}_s{args.execution_horizon}_init{args.initial_state_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4" if args.record_video else None

    # 3. Execute rollout
    print(f"\nStarting episode rollout (max {args.max_steps} steps, s={args.execution_horizon})...")
    result = rollout_episode(
        env=env,
        policy=policy,
        instruction=task_desc,
        initial_state_id=args.initial_state_id,
        execution_horizon=args.execution_horizon,
        max_steps=args.max_steps,
        task_name=f"{args.task_suite}_{args.task_id}",
        record_video=args.record_video,
        video_path=video_path,
    )

    # 4. Save standardized evaluation artifacts
    # Save episode.json
    episode_data = result.to_dict()
    episode_data["checkpoint"] = args.checkpoint
    episode_data["initial_state_id"] = args.initial_state_id
    episode_data["provenance"] = provenance
    with open(out_dir / "episode.json", "w", encoding="utf-8") as f:
        json.dump(episode_data, f, indent=2)

    # Save trajectory.npz
    np.savez_compressed(out_dir / "trajectory.npz", actions=result.actions)

    # Save timing.json
    timing_data = {
        "execution_horizon_s": args.execution_horizon,
        "mean_inference_ms": result.mean_inference_ms,
        "mean_simulation_ms": result.mean_simulation_ms,
        "inference_latencies_ms": result.inference_latencies_ms,
        "simulation_latencies_ms": result.simulation_latencies_ms,
    }
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing_data, f, indent=2)

    # Save model_output.jsonl
    result.save_model_outputs(out_dir / "model_output.jsonl")

    print("\n" + "=" * 60)
    print("Rollout Summary")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Steps Executed: {result.num_steps}")
    print(f"Total Reward: {result.total_reward:.4f}")
    print(f"Mean Inference Latency: {result.mean_inference_ms:.2f} ms")
    print(f"Mean Step Latency: {result.mean_simulation_ms:.2f} ms")
    print(f"Artifacts saved to: {out_dir}")
    if result.video_path:
        print(f"Video saved to: {result.video_path}")
    print("=" * 60)

    env.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
