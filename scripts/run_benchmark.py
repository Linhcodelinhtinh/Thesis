"""Batch Benchmark Evaluation Runner (Phase 8).

Executes automated closed-loop policy evaluation across a diverse 10-task benchmark
suite using locked initial states per SRS.md Section 8 and ADR-0010.

Outputs standardized evaluation artifacts:
- Per-episode: episode.json, diagnostics.json, timing.json, trajectory.npz, model_output.jsonl, and video.mp4.
- Batch summary: benchmark_summary.json, benchmark_summary.csv, failure_distribution.csv, and BENCHMARK_REPORT.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.evaluation.benchmark_aggregator import BenchmarkAggregator
from src.evaluation.rollout import rollout_episode
from src.models.registry import get_model_class
import src.models  # Explicitly register all policy classes (SmolVLA, MiniVLA, MiniVLA-VQ)
from src.simulator.libero_env import BenchmarkMode, LiberoEnv


# Default 10 diverse acceptance benchmark tasks per plan & pinned LIBERO commit 8f1084e
ACCEPTANCE_10_TASKS = [
    {"task_suite": "libero_object", "task_id": 0, "name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket"},
    {"task_suite": "libero_object", "task_id": 1, "name": "pick_up_the_cream_cheese_and_place_it_in_the_basket"},
    {"task_suite": "libero_object", "task_id": 2, "name": "pick_up_the_salad_dressing_and_place_it_in_the_basket"},
    {"task_suite": "libero_object", "task_id": 4, "name": "pick_up_the_ketchup_and_place_it_in_the_basket"},
    {"task_suite": "libero_spatial", "task_id": 0, "name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"},
    {"task_suite": "libero_spatial", "task_id": 2, "name": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"},
    {"task_suite": "libero_spatial", "task_id": 3, "name": "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"},
    {"task_suite": "libero_goal", "task_id": 0, "name": "open_the_middle_drawer_of_the_cabinet"},
    {"task_suite": "libero_goal", "task_id": 7, "name": "turn_on_the_stove"},
    {"task_suite": "libero_10", "task_id": 0, "name": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"},
]

# 5 representative diverse tasks across all 4 suites for fast validation
PILOT_5_TASKS = [
    {"task_suite": "libero_object", "task_id": 0, "name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket"},
    {"task_suite": "libero_spatial", "task_id": 0, "name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"},
    {"task_suite": "libero_spatial", "task_id": 3, "name": "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"},
    {"task_suite": "libero_goal", "task_id": 0, "name": "open_the_middle_drawer_of_the_cabinet"},
    {"task_suite": "libero_10", "task_id": 0, "name": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch closed-loop VLA policy benchmark (Phase 8).")
    parser.add_argument(
        "--preset",
        type=str,
        default="acceptance_10",
        choices=[
            "acceptance_10",
            "pilot_5",
            "test_5",
            "full",
            "libero_spatial",
            "spatial_10",
            "libero_object",
            "object_10",
            "libero_goal",
            "libero_10",
            "custom",
        ],
        help="Benchmark task preset to evaluate (default: acceptance_10). Shortcuts: full / all_40 (all 40 tasks), pilot_5, libero_spatial, libero_object, libero_goal, libero_10.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML benchmark configuration file (e.g., configs/benchmarks/libero_object.yaml).",
    )
    parser.add_argument(
        "--sample-level",
        type=str,
        default=None,
        choices=["pilot", "full", "custom"],
        help="Evaluation sample level: 'pilot' (10 locked states 0..9) or 'full' (50 states 0..49).",
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default=None,
        help="LIBERO task suite name when using --preset custom (e.g., libero_object).",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated task IDs when using --preset custom (e.g., --task-ids 0 1 2).",
    )
    parser.add_argument(
        "--episodes-per-task",
        "--num-episodes",
        dest="episodes_per_task",
        type=int,
        default=10,
        help="Number of initial states to evaluate per task (default: 10, using IDs 0..N-1).",
    )
    parser.add_argument(
        "--initial-state-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of initial state IDs (overrides --episodes-per-task).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="smolvla_libero",
        help="Registered model adapter name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="resources/checkpoints/smolvla_libero",
        help="Path to local official checkpoint directory.",
    )
    parser.add_argument(
        "--execution-horizon",
        "-s",
        type=int,
        default=50,
        help="Receding horizon execution steps s per predicted chunk (default: 50).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum allowable steps per episode (official benchmark standard is 1000).",
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        default=256,
        choices=[128, 256],
        help="Simulation camera resolution (default: 256 matching official LeRobot/SmolVLA training distribution).",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Record video rollouts.",
    )
    parser.add_argument(
        "--record-video-policy",
        type=str,
        default="failed_and_first",
        choices=["all", "failed_and_first", "none"],
        help="Video persistence policy: 'all', 'failed_and_first' (recommended), or 'none'.",
    )
    parser.add_argument(
        "--render-overlay",
        action="store_true",
        default=False,
        help="Render text telemetry overlay on saved video frames (default: False).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/raw_baseline/phase8_acceptance",
        help="Directory to save episode metrics, trajectories, and summary reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run policy on (cpu, cuda).",
    )
    parser.add_argument(
        "--allow-derived",
        action="store_true",
        default=False,
        help="Explicitly allow running in derived non-strict simulation configuration.",
    )
    return parser.parse_args()


def resolve_tasks(args, config_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Determine the list of tasks to evaluate."""
    if config_data and "tasks" in config_data:
        suite = config_data.get("task_suite", "libero_object")
        return [
            {
                "task_suite": suite,
                "task_id": t["task_id"],
                "name": t.get("name", f"{suite}_{t['task_id']}"),
            }
            for t in config_data["tasks"]
        ]

    if args.preset == "acceptance_10":
        return ACCEPTANCE_10_TASKS

    if args.preset in ("pilot_5", "test_5"):
        return PILOT_5_TASKS

    if args.preset in ("libero_spatial", "spatial_10"):
        return [{"task_suite": "libero_spatial", "task_id": tid, "name": f"libero_spatial_{tid}"} for tid in range(10)]

    if args.preset in ("libero_object", "object_10"):
        return [{"task_suite": "libero_object", "task_id": tid, "name": f"libero_object_{tid}"} for tid in range(10)]

    if args.preset == "libero_goal":
        return [{"task_suite": "libero_goal", "task_id": tid, "name": f"libero_goal_{tid}"} for tid in range(10)]

    if args.preset == "libero_10":
        return [{"task_suite": "libero_10", "task_id": tid, "name": f"libero_10_{tid}"} for tid in range(10)]

    if args.preset in ("full", "all_40", "full_40", "full_benchmark", "all"):
        all_tasks = []
        for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
            for tid in range(10):
                all_tasks.append({"task_suite": suite, "task_id": tid, "name": f"{suite}_{tid}"})
        return all_tasks

    if not args.task_suite:
        raise ValueError("--task-suite must be specified when using --preset custom.")

    t_ids = args.task_ids if args.task_ids is not None else [0]
    return [{"task_suite": args.task_suite, "task_id": tid, "name": f"{args.task_suite}_{tid}"} for tid in t_ids]


def main():
    args = parse_args()

    config_data = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

    tasks_to_run = resolve_tasks(args, config_data)

    # Resolve initial states and sample level tier
    if args.initial_state_ids is not None:
        init_state_ids = args.initial_state_ids
    elif args.sample_level == "full":
        init_state_ids = list(range(50))
    elif args.sample_level == "pilot":
        init_state_ids = list(range(10))
    elif config_data and "sample_levels" in config_data:
        lvl_key = args.sample_level or "pilot"
        init_state_ids = config_data["sample_levels"].get(lvl_key, {}).get("initial_state_ids", list(range(args.episodes_per_task)))
    else:
        init_state_ids = list(range(args.episodes_per_task))

    if len(init_state_ids) == 50 and set(init_state_ids) == set(range(50)):
        sample_level_tier = "FULL_50_STATES"
    else:
        sample_level_tier = f"PILOT_{len(init_state_ids)}_STATES"

    benchmark_name = "Acceptance Benchmark Baseline"
    if config_data and "benchmark_name" in config_data:
        benchmark_name = config_data["benchmark_name"]
    elif args.preset in ("libero_object", "object_10"):
        benchmark_name = "Core LIBERO-Object Benchmark (Phase 10)"
    elif args.preset == "libero_10":
        benchmark_name = "LIBERO-10 Compositional Benchmark (Phase 11)"

    print("=" * 70)
    print(f"VLA Benchmark Evaluation Runner: {benchmark_name}")
    print("=" * 70)
    print(f"Model: {args.model_name} ({args.checkpoint})")
    print(f"Execution Horizon (s): {args.execution_horizon}")
    print(f"Sample Level Tier: {sample_level_tier}")
    print(f"Tasks to Evaluate ({len(tasks_to_run)} tasks):")
    for idx, t in enumerate(tasks_to_run):
        print(f"  [{idx+1:02d}] {t['task_suite']} (Task ID {t['task_id']}): {t['name']}")
    print(f"Initial State IDs per task ({len(init_state_ids)} eps): {init_state_ids}")
    print(f"Total Episodes Planned: {len(tasks_to_run) * len(init_state_ids)}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Video Policy: {args.record_video_policy} (record_video={args.record_video})")
    print("=" * 70)

    # Fail-fast checkpoint check per AGENTS.md Rule 2
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(
            f"ERROR: Checkpoint path '{args.checkpoint}' does not exist!\n"
            "Per AGENTS.md Rule 2 & 10, mock/fallback policy execution is strictly forbidden in benchmark evaluation.\n"
            "Please download or verify the official checkpoint before running evaluation.",
            file=sys.stderr,
        )
        return 1

    # Invariants verification: 1000 steps
    if args.max_steps != 1000 and not args.allow_derived:
        print(
            f"ERROR: Benchmark invariants violation!\n"
            f"Official benchmark requires max_steps=1000 (got max_steps={args.max_steps}).\n"
            "To proceed with custom horizon configurations, pass --allow-derived.",
            file=sys.stderr,
        )
        return 1

    if args.camera_resolution == 256:
        print("[INFO] Camera resolution set to 256x256 (native LeRobot/SmolVLA training distribution).")
    else:
        print("[WARNING] Camera resolution set to 128x128 (legacy LIBERO standard). SmolVLA was trained on 256x256, so 128x128 will undergo bilinear upsampling which may cause grasping degradation.")

    # Provenance metadata per ADR-0010
    py_version = sys.version.split()[0]
    is_py38 = py_version.startswith("3.8")
    provenance = {
        "execution_tier": "STRICT_LIBERO" if is_py38 and args.max_steps == 1000 and args.camera_resolution == 128 else f"LIBERO-DERIVED (HOST_PY{sys.version_info.major}.{sys.version_info.minor})",
        "certification": "CERTIFIED_OFFICIAL" if is_py38 and args.camera_resolution == 128 else "NON-COMPARABLE_OFFICIAL_PAPER",
        "python_version": py_version,
        "sample_level": sample_level_tier,
        "benchmark_name": benchmark_name,
        "note": f"Evaluated under {sample_level_tier}. Native camera rendering per model specification.",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "execution_horizon_s": args.execution_horizon,
        "camera_resolution": args.camera_resolution,
    }

    # 1. Instantiate Policy
    print(f"\nInstantiating policy '{args.model_name}'...")
    policy_cls = get_model_class(args.model_name)
    policy = policy_cls()
    print(f"Loading checkpoint from: {args.checkpoint}...")
    policy.load(args.checkpoint, device=args.device)
    if not policy.loaded:
        print(f"ERROR: Policy '{args.model_name}' failed to load from '{args.checkpoint}'.", file=sys.stderr)
        return 1

    # 2. Initialize Aggregator
    aggregator = BenchmarkAggregator(
        benchmark_name=benchmark_name,
        provenance=provenance,
    )

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # 3. Main Benchmark Execution Loop
    total_episodes = len(tasks_to_run) * len(init_state_ids)
    global_ep_counter = 0

    for t_idx, task_spec in enumerate(tasks_to_run):
        suite = task_spec["task_suite"]
        tid = task_spec["task_id"]

        print(f"\n{'='*70}")
        print(f"Task [{t_idx+1}/{len(tasks_to_run)}]: {suite} (ID: {tid})")
        print(f"{'='*70}")

        mode = (
            BenchmarkMode.STRICT_LIBERO
            if args.max_steps == 1000 and args.camera_resolution == 128
            else BenchmarkMode.LIBERO_DERIVED
        )

        try:
            env = LiberoEnv(
                benchmark_name=suite,
                task_id=tid,
                mode=mode,
                horizon=args.max_steps,
                camera_height=args.camera_resolution,
                camera_width=args.camera_resolution,
            )
        except Exception as e:
            print(f"ERROR initializing LiberoEnv for {suite} task {tid}: {e}", file=sys.stderr)
            return 1

        actual_task_name = env.task_name
        instruction = env.language_instruction
        print(f"Canonical Task Name: {actual_task_name}")
        print(f"Instruction: \"{instruction}\"")

        for init_id in init_state_ids:
            global_ep_counter += 1
            print(f"\n--> Running Episode [{global_ep_counter}/{total_episodes}]: {actual_task_name} | Init State {init_id} (s={args.execution_horizon})...")

            ep_dir = out_base / f"{suite}_task{tid}" / f"init_{init_id}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            video_path = ep_dir / "video.mp4" if args.record_video else None

            result = rollout_episode(
                env=env,
                policy=policy,
                instruction=instruction,
                initial_state_id=init_id,
                execution_horizon=args.execution_horizon,
                max_steps=args.max_steps,
                task_name=actual_task_name,
                record_video=args.record_video,
                record_video_policy=args.record_video_policy,
                video_path=video_path,
                enable_diagnostics=True,
                render_overlay=args.render_overlay,
            )

            # Save per-episode artifacts
            ep_data = result.to_dict()
            ep_data["checkpoint"] = args.checkpoint
            ep_data["initial_state_id"] = init_id
            ep_data["provenance"] = provenance
            with open(ep_dir / "episode.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2)

            if result.diagnostics_report:
                result.save_diagnostics(ep_dir / "diagnostics.json")

            traj_kwargs = {"actions": result.actions}
            if result.states is not None and len(result.states) > 0:
                traj_kwargs["states"] = result.states
            np.savez_compressed(ep_dir / "trajectory.npz", **traj_kwargs)

            timing_data = {
                "execution_horizon_s": args.execution_horizon,
                "mean_inference_ms": result.mean_inference_ms,
                "mean_simulation_ms": result.mean_simulation_ms,
                "inference_latencies_ms": result.inference_latencies_ms,
                "simulation_latencies_ms": result.simulation_latencies_ms,
            }
            with open(ep_dir / "timing.json", "w", encoding="utf-8") as f:
                json.dump(timing_data, f, indent=2)

            result.save_model_outputs(ep_dir / "model_output.jsonl")

            # Add to aggregator
            aggregator.add_episode_result(
                task_suite=suite,
                task_id=tid,
                task_name=actual_task_name,
                instruction=instruction,
                initial_state_id=init_id,
                episode_result=result,
            )

            diag = result.diagnostics_report or {}
            print(f"    Outcome: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"    Steps: {result.num_steps} | Phase: {diag.get('failure_phase', 'N/A')} | Primary Code: {diag.get('primary_failure_code', 'N/A')}")
            print(f"    Infer Latency: {result.mean_inference_ms:.1f}ms | Sim Latency: {result.mean_simulation_ms:.1f}ms")

        env.close()

    # 4. Generate Batch Reports
    print(f"\n{'='*70}")
    print("Generating Aggregated Benchmark Reports...")
    print(f"{'='*70}")
    report_paths = aggregator.save_reports(out_base)
    for name, path in report_paths.items():
        print(f"  - {name}: {path}")

    summary = aggregator.compute_summary()
    overall = summary["overall_metrics"]
    print(f"\n{'='*70}")
    print("Benchmark Final Summary")
    print(f"{'='*70}")
    print(f"Overall Success Rate: {overall['overall_success_rate']*100:.1f}% ({overall['total_successes']}/{overall['total_episodes']})")
    print(f"95% Wilson CI: [{overall['wilson_ci_95'][0]*100:.1f}%, {overall['wilson_ci_95'][1]*100:.1f}%]")
    print(f"Mean Inference Latency: {overall['mean_inference_latency_ms']:.2f} ms")
    print(f"Reports saved to: {out_base.resolve()}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
