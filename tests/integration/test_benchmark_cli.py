"""Integration tests for scripts/run_benchmark.py CLI runner (Phase 8).

Offline verification of:
- Argument parsing and presets resolution
- Invariant validation and fail-fast triggers
- Mocked end-to-end benchmark execution flow
- Generation and integrity of all evaluation artifacts:
  - Per-episode: episode.json, diagnostics.json, timing.json, trajectory.npz, model_output.jsonl
  - Batch: benchmark_summary.json, benchmark_summary.csv, failure_distribution.csv, BENCHMARK_REPORT.md
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from scripts.run_benchmark import parse_args, resolve_tasks, PILOT_5_TASKS, ACCEPTANCE_10_TASKS
from src.evaluation.rollout import EpisodeResult


def test_resolve_tasks_presets():
    """Verify task resolution for all benchmark presets."""
    # Test acceptance_10
    args = MagicMock()
    args.preset = "acceptance_10"
    tasks = resolve_tasks(args)
    assert len(tasks) == 10
    assert tasks == ACCEPTANCE_10_TASKS

    # Test pilot_5 and test_5
    args.preset = "pilot_5"
    tasks = resolve_tasks(args)
    assert len(tasks) == 5
    assert tasks == PILOT_5_TASKS

    args.preset = "test_5"
    assert resolve_tasks(args) == PILOT_5_TASKS

    # Test suite presets
    for suite, shortcut in [("libero_spatial", "spatial_10"), ("libero_object", "object_10")]:
        args.preset = suite
        suite_tasks = resolve_tasks(args)
        assert len(suite_tasks) == 10
        assert all(t["task_suite"] == suite for t in suite_tasks)

        args.preset = shortcut
        assert len(resolve_tasks(args)) == 10

    args.preset = "libero_goal"
    goal_tasks = resolve_tasks(args)
    assert len(goal_tasks) == 10
    assert all(t["task_suite"] == "libero_goal" for t in goal_tasks)

    args.preset = "libero_10"
    l10_tasks = resolve_tasks(args)
    assert len(l10_tasks) == 10
    assert all(t["task_suite"] == "libero_10" for t in l10_tasks)

    # Test full 40 tasks presets
    for full_preset in ["full", "all_40", "full_40", "full_benchmark", "all"]:
        args.preset = full_preset
        full_tasks = resolve_tasks(args)
        assert len(full_tasks) == 40
        suites_present = set(t["task_suite"] for t in full_tasks)
        assert suites_present == {"libero_spatial", "libero_object", "libero_goal", "libero_10"}

    # Test custom preset
    args.preset = "custom"
    args.task_suite = "libero_spatial"
    args.task_ids = [0, 2, 4]
    custom_tasks = resolve_tasks(args)
    assert len(custom_tasks) == 3
    assert [t["task_id"] for t in custom_tasks] == [0, 2, 4]

    # Custom without suite must raise ValueError
    args.task_suite = None
    with pytest.raises(ValueError, match="--task-suite must be specified"):
        resolve_tasks(args)


def test_benchmark_cli_fail_fast_checkpoint(tmp_path: Path):
    """Verify fail-fast when checkpoint does not exist (Rule 2 & 10)."""
    from scripts.run_benchmark import main

    test_args = [
        "run_benchmark.py",
        "--preset", "pilot_5",
        "--checkpoint", str(tmp_path / "nonexistent_checkpoint"),
        "--output-dir", str(tmp_path / "out"),
    ]
    with patch("sys.argv", test_args):
        ret = main()
        assert ret == 1


def test_benchmark_cli_mocked_execution(tmp_path: Path):
    """Verify mocked end-to-end benchmark execution and artifact generation."""
    from scripts.run_benchmark import main

    # Create dummy checkpoint dir
    ckpt_dir = tmp_path / "dummy_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")

    out_dir = tmp_path / "benchmark_run"

    mock_diag_dict = {
        "termination_reason": "SUCCESS",
        "failure_phase": "NONE",
        "primary_failure_code": None,
        "evidence": {"ever_grasped": True, "ever_lifted": True, "ever_placed": True},
    }

    dummy_result = EpisodeResult(
        task_name="libero_object_0",
        instruction="pick up the alphabet soup",
        execution_horizon=50,
        num_steps=30,
        success=True,
        total_reward=1.0,
        actions=np.zeros((30, 7), dtype=np.float32),
        states=np.zeros((30, 9), dtype=np.float32),
        mean_inference_ms=12.5,
        mean_simulation_ms=2.0,
        inference_latencies_ms=[12.0, 13.0],
        simulation_latencies_ms=[2.0, 2.0],
        diagnostics_report=mock_diag_dict,
        model_outputs=[{"step": 0, "action": [0.0] * 7}],
    )

    test_args = [
        "run_benchmark.py",
        "--preset", "custom",
        "--task-suite", "libero_object",
        "--task-ids", "0",
        "--episodes-per-task", "1",
        "--checkpoint", str(ckpt_dir),
        "--output-dir", str(out_dir),
        "--allow-derived",
        "--camera-resolution", "256",
    ]

    with patch("sys.argv", test_args), \
         patch("scripts.run_benchmark.get_model_class") as mock_get_cls, \
         patch("scripts.run_benchmark.LiberoEnv") as mock_env_cls, \
         patch("scripts.run_benchmark.rollout_episode", return_value=dummy_result) as mock_rollout:

        mock_policy = MagicMock()
        mock_get_cls.return_value.return_value = mock_policy

        mock_env = MagicMock()
        mock_env.task_name = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
        mock_env.language_instruction = "pick up the alphabet soup and place it in the basket"
        mock_env.get_provenance_metadata.return_value = {"libero_commit": "8f1084e"}
        mock_env_cls.return_value = mock_env

        ret = main()
        assert ret is None or ret == 0

    # Verify per-episode artifacts in output dir
    ep_dir = out_dir / "libero_object_task0" / "init_0"
    assert ep_dir.exists()
    assert (ep_dir / "episode.json").exists()
    assert (ep_dir / "diagnostics.json").exists()
    assert (ep_dir / "timing.json").exists()
    assert (ep_dir / "trajectory.npz").exists()
    assert (ep_dir / "model_output.jsonl").exists()

    # Verify trajectory.npz contents
    npz_data = np.load(ep_dir / "trajectory.npz")
    assert "actions" in npz_data
    assert "states" in npz_data
    assert npz_data["actions"].shape == (30, 7)
    assert npz_data["states"].shape == (30, 9)

    # Verify batch artifacts
    assert (out_dir / "benchmark_summary.json").exists()
    assert (out_dir / "benchmark_summary.csv").exists()
    assert (out_dir / "failure_distribution.csv").exists()
    assert (out_dir / "BENCHMARK_REPORT.md").exists()

    # Verify summary JSON content
    summary_data = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary_data["overall_metrics"]["total_episodes"] == 1
    assert summary_data["overall_metrics"]["total_successes"] == 1
    assert summary_data["overall_metrics"]["overall_success_rate"] == 1.0

    task_summary = summary_data["tasks"][0]
    assert task_summary["grasp_success_rate"] == 1.0
    assert task_summary["lift_success_rate"] == 1.0
    assert task_summary["placement_success_rate"] == 1.0
    assert task_summary["mean_call_inference_latency_ms"] == pytest.approx(12.5, abs=0.1)

    # Verify report markdown
    report_md = (out_dir / "BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    assert "Acceptance Benchmark Baseline" in report_md
    assert "Grasp Rate" in report_md
    assert "Lift Rate" in report_md
    assert "Place Rate" in report_md
