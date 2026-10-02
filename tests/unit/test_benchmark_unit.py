"""Unit tests for BenchmarkAggregator and batch reporting (Phase 8).

Fast, offline unit tests (< 1s) validating:
- Wilson score interval calculation
- Aggregation across tasks and episodes
- Grasp rate and placement rate calculations with defined denominators
- Generation of benchmark_summary.json, summary.csv, failure_distribution.csv, and BENCHMARK_REPORT.md
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.evaluation.benchmark_aggregator import BenchmarkAggregator, wilson_score_interval


def test_wilson_score_interval():
    """Verify Wilson score confidence interval calculations."""
    # Case 1: 10/10 successes
    low, high = wilson_score_interval(10, 10)
    assert high == 1.0
    assert 0.70 < low < 0.75

    # Case 2: 0/10 successes
    low, high = wilson_score_interval(0, 10)
    assert low == 0.0
    assert 0.25 < high < 0.30

    # Case 3: 5/10 successes
    low, high = wilson_score_interval(5, 10)
    assert 0.20 < low < 0.28
    assert 0.72 < high < 0.80

    # Case 4: 0 total
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_benchmark_aggregator_multi_task(tmp_path: Path):
    """Verify aggregation across 2 tasks and report generation."""
    aggregator = BenchmarkAggregator(
        benchmark_name="Test Pilot Benchmark",
        provenance={"execution_tier": "LIBERO-DERIVED", "model_name": "mock_policy"},
    )

    # Task 1: 2 episodes (1 success, 1 failure)
    ep1_succ = {
        "num_steps": 25,
        "success": True,
        "mean_inference_ms": 15.0,
        "mean_simulation_ms": 2.5,
        "inference_latencies_ms": [14.0, 16.0],
        "diagnostics": {
            "termination_reason": "SUCCESS",
            "failure_phase": "NONE",
            "primary_failure_code": None,
            "evidence": {"ever_grasped": True, "ever_lifted": True, "ever_placed": True},
        },
    }
    ep1_fail = {
        "num_steps": 100,
        "success": False,
        "mean_inference_ms": 16.0,
        "mean_simulation_ms": 2.6,
        "inference_latencies_ms": [15.5, 16.5],
        "diagnostics": {
            "termination_reason": "MAX_STEPS",
            "failure_phase": "GRASP",
            "primary_failure_code": "UNATTRIBUTED",
            "evidence": {"ever_grasped": False, "ever_lifted": False, "ever_placed": False},
        },
    }
    aggregator.add_episode_result(
        task_suite="libero_object",
        task_id=0,
        task_name="task_0_soup",
        instruction="pick soup",
        initial_state_id=0,
        episode_result=ep1_succ,
    )
    aggregator.add_episode_result(
        task_suite="libero_object",
        task_id=0,
        task_name="task_0_soup",
        instruction="pick soup",
        initial_state_id=1,
        episode_result=ep1_fail,
    )

    # Task 2: 1 episode (reach failure)
    ep2_fail = {
        "num_steps": 100,
        "success": False,
        "mean_inference_ms": 14.0,
        "mean_simulation_ms": 2.4,
        "diagnostics": {
            "termination_reason": "MAX_STEPS",
            "failure_phase": "REACH",
            "primary_failure_code": "UNATTRIBUTED",
            "evidence": {"ever_lifted": False, "ever_placed": False},
        },
    }
    aggregator.add_episode_result(
        task_suite="libero_spatial",
        task_id=1,
        task_name="task_1_bowl",
        instruction="pick bowl",
        initial_state_id=0,
        episode_result=ep2_fail,
    )

    # Compute summary
    summary = aggregator.compute_summary()
    assert summary["overall_metrics"]["total_tasks"] == 2
    assert summary["overall_metrics"]["total_episodes"] == 3
    assert summary["overall_metrics"]["total_successes"] == 1
    assert summary["overall_metrics"]["overall_success_rate"] == pytest.approx(1 / 3, abs=1e-3)

    tasks = summary["tasks"]
    assert len(tasks) == 2
    t0 = next(t for t in tasks if t["task_id"] == 0)
    assert t0["total_episodes"] == 2
    assert t0["success_count"] == 1
    assert t0["success_rate"] == 0.5
    assert t0["grasp_success_rate"] == 0.5
    assert t0["lift_success_rate"] == 0.5
    assert t0["placement_success_rate"] == 0.5
    assert t0["mean_steps_to_success"] == 25.0
    assert t0["total_inference_calls"] == 4
    assert t0["failure_phase_counts"]["GRASP"] == 1

    t1 = next(t for t in tasks if t["task_id"] == 1)
    assert t1["success_rate"] == 0.0
    assert t1["failure_phase_counts"]["REACH"] == 1

    # Save reports and verify files
    report_paths = aggregator.save_reports(tmp_path)
    assert report_paths["summary_json"].exists()
    assert report_paths["summary_csv"].exists()
    assert report_paths["failure_csv"].exists()
    assert report_paths["report_md"].exists()

    # Verify JSON content
    loaded_json = json.loads(report_paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded_json["overall_metrics"]["total_episodes"] == 3

    # Verify Markdown content
    md_text = report_paths["report_md"].read_text(encoding="utf-8")
    assert "# Test Pilot Benchmark" in md_text
    assert "ADR-0010" in md_text
    assert "`libero_object`" in md_text
    assert "`GRASP`" in md_text


def test_benchmark_aggregator_subtask_progression(tmp_path: Path):
    """Verify multi-stage subtask milestone aggregation and report formatting (Phase 11)."""
    aggregator = BenchmarkAggregator(
        benchmark_name="LIBERO-10 Milestone Test",
        provenance={"execution_tier": "LIBERO-DERIVED"},
    )

    # Episode 1: Full success, both milestones achieved
    ep1 = {
        "num_steps": 40,
        "success": True,
        "mean_inference_ms": 15.0,
        "mean_simulation_ms": 2.0,
        "diagnostics": {
            "termination_reason": "SUCCESS",
            "failure_phase": "NONE",
            "primary_failure_code": None,
            "evidence": {"ever_grasped": True, "ever_lifted": True, "ever_placed": True},
            "summary_metrics": {"sequential_survival_steps": 40, "subtask_completion_rate": 1.0},
            "subtask_milestones": [
                {"name": "m0_soup", "description": "Put soup in basket", "achieved": True, "first_achieved_step": 15},
                {"name": "m1_sauce", "description": "Put sauce in basket", "achieved": True, "first_achieved_step": 38},
            ],
        },
    }

    # Episode 2: Milestone 0 achieved, failed transition on Milestone 1
    ep2 = {
        "num_steps": 100,
        "success": False,
        "mean_inference_ms": 15.0,
        "mean_simulation_ms": 2.0,
        "diagnostics": {
            "termination_reason": "MAX_STEPS",
            "failure_phase": "GRASP",
            "primary_failure_code": "UNATTRIBUTED",
            "evidence": {"ever_grasped": True, "ever_lifted": True, "ever_placed": False},
            "summary_metrics": {"sequential_survival_steps": 18, "subtask_completion_rate": 0.5},
            "subtask_milestones": [
                {"name": "m0_soup", "description": "Put soup in basket", "achieved": True, "first_achieved_step": 18},
                {"name": "m1_sauce", "description": "Put sauce in basket", "achieved": False, "first_achieved_step": None},
            ],
        },
    }

    # Episode 3: Failed before achieving milestone 0
    ep3 = {
        "num_steps": 100,
        "success": False,
        "mean_inference_ms": 15.0,
        "mean_simulation_ms": 2.0,
        "diagnostics": {
            "termination_reason": "MAX_STEPS",
            "failure_phase": "REACH",
            "primary_failure_code": "UNATTRIBUTED",
            "evidence": {"ever_grasped": False, "ever_lifted": False, "ever_placed": False},
            "summary_metrics": {"sequential_survival_steps": 0, "subtask_completion_rate": 0.0},
            "subtask_milestones": [
                {"name": "m0_soup", "description": "Put soup in basket", "achieved": False, "first_achieved_step": None},
                {"name": "m1_sauce", "description": "Put sauce in basket", "achieved": False, "first_achieved_step": None},
            ],
        },
    }

    aggregator.add_episode_result("libero_10", 0, "task_0_soup_sauce", "put both", 0, ep1)
    aggregator.add_episode_result("libero_10", 0, "task_0_soup_sauce", "put both", 1, ep2)
    aggregator.add_episode_result("libero_10", 0, "task_0_soup_sauce", "put both", 2, ep3)

    summary = aggregator.compute_summary()
    t0 = summary["tasks"][0]

    assert t0["subtask_milestones"]["m0_soup"]["achieved_count"] == 2
    assert t0["subtask_milestones"]["m0_soup"]["achieved_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert t0["subtask_milestones"]["m1_sauce"]["achieved_count"] == 1
    assert t0["subtask_milestones"]["m1_sauce"]["achieved_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert t0["sequence_transition_failures"] == 1
    assert t0["atomic_manipulation_failures"] == 1
    assert t0["mean_sequential_survival_steps"] == pytest.approx((40 + 18 + 0) / 3, abs=1e-2)

    report_paths = aggregator.save_reports(tmp_path)
    md_text = report_paths["report_md"].read_text(encoding="utf-8")
    assert "Multi-Stage Subtask Progression" in md_text
    assert "m0_soup" in md_text
    assert "m1_sauce" in md_text
