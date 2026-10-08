"""Unit tests for paired memory benchmark comparison script (P6)."""

import json
from pathlib import Path
import pytest

from scripts.compare_memory_runs import compute_paired_stats, generate_paired_report


def test_compute_paired_stats_empty():
    assert compute_paired_stats([]) == {}


def test_compute_paired_stats_concordant():
    # 10 episodes: 6 both succeed, 4 both fail
    pairs = [(True, True)] * 6 + [(False, False)] * 4
    stats = compute_paired_stats(pairs)

    assert stats["total_pairs"] == 10
    assert stats["baseline_success_count"] == 6
    assert stats["treatment_success_count"] == 6
    assert stats["delta_success_rate"] == 0.0
    assert stats["mcnemar_chi2"] == 0.0
    assert stats["contingency_matrix"]["both_success"] == 6
    assert stats["contingency_matrix"]["both_failed"] == 4
    assert stats["contingency_matrix"]["baseline_only_success"] == 0
    assert stats["contingency_matrix"]["treatment_only_success"] == 0


def test_compute_paired_stats_discordant():
    # 20 episodes:
    # 5 both succeed
    # 3 baseline only
    # 8 treatment only
    # 4 both fail
    pairs = (
        [(True, True)] * 5
        + [(True, False)] * 3
        + [(False, True)] * 8
        + [(False, False)] * 4
    )
    stats = compute_paired_stats(pairs)

    assert stats["total_pairs"] == 20
    assert stats["baseline_success_count"] == 8
    assert stats["treatment_success_count"] == 13
    assert stats["baseline_success_rate"] == 0.4
    assert stats["treatment_success_rate"] == 0.65
    assert stats["delta_success_rate"] == 0.25
    assert stats["contingency_matrix"]["baseline_only_success"] == 3
    assert stats["contingency_matrix"]["treatment_only_success"] == 8
    # McNemar with continuity correction: (|3 - 8| - 1)^2 / (3 + 8) = 16 / 11 ~ 1.4545
    assert pytest.approx(stats["mcnemar_chi2"], 0.01) == 1.45


def test_generate_paired_report(tmp_path: Path):
    base_dir = tmp_path / "baseline_off"
    treat_dir = tmp_path / "treatment_text"
    base_dir.mkdir()
    treat_dir.mkdir()

    # Create 2 tasks with 2 inits each
    tasks = ["libero_spatial_pick_up", "libero_spatial_open_drawer"]
    inits = [0, 1]

    for t_idx, t in enumerate(tasks):
        for i_idx, i in enumerate(inits):
            b_ep_dir = base_dir / t / f"init_{i}"
            t_ep_dir = treat_dir / t / f"init_{i}"
            b_ep_dir.mkdir(parents=True)
            t_ep_dir.mkdir(parents=True)

            # baseline: succeed only on init 0
            b_data = {
                "task_name": t,
                "initial_state_id": i,
                "success": (i == 0),
                "num_steps": 100,
            }
            # treatment: succeed on both inits
            t_data = {
                "task_name": t,
                "initial_state_id": i,
                "success": True,
                "num_steps": 80,
                "memory": {
                    "condition": "text_only",
                    "trace": [
                        {"rendered_context_length_chars": 45},
                        {"rendered_context_length_chars": 60},
                    ],
                },
            }

            with open(b_ep_dir / "episode.json", "w", encoding="utf-8") as f:
                json.dump(b_data, f)
            with open(t_ep_dir / "episode.json", "w", encoding="utf-8") as f:
                json.dump(t_data, f)

    out_dir = tmp_path / "out"
    md_content, stats = generate_paired_report(base_dir, treat_dir, out_dir)

    assert stats["total_pairs"] == 4
    assert stats["baseline_success_count"] == 2
    assert stats["treatment_success_count"] == 4
    assert stats["delta_success_rate"] == 0.5
    assert stats["context_length_diagnostics"]["total_queries"] == 8
    assert stats["context_length_diagnostics"]["mean_chars"] == 52.5
    assert stats["context_length_diagnostics"]["max_chars"] == 60

    assert "# V2 Paired Memory Evaluation Report" in md_content
    assert "LIBERO-DERIVED (NON-COMPARABLE_OFFICIAL_PAPER)" in md_content
    assert "libero_spatial_pick_up" in md_content
