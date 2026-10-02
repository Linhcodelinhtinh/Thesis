"""Unit tests for LIBERO-10 Benchmark Configuration (Phase 11).

Verifies:
- Loading and schema validation of configs/benchmarks/libero_10.yaml.
- Exact canonical 10-task ordering matching official commit 8f1084e.
- Sample levels: PILOT_10_STATES vs FULL_50_STATES.
- Diagnostics subtask tracking configuration.
"""

from pathlib import Path
import yaml
import pytest


def test_libero_10_config_loading():
    """Verify loading configs/benchmarks/libero_10.yaml."""
    cfg_path = Path("configs/benchmarks/libero_10.yaml")
    assert cfg_path.exists(), "configs/benchmarks/libero_10.yaml must exist."

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["task_suite"] == "libero_10"
    assert cfg["num_tasks"] == 10
    assert len(cfg["tasks"]) == 10


def test_libero_10_canonical_task_ordering():
    """Verify canonical 10 tasks in exact commit 8f1084e ordering."""
    cfg_path = Path("configs/benchmarks/libero_10.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_tasks = [
        (0, "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"),
        (1, "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"),
        (2, "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
        (3, "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"),
        (4, "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
        (5, "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"),
        (6, "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
        (7, "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"),
        (8, "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"),
        (9, "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"),
    ]

    for expected_id, expected_name in expected_tasks:
        task_entry = cfg["tasks"][expected_id]
        assert task_entry["task_id"] == expected_id
        assert task_entry["name"] == expected_name
        assert "subtasks" in task_entry
        assert len(task_entry["subtasks"]) >= 2


def test_libero_10_sample_levels():
    """Verify pilot (10 episodes) and full (50 episodes) sample tiers."""
    cfg_path = Path("configs/benchmarks/libero_10.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    levels = cfg["sample_levels"]
    assert "pilot" in levels
    assert "full" in levels

    pilot = levels["pilot"]
    assert pilot["tier"] == "PILOT_10_STATES"
    assert pilot["episodes_per_task"] == 10
    assert len(pilot["initial_state_ids"]) == 10

    full = levels["full"]
    assert full["tier"] == "FULL_50_STATES"
    assert full["episodes_per_task"] == 50
    assert len(full["initial_state_ids"]) == 50


def test_libero_10_diagnostics_configuration():
    """Verify diagnostics configuration for compositional evaluation."""
    cfg_path = Path("configs/benchmarks/libero_10.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    diag = cfg["diagnostics"]
    assert diag["subtask_tracking"] is True
    assert diag["track_milestone_timestamps"] is True
    assert diag["track_sequential_survival"] is True
