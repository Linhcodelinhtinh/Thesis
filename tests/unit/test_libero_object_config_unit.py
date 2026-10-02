"""Unit tests for LIBERO-Object Benchmark Configuration (Phase 10).

Verifies:
- Loading and schema validation of configs/benchmarks/libero_object.yaml.
- 10 official single-object tasks matching pinned LIBERO commit 8f1084e.
- Sample levels: PILOT_10_STATES vs FULL_50_STATES.
- Simulation invariants: Franka Panda, OSC_POSE controller, 20Hz, 1000 max steps.
"""

from pathlib import Path
import yaml
import pytest


def test_libero_object_config_loading():
    """Verify loading configs/benchmarks/libero_object.yaml."""
    cfg_path = Path("configs/benchmarks/libero_object.yaml")
    assert cfg_path.exists(), "configs/benchmarks/libero_object.yaml must exist."

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["task_suite"] == "libero_object"
    assert cfg["num_tasks"] == 10
    assert len(cfg["tasks"]) == 10


def test_libero_object_tasks_specification():
    """Verify the 10 official tasks in libero_object."""
    cfg_path = Path("configs/benchmarks/libero_object.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_tasks = [
        (0, "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
        (1, "pick_up_the_cream_cheese_and_place_it_in_the_basket"),
        (2, "pick_up_the_salad_dressing_and_place_it_in_the_basket"),
        (3, "pick_up_the_bbq_sauce_and_place_it_in_the_basket"),
        (4, "pick_up_the_ketchup_and_place_it_in_the_basket"),
        (5, "pick_up_the_tomato_sauce_and_place_it_in_the_basket"),
        (6, "pick_up_the_butter_and_place_it_in_the_basket"),
        (7, "pick_up_the_milk_and_place_it_in_the_basket"),
        (8, "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"),
        (9, "pick_up_the_orange_juice_and_place_it_in_the_basket"),
    ]

    for expected_id, expected_name in expected_tasks:
        task_entry = cfg["tasks"][expected_id]
        assert task_entry["task_id"] == expected_id
        assert task_entry["name"] == expected_name


def test_libero_object_sample_levels():
    """Verify pilot (10 episodes) and full (50 episodes) sample tiers."""
    cfg_path = Path("configs/benchmarks/libero_object.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    levels = cfg["sample_levels"]
    assert "pilot" in levels
    assert "full" in levels

    # Pilot level: 10 episodes
    pilot = levels["pilot"]
    assert pilot["tier"] == "PILOT_10_STATES"
    assert pilot["episodes_per_task"] == 10
    assert len(pilot["initial_state_ids"]) == 10
    assert pilot["initial_state_ids"] == list(range(10))

    # Full level: 50 episodes
    full = levels["full"]
    assert full["tier"] == "FULL_50_STATES"
    assert full["episodes_per_task"] == 50
    assert len(full["initial_state_ids"]) == 50
    assert full["initial_state_ids"] == list(range(50))


def test_libero_object_simulation_invariants():
    """Verify official simulation invariants."""
    cfg_path = Path("configs/benchmarks/libero_object.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation"]
    assert sim["robot"] == "Panda"
    assert sim["controller"] == "OSC_POSE"
    assert sim["control_frequency_hz"] == 20
    assert sim["max_steps_per_episode"] == 1000
    assert sim["camera_resolution"] == [256, 256]
