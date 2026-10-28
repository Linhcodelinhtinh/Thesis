"""Unit tests for LIBERO-Mem pilot benchmark configuration."""

from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_libero_mem_pilot_config_exists_and_valid():
    config_path = REPO_ROOT / "configs/benchmarks/libero_mem_pilot.yaml"
    assert config_path.is_file(), f"Missing config file: {config_path}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["task_suite"] == "libero_mem_pilot"
    assert config["provenance_tag"] == "CUSTOM_V2_LIBERO_MEM"
    assert config["official_paper_comparable"] is False
    assert "NON-COMPARABLE_OFFICIAL_PAPER" in config["comparability_note"]


def test_libero_mem_pilot_tasks_specification():
    config_path = REPO_ROOT / "configs/benchmarks/libero_mem_pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = config["tasks"]
    assert len(tasks) >= 2

    # Check T1
    t1 = next(t for t in tasks if t["alias"] == "T1")
    assert t1["target_cycles"] == 1
    assert t1["target_object"] == "bowl"
    assert t1["support_object"] == "plate"
    assert len(t1["subtasks"]) == 1

    # Check T3
    t3 = next(t for t in tasks if t["alias"] == "T3")
    assert t3["target_cycles"] == 3
    assert t3["target_object"] == "bowl"
    assert t3["support_object"] == "plate"
    assert len(t3["subtasks"]) == 3


def test_libero_mem_simulation_invariants():
    config_path = REPO_ROOT / "configs/benchmarks/libero_mem_pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    sim = config["simulation"]
    assert sim["robot"] == "Panda"
    assert sim["controller"] == "OSC_POSE"
    assert sim["control_frequency_hz"] == 20
    assert sim["camera_names"] == ["agentview", "robot0_eye_in_hand"]

    mem = config["memory_contract"]
    assert mem["schema_version"] == 2
    assert mem["max_total_tokens"] == 48
    assert mem["observation_action_offset"] == 0
    assert mem["allow_oracle_in_system"] is False
