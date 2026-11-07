"""Unit tests for prompt index resumption and pruning contracts."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.prepare_libero_mem_sft import recover_completed_index

ROOT = Path(__file__).resolve().parents[2]


def make_sample(ep_id, step, has_mem=True):
    return json.dumps({
        "episode_id": ep_id,
        "split": "train",
        "step": step,
        "source_hdf5": "test.hdf5",
        "source_demo": "demo_1",
        "source_obs_idx": step,
        "source_action_start_idx": step,
        "instruction": "test instruction",
        "rendered_prompt": "test prompt",
        "has_memory": has_mem,
        "tokens": 20,
        "skipped_reason": None,
    }) + "\n"


def make_trace(ep_id, step, confirmed=1, complete=True):
    return json.dumps({
        "episode_id": ep_id,
        "step": step,
        "signals": {"step": step},
        "snapshot": {
            "history_complete": complete,
            "cycles": {"bowl_plate": {"confirmed": confirmed}},
        },
    }) + "\n"


def make_detection(ep_id):
    return json.dumps({
        "episode_id": ep_id,
        "evidence": {"id": "ev_1"},
    }) + "\n"


def test_recover_empty_dir(tmp_path):
    meta, total, with_mem, completed = recover_completed_index(tmp_path)
    assert meta == []
    assert total == 0
    assert with_mem == 0
    assert completed == set()


def test_recover_completed_episodes(tmp_path):
    s_file = tmp_path / "samples.jsonl"
    t_file = tmp_path / "memory_trace.jsonl"
    d_file = tmp_path / "detections.jsonl"

    # Ep 1: 3 frames
    s_lines = [make_sample("ep1", 0, True), make_sample("ep1", 1, True), make_sample("ep1", 2, False)]
    t_lines = [make_trace("ep1", 0), make_trace("ep1", 1), make_trace("ep1", 2, confirmed=2, complete=True)]
    d_lines = [make_detection("ep1"), make_detection("ep1"), make_detection("ep1")]

    # Ep 2: 2 frames
    s_lines += [make_sample("ep2", 0, True), make_sample("ep2", 1, True)]
    t_lines += [make_trace("ep2", 0), make_trace("ep2", 1, confirmed=0, complete=False)]
    d_lines += [make_detection("ep2"), make_detection("ep2")]

    s_file.write_text("".join(s_lines), encoding="utf-8")
    t_file.write_text("".join(t_lines), encoding="utf-8")
    d_file.write_text("".join(d_lines), encoding="utf-8")

    meta, total, with_mem, completed = recover_completed_index(tmp_path)
    assert completed == {"ep1", "ep2"}
    assert total == 5
    assert with_mem == 4
    assert len(meta) == 2
    assert meta[0] == {
        "id": "ep1",
        "split": "train",
        "frames": 3,
        "memory_frames": 2,
        "confirmed_returns": 2,
        "history_complete": True,
    }
    assert meta[1] == {
        "id": "ep2",
        "split": "train",
        "frames": 2,
        "memory_frames": 2,
        "confirmed_returns": 0,
        "history_complete": False,
    }


def test_recover_prunes_incomplete_trailing_episode(tmp_path):
    s_file = tmp_path / "samples.jsonl"
    t_file = tmp_path / "memory_trace.jsonl"
    d_file = tmp_path / "detections.jsonl"

    # Ep 1: complete (2 frames)
    s_lines = [make_sample("ep1", 0, True), make_sample("ep1", 1, False)]
    t_lines = [make_trace("ep1", 0), make_trace("ep1", 1, confirmed=1, complete=True)]
    d_lines = [make_detection("ep1"), make_detection("ep1")]

    # Ep 2: cut off! samples only has 1 line, while trace and det have 2 lines
    s_lines += [make_sample("ep2", 0, True)]
    t_lines += [make_trace("ep2", 0), make_trace("ep2", 1)]
    d_lines += [make_detection("ep2"), make_detection("ep2")]

    s_file.write_text("".join(s_lines), encoding="utf-8")
    t_file.write_text("".join(t_lines), encoding="utf-8")
    d_file.write_text("".join(d_lines), encoding="utf-8")

    meta, total, with_mem, completed = recover_completed_index(tmp_path)
    assert completed == {"ep1"}
    assert total == 2
    assert with_mem == 1
    assert len(meta) == 1
    assert meta[0]["id"] == "ep1"

    # Check files were rewritten/pruned to remove ep2
    assert s_file.read_text(encoding="utf-8").count("episode_id") == 2
    assert t_file.read_text(encoding="utf-8").count("episode_id") == 2
    assert d_file.read_text(encoding="utf-8").count("episode_id") == 2
    assert "ep2" not in s_file.read_text(encoding="utf-8")


def test_cli_accepts_resume_flag():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_libero_mem_sft.py"), "--help"],
        capture_output=True, text=True, check=True
    )
    assert "--resume" in result.stdout

