"""Source-discovery integrity tests; byte fixtures are never benchmark data."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_memory_training_sources import source_inventory, plan_sources, materialize_source
from src.training.split_lock import load_split_lock

ROOT = Path(__file__).resolve().parents[2]


def fixture_source(tmp_path):
    return dict(repo="test-only", revision="test-only", relative="suite/file.hdf5",
                sha256=hashlib.sha256(b"abc").hexdigest(), size_bytes=3,
                root=str(tmp_path / "destination"))


def test_plan_identifies_download_without_network_or_writes(tmp_path):
    plan = plan_sources([fixture_source(tmp_path)], tmp_path / "input")
    assert plan["status"] == "planned_not_hash_verified"
    assert plan["download_bytes"] == plan["total_source_bytes"] == 3
    assert plan["sources"][0]["origin"] == "download"
    assert not (tmp_path / "destination").exists()


def test_attached_discovery_and_duplicate_rejection(tmp_path):
    attached = tmp_path / "input" / "one" / "file.hdf5"
    attached.parent.mkdir(parents=True)
    attached.write_bytes(b"abc")
    plan = plan_sources([fixture_source(tmp_path)], tmp_path / "input")
    assert plan["download_bytes"] == 0
    assert plan["sources"][0]["origin"] == "attached"
    assert Path(plan["sources"][0]["local_path"]) == attached.resolve()
    duplicate = tmp_path / "input" / "two" / "file.hdf5"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Ambiguous"):
        plan_sources([fixture_source(tmp_path)], tmp_path / "input")


def test_partial_existing_source_is_not_replaced(tmp_path):
    source = fixture_source(tmp_path)
    path = Path(source["root"]) / source["relative"]
    path.parent.mkdir(parents=True)
    path.touch()
    with pytest.raises(ValueError, match="size mismatch"):
        plan_sources([source])
    assert path.read_bytes() == b""


@pytest.mark.parametrize("attached", [False, True])
def test_same_size_corruption_fails_before_reuse(tmp_path, attached):
    source = fixture_source(tmp_path)
    path = tmp_path / "input/file.hdf5" if attached else Path(source["root"]) / source["relative"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"xyz")
    plan = plan_sources([source], tmp_path / "input")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        materialize_source(plan["sources"][0])
    assert path.read_bytes() == b"xyz"
    if attached:
        assert not Path(plan["sources"][0]["destination"]).exists()


def test_verified_existing_source_requires_no_download(tmp_path):
    source = fixture_source(tmp_path)
    path = Path(source["root"]) / source["relative"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    plan = plan_sources([source])
    materialize_source(plan["sources"][0])
    assert plan["download_bytes"] == 0
    assert path.read_bytes() == b"abc"


def test_bundled_expanded_lock_preserves_all_previous_assignments(tmp_path):
    lock = load_split_lock(ROOT / "resources/manifests/libero_mem_t1_t3_base10_split_v1.json")
    previous = load_split_lock(ROOT / "resources/manifests/libero_mem_t1_t3_base7_split_v1.json")
    assert lock["lock_sha256"] == "bb5124d2a52912af941907ed94600deebbf467557e85575c010be8d41b35a3c3"
    assert lock["previous_lock_sha256"] == previous["lock_sha256"]
    assert len(lock["episodes"]) == 698
    assert Counter(e["split"] for e in lock["episodes"]) == {"train": 558, "val": 70, "test": 70}
    assignments = {e["initial_state_sha256"]: e["split"] for e in lock["episodes"]}
    assert all(assignments[e["initial_state_sha256"]] == e["split"] for e in previous["episodes"])
    sources = source_inventory(tmp_path / "base", tmp_path / "mem")
    assert len(sources) == 13
    assert sum(s["size_bytes"] for s in sources) == 42544733202
    assert {(s["relative"], s["sha256"]) for s in sources} == {(s["path"], s["sha256"]) for s in lock["sources"]}


def test_notebook_code_cells_compile_without_saved_execution():
    notebook = json.loads((ROOT / "notebooks/kaggle_mem_adapter_v1.ipynb").read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{i}", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
