"""Loader contracts without creating a simulator in the unit-test environment."""
from pathlib import Path
import pytest
from src.simulator.libero_mem_loader import LiberoMemBDDLLoader
from src.simulator.libero_mem_env import LiberoMemEnv
from src.evaluation.libero_mem_protocol import load_protocol

ROOT=Path(__file__).resolve().parents[2]


def test_source_hashes_and_exact_task_resolution():
    loader=LiberoMemBDDLLoader()
    assert loader.resolve_bddl_path('T3').is_file()
    with pytest.raises(ValueError):loader.resolve_bddl_path('KITCHEN_SCENE1')


def test_no_fixture_fallback():
    with pytest.raises(ValueError,match='Fixture'):
        LiberoMemEnv('T1',split_lock='unused',use_real_sim=False)


def test_training_states_cannot_be_evaluation_states():
    with pytest.raises(ValueError,match='Training initial'):
        LiberoMemEnv('T1',split_lock='unused',split='train')


def test_source_modification_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):load_protocol(source_root=tmp_path)
