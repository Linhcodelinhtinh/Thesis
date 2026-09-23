"""Fast unit tests for DemonstrationReplayer (Phase 3)."""

import json
from pathlib import Path
import h5py
import numpy as np
import pytest

from src.evaluation.demo_replay import DemonstrationReplayer, ReplayMetrics


def test_missing_hdf5_file_raises_error(tmp_path):
    """Verify that a nonexistent HDF5 path raises FileNotFoundError."""
    missing_path = tmp_path / "nonexistent.hdf5"
    with pytest.raises(FileNotFoundError, match="not found"):
        DemonstrationReplayer(missing_path)


def test_hdf5_structure_and_metadata_parsing(tmp_path):
    """Verify parsing of demo keys, env_args, actions, and states from HDF5."""
    hdf5_file = tmp_path / "test_demo.hdf5"

    env_args = {
        "env_name": "LiberoEnv",
        "problem_name": "Libero_Tabletop_Manipulation",
        "env_kwargs": {"control_freq": 20, "horizon": 100},
    }

    with h5py.File(hdf5_file, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_args)

        demo_0 = data_grp.create_group("demo_0")
        demo_0.attrs["model_file"] = "<mujoco model='test'></mujoco>"
        demo_0.create_dataset("actions", data=np.zeros((10, 7), dtype=np.float32))
        demo_0.create_dataset("states", data=np.zeros((11, 20), dtype=np.float32))

        demo_1 = data_grp.create_group("demo_1")
        demo_1.attrs["model_file"] = "<mujoco model='test'></mujoco>"
        demo_1.create_dataset("actions", data=np.ones((5, 7), dtype=np.float32))
        demo_1.create_dataset("states", data=np.ones((6, 20), dtype=np.float32))

    replayer = DemonstrationReplayer(hdf5_file)

    demos = replayer.list_demos()
    assert demos == ["demo_0", "demo_1"]

    metadata = replayer.get_env_metadata()
    assert metadata["problem_name"] == "Libero_Tabletop_Manipulation"
    assert metadata["env_kwargs"]["control_freq"] == 20


def test_verify_hdf5_hash_and_manifest(tmp_path):
    """Verify cryptographic SHA-256 verification of demonstration HDF5."""
    import hashlib
    import yaml

    hdf5_file = tmp_path / "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    content = b"dummy_hdf5_content_for_hash_testing"
    hdf5_file.write_bytes(content)
    actual_hash = hashlib.sha256(content).hexdigest()

    manifest_file = tmp_path / "demonstrations_manifest.yaml"
    manifest_data = {
        "demonstrations": [
            {
                "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                "sha256": actual_hash,
            }
        ]
    }
    with open(manifest_file, "w") as f:
        yaml.dump(manifest_data, f)

    replayer = DemonstrationReplayer(hdf5_file)
    assert replayer.verify_hdf5_hash(manifest_file) is True

    # Test mismatch raises ValueError
    manifest_data["demonstrations"][0]["sha256"] = "wrong_hash"
    with open(manifest_file, "w") as f:
        yaml.dump(manifest_data, f)

    with pytest.raises(ValueError, match="Cryptographic integrity check failed"):
        replayer.verify_hdf5_hash(manifest_file)

