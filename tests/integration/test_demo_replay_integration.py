"""Integration tests for official LIBERO demonstration replay (Phase 3).

Requires official LIBERO demonstration HDF5 and simulation stack.
Marked with @pytest.mark.libero.
"""

from pathlib import Path
import pytest

from src.evaluation.demo_replay import DemonstrationReplayer

DEMO_HDF5 = Path("resources/demonstrations/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5")


@pytest.mark.libero
def test_official_demo_replay_success():
    """Verify that replaying official demo_0 inside exactly reconstructed env achieves success."""
    if not DEMO_HDF5.exists():
        pytest.skip(f"Demonstration file not present at {DEMO_HDF5}")

    replayer = DemonstrationReplayer(DEMO_HDF5)
    # 1. Cryptographic provenance verification against manifest
    manifest_path = Path("resources/manifests/demonstrations_manifest.yaml")
    assert manifest_path.exists()
    assert replayer.verify_hdf5_hash(manifest_path) is True

    demos = replayer.list_demos()
    assert "demo_0" in demos

    metadata = replayer.get_env_metadata()
    assert "problem_name" in metadata

    # 2. Replay episode with tracking tolerance enforcement
    metrics = replayer.replay_episode("demo_0", tracking_tolerance=0.1, strict_tolerance=False, render=True)
    assert metrics.num_steps == 148
    assert metrics.action_dim == 7
    assert metrics.env_success is True
    assert metrics.tracking_success is True
    assert metrics.success is True
    assert metrics.mean_tracking_error < 0.1
    assert len(metrics.rendered_frames) == 148
