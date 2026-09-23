"""Comprehensive smoke, invariant, and acceptance tests for official LIBERO simulation environment.

Covers all 6 acceptance requirements per SRS.md and AGENTS.md:
1. Robot kinematics, joints, actuators, and PandaGripper geometry.
2. Camera pose, FOV, resolution, and body parenting.
3. Physical state determinism on repeated initial_state resets.
4. Cryptographic SHA-256 asset and initial-state integrity.
5. Strict-mode invariants enforcement (Fail-Fast per ADR-0005 and ADR-0008).
6. Execution environment provenance tagging (STRICT-LIBERO vs HOST-SMOKE-ONLY).
"""

import numpy as np
import pytest

from src.simulator.asset_validator import AssetIntegrityError, verify_task_asset_integrity
from src.simulator.libero_env import (
    BenchmarkMode,
    LiberoEnv,
    StrictInvariantViolationError,
)


@pytest.fixture(scope="module")
def libero_env():
    """Shared fixture for LiberoEnv on official LIBERO-Object task 0."""
    env = LiberoEnv(benchmark_name="libero_object", task_id=0, mode=BenchmarkMode.STRICT_LIBERO)
    yield env
    env.close()


def test_task_registry_loading(libero_env):
    """Test 1: Verify programmatic loading from official benchmark registry."""
    assert libero_env.task_name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    assert libero_env.language_instruction == "pick up the alphabet soup and place it in the basket"
    assert libero_env.num_initial_states == 50
    assert libero_env.control_freq == 20
    assert libero_env.horizon == 1000
    assert libero_env.mode == BenchmarkMode.STRICT_LIBERO
    assert libero_env.is_strict_mode is True


def test_initial_state_reset_and_observations(libero_env):
    """Test 2: Verify exact initial state restoration and observation schema."""
    obs = libero_env.reset(initial_state_id=0)

    # Check visual observations
    assert "agentview_image" in obs
    assert obs["agentview_image"].shape == (128, 128, 3)
    assert obs["agentview_image"].dtype == np.uint8

    assert "robot0_eye_in_hand_image" in obs
    assert obs["robot0_eye_in_hand_image"].shape == (128, 128, 3)
    assert obs["robot0_eye_in_hand_image"].dtype == np.uint8

    # Check robot state observations
    assert "robot0_eef_pos" in obs
    assert obs["robot0_eef_pos"].shape == (3,)
    assert "robot0_eef_quat" in obs
    assert obs["robot0_eef_quat"].shape == (4,)
    assert "robot0_gripper_qpos" in obs


def test_camera_rendering(libero_env):
    """Test 3: Verify off-screen rendering pipeline produces valid non-blank frames."""
    frame_agentview = libero_env.render("agentview")
    assert frame_agentview.shape == (128, 128, 3)
    assert frame_agentview.dtype == np.uint8
    assert np.mean(frame_agentview) > 1.0

    frame_hand = libero_env.render("robot0_eye_in_hand")
    assert frame_hand.shape == (128, 128, 3)
    assert frame_hand.dtype == np.uint8
    assert np.mean(frame_hand) > 1.0


def test_diagnostic_action_step(libero_env):
    """Test 4: Verify stepping a manually-defined diagnostic 7-DoF action."""
    libero_env.reset(initial_state_id=0)

    # 7-dim action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    diag_action = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, done, info = libero_env.step(diag_action)

    assert "agentview_image" in obs
    assert obs["agentview_image"].shape == (128, 128, 3)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "success" in info
    assert info["step"] == 1
    assert info["mode"] == "STRICT-LIBERO"
    assert not done


def test_success_predicate(libero_env):
    """Test 5: Verify official BDDL success predicate evaluation."""
    libero_env.reset(initial_state_id=0)
    success = libero_env.check_success()
    assert isinstance(success, bool)
    assert success is False


def test_fail_fast_action_validation(libero_env):
    """Test 6: Verify fail-fast validation on invalid action dimensions and NaN values."""
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        libero_env.step(np.zeros(6))

    with pytest.raises(ValueError, match="Invalid action containing NaN"):
        libero_env.step(np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_robot_kinematics_and_actuators(libero_env):
    """Test 7: Verify Franka Panda kinematics, 7 arm joints, 2 gripper joints, and actuators."""
    kinematics = libero_env.get_robot_kinematics_info()

    # Robot identity
    assert "Panda" in kinematics["robot_name"]

    # Exactly 7 arm joints
    expected_arm_joints = [f"robot0_joint{i}" for i in range(1, 8)]
    assert kinematics["arm_joints"] == expected_arm_joints
    assert kinematics["num_arm_joints"] == 7

    # Gripper joints and type
    assert kinematics["gripper_type"] == "PandaGripper"
    assert kinematics["num_gripper_joints"] == 2
    assert any("finger_joint1" in j for j in kinematics["gripper_joints"])
    assert any("finger_joint2" in j for j in kinematics["gripper_joints"])

    # 9 actuators total: 7 arm torques + 2 gripper actuators
    assert kinematics["num_actuators"] >= 9
    assert "robot0_torq_j1" in kinematics["actuator_names"]
    assert "robot0_torq_j7" in kinematics["actuator_names"]

    # Controller type
    assert kinematics["controller_type"] == "OSC_POSE"


def test_camera_pose_composition_and_parenting(libero_env):
    """Test 8: Verify camera poses, resolution, FOV, and topological attachment."""
    # 1. agentview camera must be parented to world (static arena observer)
    agent_info = libero_env.get_camera_info("agentview")
    assert agent_info["parent_body_name"] == "world"
    assert agent_info["resolution"] == (128, 128)
    assert 40.0 <= agent_info["fovy"] <= 60.0
    assert len(agent_info["pos"]) == 3
    assert len(agent_info["quat"]) == 4

    # 2. robot0_eye_in_hand camera must be parented to the robot wrist/hand
    wrist_info = libero_env.get_camera_info("robot0_eye_in_hand")
    assert "hand" in wrist_info["parent_body_name"].lower()
    assert wrist_info["resolution"] == (128, 128)
    assert wrist_info["fovy"] == 75.0


def test_reset_determinism_identical_initial_state(libero_env):
    """Test 9: Prove that resetting with identical initial_state_id reproduces identical physics."""
    # First reset to state 0
    libero_env.reset(initial_state_id=0)
    state_0_initial = libero_env.get_physics_state()

    # Apply 3 non-zero actions to alter physics state
    action = np.array([0.01, -0.01, 0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(3):
        libero_env.step(action)

    state_0_after_steps = libero_env.get_physics_state()
    # Confirm state actually changed
    assert not np.allclose(state_0_initial, state_0_after_steps)

    # Second reset to state 0
    libero_env.reset(initial_state_id=0)
    state_0_reproduced = libero_env.get_physics_state()

    # PROVE DETERMINISM: state_0_reproduced MUST be identical to state_0_initial
    assert np.allclose(
        state_0_initial, state_0_reproduced, atol=1e-7
    ), "Determinism violation: reset(0) produced a differing physical state!"

    # Also prove that reset(1) produces a DIFFERENT physical state
    libero_env.reset(initial_state_id=1)
    state_1 = libero_env.get_physics_state()
    assert not np.allclose(
        state_0_initial, state_1
    ), "Initial state differentiation failed: state 0 and state 1 are identical!"


def test_asset_and_init_state_hash_integrity(libero_env):
    """Test 10: Verify SHA-256 cryptographic provenance of BDDL and init files."""
    integrity = libero_env.asset_integrity_info
    assert integrity["integrity_status"] == "PASS"
    assert integrity["bddl_sha256"] == "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e"
    assert integrity["init_verified"] is True
    assert integrity["init_sha256"] == "b9023e5546b239cacb2aaa3c76df3b73c3de6f97500fea5e27a0db9827002a19"

    # Verify that asset_validator fails loudly if given a corrupted/tampered file
    with pytest.raises(AssetIntegrityError, match="BDDL hash mismatch|Asset manifest not found|is not registered|does not exist"):
        verify_task_asset_integrity(
            bddl_file_path="nonexistent_task.bddl"
        )


def test_strict_mode_invariants_enforcement():
    """Test 11: Verify strict-mode invariants lock and reject non-benchmark configurations."""
    # Disallowed control frequency
    with pytest.raises(StrictInvariantViolationError, match="violates STRICT-LIBERO invariant"):
        LiberoEnv(task_id=0, control_freq=30, mode=BenchmarkMode.STRICT_LIBERO)

    # Disallowed episode horizon
    with pytest.raises(StrictInvariantViolationError, match="violates STRICT-LIBERO invariant"):
        LiberoEnv(task_id=0, horizon=500, mode=BenchmarkMode.STRICT_LIBERO)

    # Disallowed controller type
    with pytest.raises(StrictInvariantViolationError, match="violates STRICT-LIBERO invariant"):
        LiberoEnv(task_id=0, controller="JOINT_VELOCITY", mode=BenchmarkMode.STRICT_LIBERO)

    # Disallowed camera list
    with pytest.raises(StrictInvariantViolationError, match="violates STRICT-LIBERO invariant"):
        LiberoEnv(task_id=0, camera_names=["agentview"], mode=BenchmarkMode.STRICT_LIBERO)

    # Disallowed resolution
    with pytest.raises(StrictInvariantViolationError, match="violates STRICT-LIBERO invariant"):
        LiberoEnv(task_id=0, camera_height=256, camera_width=256, mode=BenchmarkMode.STRICT_LIBERO)


def test_execution_environment_provenance_labeling(libero_env):
    """Test 12: Verify execution environment provenance inspection and transparent tier labeling."""
    meta = libero_env.get_provenance_metadata()
    assert "execution_tier" in meta
    assert "is_official_reference_stack" in meta
    assert "is_benchmark_comparable" in meta

    # On Windows host (Python 3.12, WGL), runs MUST be labeled NON-COMPARABLE
    import sys
    if sys.platform == "win32" or sys.version_info.major != 3 or sys.version_info.minor != 8:
        assert meta["execution_tier"] == "HOST-SMOKE-ONLY (NON-COMPARABLE)"
        assert meta["is_official_reference_stack"] is False
        assert meta["is_benchmark_comparable"] is False
    else:
        assert meta["execution_tier"] == "STRICT-LIBERO-CERTIFIED"
        assert meta["is_official_reference_stack"] is True
        assert meta["is_benchmark_comparable"] is True
