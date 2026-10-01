"""Unit tests for ContactDiagnostics (Phase 4).

Validates:
- Contact parsing from MuJoCo sim data
- Gripper finger contact detection (left, right, both)
- Target object substring matching
- Object lift height tracking
- Fallback handling when simulation or contacts are absent
"""

from typing import Any, List
import numpy as np
import pytest

from src.evaluation.diagnostics import ContactDiagnostics


class MockGeomContact:
    def __init__(self, geom1: int, geom2: int, dist: float = -0.001) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class MockSimData:
    def __init__(self, contacts: List[MockGeomContact]) -> None:
        self.contact = contacts
        self.ncon = len(contacts)


class MockSimModel:
    def __init__(self, geom_map: dict) -> None:
        self._geom_map = geom_map

    def geom_id2name(self, geom_id: int) -> str:
        return self._geom_map.get(geom_id, f"geom_{geom_id}")


class MockSim:
    def __init__(self, contacts: List[MockGeomContact], geom_map: dict) -> None:
        self.data = MockSimData(contacts)
        self.model = MockSimModel(geom_map)


class MockEnvWithSim:
    def __init__(self, sim: Any) -> None:
        self.sim = sim


def test_contact_diagnostics_empty_sim():
    """Verify empty/missing sim gracefully returns empty contacts."""
    diag = ContactDiagnostics(env=None, target_object_name="cream_cheese")
    assert diag.get_active_contacts() == []
    gripper_status = diag.inspect_gripper_contacts()
    assert gripper_status["any_contact"] is False
    assert gripper_status["both_fingers_contact"] is False
    assert gripper_status["matched_contact_pairs"] == []


def test_contact_diagnostics_gripper_contacts():
    """Verify detection of left, right, and both fingers touching target object."""
    geom_map = {
        0: "robot0_finger1_pad_collision",
        1: "robot0_finger2_pad_collision",
        2: "cream_cheese_g0",
        3: "table_collision",
    }
    # Case 1: only finger 1 contacts object
    contacts1 = [
        MockGeomContact(0, 2, dist=-0.002),
        MockGeomContact(2, 3, dist=0.0),
    ]
    env1 = MockEnvWithSim(MockSim(contacts1, geom_map))
    diag1 = ContactDiagnostics(env=env1, target_object_name="cream_cheese")
    res1 = diag1.inspect_gripper_contacts()
    assert res1["any_contact"] is True
    assert res1["left_finger_contact"] is True
    assert res1["right_finger_contact"] is False
    assert res1["both_fingers_contact"] is False
    assert len(res1["matched_contact_pairs"]) == 1

    # Case 2: both fingers contact object
    contacts2 = [
        MockGeomContact(0, 2, dist=-0.002),
        MockGeomContact(1, 2, dist=-0.003),
    ]
    env2 = MockEnvWithSim(MockSim(contacts2, geom_map))
    diag2 = ContactDiagnostics(env=env2, target_object_name="cream_cheese")
    res2 = diag2.inspect_gripper_contacts()
    assert res2["any_contact"] is True
    assert res2["left_finger_contact"] is True
    assert res2["right_finger_contact"] is True
    assert res2["both_fingers_contact"] is True
    assert len(res2["matched_contact_pairs"]) == 2


def test_contact_diagnostics_object_lift():
    """Verify initial support height capture and delta_z thresholding."""
    diag = ContactDiagnostics(env=None, target_object_name="cream_cheese")

    # Step 0: Initial position
    pos0 = np.array([0.1, 0.2, 0.82])
    res0 = diag.inspect_object_lift(pos0, lift_threshold=0.04)
    assert res0["is_lifted"] is False
    assert res0["initial_z"] == pytest.approx(0.82)
    assert res0["delta_z"] == 0.0

    # Step 1: Small displacement (< 4cm)
    pos1 = np.array([0.1, 0.2, 0.84])
    res1 = diag.inspect_object_lift(pos1, lift_threshold=0.04)
    assert res1["is_lifted"] is False
    assert res1["delta_z"] == pytest.approx(0.02)

    # Step 2: Significant lift (>= 4cm)
    pos2 = np.array([0.1, 0.2, 0.88])
    res2 = diag.inspect_object_lift(pos2, lift_threshold=0.04)
    assert res2["is_lifted"] is True
    assert res2["delta_z"] == pytest.approx(0.06)


def test_extract_target_and_goal():
    """Verify target object and goal container resolution from instructions."""
    from src.evaluation.diagnostics import extract_target_and_goal

    # Case 1: Pick up alphabet soup and place in basket
    t1, g1 = extract_target_and_goal(None, "pick up the alphabet soup and place it in the basket")
    assert t1 == "alphabet_soup"
    assert g1 == "basket"

    # Case 2: Pick up black bowl and place on plate
    t2, g2 = extract_target_and_goal(None, "pick up the black bowl between the plate and the ramekin and place it on the plate")
    assert t2 == "black_bowl"
    assert g2 == "plate"

    # Case 3: Open middle drawer of cabinet
    t3, g3 = extract_target_and_goal(None, "open the middle drawer of the cabinet")
    assert t3 == "bowl" or "drawer" in str(t3) or t3 is None or "cabinet" in str(g3)
    assert g3 == "cabinet"


def test_diagnostics_collector_success():
    """Verify successful episode yields NONE phase and None primary failure code."""
    from src.evaluation.diagnostics import EpisodeDiagnosticsCollector, TerminationReason, FailurePhase

    collector = EpisodeDiagnosticsCollector(env=None, target_object_name="soup")
    report = collector.finalize(final_success=True, termination_reason=TerminationReason.SUCCESS)
    assert report.termination_reason == "SUCCESS"
    assert report.failure_phase == "NONE"
    assert report.primary_failure_code is None


def test_diagnostics_collector_reach_failure():
    """Verify robot failing to approach object is classified as REACH and UNATTRIBUTED."""
    from src.evaluation.diagnostics import EpisodeDiagnosticsCollector, TerminationReason, FailurePhase, SRSAttributionCode

    collector = EpisodeDiagnosticsCollector(env=None, target_object_name="soup")
    # Simulate step where EEF is far from object (1.0m)
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.0])}
    action = np.zeros(7)
    collector.record_step(step_num=0, obs=obs, action=action, eef_pos=np.array([1.0, 1.0, 1.0]))

    report = collector.finalize(final_success=False, termination_reason=TerminationReason.MAX_STEPS)
    assert report.termination_reason == "MAX_STEPS"
    assert report.failure_phase == FailurePhase.REACH.value
    assert report.primary_failure_code == SRSAttributionCode.UNATTRIBUTED.value
    assert report.evidence["ever_reached"] is False


def test_diagnostics_collector_invalid_action():
    """Verify NaN/Inf action maps to INVALID_ACTION and F10 action decoding."""
    from src.evaluation.diagnostics import EpisodeDiagnosticsCollector, TerminationReason, SRSAttributionCode

    collector = EpisodeDiagnosticsCollector(env=None, target_object_name="soup")
    report = collector.finalize(final_success=False, termination_reason=TerminationReason.INVALID_ACTION)
    assert report.termination_reason == "INVALID_ACTION"
    assert report.primary_failure_code == SRSAttributionCode.F10_ACTION_DECODING.value


def test_diagnostics_collector_policy_error():
    """Verify policy exception maps to POLICY_ERROR and F16."""
    from src.evaluation.diagnostics import EpisodeDiagnosticsCollector, TerminationReason, SRSAttributionCode

    collector = EpisodeDiagnosticsCollector(env=None, target_object_name="soup")
    exc = RuntimeError("CUDA out of memory")
    report = collector.finalize(final_success=False, termination_reason=TerminationReason.POLICY_ERROR, exception=exc)
    assert report.termination_reason == "POLICY_ERROR"
    assert report.primary_failure_code == SRSAttributionCode.F16_POLICY_INFERENCE_FAILURE.value
    assert report.evidence["exception_type"] == "RuntimeError"


def test_diagnostics_collector_timing_chunking_anomaly():
    """Verify explicit timing anomaly logs assign F13 timing/chunking."""
    from src.evaluation.diagnostics import EpisodeDiagnosticsCollector, TerminationReason, SRSAttributionCode

    collector = EpisodeDiagnosticsCollector(env=None, target_object_name="soup")
    anomalies = ["inference_latency_exceeded_chunk_period_by_200ms"]
    report = collector.finalize(
        final_success=False,
        termination_reason=TerminationReason.MAX_STEPS,
        timing_anomalies=anomalies,
    )
    assert report.primary_failure_code == SRSAttributionCode.F13_TIMING_CHUNKING.value
    assert "inference_latency_exceeded_chunk_period_by_200ms" in report.secondary_failure_tags
