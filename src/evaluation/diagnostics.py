"""Physics, Contact, and Containment Diagnostics for LIBERO Evaluation (Phase 7).

Monitors:
- Finger-to-object contact state via MuJoCo collision geometry.
- Object initial support height and lift state (captured at t=0 immediately after reset).
- EEF, object, and goal container kinematic tracking.
- Multi-tier decoupled outcome classification:
    * termination_reason (SUCCESS, MAX_STEPS, INVALID_ACTION, POLICY_ERROR, SIMULATOR_ERROR)
    * failure_phase (NONE, REACH, GRASP, LIFT, TRANSPORT, PLACEMENT, TIMEOUT)
    * primary_failure_code (SRS Section 22: F1-F17 or UNATTRIBUTED)
    * evidence (concrete measurements, kinematic logs, and exceptions)
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TerminationReason(str, Enum):
    """Reason for evaluation episode termination."""
    SUCCESS = "SUCCESS"
    MAX_STEPS = "MAX_STEPS"
    INVALID_ACTION = "INVALID_ACTION"
    POLICY_ERROR = "POLICY_ERROR"
    SIMULATOR_ERROR = "SIMULATOR_ERROR"


class FailurePhase(str, Enum):
    """Behavioral phase reached by the robot before failure."""
    NONE = "NONE"
    REACH = "REACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACEMENT = "PLACEMENT"
    TIMEOUT = "TIMEOUT"


class SRSAttributionCode(str, Enum):
    """Formal failure attribution taxonomy per SRS.md Section 22."""
    F1_VISUAL_PERCEPTION = "F1 — visual perception"
    F2_LANGUAGE_GROUNDING = "F2 — language grounding"
    F3_TARGET_LOCALIZATION = "F3 — target localization"
    F4_REACHING = "F4 — reaching"
    F5_GRASP_ACQUISITION = "F5 — grasp acquisition"
    F6_GRASP_RETENTION = "F6 — grasp retention"
    F7_TRANSPORT = "F7 — transport"
    F8_PLACEMENT = "F8 — placement"
    F9_GRIPPER_SEMANTICS = "F9 — gripper semantics"
    F10_ACTION_DECODING = "F10 — action decoding"
    F11_CONTROLLER_MISMATCH = "F11 — controller mismatch"
    F12_CAMERA_PREPROCESSING_MISMATCH = "F12 — camera/preprocessing mismatch"
    F13_TIMING_CHUNKING = "F13 — timing/chunking"
    F14_TASK_SEQUENCING = "F14 — task sequencing"
    F15_SIMULATOR_INSTABILITY = "F15 — simulator instability"
    F16_POLICY_INFERENCE_FAILURE = "F16 — policy inference failure"
    F17_ENVIRONMENT_RESOURCE_MISMATCH = "F17 — environment/resource mismatch"
    UNATTRIBUTED = "UNATTRIBUTED"


@dataclass
class DiagnosticsConfig:
    """Auxiliary diagnostic thresholds.

    NOTE: These thresholds are auxiliary instrumentation metrics for failure
    classification and do NOT replace the ground-truth env.check_success() predicate.
    """
    reach_distance_threshold: float = 0.05       # Max distance (m) between EEF and object considered 'reached'
    lift_height_threshold: float = 0.04          # Delta z (m) above baseline support height considered 'lifted'
    transport_distance_reduction: float = 0.05   # Distance (m) closer to goal container relative to lift moment
    placement_distance_threshold: float = 0.08   # Max distance (m) between object and goal container for 'placement'
    gripper_close_threshold: float = 0.0         # robosuite Panda gripper action >= 0 is closing/closed


@dataclass
class DiagnosticReport:
    """Telemetry report with decoupled termination reason, phase, and evidence."""
    termination_reason: str
    failure_phase: str
    primary_failure_code: Optional[str]
    secondary_failure_tags: List[str] = field(default_factory=list)
    first_failure_step: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_target_and_goal(env: Any, instruction: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Resolve target object and goal fixture names from env metadata or instruction.

    Args:
        env: LiberoEnv or gym environment.
        instruction: Optional natural language instruction string.

    Returns:
        (target_object_substr, goal_container_substr)
    """
    text = (instruction or getattr(env, "language_instruction", "") or getattr(env, "task_description", "") or "").lower()

    # 1. Target objects in LIBERO suites
    known_objects = [
        "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
        "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
        "black_bowl", "bowl", "wine_bottle", "moka_pot", "white_mug",
        "yellow_and_white_mug", "book", "plate"
    ]
    target_obj = None
    for obj in known_objects:
        obj_words = obj.replace("_", " ")
        if obj_words in text or obj in text:
            target_obj = obj
            break

    # 2. Goal containers / fixtures in LIBERO suites
    known_goals = [
        "basket", "plate", "stove", "cabinet", "drawer", "microwave",
        "caddy", "rack", "ramekin", "cookie_box"
    ]
    goal_container = None
    for goal in known_goals:
        if f"in the {goal}" in text or f"on the {goal}" in text or f"to the {goal}" in text or goal in text:
            goal_container = goal
            break

    return target_obj, goal_container


class ContactDiagnostics:
    """Diagnostic utility inspecting MuJoCo contact pairs and object kinematics."""

    def __init__(
        self,
        env: Any,
        target_object_name: Optional[str] = None,
        goal_container_name: Optional[str] = None,
        config: Optional[DiagnosticsConfig] = None,
    ) -> None:
        self.env = env
        self.config = config or DiagnosticsConfig()

        # If not provided, attempt automatic resolution from env
        auto_target, auto_goal = extract_target_and_goal(env)
        self.target_object_name = target_object_name or auto_target
        self.goal_container_name = goal_container_name or auto_goal

        self.initial_object_pos: Optional[np.ndarray] = None
        self.initial_object_z: Optional[float] = None

    def get_sim(self) -> Any:
        """Resolve MuJoCo sim instance across wrapper layers."""
        sim = getattr(self.env, "sim", None)
        if sim is None and hasattr(self.env, "_env"):
            sim = getattr(self.env._env, "sim", None)
        if sim is None and hasattr(self.env, "env"):
            sim = getattr(self.env.env, "sim", None)
        return sim

    def capture_baseline(self, obs: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """Capture baseline object position immediately upon reset (t=0).

        Ensures delta_z measurements start from pristine pre-step ground truth.
        """
        pos = self.get_target_object_pos()
        if pos is not None:
            self.initial_object_pos = np.array(pos, copy=True)
            self.initial_object_z = float(pos[2])
        return self.initial_object_pos

    def get_target_object_pos(self, object_substr: Optional[str] = None) -> Optional[np.ndarray]:
        """Extract Cartesian position (x, y, z) of target object body from sim."""
        sim = self.get_sim()
        if sim is None or not hasattr(sim, "model") or not hasattr(sim, "data"):
            return None
        target = (object_substr or self.target_object_name or "").lower()
        if not target:
            return None
        try:
            nbody = getattr(sim.model, "nbody", 0)
            for i in range(nbody):
                name = sim.model.body_id2name(i)
                if name and target in name.lower():
                    return np.array(sim.data.body_xpos[i], copy=True)
        except Exception:
            pass
        return None

    def get_goal_container_pos(self, container_substr: Optional[str] = None) -> Optional[np.ndarray]:
        """Extract Cartesian position (x, y, z) of goal fixture/container from sim."""
        sim = self.get_sim()
        if sim is None or not hasattr(sim, "model") or not hasattr(sim, "data"):
            return None
        target = (container_substr or self.goal_container_name or "").lower()
        if not target:
            return None
        try:
            nbody = getattr(sim.model, "nbody", 0)
            for i in range(nbody):
                name = sim.model.body_id2name(i)
                if name and target in name.lower():
                    return np.array(sim.data.body_xpos[i], copy=True)
        except Exception:
            pass
        return None

    def get_active_contacts(self) -> List[Tuple[str, str, float]]:
        """Extract all active contact geom name pairs and normal force distance."""
        sim = self.get_sim()
        if sim is None or not hasattr(sim, "data") or not hasattr(sim.data, "ncon"):
            return []

        contacts: List[Tuple[str, str, float]] = []
        ncon = sim.data.ncon
        for i in range(ncon):
            contact = sim.data.contact[i]
            geom1_name = sim.model.geom_id2name(contact.geom1) or ""
            geom2_name = sim.model.geom_id2name(contact.geom2) or ""
            dist = float(contact.dist)
            contacts.append((geom1_name, geom2_name, dist))

        return contacts

    def inspect_gripper_contacts(self, object_substr: Optional[str] = None) -> Dict[str, Any]:
        """Check whether robot gripper pads are actively contacting target object."""
        target = object_substr or self.target_object_name or ""
        contacts = self.get_active_contacts()

        left_contact = False
        right_contact = False
        matched_contacts = []
        gripper_keywords = ["gripper", "finger", "pad"]

        for g1, g2, dist in contacts:
            is_g1_gripper = any(k in g1.lower() for k in gripper_keywords)
            is_g2_gripper = any(k in g2.lower() for k in gripper_keywords)

            is_g1_obj = target.lower() in g1.lower() if target else False
            is_g2_obj = target.lower() in g2.lower() if target else False

            if (is_g1_gripper and is_g2_obj) or (is_g2_gripper and is_g1_obj):
                matched_contacts.append((g1, g2, dist))
                contact_str = (g1 if is_g1_gripper else g2).lower()
                if "1" in contact_str or "left" in contact_str:
                    left_contact = True
                if "2" in contact_str or "right" in contact_str:
                    right_contact = True

        both_contact = left_contact and right_contact
        any_contact = len(matched_contacts) > 0

        return {
            "target": target,
            "any_contact": any_contact,
            "both_fingers_contact": both_contact,
            "left_finger_contact": left_contact,
            "right_finger_contact": right_contact,
            "matched_contact_pairs": matched_contacts,
        }

    def inspect_object_lift(
        self, current_object_pos: Optional[np.ndarray] = None, lift_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if target object has been lifted above its initial support height."""
        thresh = lift_threshold if lift_threshold is not None else self.config.lift_height_threshold
        if current_object_pos is None:
            current_object_pos = self.get_target_object_pos()

        if current_object_pos is None:
            return {"is_lifted": False, "delta_z": 0.0, "initial_z": self.initial_object_z, "current_z": None}

        curr_z = float(current_object_pos[2])
        if self.initial_object_z is None:
            self.initial_object_z = curr_z

        delta_z = curr_z - self.initial_object_z
        is_lifted = delta_z >= thresh

        return {
            "initial_z": self.initial_object_z,
            "current_z": curr_z,
            "delta_z": delta_z,
            "is_lifted": is_lifted,
        }


class EpisodeDiagnosticsCollector:
    """Step-by-step telemetry collector and evidence-based failure classifier (Phase 7)."""

    def __init__(
        self,
        env: Any,
        target_object_name: Optional[str] = None,
        goal_container_name: Optional[str] = None,
        config: Optional[DiagnosticsConfig] = None,
    ) -> None:
        self.config = config or DiagnosticsConfig()
        self.diagnostics = ContactDiagnostics(
            env=env,
            target_object_name=target_object_name,
            goal_container_name=goal_container_name,
            config=self.config,
        )

        # Progression flags
        self.min_eef_to_object_dist: float = float("inf")
        self.min_object_to_goal_dist: float = float("inf")
        self.max_lift_delta_z: float = 0.0

        self.ever_reached: bool = False
        self.ever_grasped: bool = False
        self.ever_lifted: bool = False
        self.ever_transported: bool = False
        self.ever_placed: bool = False

        self.reach_step: Optional[int] = None
        self.grasp_step: Optional[int] = None
        self.lift_step: Optional[int] = None
        self.transport_step: Optional[int] = None
        self.placement_step: Optional[int] = None
        self.min_eef_to_obj_step: Optional[int] = None
        self.lost_contact_step: Optional[int] = None
        self.object_to_goal_dist_at_lift: Optional[float] = None

        self.step_records: List[Dict[str, Any]] = []

    def capture_baseline(self, obs: Optional[Dict[str, Any]] = None) -> None:
        """Capture t=0 baseline before first step."""
        self.diagnostics.capture_baseline(obs)

    def record_step(
        self,
        step_num: int,
        obs: Dict[str, Any],
        action: np.ndarray,
        eef_pos: Optional[np.ndarray] = None,
        is_replanned: bool = False,
        infer_latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Record step-level kinematics and contact telemetry."""
        # 1. Resolve EEF pos
        if eef_pos is None:
            if "robot0_eef_pos" in obs:
                eef_pos = np.asarray(obs["robot0_eef_pos"])

        # 2. Kinematics & contacts
        obj_pos = self.diagnostics.get_target_object_pos()
        goal_pos = self.diagnostics.get_goal_container_pos()
        contact_info = self.diagnostics.inspect_gripper_contacts()
        lift_info = self.diagnostics.inspect_object_lift(obj_pos)

        d_eef_obj = float("inf")
        if eef_pos is not None and obj_pos is not None:
            d_eef_obj = float(np.linalg.norm(eef_pos - obj_pos))
            if d_eef_obj < self.min_eef_to_object_dist:
                self.min_eef_to_object_dist = d_eef_obj
                self.min_eef_to_obj_step = step_num

        d_obj_goal = float("inf")
        if obj_pos is not None and goal_pos is not None:
            d_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))
            if d_obj_goal < self.min_object_to_goal_dist:
                self.min_object_to_goal_dist = d_obj_goal

        delta_z = lift_info.get("delta_z", 0.0)
        if delta_z > self.max_lift_delta_z:
            self.max_lift_delta_z = delta_z

        # 3. Update progression milestones
        # Reach: EEF within auxiliary threshold
        if d_eef_obj <= self.config.reach_distance_threshold:
            self.ever_reached = True
            if self.reach_step is None:
                self.reach_step = step_num

        # Grasp: both fingers in contact with target object and gripper closed
        gripper_action = float(action[-1]) if action is not None and len(action) > 0 else 0.0
        is_gripper_closing = gripper_action >= self.config.gripper_close_threshold
        if contact_info.get("both_fingers_contact", False) and is_gripper_closing:
            self.ever_grasped = True
            if self.grasp_step is None:
                self.grasp_step = step_num

        # Lift: object delta z above support threshold
        if lift_info.get("is_lifted", False):
            self.ever_lifted = True
            if self.lift_step is None:
                self.lift_step = step_num
                self.object_to_goal_dist_at_lift = d_obj_goal

        # Track contact retention after lift
        if self.ever_lifted and not contact_info.get("both_fingers_contact", False):
            if self.lost_contact_step is None:
                self.lost_contact_step = step_num

        # Transport: object was lifted and has made substantial progress towards goal container
        transport_reduction = getattr(self.config, "transport_distance_reduction", 0.05)
        if self.ever_lifted and self.object_to_goal_dist_at_lift is not None:
            if d_obj_goal <= (self.object_to_goal_dist_at_lift - transport_reduction):
                self.ever_transported = True
                if self.transport_step is None:
                    self.transport_step = step_num

        # Placement heuristic: object close to goal container AND gripper released (opening/opened)
        is_gripper_opening = gripper_action < self.config.gripper_close_threshold
        if d_obj_goal <= self.config.placement_distance_threshold and is_gripper_opening:
            self.ever_placed = True
            if self.placement_step is None:
                self.placement_step = step_num

        # 4. Extract 3D coordinates and joint positions for comprehensive telemetry
        gripper_qpos = None
        if "robot0_gripper_qpos" in obs:
            gripper_qpos = [round(float(v), 5) for v in np.asarray(obs["robot0_gripper_qpos"]).flatten()]

        step_telemetry = {
            "step": step_num,
            "dist_eef_to_object": round(d_eef_obj, 4) if d_eef_obj != float("inf") else None,
            "dist_object_to_goal": round(d_obj_goal, 4) if d_obj_goal != float("inf") else None,
            "object_delta_z": round(delta_z, 4),
            "both_fingers_contact": contact_info.get("both_fingers_contact", False),
            "gripper_action": round(gripper_action, 4),
            "is_lifted": lift_info.get("is_lifted", False),
            "eef_pos": [round(float(v), 4) for v in eef_pos] if eef_pos is not None else None,
            "object_pos": [round(float(v), 4) for v in obj_pos] if obj_pos is not None else None,
            "goal_pos": [round(float(v), 4) for v in goal_pos] if goal_pos is not None else None,
            "gripper_qpos": gripper_qpos,
        }
        self.step_records.append(step_telemetry)
        return step_telemetry

    def finalize(
        self,
        final_success: bool,
        termination_reason: TerminationReason,
        exception: Optional[Exception] = None,
        timing_anomalies: Optional[List[str]] = None,
    ) -> DiagnosticReport:
        """Produce an evidence-based DiagnosticReport per SRS Section 22 rules."""
        # If ground-truth environment predicate confirmed success, placement was physically achieved
        if final_success:
            self.ever_placed = True
            if self.placement_step is None:
                self.placement_step = len(self.step_records) - 1

        evidence: Dict[str, Any] = {
            "min_eef_to_object_dist": round(self.min_eef_to_object_dist, 4) if self.min_eef_to_object_dist != float("inf") else None,
            "min_object_to_goal_dist": round(self.min_object_to_goal_dist, 4) if self.min_object_to_goal_dist != float("inf") else None,
            "max_lift_delta_z": round(self.max_lift_delta_z, 4),
            "ever_reached": self.ever_reached,
            "ever_grasped": self.ever_grasped,
            "ever_lifted": self.ever_lifted,
            "ever_transported": self.ever_transported,
            "ever_placed": self.ever_placed,
            "reach_step": self.reach_step,
            "grasp_step": self.grasp_step,
            "lift_step": self.lift_step,
            "transport_step": self.transport_step,
            "placement_step": self.placement_step,
        }
        if exception:
            evidence["exception_type"] = type(exception).__name__
            evidence["exception_message"] = str(exception)

        summary_metrics = {
            "total_recorded_steps": len(self.step_records),
            "min_eef_to_object_dist": evidence["min_eef_to_object_dist"],
            "max_lift_delta_z": evidence["max_lift_delta_z"],
            "ever_reached": self.ever_reached,
            "ever_grasped": self.ever_grasped,
            "ever_lifted": self.ever_lifted,
        }

        # Case 1: Task completed successfully
        if final_success:
            return DiagnosticReport(
                termination_reason=TerminationReason.SUCCESS.value,
                failure_phase=FailurePhase.NONE.value,
                primary_failure_code=None,
                secondary_failure_tags=[],
                first_failure_step=None,
                evidence=evidence,
                summary_metrics=summary_metrics,
                config=asdict(self.config),
            )

        # Case 2: Explicit abnormal exceptions / invalid actions
        secondary_tags: List[str] = []
        if termination_reason == TerminationReason.INVALID_ACTION:
            return DiagnosticReport(
                termination_reason=termination_reason.value,
                failure_phase=FailurePhase.REACH.value if not self.ever_reached else FailurePhase.GRASP.value,
                primary_failure_code=SRSAttributionCode.F10_ACTION_DECODING.value,
                secondary_failure_tags=["action_bounds_violation"],
                first_failure_step=len(self.step_records) - 1,
                evidence=evidence,
                summary_metrics=summary_metrics,
                config=asdict(self.config),
            )

        if termination_reason == TerminationReason.SIMULATOR_ERROR:
            return DiagnosticReport(
                termination_reason=termination_reason.value,
                failure_phase=FailurePhase.TIMEOUT.value,
                primary_failure_code=SRSAttributionCode.F15_SIMULATOR_INSTABILITY.value,
                secondary_failure_tags=["mujoco_physics_error"],
                first_failure_step=len(self.step_records) - 1,
                evidence=evidence,
                summary_metrics=summary_metrics,
                config=asdict(self.config),
            )

        if termination_reason == TerminationReason.POLICY_ERROR:
            return DiagnosticReport(
                termination_reason=termination_reason.value,
                failure_phase=FailurePhase.TIMEOUT.value,
                primary_failure_code=SRSAttributionCode.F16_POLICY_INFERENCE_FAILURE.value,
                secondary_failure_tags=["torch_inference_exception"],
                first_failure_step=len(self.step_records) - 1,
                evidence=evidence,
                summary_metrics=summary_metrics,
                config=asdict(self.config),
            )

        # Case 3: Reached max steps (timeout) without explicit crash
        # 3a. Determine physical failure phase and first failure step
        if not self.ever_reached:
            phase = FailurePhase.REACH
            first_fail_step = self.min_eef_to_obj_step if self.min_eef_to_obj_step is not None else 0
            secondary_tags.append("failed_to_approach_target_object")
            # Without active perception probing, visual perception (F1) vs language grounding (F2) vs target localization (F3) cannot be distinguished
            primary_code = SRSAttributionCode.UNATTRIBUTED.value
            secondary_tags.append("perception_grounding_unisolated")
        elif not self.ever_grasped:
            phase = FailurePhase.GRASP
            first_fail_step = self.reach_step
            secondary_tags.append("failed_gripper_closure_or_contacts")
            # Robot successfully reached target object within threshold, but grasp acquisition failed
            primary_code = SRSAttributionCode.F5_GRASP_ACQUISITION.value
        elif not self.ever_lifted:
            phase = FailurePhase.LIFT
            first_fail_step = self.grasp_step
            secondary_tags.append("insufficient_vertical_lift")
            # Grasp was established, but lift was not achieved (slip or insufficient torque)
            primary_code = SRSAttributionCode.F6_GRASP_RETENTION.value if self.lost_contact_step is not None else SRSAttributionCode.UNATTRIBUTED.value
        elif not self.ever_transported:
            phase = FailurePhase.TRANSPORT
            first_fail_step = self.lost_contact_step if self.lost_contact_step is not None else self.lift_step
            secondary_tags.append("dropped_or_stalled_in_transit")
            primary_code = SRSAttributionCode.F6_GRASP_RETENTION.value if self.lost_contact_step is not None else SRSAttributionCode.F7_TRANSPORT.value
        elif not self.ever_placed:
            phase = FailurePhase.PLACEMENT
            first_fail_step = self.transport_step if self.transport_step is not None else self.lift_step
            secondary_tags.append("failed_to_place_inside_goal_container")
            primary_code = SRSAttributionCode.F8_PLACEMENT.value
        else:
            phase = FailurePhase.TIMEOUT
            first_fail_step = len(self.step_records) - 1
            secondary_tags.append("exceeded_max_horizon")
            primary_code = SRSAttributionCode.UNATTRIBUTED.value

        # 3b. Override primary failure code if there is explicit timing/chunking anomaly evidence
        if timing_anomalies and len(timing_anomalies) > 0:
            primary_code = SRSAttributionCode.F13_TIMING_CHUNKING.value
            secondary_tags.extend(timing_anomalies)

        return DiagnosticReport(
            termination_reason=termination_reason.value,
            failure_phase=phase.value,
            primary_failure_code=primary_code,
            secondary_failure_tags=secondary_tags,
            first_failure_step=first_fail_step,
            evidence=evidence,
            summary_metrics=summary_metrics,
            config=asdict(self.config),
        )
