"""Physics, Contact, and Containment Diagnostics for LIBERO Evaluation (Phase 4).

Monitors:
- Finger-to-object contact state via MuJoCo collision geometry.
- Grasp retention and lifting state.
- Distance to target container / basket.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ContactDiagnostics:
    """Diagnostic utility inspecting MuJoCo contact pairs and object kinematics."""

    def __init__(self, env: Any, target_object_name: Optional[str] = None) -> None:
        self.env = env
        self.target_object_name = target_object_name
        self.initial_object_z: Optional[float] = None

    def get_sim(self) -> Any:
        """Resolve MuJoCo sim instance across wrapper layers."""
        sim = getattr(self.env, "sim", None)
        if sim is None and hasattr(self.env, "_env"):
            sim = getattr(self.env._env, "sim", None)
        if sim is None and hasattr(self.env, "env"):
            sim = getattr(self.env.env, "sim", None)
        return sim

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

    def get_active_contacts(self) -> List[Tuple[str, str, float]]:
        """Extract all active contact geom name pairs and normal force."""
        sim = self.get_sim()
        if sim is None or not hasattr(sim, "data") or not hasattr(sim.data, "ncon"):
            return []

        contacts: List[Tuple[str, str, float]] = []
        ncon = sim.data.ncon
        for i in range(ncon):
            contact = sim.data.contact[i]
            geom1_name = sim.model.geom_id2name(contact.geom1) or ""
            geom2_name = sim.model.geom_id2name(contact.geom2) or ""

            # Normal contact force approximation
            dist = float(contact.dist)
            contacts.append((geom1_name, geom2_name, dist))

        return contacts

    def inspect_gripper_contacts(
        self, object_substr: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check whether robot gripper pads are actively contacting target object.

        Args:
            object_substr: Substring matching the target object geom (e.g. 'soup', 'cream_cheese').
                           If None, uses self.target_object_name.

        Returns:
            Dictionary with contact booleans and contact pairs.
        """
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
        self, current_object_pos: Optional[np.ndarray] = None, lift_threshold: float = 0.04
    ) -> Dict[str, Any]:
        """Check if target object has been lifted above its initial support height."""
        if current_object_pos is None:
            current_object_pos = self.get_target_object_pos()

        if current_object_pos is None:
            return {"is_lifted": False, "delta_z": 0.0, "initial_z": None, "current_z": None}

        curr_z = float(current_object_pos[2])
        if self.initial_object_z is None:
            self.initial_object_z = curr_z

        delta_z = curr_z - self.initial_object_z
        is_lifted = delta_z >= lift_threshold

        return {
            "initial_z": self.initial_object_z,
            "current_z": curr_z,
            "delta_z": delta_z,
            "is_lifted": is_lifted,
        }
