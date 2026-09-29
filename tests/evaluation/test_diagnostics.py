"""Test diagnostics integration and coverage under tests/evaluation/."""

from tests.unit.test_diagnostics_unit import (
    test_contact_diagnostics_empty_sim,
    test_contact_diagnostics_gripper_contacts,
    test_contact_diagnostics_object_lift,
)

__all__ = [
    "test_contact_diagnostics_empty_sim",
    "test_contact_diagnostics_gripper_contacts",
    "test_contact_diagnostics_object_lift",
]
