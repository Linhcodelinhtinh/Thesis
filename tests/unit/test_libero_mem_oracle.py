"""Unit tests for privileged LiberoMemOracleWriter."""

from src.memory.libero_mem_oracle import LiberoMemOracleWriter
from src.memory.models import EvidenceSource


def test_oracle_writer_cycle_progression():
    oracle = LiberoMemOracleWriter(
        episode_id="oracle_ep_01",
        target_object_id="bowl_A",
        support_id="plate_A",
    )

    # Step 0: at rest
    snap0 = oracle.step(0, {"is_on_support": True, "is_held": False})
    assert snap0.cycles["bowl_A_plate_A"].confirmed == 0
    assert not snap0.cycles["bowl_A_plate_A"].in_progress

    # Step 10: departed
    snap1 = oracle.step(10, {"is_on_support": False, "is_held": True})
    assert snap1.cycles["bowl_A_plate_A"].in_progress
    assert len(snap1.events) == 1
    assert snap1.events[0].type == "departed_support"
    assert snap1.events[0].source == EvidenceSource.ORACLE_SIMULATOR

    # Step 20: returned
    snap2 = oracle.step(20, {"is_on_support": True, "is_held": False})
    assert snap2.cycles["bowl_A_plate_A"].confirmed == 1
    assert not snap2.cycles["bowl_A_plate_A"].in_progress
    assert len(snap2.events) == 2
    assert snap2.events[1].type == "returned_to_support"
    assert snap2.events[1].source == EvidenceSource.ORACLE_SIMULATOR
