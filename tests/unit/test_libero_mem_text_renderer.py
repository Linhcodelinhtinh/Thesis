"""Unit tests for LiberoMemTextRenderer."""

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.models import (
    CycleCounter,
    ObjectIdentity,
    SchemaV2EpisodeMemory,
)


def test_renderer_exact_pass_through_when_disabled():
    renderer = LiberoMemTextRenderer(max_total_tokens=48)
    instr = "lift the bowl and place it back on the plate 3 times"

    res_disabled = renderer.render(instr, None, enabled=False)
    assert res_disabled.text == instr
    assert res_disabled.rendered_context_chars == 0
    assert not res_disabled.truncated


def test_renderer_cycle_rendering():
    renderer = LiberoMemTextRenderer(max_total_tokens=48)
    instr = "lift the bowl and place it back on the plate"

    counter1 = CycleCounter(
        cycle_key="bowl_A_plate_A",
        object_id="bowl_A",
        support_id="plate_A",
        confirmed=1,
        in_progress=False,
    )
    mem1 = SchemaV2EpisodeMemory(
        episode_id="ep1",
        as_of_observation=50,
        last_consumed_action=49,
        history_complete=True,
        objects={"bowl_A": ObjectIdentity("bowl_A", "origin")},
        relations=(),
        cycles={"bowl_A_plate_A": counter1},
        events=(),
    )

    res = renderer.render(instr, mem1, enabled=True)
    assert "bowl A returned to its original support once; no cycle in progress." in res.text
    assert res.text.startswith(instr)
    assert len(res.facts_included) == 1
    assert not res.truncated


def test_renderer_in_progress_rendering():
    renderer = LiberoMemTextRenderer(max_total_tokens=48)
    instr = "lift the bowl and place it back on the plate 3 times"

    counter2 = CycleCounter(
        cycle_key="bowl_A_plate_A",
        object_id="bowl_A",
        support_id="plate_A",
        confirmed=2,
        in_progress=True,
    )
    mem2 = SchemaV2EpisodeMemory(
        episode_id="ep2",
        as_of_observation=100,
        last_consumed_action=99,
        history_complete=True,
        objects={"bowl_A": ObjectIdentity("bowl_A", "origin")},
        relations=(),
        cycles={"bowl_A_plate_A": counter2},
        events=(),
    )

    res = renderer.render(instr, mem2, enabled=True)
    assert "bowl A returned to its original support twice; one cycle in progress." in res.text
    assert not res.truncated


def test_renderer_strict_budget_never_truncates_instruction():
    # Very small budget where memory cannot fit
    renderer = LiberoMemTextRenderer(max_total_tokens=15)
    instr = "lift the bowl and place it back on the plate 3 times"

    counter = CycleCounter(
        cycle_key="bowl_A_plate_A",
        object_id="bowl_A",
        support_id="plate_A",
        confirmed=1,
        in_progress=False,
    )
    mem = SchemaV2EpisodeMemory(
        episode_id="ep3",
        as_of_observation=50,
        last_consumed_action=49,
        history_complete=True,
        objects={"bowl_A": ObjectIdentity("bowl_A", "origin")},
        relations=(),
        cycles={"bowl_A_plate_A": counter},
        events=(),
    )

    res = renderer.render(instr, mem, enabled=True)
    # Instruction is preserved completely!
    assert res.text == instr
    assert res.truncated
    assert res.memory_skipped_reason == "budget_exceeded"
    assert len(res.facts_included) == 0
