"""Renderer tests require actual event provenance, not unsupported counters."""
from dataclasses import replace
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.event_writer import LiberoMemEventWriter
from test_rgb_cycle_writer import rest, cycle


def snapshot(cycles=1, in_progress=False):
    writer = LiberoMemEventWriter("text")
    step = rest(writer, 0)
    for _ in range(cycles):
        step = cycle(writer, step)
    if in_progress:
        rest(writer, step, lift=.15, gap=.005)
    return writer.get_snapshot()


def test_renderer_exact_pass_through_when_disabled():
    renderer = LiberoMemTextRenderer()
    instruction = "  lift the bowl and place it back on the plate 3 times\n"
    assert renderer.render(instruction, snapshot(), enabled=False).text == instruction


def test_renderer_cycle_rendering():
    result = LiberoMemTextRenderer().render("lift the bowl", snapshot())
    assert "bowl A: 1 returns to original plate." in result.text
    assert result.included_facts_count == 1
    assert not result.truncated


def test_renderer_in_progress_rendering():
    result = LiberoMemTextRenderer().render("lift the bowl", snapshot(2, True))
    assert "2 returns to original plate. Cycle underway." in result.text


def test_renderer_strict_budget_never_truncates_instruction():
    instruction = "lift the bowl and place it back on the plate 3 times"
    result = LiberoMemTextRenderer(max_total_tokens=18).render(instruction, snapshot())
    assert result.text == instruction
    assert result.truncated
    assert result.memory_skipped_reason == "budget_exceeded"


def test_one_allowed_event_cannot_authorize_an_entire_counter():
    memory = snapshot(2)
    memory = replace(memory, events=memory.events[:-1])
    assert LiberoMemTextRenderer().render("lift bowl", memory).text == "lift bowl"
