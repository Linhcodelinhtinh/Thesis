"""Treatments accepted by the V2 evaluation harness."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    weights: str
    memory_source: str
    projection: str
    intervention: str
    text_budget: int
    allow_oracle: bool = False


CONDITIONS = {
    "B0": ConditionSpec("base", "none", "instruction_only", "none", 48),
    "F0": ConditionSpec("no_memory", "none", "instruction_only", "none", 48),
    "F1": ConditionSpec("memory", "predicted_rgb_v3", "event_text", "none", 48),
    "F1-off": ConditionSpec("memory", "none", "instruction_only", "suppress_memory", 48),
    "O1": ConditionSpec("base", "oracle", "event_text", "none", 48, allow_oracle=True),
}
CLI_CONDITIONS = ("B0", "F0", "F1", "F1-off")


def condition_spec(name):
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"Unknown or unsupported memory condition: {name}") from None
