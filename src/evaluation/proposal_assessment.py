"""Offline precondition checks for audited proposal labels."""
from enum import Enum
import re


class Verdict(str, Enum):
    SUPPORTED = "supported"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


def return_cycle_target(instruction):
    """Parse the supported language contract without reading simulator goals."""
    match = re.fullmatch(
        r"(?:pick up|lift) the bowl and place it (?:back )?on the plate(?: ([1-9][0-9]*) times)?",
        " ".join(instruction.lower().split()),
    )
    if match is None:
        raise ValueError("Unsupported return-cycle instruction")
    return int(match[1] or 1)


def combine_hypotheses(verdicts, *, residual_unknown=False):
    """Require agreement across every still-plausible history."""
    verdicts = set(verdicts)
    if residual_unknown or len(verdicts) != 1:
        return Verdict.UNKNOWN
    return Verdict(next(iter(verdicts)))


def count_precondition(fact, target, intent):
    """Check a labelled restart or finish against observed-return evidence.

    The target and intent must be independently supplied. This function does
    not infer them from controller actions and has no authority in the runner.
    """
    if intent not in ("start_cycle", "finish") or type(target) is not int or target < 1:
        raise ValueError("Expected a labelled start/finish intent and a positive target")
    if fact is None or fact.validity != "supported" or fact.value[0] is None:
        return Verdict.UNKNOWN
    observed = fact.value[0]
    if observed > target:
        return Verdict.VIOLATED
    if fact.value[1]:
        return Verdict.VIOLATED
    if intent == "finish":
        return Verdict.SUPPORTED if observed == target else Verdict.VIOLATED
    return Verdict.SUPPORTED if observed < target else Verdict.VIOLATED
