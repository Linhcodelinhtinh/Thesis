"""Perception profiles for the pinned LIBERO-Mem task inventory."""
from dataclasses import dataclass

from src.evaluation.libero_mem_protocol import task_spec


@dataclass(frozen=True)
class MemoryProfile:
    subject: str
    subject_count: int
    reference: str
    reference_count: int
    predicate: str
    count_cycles: bool = False

    @property
    def queries(self):
        vocabulary = {"bowl": "a metal bowl.", "plate": "a white plate.",
                      "bottle": "a wine bottle.", "basket": "a basket.",
                      "cream cheese": "a cream cheese box."}
        return {label: vocabulary[label] for label in (self.subject, self.reference)}


def profile_for_task(suite, task):
    index = task_spec(suite, task)["index"]
    if index < 6:
        label = "bottle" if index in (1, 3) else "bowl"
        return MemoryProfile(label, 1, "plate", 3, "on", count_cycles=True)
    if index in (6, 7):
        return MemoryProfile("bowl", index - 4, "plate", index - 3, "on")
    if index in (8, 9):
        return MemoryProfile("cream cheese", 1, "basket", 2, "in")
    raise ValueError("No LIBERO-Mem perception profile for this task")
