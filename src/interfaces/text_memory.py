"""Deterministic text rendering for retrieved, provenance-aware memory."""

from typing import FrozenSet, List, Optional

from src.memory.models import EvidenceSource, MemoryStatus, RetrievedMemory


class TextMemoryRenderer:
    """Append concise facts while preserving an exact no-memory pass-through.

    Oracle evidence is excluded by default. Stage A upper-bound experiments
    must opt in explicitly and retain provenance labels in rendered context.
    ``max_context_chars`` bounds the memory block, not the original instruction.
    """

    def __init__(
        self,
        max_context_chars: int = 512,
        allowed_sources: Optional[FrozenSet[EvidenceSource]] = None,
    ) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        self.max_context_chars = max_context_chars
        if allowed_sources is None:
            self.allowed_sources = frozenset(
                source
                for source in EvidenceSource
                if source != EvidenceSource.ORACLE_SIMULATOR
            )
        else:
            self.allowed_sources = frozenset(allowed_sources)

    def render(self, instruction: str, memory: RetrievedMemory) -> str:
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        facts: List[str] = []
        for item in memory.objects:
            if item.evidence_source not in self.allowed_sources:
                continue
            qualifier = "Possibly " if item.validity == MemoryStatus.UNCERTAIN else ""
            facts.append(
                "{}{} ({}) is {} [status={}, step {}, confidence {:.2f}, source={}].".format(
                    qualifier,
                    item.semantic_label,
                    item.object_id,
                    item.state,
                    item.validity.value,
                    item.last_updated_step,
                    item.confidence,
                    item.evidence_source.value,
                )
            )

        for item in memory.events:
            if item.evidence_source not in self.allowed_sources:
                continue
            qualifier = "Possible event: " if item.validity == MemoryStatus.UNCERTAIN else ""
            statement = "{}{}: {}".format(qualifier, item.semantic_label, item.event_type)
            if item.object_state_after:
                statement += " → {}".format(item.object_state_after)
            if item.target_id:
                statement += " → {}".format(item.target_id)
            facts.append(
                "{} [step {}, confidence {:.2f}, source={}].".format(
                    statement,
                    item.step,
                    item.confidence,
                    item.evidence_source.value,
                )
            )

        if not facts:
            return instruction

        lines = ["[Episode memory]"]
        for fact in facts:
            candidate = "\n".join(lines + ["- " + fact])
            if len(candidate) > self.max_context_chars:
                break
            lines.append("- " + fact)
        if len(lines) == 1:
            return instruction
        block = "\n".join(lines)
        return instruction + "\n\n" + block
