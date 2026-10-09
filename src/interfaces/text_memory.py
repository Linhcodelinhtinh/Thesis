"""Deterministic text rendering for retrieved, provenance-aware memory."""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from src.memory.models import EvidenceSource, MemoryStatus, RetrievedMemory


@dataclass(frozen=True)
class RenderResult:
    """Detailed audit metadata for rendered memory context."""

    text: str
    truncated: bool
    total_facts_considered: int
    included_facts_count: int
    dropped_facts_count: int
    dropped_facts: Tuple[str, ...]
    rendered_context_chars: int
    budget_chars: int


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

    def render_with_audit(self, instruction: str, memory: RetrievedMemory) -> RenderResult:
        """Render memory facts into instruction prompt and return full audit metadata."""
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        candidate_facts: List[str] = []
        for item in memory.objects:
            if item.evidence_source not in self.allowed_sources:
                continue
            qualifier = "Possibly " if item.validity == MemoryStatus.UNCERTAIN else ""
            candidate_facts.append(
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
            candidate_facts.append(
                "{} [step {}, confidence {:.2f}, source={}].".format(
                    statement,
                    item.step,
                    item.confidence,
                    item.evidence_source.value,
                )
            )

        if not candidate_facts:
            return RenderResult(
                text=instruction,
                truncated=False,
                total_facts_considered=0,
                included_facts_count=0,
                dropped_facts_count=0,
                dropped_facts=(),
                rendered_context_chars=0,
                budget_chars=self.max_context_chars,
            )

        lines = ["[Episode memory]"]
        included_facts: List[str] = []
        dropped_facts: List[str] = []
        truncated = False

        for fact in candidate_facts:
            candidate = "\n".join(lines + ["- " + fact])
            if len(candidate) > self.max_context_chars:
                truncated = True
                dropped_facts.append(fact)
            else:
                lines.append("- " + fact)
                included_facts.append(fact)

        if len(lines) == 1:
            # Memory header alone exceeds budget or no facts could fit
            return RenderResult(
                text=instruction,
                truncated=truncated,
                total_facts_considered=len(candidate_facts),
                included_facts_count=0,
                dropped_facts_count=len(dropped_facts),
                dropped_facts=tuple(dropped_facts),
                rendered_context_chars=0,
                budget_chars=self.max_context_chars,
            )

        block = "\n".join(lines)
        rendered_text = instruction + "\n\n" + block
        rendered_chars = len(block)

        return RenderResult(
            text=rendered_text,
            truncated=truncated,
            total_facts_considered=len(candidate_facts),
            included_facts_count=len(included_facts),
            dropped_facts_count=len(dropped_facts),
            dropped_facts=tuple(dropped_facts),
            rendered_context_chars=rendered_chars,
            budget_chars=self.max_context_chars,
        )

    def render(self, instruction: str, memory: RetrievedMemory) -> str:
        """Render facts and return string prompt (backward-compatible API)."""
        return self.render_with_audit(instruction, memory).text
