"""Deterministic text rendering for retrieved, provenance-aware memory."""

from dataclasses import dataclass
from typing import Any, FrozenSet, List, Optional, Tuple

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
    rendered_context_tokens: int = 0
    budget_tokens: int = 0
    token_truncated: bool = False
    total_tokens_with_instruction: int = 0


class TextMemoryRenderer:
    """Append concise facts while preserving an exact no-memory pass-through.

    Oracle evidence is excluded by default. Stage A upper-bound experiments
    must opt in explicitly and retain provenance labels in rendered context.
    ``max_context_chars`` bounds the memory block, not the original instruction.
    ``max_total_tokens`` enforces strict token budget matching model preprocessor.
    """

    def __init__(
        self,
        max_context_chars: int = 512,
        allowed_sources: Optional[FrozenSet[EvidenceSource]] = None,
        max_total_tokens: Optional[int] = None,
        compact_mode: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self.max_context_chars = max_context_chars
        self.max_total_tokens = max_total_tokens
        self.compact_mode = compact_mode
        self.tokenizer = tokenizer
        if allowed_sources is None:
            self.allowed_sources = frozenset(
                source
                for source in EvidenceSource
                if source != EvidenceSource.ORACLE_SIMULATOR
            )
        else:
            self.allowed_sources = frozenset(allowed_sources)

    def _count_tokens(self, text: str) -> int:
        """Estimate or compute token count using official tokenizer or conservative fallback."""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                elif callable(self.tokenizer):
                    res = self.tokenizer(text)
                    if isinstance(res, dict) and "input_ids" in res:
                        return len(res["input_ids"])
            except Exception:
                pass
        # Conservative fallback: words * 1.35 + 1
        words = text.strip().split()
        if not words:
            return 0
        return max(1, int(len(words) * 1.35) + 1)

    def render_with_audit(self, instruction: str, memory: RetrievedMemory) -> RenderResult:
        """Render memory facts into instruction prompt and return full audit metadata."""
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        candidate_facts: List[str] = []
        for item in memory.objects:
            if item.evidence_source not in self.allowed_sources:
                continue
            qualifier = "Possibly " if item.validity == MemoryStatus.UNCERTAIN else ""
            if self.compact_mode:
                candidate_facts.append(
                    "{}{} is {}.".format(
                        qualifier,
                        item.semantic_label,
                        item.state.replace("_", " "),
                    )
                )
            else:
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
            if self.compact_mode:
                statement = "{}{}: {}".format(qualifier, item.semantic_label, item.event_type.replace("_", " "))
                if item.object_state_after:
                    statement += " -> {}".format(item.object_state_after.replace("_", " "))
                candidate_facts.append(statement + ".")
            else:
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

        inst_tokens = self._count_tokens(instruction)

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
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=False,
                total_tokens_with_instruction=inst_tokens,
            )

        # If instruction alone exceeds or meets max_total_tokens, memory cannot be injected
        if self.max_total_tokens is not None and inst_tokens >= self.max_total_tokens:
            return RenderResult(
                text=instruction,
                truncated=True,
                total_facts_considered=len(candidate_facts),
                included_facts_count=0,
                dropped_facts_count=len(candidate_facts),
                dropped_facts=tuple(candidate_facts),
                rendered_context_chars=0,
                budget_chars=self.max_context_chars,
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=True,
                total_tokens_with_instruction=inst_tokens,
            )

        lines = ["[Episode memory]"]
        included_facts: List[str] = []
        dropped_facts: List[str] = []
        char_truncated = False
        token_truncated = False

        for fact in candidate_facts:
            candidate_lines = lines + ["- " + fact]
            candidate_block = "\n".join(candidate_lines)
            candidate_full_text = instruction + "\n\n" + candidate_block
            candidate_total_tokens = self._count_tokens(candidate_full_text)

            exceeds_chars = len(candidate_block) > self.max_context_chars
            exceeds_tokens = (
                self.max_total_tokens is not None
                and candidate_total_tokens > self.max_total_tokens
            )

            if exceeds_chars or exceeds_tokens:
                if exceeds_chars:
                    char_truncated = True
                if exceeds_tokens:
                    token_truncated = True
                dropped_facts.append(fact)
            else:
                lines.append("- " + fact)
                included_facts.append(fact)

        truncated = char_truncated or token_truncated

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
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=token_truncated,
                total_tokens_with_instruction=inst_tokens,
            )

        block = "\n".join(lines)
        rendered_text = instruction + "\n\n" + block
        rendered_chars = len(block)
        final_total_tokens = self._count_tokens(rendered_text)
        rendered_context_tokens = max(0, final_total_tokens - inst_tokens)

        return RenderResult(
            text=rendered_text,
            truncated=truncated,
            total_facts_considered=len(candidate_facts),
            included_facts_count=len(included_facts),
            dropped_facts_count=len(dropped_facts),
            dropped_facts=tuple(dropped_facts),
            rendered_context_chars=rendered_chars,
            budget_chars=self.max_context_chars,
            rendered_context_tokens=rendered_context_tokens,
            budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
            token_truncated=token_truncated,
            total_tokens_with_instruction=final_total_tokens,
        )

    def render(self, instruction: str, memory: RetrievedMemory) -> str:
        """Render facts and return string prompt (backward-compatible API)."""
        return self.render_with_audit(instruction, memory).text
