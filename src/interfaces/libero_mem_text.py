"""Deterministic text rendering for LIBERO-Mem Schema V2 memory context.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 7:
- Deterministic English phrasing for cycle counts, in-progress cycles, and relations.
- Strict 48-token total budget (instruction + memory).
- Instruction is NEVER truncated; if budget exceeded, memory facts are dropped cleanly.
- Exact pass-through when memory is empty, disabled, or no facts are confirmed.
- Returns comprehensive RenderResult audit metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.interfaces.text_memory import RenderResult
from src.memory.models import (
    CycleCounter,
    ObjectIdentity,
    RelationFact,
    RelationStatus,
    SchemaV2EpisodeMemory,
    VisibilityStatus,
)


def _number_to_word(n: int) -> str:
    words = {
        0: "zero times",
        1: "once",
        2: "twice",
        3: "three times",
        4: "four times",
        5: "five times",
    }
    return words.get(n, f"{n} times")


class LiberoMemTextRenderer:
    """Renders Schema V2 memory into grounded, token-budgeted prompt additions."""

    def __init__(
        self,
        max_total_tokens: int = 48,
        tokenizer: Optional[Any] = None,
        strict_tokenizer: bool = False,
    ) -> None:
        if max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.tokenizer = tokenizer
        self.strict_tokenizer = strict_tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer if provided, else accurate whitespace/subword estimate."""
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "encode"):
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            elif callable(self.tokenizer):
                res = self.tokenizer(text)
                if isinstance(res, dict) and "input_ids" in res:
                    return len(res["input_ids"])
                elif isinstance(res, list):
                    return len(res)
        if self.strict_tokenizer:
            raise ValueError("strict_tokenizer requires a valid Hugging Face tokenizer instance")
        # Approximate tokenizer token count (~1.3 tokens per word)
        words = text.strip().split()
        if not words:
            return 0
        return max(len(words), int(len(text) / 3.8) + 1)

    def render(
        self,
        instruction: str,
        memory: Optional[SchemaV2EpisodeMemory],
        enabled: bool = True,
    ) -> RenderResult:
        """Render memory string appended to instruction within max_total_tokens."""
        clean_instruction = instruction.strip()
        instr_tokens = self.count_tokens(clean_instruction)

        # Exact pass-through if disabled or no memory
        if not enabled or memory is None:
            return RenderResult(
                text=clean_instruction,
                truncated=False,
                total_facts_considered=0,
                included_facts_count=0,
                dropped_facts_count=0,
                dropped_facts=(),
                rendered_context_chars=0,
                budget_chars=len(clean_instruction),
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens,
                token_truncated=False,
                total_tokens_with_instruction=instr_tokens,
                memory_skipped_reason="memory_disabled_or_none" if not enabled else "memory_empty",
                facts_included=(),
            )

        facts_considered: List[str] = []

        # 1. Cycle facts
        for cycle_key, cycle in memory.cycles.items():
            if cycle.uncertain_count > 0:
                confirmed_str = _number_to_word(cycle.confirmed)
                fact = f"{cycle.confirmed} returns confirmed; {cycle.uncertain_count} later return uncertain."
            else:
                n_str = _number_to_word(cycle.confirmed)
                progress_str = "one cycle in progress" if cycle.in_progress else "no cycle in progress"
                obj_name = cycle.object_id.split("_")[0]
                fact = f"{obj_name} returned to its original support {n_str}; {progress_str}."
            facts_considered.append(fact)

        # 2. Containment / Occlusion facts
        for obj_id, obj in memory.objects.items():
            if obj.last_confirmed_in is not None:
                short_name = obj_id.split("_")[0]
                container = obj.last_confirmed_in.split("_")[0]
                if obj.visibility == VisibilityStatus.OCCLUDED:
                    facts_considered.append(f"{short_name} was placed in {container}; {short_name} is now hidden.")
                else:
                    facts_considered.append(f"{short_name} was placed in {container}.")

        # If no facts considered, return exact pass-through
        if not facts_considered:
            return RenderResult(
                text=clean_instruction,
                truncated=False,
                total_facts_considered=0,
                included_facts_count=0,
                dropped_facts_count=0,
                dropped_facts=(),
                rendered_context_chars=0,
                budget_chars=len(clean_instruction),
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens,
                token_truncated=False,
                total_tokens_with_instruction=instr_tokens,
                memory_skipped_reason="no_confirmed_facts",
                facts_included=(),
            )

        # Build candidate memory strings with budget checking
        included_facts: List[str] = []
        dropped_facts: List[str] = []

        for fact in facts_considered:
            test_facts = list(included_facts) + [fact]
            test_memory_str = " Memory: " + " ".join(test_facts)
            candidate_text = clean_instruction + test_memory_str
            total_toks = self.count_tokens(candidate_text)

            if total_toks <= self.max_total_tokens:
                included_facts.append(fact)
            else:
                dropped_facts.append(fact)

        if not included_facts:
            # Memory could not fit within budget; preserve instruction untouched
            return RenderResult(
                text=clean_instruction,
                truncated=True,
                total_facts_considered=len(facts_considered),
                included_facts_count=0,
                dropped_facts_count=len(dropped_facts),
                dropped_facts=tuple(dropped_facts),
                rendered_context_chars=0,
                budget_chars=len(clean_instruction),
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens,
                token_truncated=True,
                total_tokens_with_instruction=instr_tokens,
                memory_skipped_reason="budget_exceeded",
                facts_included=(),
            )

        final_memory_text = " Memory: " + " ".join(included_facts)
        final_full_text = clean_instruction + final_memory_text
        final_total_tokens = self.count_tokens(final_full_text)
        final_memory_tokens = self.count_tokens(final_memory_text.strip())

        return RenderResult(
            text=final_full_text,
            truncated=len(dropped_facts) > 0,
            total_facts_considered=len(facts_considered),
            included_facts_count=len(included_facts),
            dropped_facts_count=len(dropped_facts),
            dropped_facts=tuple(dropped_facts),
            rendered_context_chars=len(final_memory_text),
            budget_chars=len(final_full_text),
            rendered_context_tokens=final_memory_tokens,
            budget_tokens=self.max_total_tokens,
            token_truncated=len(dropped_facts) > 0,
            total_tokens_with_instruction=final_total_tokens,
            memory_skipped_reason=None,
            facts_included=tuple(included_facts),
        )
