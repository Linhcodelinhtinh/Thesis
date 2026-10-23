"""Deterministic text rendering for LIBERO-Mem Schema V2 memory context.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 7:
- Deterministic English phrasing for cycle counts, in-progress cycles, and relations.
- Strict token budgeting with exact pass-through when disabled or no-memory.
- Qualifies incomplete history (never falsely claims zero times when history is unknown).
- Preserves object and container identities across instances (e.g. basket A vs basket B).
- Enforces causal filtering: facts available after snapshot.as_of_observation are ignored.
- Enforces privilege gate: oracle-sourced events are excluded unless allow_oracle=True.
- Returns comprehensive RenderResult audit metadata.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from src.interfaces.text_memory import RenderResult
from src.memory.models import (
    CycleCounter,
    EvidenceSource,
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


def _format_entity_name(raw_name: str) -> str:
    """Format entity identifier cleanly preserving identity (e.g. basket_A -> basket A)."""
    return raw_name.replace("_", " ").strip()


class LiberoMemTextRenderer:
    """Renders Schema V2 memory into grounded, token-budgeted prompt additions."""

    def __init__(
        self,
        max_total_tokens: int = 48,
        tokenizer: Optional[Any] = None,
        strict_tokenizer: bool = False,
        allow_oracle: bool = False,
    ) -> None:
        if max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.tokenizer = tokenizer
        self.strict_tokenizer = strict_tokenizer
        self.allow_oracle = allow_oracle

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer if provided, else accurate estimate."""
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
        instr_tokens = self.count_tokens(instruction)
        if instr_tokens > self.max_total_tokens:
            raise ValueError(
                f"Instruction length ({instr_tokens} tokens) exceeds total token budget ({self.max_total_tokens})"
            )

        # Exact pass-through if disabled or no memory (do NOT modify instruction string)
        if not enabled or memory is None:
            return RenderResult(
                text=instruction,
                truncated=False,
                total_facts_considered=0,
                included_facts_count=0,
                dropped_facts_count=0,
                dropped_facts=(),
                rendered_context_chars=0,
                budget_chars=len(instruction),
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens,
                token_truncated=False,
                total_tokens_with_instruction=instr_tokens,
                memory_skipped_reason="memory_disabled_or_none" if not enabled else "memory_empty",
                facts_included=(),
            )

        as_of = memory.as_of_observation

        # Build map of allowed event IDs
        allowed_event_ids: Set[str] = set()
        for ev in memory.events:
            # Causal check: event must be available at or before as_of
            if ev.available_at > as_of:
                continue
            # Privilege check: exclude oracle events unless allow_oracle is set
            if ev.source == EvidenceSource.ORACLE_SIMULATOR and not self.allow_oracle:
                continue
            allowed_event_ids.add(ev.id)

        facts_considered: List[str] = []

        # 1. Cycle facts
        for cycle_key, cycle in memory.cycles.items():
            # Check history completeness
            if not memory.history_complete or not cycle.history_complete:
                facts_considered.append("cycle history incomplete; completed count unknown.")
                continue

            # Check if cycle event is allowed
            has_valid_event = True
            if cycle.event_ids:
                has_valid_event = any(eid in allowed_event_ids for eid in cycle.event_ids)
            elif not self.allow_oracle and any(ev.source == EvidenceSource.ORACLE_SIMULATOR for ev in memory.events):
                has_valid_event = False

            if not has_valid_event and (cycle.confirmed > 0 or cycle.in_progress or not self.allow_oracle):
                # Cycle references future or disallowed oracle event
                continue

            if cycle.uncertain_count > 0:
                fact = f"{cycle.confirmed} returns confirmed; {cycle.uncertain_count} later return uncertain."
            else:
                n_str = _number_to_word(cycle.confirmed)
                progress_str = "one cycle in progress" if cycle.in_progress else "no cycle in progress"
                obj_name = _format_entity_name(cycle.object_id)
                fact = f"{obj_name} returned to its original support {n_str}; {progress_str}."
            facts_considered.append(fact)

        # 2. Containment / Occlusion facts
        for obj_id, obj in memory.objects.items():
            if obj.last_confirmed_in is not None:
                is_oracle_obj = obj.metadata is not None and obj.metadata.get("source") == EvidenceSource.ORACLE_SIMULATOR
                if is_oracle_obj and not self.allow_oracle:
                    continue
                short_name = _format_entity_name(obj_id)
                container = _format_entity_name(obj.last_confirmed_in)
                if obj.visibility == VisibilityStatus.OCCLUDED:
                    facts_considered.append(f"{short_name} was placed in {container}; {short_name} is now hidden.")
                else:
                    facts_considered.append(f"{short_name} was placed in {container}.")

        # If no facts considered, return exact pass-through
        if not facts_considered:
            return RenderResult(
                text=instruction,
                truncated=False,
                total_facts_considered=0,
                included_facts_count=0,
                dropped_facts_count=0,
                dropped_facts=(),
                rendered_context_chars=0,
                budget_chars=len(instruction),
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
        base_text = instruction.rstrip()

        for fact in facts_considered:
            test_facts = list(included_facts) + [fact]
            test_memory_str = " Memory: " + " ".join(test_facts)
            candidate_text = base_text + test_memory_str
            total_toks = self.count_tokens(candidate_text)

            if total_toks <= self.max_total_tokens:
                included_facts.append(fact)
            else:
                dropped_facts.append(fact)

        if not included_facts:
            return RenderResult(
                text=instruction,
                truncated=True,
                total_facts_considered=len(facts_considered),
                included_facts_count=0,
                dropped_facts_count=len(dropped_facts),
                dropped_facts=tuple(dropped_facts),
                rendered_context_chars=0,
                budget_chars=len(instruction),
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens,
                token_truncated=True,
                total_tokens_with_instruction=instr_tokens,
                memory_skipped_reason="budget_exceeded",
                facts_included=(),
            )

        final_memory_text = " Memory: " + " ".join(included_facts)
        final_full_text = base_text + final_memory_text
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
