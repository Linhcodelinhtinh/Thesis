"""Compact, provenance-checked causal memory text for LIBERO-Mem."""
from src.interfaces.text_memory import RenderResult
from src.memory.models import EvidenceSource


class LiberoMemTextRenderer:
    def __init__(self, max_total_tokens=48, tokenizer=None, strict_tokenizer=False, allow_oracle=False):
        if max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.tokenizer = tokenizer
        self.strict_tokenizer = strict_tokenizer
        self.allow_oracle = allow_oracle

    def count_tokens(self, text):
        # Same newline/special-token convention as the SmolVLA preprocessor.
        text = text if text.endswith("\n") else text + "\n"
        if self.tokenizer is not None:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            return len(encoded.ids if hasattr(encoded, "ids") else encoded)
        if self.strict_tokenizer:
            raise ValueError("Official tokenizer required; token estimation is test-only")
        return max(1, int(len(text.split()) * 1.35) + 1)

    def _facts(self, memory):
        privileged = {EvidenceSource.ORACLE_SIMULATOR, EvidenceSource.HUMAN_ANNOTATION}
        # Reject the entire privileged snapshot so counters/object projections
        # cannot leak information even when their event list is empty.
        if not self.allow_oracle and (any(ev.source in privileged for ev in memory.events) or
                any((obj.metadata or {}).get("source") in privileged for obj in memory.objects.values())):
            return []
        allowed = {ev.id: ev for ev in memory.events if ev.available_at <= memory.as_of_observation
                   and (self.allow_oracle or ev.source == EvidenceSource.PREDICTED_RGB_TRACKER)}
        facts = []
        for cycle in memory.cycles.values():
            if len(cycle.event_ids) != len(set(cycle.event_ids)):
                continue
            referenced = [allowed[eid] for eid in cycle.event_ids if eid in allowed]
            if len(referenced) != len(cycle.event_ids):
                continue
            returns = [ev for ev in referenced if ev.type == "returned_to_support"]
            if len(returns) != cycle.confirmed:
                continue
            if not self.allow_oracle and any(
                (ev.details or {}).get("object_id") != cycle.object_id or
                (ev.details or {}).get("support_id") != cycle.support_id or
                (ev.details or {}).get("cycle_count") != index
                for index, ev in enumerate(returns, 1)
            ):
                continue
            # Zero is a fact only after the initial support has been observed.
            initial = [ev for ev in allowed.values() if ev.type == "initial_support_observed"
                       and (ev.details or {}).get("object_id") == cycle.object_id
                       and (ev.details or {}).get("support_id") == cycle.support_id]
            if not initial and not (self.allow_oracle and referenced):
                continue
            relevant = [ev for ev in allowed.values() if (ev.details or {}).get("object_id") == cycle.object_id
                        or (self.allow_oracle and ev.id in cycle.event_ids)]
            if cycle.in_progress and (not relevant or max(relevant, key=lambda ev: ev.available_at).type != "departed_support"):
                continue
            name = cycle.object_id.replace("_", " ")
            complete = memory.history_complete and cycle.history_complete and cycle.uncertain_count == 0
            if complete:
                fact = f"{name}: {cycle.confirmed} returns to original plate."
                if cycle.in_progress:
                    fact += " Cycle underway."
            elif cycle.confirmed:
                fact = f"{name}: at least {cycle.confirmed} returns to original plate; history incomplete."
            else:
                fact = f"{name}: return count unknown; history incomplete."
            facts.append(fact)
        return facts

    def render(self, instruction, memory, enabled=True):
        tokens = self.count_tokens(instruction)
        facts = self._facts(memory) if enabled and memory is not None else []
        included, dropped = [], []
        for fact in facts:
            candidate = instruction + " Memory: " + " ".join(included + [fact])
            if self.count_tokens(candidate) <= self.max_total_tokens:
                included.append(fact)
            else:
                dropped.append(fact)
        context = " Memory: " + " ".join(included) if included else ""
        text = instruction + context
        reason = None
        if not included:
            reason = ("memory_disabled_or_none" if not enabled or memory is None else
                      "budget_exceeded" if dropped else "no_confirmed_facts")
        overflow = tokens > self.max_total_tokens
        return RenderResult(
            text=text, truncated=bool(dropped) or overflow, total_facts_considered=len(facts),
            included_facts_count=len(included), dropped_facts_count=len(dropped),
            dropped_facts=tuple(dropped), rendered_context_chars=len(context), budget_chars=len(text),
            rendered_context_tokens=self.count_tokens(context) if context else 0,
            budget_tokens=self.max_total_tokens, token_truncated=bool(dropped) or overflow,
            total_tokens_with_instruction=self.count_tokens(text), memory_skipped_reason=reason,
            facts_included=tuple(included), tokenizer_truncated=overflow,
        )
