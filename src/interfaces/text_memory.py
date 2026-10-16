"""Deterministic text rendering for retrieved, provenance-aware memory."""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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
    memory_skipped_reason: Optional[str] = None
    facts_included: Tuple[str, ...] = ()
    tokenizer_truncated: bool = False


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
        strict_tokenizer: bool = False,
    ) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self.max_context_chars = max_context_chars
        self.max_total_tokens = max_total_tokens
        self.compact_mode = compact_mode
        self.tokenizer = tokenizer
        self.strict_tokenizer = strict_tokenizer
        if allowed_sources is None:
            self.allowed_sources = frozenset(
                source
                for source in EvidenceSource
                if source != EvidenceSource.ORACLE_SIMULATOR
            )
        else:
            self.allowed_sources = frozenset(allowed_sources)

    def count_tokens(self, text: str) -> int:
        """Count tokens using official tokenizer with newline matching preprocessor."""
        if not text:
            return 0
        prompt_text = text if text.endswith("\n") else f"{text}\n"
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
                elif callable(self.tokenizer):
                    res = self.tokenizer(prompt_text)
                    if isinstance(res, dict) and "input_ids" in res:
                        return len(res["input_ids"])
                    elif isinstance(res, list):
                        return len(res)
            except Exception as e:
                if self.strict_tokenizer:
                    raise RuntimeError(f"Tokenizer encoding failed: {e}") from e
        if self.strict_tokenizer:
            raise RuntimeError(
                "Official tokenizer is required for token budget calculation in benchmark evaluation. "
                "Token estimation fallback is strictly forbidden per AGENTS.md."
            )
        # Conservative fallback for tests without tokenizer: words * 1.35 + 1
        words = prompt_text.strip().split()
        if not words:
            return 0
        return max(1, int(len(words) * 1.35) + 1)

    # Maintain backward compatibility
    _count_tokens = count_tokens

    def _extract_compact_facts(self, memory: RetrievedMemory) -> List[Tuple[int, str, str, Dict[str, Any]]]:
        """Extract prioritized, non-redundant compact facts for prompt injection.

        Returns list of (priority, fact_text, entity_id, metadata).
        Lower priority number = higher ranking.

        Priorities:
        1. Active, non-superseded recovery events (e.g. slip, lost contact)
        2. Completed target states (e.g. in_basket, on_plate, placed)
        3. Current active manipulation state (e.g. holding, grasped, lifted)
        (Unmanipulated initial/resting states like on_table, detected, target_destination,
         empty are EXCLUDED to prevent prompt contamination and distribution shift).
        """
        facts: List[Tuple[int, str, str, Dict[str, Any]]] = []
        handled_entities: Set[str] = set()

        # Build map of latest object state and last_updated_step
        obj_state_map: Dict[str, str] = {}
        obj_step_map: Dict[str, int] = {}
        for obj in memory.objects:
            obj_state_map[obj.object_id] = obj.state.lower()
            obj_step_map[obj.object_id] = obj.last_updated_step

        # 1. Recovery events (only active, non-superseded events)
        sorted_events = sorted(memory.events, key=lambda e: e.step, reverse=True)
        for ev in sorted_events:
            if ev.evidence_source not in self.allowed_sources:
                continue
            if ev.validity not in (MemoryStatus.CONFIRMED, MemoryStatus.UNCERTAIN):
                continue
            # Do not render unidentified objects in prompt
            if ev.object_id in ("unidentified_object", "unknown_object", "target_object"):
                continue

            ev_type = ev.event_type.lower()
            if any(term in ev_type for term in ("slip", "dropped", "lost_contact", "contact_lost")):
                latest_obj_step = obj_step_map.get(ev.object_id, -1)
                latest_obj_state = obj_state_map.get(ev.object_id, "")

                # A recovery event is SUPERSEDED if:
                # 1) The object was updated after the recovery event with a resolving state
                #    (grasped, holding, lifted, released, placed, in_*, on_*, contains_*)
                # 2) Or the current object state is not contact_lost
                if latest_obj_step > ev.step:
                    continue
                if latest_obj_state in ("grasped", "holding", "lifted", "released", "placed"):
                    continue
                if any(latest_obj_state.startswith(p) for p in ("in_", "on_", "contains_")):
                    continue
                if latest_obj_state and latest_obj_state != "contact_lost":
                    continue

                qualifier = "Possibly lost" if ev.validity == MemoryStatus.UNCERTAIN else "Lost"
                label = ev.semantic_label.lower()
                fact_str = f"{qualifier} contact with {label}."
                meta = {
                    "type": "recovery_event",
                    "event_id": ev.event_id,
                    "step": ev.step,
                    "confidence": ev.confidence,
                    "source": ev.evidence_source.value,
                    "validity": ev.validity.value,
                }
                facts.append((1, fact_str, ev.object_id, meta))
                handled_entities.add(ev.object_id)
                break

        # 2-3. Object states
        for item in memory.objects:
            if item.evidence_source not in self.allowed_sources:
                continue
            if item.validity not in (MemoryStatus.CONFIRMED, MemoryStatus.UNCERTAIN):
                continue
            if item.object_id in handled_entities:
                continue
            # Exclude unidentified/unknown objects from prompt injection
            if item.object_id in ("unidentified_object", "unknown_object", "target_object"):
                continue

            state = item.state.lower()
            label = item.semantic_label
            clean_label = label.capitalize()

            # Priority 2: Completed subtasks / goal reached
            is_completed = False
            for target_ind in ("in_", "on_plate", "placed", "goal_reached", "inside_"):
                if target_ind in state:
                    is_completed = True
                    break

            if is_completed:
                if "in_" in state:
                    container = state.split("in_")[-1].replace("_", " ")
                    if item.validity == MemoryStatus.UNCERTAIN:
                        fact_str = f"Possibly {label} is in the {container}."
                    else:
                        fact_str = f"{clean_label} is already in the {container}."
                elif state == "placed":
                    if item.validity == MemoryStatus.UNCERTAIN:
                        fact_str = f"Possibly {label} is placed."
                    else:
                        fact_str = f"{clean_label} is placed."
                elif "on_plate" in state:
                    if item.validity == MemoryStatus.UNCERTAIN:
                        fact_str = f"Possibly {label} is on the plate."
                    else:
                        fact_str = f"{clean_label} is already on the plate."
                else:
                    clean_st = state.replace("_", " ")
                    if item.validity == MemoryStatus.UNCERTAIN:
                        fact_str = f"Possibly {label} is {clean_st}."
                    else:
                        fact_str = f"{clean_label} is {clean_st}."

                meta = {
                    "type": "completed_state",
                    "object_id": item.object_id,
                    "step": item.last_updated_step,
                    "confidence": item.confidence,
                    "source": item.evidence_source.value,
                    "validity": item.validity.value,
                }
                facts.append((2, fact_str, item.object_id, meta))
                handled_entities.add(item.object_id)
                continue

            # Priority 3: Current manipulation state
            if any(term in state for term in ("grasped", "holding", "lifted")):
                if item.validity == MemoryStatus.UNCERTAIN:
                    fact_str = f"Possibly holding {label}."
                else:
                    fact_str = f"Holding {label}."
                meta = {
                    "type": "manipulation_state",
                    "object_id": item.object_id,
                    "step": item.last_updated_step,
                    "confidence": item.confidence,
                    "source": item.evidence_source.value,
                    "validity": item.validity.value,
                }
                facts.append((3, fact_str, item.object_id, meta))
                handled_entities.add(item.object_id)
                continue

            # NOTE: Priority 4 (resting states like on_table, detected, target_destination, empty)
            # is omitted to prevent appending non-informative static suffixes that degrade policy execution.

        facts.sort(key=lambda x: x[0])
        return facts

    def render_with_audit(self, instruction: str, memory: RetrievedMemory) -> RenderResult:
        """Render memory facts into instruction prompt and return full audit metadata."""
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        inst_tokens = self.count_tokens(instruction)

        # -------------------------------------------------------------
        # 1. Compact Mode (Targeted, non-redundant prompt injection)
        # -------------------------------------------------------------
        if self.compact_mode:
            candidate_facts = self._extract_compact_facts(memory)
            total_considered = len(candidate_facts)

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
                    memory_skipped_reason="no_candidate_facts",
                    facts_included=(),
                    tokenizer_truncated=False,
                )

            # If instruction itself reaches or exceeds budget, memory is completely skipped
            if self.max_total_tokens is not None and inst_tokens >= self.max_total_tokens:
                return RenderResult(
                    text=instruction,
                    truncated=True,
                    total_facts_considered=total_considered,
                    included_facts_count=0,
                    dropped_facts_count=total_considered,
                    dropped_facts=tuple(f[1] for f in candidate_facts),
                    rendered_context_chars=0,
                    budget_chars=self.max_context_chars,
                    rendered_context_tokens=0,
                    budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                    token_truncated=True,
                    total_tokens_with_instruction=inst_tokens,
                    memory_skipped_reason="instruction_overflow",
                    facts_included=(),
                    tokenizer_truncated=(inst_tokens > self.max_total_tokens),
                )

            base_inst = instruction.rstrip()
            if not base_inst.endswith((".", "!", "?")):
                base_inst += "."

            included_facts: List[str] = []
            dropped_facts: List[str] = []
            char_truncated = False
            token_truncated = False

            for priority, fact_text, ent_id, meta in candidate_facts:
                trial_facts = included_facts + [fact_text]
                trial_context = "Memory: " + " ".join(trial_facts)
                trial_prompt = f"{base_inst} {trial_context}"

                # Character budget check
                if len(trial_context) > self.max_context_chars:
                    char_truncated = True
                    dropped_facts.append(fact_text)
                    continue

                # Token budget check
                trial_tokens = self.count_tokens(trial_prompt)
                if self.max_total_tokens is not None and trial_tokens > self.max_total_tokens:
                    token_truncated = True
                    dropped_facts.append(fact_text)
                    continue

                included_facts.append(fact_text)

            truncated = char_truncated or token_truncated

            if not included_facts:
                return RenderResult(
                    text=instruction,
                    truncated=truncated,
                    total_facts_considered=total_considered,
                    included_facts_count=0,
                    dropped_facts_count=len(dropped_facts),
                    dropped_facts=tuple(dropped_facts),
                    rendered_context_chars=0,
                    budget_chars=self.max_context_chars,
                    rendered_context_tokens=0,
                    budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                    token_truncated=token_truncated,
                    total_tokens_with_instruction=inst_tokens,
                    memory_skipped_reason="token_budget",
                    facts_included=(),
                    tokenizer_truncated=False,
                )

            final_context = "Memory: " + " ".join(included_facts)
            final_prompt = f"{base_inst} {final_context}"
            final_total_tokens = self.count_tokens(final_prompt)
            rendered_context_tokens = max(0, final_total_tokens - inst_tokens)

            return RenderResult(
                text=final_prompt,
                truncated=truncated,
                total_facts_considered=total_considered,
                included_facts_count=len(included_facts),
                dropped_facts_count=len(dropped_facts),
                dropped_facts=tuple(dropped_facts),
                rendered_context_chars=len(final_context),
                budget_chars=self.max_context_chars,
                rendered_context_tokens=rendered_context_tokens,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=token_truncated,
                total_tokens_with_instruction=final_total_tokens,
                memory_skipped_reason=None,
                facts_included=tuple(included_facts),
                tokenizer_truncated=False,
            )

        # -------------------------------------------------------------
        # 2. Legacy / Non-compact Mode (Verbose bracketed block)
        # -------------------------------------------------------------
        candidate_facts_legacy: List[str] = []
        for item in memory.objects:
            if item.evidence_source not in self.allowed_sources:
                continue
            qualifier = "Possibly " if item.validity == MemoryStatus.UNCERTAIN else ""
            candidate_facts_legacy.append(
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
            candidate_facts_legacy.append(
                "{} [step {}, confidence {:.2f}, source={}].".format(
                    statement,
                    item.step,
                    item.confidence,
                    item.evidence_source.value,
                )
            )

        if not candidate_facts_legacy:
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
                memory_skipped_reason="no_candidate_facts",
                facts_included=(),
                tokenizer_truncated=False,
            )

        if self.max_total_tokens is not None and inst_tokens >= self.max_total_tokens:
            return RenderResult(
                text=instruction,
                truncated=True,
                total_facts_considered=len(candidate_facts_legacy),
                included_facts_count=0,
                dropped_facts_count=len(candidate_facts_legacy),
                dropped_facts=tuple(candidate_facts_legacy),
                rendered_context_chars=0,
                budget_chars=self.max_context_chars,
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=True,
                total_tokens_with_instruction=inst_tokens,
                memory_skipped_reason="instruction_overflow",
                facts_included=(),
                tokenizer_truncated=(inst_tokens > self.max_total_tokens),
            )

        lines = ["[Episode memory]"]
        included_facts_legacy: List[str] = []
        dropped_facts_legacy: List[str] = []
        char_truncated = False
        token_truncated = False

        for fact in candidate_facts_legacy:
            candidate_lines = lines + ["- " + fact]
            candidate_block = "\n".join(candidate_lines)
            candidate_full_text = instruction + "\n\n" + candidate_block
            candidate_total_tokens = self.count_tokens(candidate_full_text)

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
                dropped_facts_legacy.append(fact)
            else:
                lines.append("- " + fact)
                included_facts_legacy.append(fact)

        truncated = char_truncated or token_truncated

        if len(lines) == 1:
            return RenderResult(
                text=instruction,
                truncated=truncated,
                total_facts_considered=len(candidate_facts_legacy),
                included_facts_count=0,
                dropped_facts_count=len(dropped_facts_legacy),
                dropped_facts=tuple(dropped_facts_legacy),
                rendered_context_chars=0,
                budget_chars=self.max_context_chars,
                rendered_context_tokens=0,
                budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
                token_truncated=token_truncated,
                total_tokens_with_instruction=inst_tokens,
                memory_skipped_reason="token_budget",
                facts_included=(),
                tokenizer_truncated=False,
            )

        block = "\n".join(lines)
        rendered_text = instruction + "\n\n" + block
        rendered_chars = len(block)
        final_total_tokens = self.count_tokens(rendered_text)
        rendered_context_tokens = max(0, final_total_tokens - inst_tokens)

        return RenderResult(
            text=rendered_text,
            truncated=truncated,
            total_facts_considered=len(candidate_facts_legacy),
            included_facts_count=len(included_facts_legacy),
            dropped_facts_count=len(dropped_facts_legacy),
            dropped_facts=tuple(dropped_facts_legacy),
            rendered_context_chars=rendered_chars,
            budget_chars=self.max_context_chars,
            rendered_context_tokens=rendered_context_tokens,
            budget_tokens=self.max_total_tokens if self.max_total_tokens is not None else 0,
            token_truncated=token_truncated,
            total_tokens_with_instruction=final_total_tokens,
            memory_skipped_reason=None,
            facts_included=tuple(included_facts_legacy),
            tokenizer_truncated=False,
        )

    def render(self, instruction: str, memory: RetrievedMemory) -> str:
        """Render facts and return string prompt (backward-compatible API)."""
        return self.render_with_audit(instruction, memory).text

