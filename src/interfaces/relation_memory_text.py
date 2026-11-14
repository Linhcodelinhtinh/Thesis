"""Project whole relation facts with their identity and uncertainty qualifiers."""
from dataclasses import dataclass
import re

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.interfaces.text_memory import RenderResult
from src.memory.models import EvidenceSource, IdentityStatus


class RelationMemoryRenderer(LiberoMemTextRenderer):
    """Use the pilot tokenizer convention with a separate projection version."""
    VERSION = "relation_text_v1"

    def __init__(self, tokenizer, max_total_tokens=48, strategy="relevant", token_counter=None):
        if tokenizer is None or strategy not in ("relevant", "recent"):
            raise ValueError("An audited tokenizer and a supported projection are required")
        super().__init__(max_total_tokens, tokenizer, strict_tokenizer=True, allow_oracle=False)
        self.strategy = strategy
        self.token_counter = token_counter

    def count_tokens(self, text):
        if self.token_counter is not None:
            return self.token_counter(text)
        return super().count_tokens(text)

    @staticmethod
    def _current_name(snapshot, key):
        obj = snapshot.objects.get(key)
        if obj is None or obj.identity_status != IdentityStatus.TRACKED:
            return None
        evidence = obj.metadata.get("location_evidence", ())
        if not evidence or any(eid not in snapshot.evidence or
                               snapshot.evidence[eid]["step"] != snapshot.as_of_observation for eid in evidence):
            return None
        return obj.current_location_desc

    def _candidate(self, snapshot, key, state):
        event = (state.last_return or state.last_event) if snapshot.profile.count_cycles else state.last_event
        if event is None:
            return None
        dependencies = (event, state.departure) if state.departure else (event,)
        for dependency in dependencies:
            if (dependency.source != EvidenceSource.PREDICTED_RGB_TRACKER
                    or dependency.available_at > snapshot.as_of_observation
                    or dependency.details["subject"] != key or not dependency.evidence_ids
                    or any(eid not in snapshot.evidence or snapshot.evidence[eid]["step"] > dependency.available_at
                           for eid in dependency.evidence_ids)):
                return event, None
        profile = snapshot.profile
        if profile.count_cycles:
            if event.details["observed_returns"] != state.returns:
                return event, None
            qualifier = " History incomplete." if not state.history_complete else ""
            progress = " Lift observed." if state.armed else ""
            return event, f"{profile.subject}: {state.returns} observed returns.{qualifier}{progress}"
        reference = event.details["reference"]
        ref_name = self._current_name(snapshot, reference)
        if not ref_name:
            return event, None
        current = state.reference == reference and bool(state.current_evidence_ids)
        if current and any(eid not in snapshot.evidence or
                           snapshot.evidence[eid]["step"] > snapshot.as_of_observation
                           for eid in state.current_evidence_ids):
            return event, None
        if profile.predicate == "in":
            # The tracked container's current description locates the earlier
            # observation. A missing object never proves current occupancy.
            if not state.historical_relation_valid:
                return event, None
            return event, (f"{profile.subject} visually in {ref_name}." if current else
                           f"{profile.subject} last seen in {ref_name}, now uncertain.")
        obj = snapshot.objects[key]
        origin_evidence = obj.metadata.get("origin_evidence")
        if (not current or obj.identity_status != IdentityStatus.TRACKED
                or origin_evidence not in snapshot.evidence):
            return event, None
        rank = key.rsplit("_", 1)[1]
        return event, f"Original {profile.subject} {rank} from left on {ref_name}."

    def render(self, instruction, memory, enabled=True):
        instruction_tokens = self.count_tokens(instruction)
        if instruction_tokens > self.max_total_tokens:
            return RelationRenderResult(instruction, True, 0, 0, 0, (), 0, len(instruction),
                                budget_tokens=self.max_total_tokens, token_truncated=True,
                                total_tokens_with_instruction=instruction_tokens, tokenizer_truncated=True,
                                memory_skipped_reason="instruction_over_budget")
        candidates = []
        if enabled and memory is not None:
            words = set(re.findall(r"[a-z]+", instruction.lower()))
            labels = set(memory.profile.subject.split()) | {memory.profile.reference}
            if labels & {word.removesuffix("s") if word not in labels else word for word in words}:
                for key, state in memory.relations.items():
                    candidate = self._candidate(memory, key, state)
                    if candidate:
                        candidates.append((key, *candidate))
        if self.strategy == "recent":
            # Same admissible facts, ordered by availability without relevance weighting.
            candidates.sort(key=lambda row: (-row[1].available_at, row[0]))
        else:
            def priority(row):
                _, event, _ = row
                return (event.type != "return", event.type == "initial_relation",
                        -event.available_at, row[0])

            candidates.sort(key=priority)
        included, dropped, failures = [], [], []
        for key, event, fact in candidates:
            if fact is None:
                failures.append(event.id)
                continue
            if self.count_tokens(instruction + " Memory: " + " ".join(included + [fact])) <= self.max_total_tokens:
                included.append(fact)
            else:
                dropped.append(fact)
        suffix = " Memory: " + " ".join(included) if included else ""
        reason = ("budget_exceeded" if dropped else "dependency_unavailable" if failures else
                  "no_relevant_fact" if enabled else "memory_disabled")
        rendered = RenderResult(instruction + suffix, bool(dropped), len(candidates), len(included),
                                len(dropped), tuple(dropped), len(suffix), len(instruction + suffix),
                                self.count_tokens(suffix) if suffix else 0, self.max_total_tokens, bool(dropped),
                                self.count_tokens(instruction + suffix), None if included else reason, tuple(included))
        return RelationRenderResult(**vars(rendered), dependency_failures=tuple(failures))


@dataclass(frozen=True)
class RelationRenderResult(RenderResult):
    dependency_failures: tuple = ()
