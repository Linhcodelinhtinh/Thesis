"""Decision facts from a causal relation snapshot. Rendering stays source-locked."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re

from src.memory.models import EvidenceSource


@dataclass(frozen=True)
class DecisionFact:
    key: str
    value: tuple
    validity: str
    event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    @property
    def version(self):
        # New evidence for the same conclusion refreshes provenance, not meaning.
        meaning = json.dumps([self.value, self.validity], separators=(",", ":"))
        return hashlib.sha256(meaning.encode("utf-8")).hexdigest()

    def trace(self):
        return dict(vars(self), version=self.version)


@dataclass(frozen=True)
class DecisionContext:
    query_id: int
    observation: int
    exposure: str
    facts: tuple[DecisionFact, ...]

    def trace(self):
        return dict(query_id=self.query_id, observation=self.observation,
                    exposure=self.exposure, facts=[fact.trace() for fact in self.facts])


def _event_supported(event, evidence, step):
    return bool(event is not None and event["source"] == EvidenceSource.PREDICTED_RGB_TRACKER.value
                and event["available_at"] <= step and event["evidence_ids"]
                and all(eid in evidence and evidence[eid]["step"] <= event["available_at"]
                        for eid in event["evidence_ids"]))


def relevant_instruction(instruction, profile):
    words = set(re.findall(r"[a-z]+", instruction.lower()))
    words |= {word.removesuffix("s") for word in words}
    return set(profile["subject"].split()) <= words and profile["reference"] in words


def projected_facts(renderer, snapshot, rendered):
    """Attach IDs to the actual included text without changing the renderer contract."""
    remaining = Counter(rendered.facts_included)
    facts = []
    for subject, state in sorted(snapshot.relations.items()):
        candidate = renderer._candidate(snapshot, subject, state)
        if candidate is None:
            continue
        event, phrase = candidate
        if phrase not in remaining:
            continue
        evidence = list(event.evidence_ids)
        events = [event.id]
        if state.departure is not None:
            evidence.extend(state.departure.evidence_ids)
            events.append(state.departure.id)
        if not snapshot.profile.count_cycles:
            reference = snapshot.objects[event.details["reference"]]
            evidence.extend(reference.metadata["location_evidence"])
            if state.reference == event.details["reference"]:
                evidence.extend(state.current_evidence_ids)
            origin = snapshot.objects[subject].metadata.get("origin_evidence")
            if origin is not None:
                evidence.append(origin)
        facts.append(DecisionFact(subject, (phrase,), "projected", tuple(events),
                                  tuple(dict.fromkeys(evidence))))
    if Counter(fact.value[0] for fact in facts) != remaining:
        raise ValueError("Cannot uniquely link projected relation text to its facts")
    return tuple(facts)


def snapshot_facts(snapshot):
    """Extract from the same serialized schema used in the offline memory journal."""
    step = snapshot["as_of_observation"]
    evidence = snapshot["evidence"]
    objects = snapshot["objects"]
    facts = []
    for subject, state in sorted(snapshot["relations"].items()):
        obj = objects.get(subject, {})
        identity = obj.get("identity_status", "unknown")
        metadata = obj.get("metadata") or {}
        identity_evidence = tuple(eid for eid in (
            metadata.get("origin_evidence"), *metadata.get("location_evidence", ()))
            if eid in evidence and evidence[eid]["step"] <= step)
        conflict = state["status"] == "conflicted"
        if snapshot["profile"]["count_cycles"]:
            event = state["last_return"] or state["last_event"]
            departure = state["departure"]
            supported = (_event_supported(event, evidence, step)
                         and event["details"]["subject"] == subject
                         and event["details"]["observed_returns"] == state["returns"])
            armed = bool(departure and _event_supported(departure, evidence, step))
            # Retain the number of observed returns as a qualified claim only.
            # The reducer cannot establish an exact count after a coverage gap.
            validity = ("conflicted" if conflict else "unknown" if not supported else
                        "supported" if state["history_complete"] else "incomplete")
            dependencies = [item for item in (event, departure) if _event_supported(item, evidence, step)]
            facts.append(DecisionFact(
                subject, (state["returns"] if supported and not conflict else None, armed, identity),
                validity, tuple(item["id"] for item in dependencies),
                tuple(dict.fromkeys((*identity_evidence,
                                     *(eid for item in dependencies for eid in item["evidence_ids"]))))))
        else:
            event = state["last_event"]
            reference = state["reference"]
            current = tuple(state["current_evidence_ids"])
            supported = (reference is not None and bool(current) and identity == "tracked"
                         and objects.get(reference, {}).get("identity_status") == "tracked"
                         and all(eid in evidence and evidence[eid]["step"] <= step for eid in current))
            historical = (not conflict and state["historical_relation_valid"]
                          and _event_supported(event, evidence, step))
            validity = ("conflicted" if conflict else "supported" if supported else
                        "historical" if historical else "unknown")
            facts.append(DecisionFact(
                subject, (reference if supported else None,
                          event["details"]["reference"] if historical else None, identity),
                validity, (event["id"],) if historical else (),
                tuple(dict.fromkeys(identity_evidence + (current if supported else ()) +
                                    (tuple(event["evidence_ids"]) if historical else ())))))
    return tuple(facts)
