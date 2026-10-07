"""Evidence-driven event ingestion and state projection."""

from src.memory.models import MemoryEvent, MemoryStatus, ObjectMemory
from src.memory.store import EpisodeMemoryStore


class MemoryUpdater:
    """Apply explicitly supplied evidence without inferring hidden facts."""

    def __init__(self, store: EpisodeMemoryStore) -> None:
        self.store = store

    def record(self, event: MemoryEvent) -> None:
        """Append an event and update its object's state when evidence permits.

        Events without ``object_state_after`` remain in history but do not
        mutate the current state. INVALIDATED and UNKNOWN events update
        validity while retaining the last explicit state as an uncertain
        historical value; retrieval filters these statuses by default.
        """
        self.store.append_event(event)
        previous = self.store.get_object(event.object_id)

        if event.object_state_after is None:
            if previous is not None and event.validity in (
                MemoryStatus.INVALIDATED,
                MemoryStatus.STALE,
                MemoryStatus.UNKNOWN,
            ):
                self.store.upsert_object(
                    ObjectMemory(
                        episode_id=event.episode_id,
                        object_id=previous.object_id,
                        semantic_label=previous.semantic_label,
                        state=previous.state,
                        first_seen_step=previous.first_seen_step,
                        last_updated_step=event.step,
                        last_confirmed_step=previous.last_confirmed_step,
                        confidence=min(previous.confidence, event.confidence),
                        evidence_source=event.evidence_source,
                        validity=event.validity,
                        last_event_id=event.event_id,
                    )
                )
            return

        if event.validity in (MemoryStatus.INVALIDATED, MemoryStatus.STALE):
            # A contradictory/stale event cannot establish a new object state.
            if previous is not None:
                self.store.upsert_object(
                    ObjectMemory(
                        episode_id=event.episode_id,
                        object_id=previous.object_id,
                        semantic_label=previous.semantic_label,
                        state=previous.state,
                        first_seen_step=previous.first_seen_step,
                        last_updated_step=event.step,
                        last_confirmed_step=previous.last_confirmed_step,
                        confidence=min(previous.confidence, event.confidence),
                        evidence_source=event.evidence_source,
                        validity=event.validity,
                        last_event_id=event.event_id,
                    )
                )
            return

        first_seen_step = previous.first_seen_step if previous else event.step
        last_confirmed_step = (
            event.step
            if event.validity == MemoryStatus.CONFIRMED
            else (previous.last_confirmed_step if previous else None)
        )
        self.store.upsert_object(
            ObjectMemory(
                episode_id=event.episode_id,
                object_id=event.object_id,
                semantic_label=event.semantic_label,
                state=event.object_state_after,
                first_seen_step=first_seen_step,
                last_updated_step=event.step,
                last_confirmed_step=last_confirmed_step,
                confidence=event.confidence,
                evidence_source=event.evidence_source,
                validity=event.validity,
                last_event_id=event.event_id,
            )
        )
