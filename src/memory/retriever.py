"""Stable task-conditioned retrieval for episode-local memory."""

from typing import List, Set, Tuple

from src.memory.models import (
    MemoryEvent,
    MemoryQuery,
    MemoryStatus,
    ObjectMemory,
    RetrievedMemory,
    WorldMemorySnapshot,
)


class DeterministicMemoryRetriever:
    """Retrieve only explicit task entities and evidence available as of a step."""

    _HIDDEN_STATUSES = {MemoryStatus.STALE, MemoryStatus.INVALIDATED, MemoryStatus.UNKNOWN}

    def retrieve(
        self, snapshot: WorldMemorySnapshot, query: MemoryQuery
    ) -> RetrievedMemory:
        if snapshot.episode_id != query.episode_id:
            raise ValueError("query episode_id does not match memory snapshot")
        if query.as_of_step > snapshot.as_of_step:
            raise ValueError("query cannot look beyond the supplied snapshot")

        entity_ids: Set[str] = set(query.entity_ids)
        if not entity_ids:
            return RetrievedMemory(episode_id=query.episode_id, as_of_step=query.as_of_step)

        objects = [
            item
            for item in snapshot.objects
            if item.object_id in entity_ids
            and item.last_updated_step <= query.as_of_step
            and item.validity not in self._HIDDEN_STATUSES
            and item.confidence >= query.minimum_confidence
            and self._is_fresh(item.last_updated_step, query)
        ]
        objects.sort(
            key=lambda item: (
                -item.confidence,
                -item.last_updated_step,
                item.object_id,
            )
        )

        events = [
            item
            for item in snapshot.events
            if item.step <= query.as_of_step
            and (item.object_id in entity_ids or item.target_id in entity_ids)
            and item.validity not in self._HIDDEN_STATUSES
            and item.confidence >= query.minimum_confidence
            and self._is_fresh(item.step, query)
        ]
        events.sort(
            key=lambda item: (-item.confidence, -item.step, item.event_id)
        )

        return RetrievedMemory(
            episode_id=query.episode_id,
            as_of_step=query.as_of_step,
            objects=tuple(objects[: query.max_objects]),
            events=tuple(events[: query.max_events]),
        )

    @staticmethod
    def _is_fresh(step: int, query: MemoryQuery) -> bool:
        if query.max_age_steps is None:
            return True
        age = query.as_of_step - step
        return 0 <= age <= query.max_age_steps
