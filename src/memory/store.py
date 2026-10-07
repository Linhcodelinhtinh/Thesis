"""Deterministic in-memory store scoped to one evaluation episode."""

from typing import Dict, List, Optional

from src.memory.models import MemoryEvent, ObjectMemory, WorldMemorySnapshot


class EpisodeMemoryStore:
    """Keep an append-only event log and latest state for one episode.

    This store is deliberately not a cross-episode database. Construct a new
    store or call ``reset`` on every environment reset to prevent information
    leaking between benchmark episodes.
    """

    def __init__(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        self.episode_id = episode_id
        self._events: List[MemoryEvent] = []
        self._event_ids = set()
        self._objects: Dict[str, ObjectMemory] = {}

    def reset(self, episode_id: str) -> None:
        """Clear all state and bind the store to a new episode."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        self.episode_id = episode_id
        self._events.clear()
        self._event_ids.clear()
        self._objects.clear()

    def append_event(self, event: MemoryEvent) -> None:
        if event.episode_id != self.episode_id:
            raise ValueError("event episode_id does not match this store")
        if event.event_id in self._event_ids:
            raise ValueError("duplicate memory event_id: {}".format(event.event_id))
        self._event_ids.add(event.event_id)
        self._events.append(event)
        self._events.sort(key=lambda item: (item.step, item.timestamp, item.event_id))

    def upsert_object(self, memory: ObjectMemory) -> None:
        if memory.episode_id != self.episode_id:
            raise ValueError("object episode_id does not match this store")
        previous = self._objects.get(memory.object_id)
        if previous is not None and memory.last_updated_step < previous.last_updated_step:
            raise ValueError("object memory update would move backwards in time")
        self._objects[memory.object_id] = memory

    def get_object(self, object_id: str) -> Optional[ObjectMemory]:
        return self._objects.get(object_id)

    def snapshot(self, as_of_step: int) -> WorldMemorySnapshot:
        if isinstance(as_of_step, bool) or not isinstance(as_of_step, int) or as_of_step < 0:
            raise ValueError("as_of_step must be a non-negative integer")
        objects = tuple(
            self._objects[key]
            for key in sorted(self._objects)
            if self._objects[key].last_updated_step <= as_of_step
        )
        events = tuple(event for event in self._events if event.step <= as_of_step)
        return WorldMemorySnapshot(
            episode_id=self.episode_id,
            as_of_step=as_of_step,
            objects=objects,
            events=events,
        )

    def to_dict(self, as_of_step: int) -> dict:
        """Return a stable JSON-compatible snapshot for run artifacts."""
        snapshot = self.snapshot(as_of_step)
        return {
            "schema_version": 1,
            "episode_id": snapshot.episode_id,
            "as_of_step": snapshot.as_of_step,
            "objects": [item.to_dict() for item in snapshot.objects],
            "events": [item.to_dict() for item in snapshot.events],
        }
