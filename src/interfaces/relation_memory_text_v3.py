"""Prompt projection that withholds cycle counts without a verified origin."""

from src.interfaces.relation_memory_text import RelationMemoryRenderer


class RelationMemoryRendererV3(RelationMemoryRenderer):
    VERSION = "relation_text_v2"

    def _candidate(self, snapshot, key, state):
        if snapshot.profile.count_cycles and not state.origin_verified:
            event = state.last_return or state.last_event
            return (event, None) if event is not None else None
        return super()._candidate(snapshot, key, state)
