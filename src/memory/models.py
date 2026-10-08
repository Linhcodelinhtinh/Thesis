"""Versioned, provenance-aware schemas for episode-local memory."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, Optional, Tuple


class EvidenceSource(str, Enum):
    """Where a memory fact came from; oracle evidence is always explicit."""

    OBSERVATION_TRACKER = "observation_tracker"
    POLICY_ACTION = "policy_action"
    ORACLE_SIMULATOR = "oracle_simulator"
    HUMAN_ANNOTATION = "human_annotation"


class MemoryStatus(str, Enum):
    """Validity of a state or event at the time it is retrieved."""

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    INVALIDATED = "invalidated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryEvent:
    """An immutable, evidence-backed event observed during one episode."""

    event_id: str
    episode_id: str
    step: int
    timestamp: float
    event_type: str
    object_id: str
    semantic_label: str
    evidence_source: EvidenceSource
    confidence: float
    object_state_before: Optional[str] = None
    object_state_after: Optional[str] = None
    target_id: Optional[str] = None
    validity: MemoryStatus = MemoryStatus.CONFIRMED
    evidence_ids: Tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text("event_id", self.event_id)
        _require_text("episode_id", self.episode_id)
        _require_text("event_type", self.event_type)
        _require_text("object_id", self.object_id)
        _require_text("semantic_label", self.semantic_label)
        _require_step(self.step)
        _require_timestamp(self.timestamp)
        _require_confidence(self.confidence)
        if not isinstance(self.evidence_source, EvidenceSource):
            raise TypeError("evidence_source must be an EvidenceSource")
        if not isinstance(self.validity, MemoryStatus):
            raise TypeError("validity must be a MemoryStatus")
        if self.schema_version != 1:
            raise ValueError("unsupported MemoryEvent schema_version")
        if self.target_id is not None:
            _require_text("target_id", self.target_id)
        for evidence_id in self.evidence_ids:
            _require_text("evidence_id", evidence_id)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation with stable field names."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "object_id": self.object_id,
            "semantic_label": self.semantic_label,
            "evidence_source": self.evidence_source.value,
            "confidence": self.confidence,
            "object_state_before": self.object_state_before,
            "object_state_after": self.object_state_after,
            "target_id": self.target_id,
            "validity": self.validity.value,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ObjectMemory:
    """Latest evidence-backed textual state for one episode-local entity."""

    episode_id: str
    object_id: str
    semantic_label: str
    state: str
    first_seen_step: int
    last_updated_step: int
    confidence: float
    evidence_source: EvidenceSource
    validity: MemoryStatus = MemoryStatus.CONFIRMED
    last_confirmed_step: Optional[int] = None
    last_event_id: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text("episode_id", self.episode_id)
        _require_text("object_id", self.object_id)
        _require_text("semantic_label", self.semantic_label)
        _require_text("state", self.state)
        _require_step(self.first_seen_step)
        _require_step(self.last_updated_step)
        if self.first_seen_step > self.last_updated_step:
            raise ValueError("first_seen_step cannot exceed last_updated_step")
        if self.last_confirmed_step is not None:
            _require_step(self.last_confirmed_step)
            if self.last_confirmed_step > self.last_updated_step:
                raise ValueError("last_confirmed_step cannot exceed last_updated_step")
        _require_confidence(self.confidence)
        if not isinstance(self.evidence_source, EvidenceSource):
            raise TypeError("evidence_source must be an EvidenceSource")
        if not isinstance(self.validity, MemoryStatus):
            raise TypeError("validity must be a MemoryStatus")
        if self.last_event_id is not None:
            _require_text("last_event_id", self.last_event_id)
        if self.schema_version != 1:
            raise ValueError("unsupported ObjectMemory schema_version")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "object_id": self.object_id,
            "semantic_label": self.semantic_label,
            "state": self.state,
            "first_seen_step": self.first_seen_step,
            "last_updated_step": self.last_updated_step,
            "last_confirmed_step": self.last_confirmed_step,
            "confidence": self.confidence,
            "evidence_source": self.evidence_source.value,
            "validity": self.validity.value,
            "last_event_id": self.last_event_id,
        }


@dataclass(frozen=True)
class MemoryQuery:
    """A deterministic retrieval request using explicit task entity IDs."""

    episode_id: str
    as_of_step: int
    entity_ids: Tuple[str, ...]
    minimum_confidence: float = 0.5
    max_age_steps: Optional[int] = 200
    max_events: int = 5
    max_objects: int = 5

    def __post_init__(self) -> None:
        _require_text("episode_id", self.episode_id)
        _require_step(self.as_of_step)
        _require_confidence(self.minimum_confidence)
        if self.max_age_steps is not None:
            _require_step(self.max_age_steps)
        if self.max_events < 0 or self.max_objects < 0:
            raise ValueError("retrieval limits must be non-negative")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must not contain duplicates")
        for entity_id in self.entity_ids:
            _require_text("entity_id", entity_id)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "as_of_step": self.as_of_step,
            "entity_ids": list(self.entity_ids),
            "minimum_confidence": self.minimum_confidence,
            "max_age_steps": self.max_age_steps,
            "max_events": self.max_events,
            "max_objects": self.max_objects,
        }


@dataclass(frozen=True)
class RetrievedMemory:
    """Auditable retrieval result; source data remains available to the renderer."""

    episode_id: str
    as_of_step: int
    objects: Tuple[ObjectMemory, ...] = ()
    events: Tuple[MemoryEvent, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "as_of_step": self.as_of_step,
            "objects": [item.to_dict() for item in self.objects],
            "events": [item.to_dict() for item in self.events],
        }


@dataclass(frozen=True)
class WorldMemorySnapshot:
    """Immutable, chronologically ordered snapshot for a single episode."""

    episode_id: str
    as_of_step: int
    objects: Tuple[ObjectMemory, ...]
    events: Tuple[MemoryEvent, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "as_of_step": self.as_of_step,
            "objects": [item.to_dict() for item in self.objects],
            "events": [item.to_dict() for item in self.events],
        }


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))


def _require_step(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("step values must be non-negative integers")


def _require_timestamp(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("timestamp must be finite and non-negative")


def _require_confidence(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
