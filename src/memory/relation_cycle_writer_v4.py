"""Bowl-cycle writer V4: bridge detector losses of the stationary plate with its observed anchor.

V3 needs a plate box on every confirming frame. While the bowl is lifted or being set down the
detector often loses the plate or reports several overlapping plates, so a return that the author
counts is never confirmed. The plate does not move in the T1/T3 profiles (the tracker already
keys it to an anchor measured before the first lift), so once the writer holds a verified origin
the anchor box stands in for a missing or ambiguous plate measurement.

Everything else is V3: thresholds, confirmation, motion checks, prompt projection and snapshot
schema. The stand-in is recorded in ``last_signals["plate_prior"]`` and in ``stats``; the event
evidence ids point at the anchor's real detection. This module is new and registers its own
version, so the hash-pinned V1 to V3 sources stay byte-identical.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from src.memory.models import IdentityStatus
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.utils.file_hash import sha256_file

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/memory/relation_plate_prior_v1.json"


@dataclass(frozen=True)
class PlatePriorConfig:
    # Use the anchor when the plate has no box this frame.
    when_missing: bool = True
    # Use the anchor when several plates compete and none can be chosen.
    when_ambiguous: bool = True
    # Longest run of consecutive stand-in frames; None means unlimited.
    max_consecutive_frames: int | None = None

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in ("when_missing", "when_ambiguous")):
            raise ValueError("Plate prior switches must be booleans")
        limit = self.max_consecutive_frames
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("max_consecutive_frames must be a positive integer or null")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        config = cls(**values)
        if config.to_dict() != values:
            raise ValueError("Plate prior config does not match its validated schema")
        return config

    @staticmethod
    def file_sha256(path=CONFIG_PATH):
        return sha256_file(path)


class PlateAnchorWriterV4(BowlCycleWriterV3):
    """V3 plus the stationary-plate stand-in. Same prompt contract as V3 (``relation_text_v2``)."""

    VERSION = "rgb_relations_v4"
    CORE_VERSION = BowlCycleWriterV3.VERSION

    def __init__(self, episode_id, profile, detector, image_convention, config=None, prior=None):
        super().__init__(episode_id, profile, detector, image_convention, config=config)
        self.prior = prior or PlatePriorConfig.load()
        self.stats = dict(frames=0, prior_frames=0, prior_missing=0, prior_ambiguous=0, longest_run=0)
        self._run = 0
        self._used_prior = False

    def step(self, step, observation, last_action=None):
        duplicate = step == self._step
        if not duplicate:
            self.stats["frames"] += 1
            self._used_prior = False
        snapshot = super().step(step, observation, last_action)
        if not duplicate and self.last_signals is not None:
            self.last_signals = dict(self.last_signals, plate_prior=self._used_prior)
        return snapshot

    def _measure(self, subject, references, reducer, released):
        plate = references["plate_1"]
        ambiguous = plate.identity.identity_status == IdentityStatus.AMBIGUOUS
        missing = plate.box is None and not ambiguous
        usable = ((missing and self.prior.when_missing) or (ambiguous and self.prior.when_ambiguous))
        limit = self.prior.max_consecutive_frames
        eligible = usable and plate.anchor is not None and reducer.origin is not None
        if eligible and limit is not None and self._run >= limit:
            return super()._measure(subject, references, reducer, released)  # budget spent: stay on V3
        if eligible:
            anchor = plate.anchor
            references = dict(references, plate_1=replace(
                plate, box=anchor, previous=anchor,
                identity=replace(plate.identity, identity_status=IdentityStatus.TRACKED)))
            self._run += 1
            self._used_prior = True
            self.stats["prior_frames"] += 1
            self.stats["prior_ambiguous" if ambiguous else "prior_missing"] += 1
            self.stats["longest_run"] = max(self.stats["longest_run"], self._run)
        elif not usable:
            self._run = 0
        return super()._measure(subject, references, reducer, released)


V4_WRITERS = {PlateAnchorWriterV4.VERSION: PlateAnchorWriterV4}
V4_SOURCE_FILES = ("src/memory/relation_cycle_writer_v4.py", "configs/memory/relation_plate_prior_v1.json")


def make_v4_pipeline(suite, task, detector, convention, tokenizer, strategy, prior=None):
    """Return (writer_factory, renderer). The renderer is V3's own: the prompt text is unchanged."""
    from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
    from src.memory.libero_mem_profiles import profile_for_task
    from src.memory.relation_cycle_config import BowlCycleConfig

    profile = profile_for_task(suite, task)
    if (profile.subject, profile.subject_count, profile.reference, profile.predicate,
            profile.count_cycles) != ("bowl", 1, "plate", "on", True):
        raise ValueError(f"{PlateAnchorWriterV4.VERSION} supports only single-bowl return profiles")
    config, prior = BowlCycleConfig.load(), prior or PlatePriorConfig.load()

    def factory(episode_id, **_):
        return PlateAnchorWriterV4(episode_id, profile, detector, convention, config=config, prior=prior)

    return factory, RelationMemoryRendererV3(tokenizer, strategy=strategy)


def v4_source_hashes(root):
    from src.memory.relation_writer_versions import versioned_relation_source_hashes

    hashes = versioned_relation_source_hashes(root, BowlCycleWriterV3.VERSION)
    hashes.update({name: sha256_file(Path(root) / name) for name in V4_SOURCE_FILES})
    return hashes
