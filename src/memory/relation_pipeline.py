"""Shared offline/online construction of the relation writer and projection."""
from pathlib import Path

from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory.libero_mem_profiles import profile_for_task
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.utils.file_hash import sha256_file


def relation_config(strategy="relevant", writer_version=RelationMemoryWriter.VERSION):
    if strategy not in ("relevant", "recent"):
        raise ValueError("Unknown relation projection")
    projection = (RelationMemoryRendererV3.VERSION if writer_version == BowlCycleWriterV3.VERSION
                  else RelationMemoryRenderer.VERSION)
    config = dict(projection=projection, strategy=strategy, max_total_tokens=48)
    if writer_version == BowlCycleWriterV3.VERSION:
        config["cycle"] = BowlCycleConfig.load().to_dict()
    return config


def relation_pipeline(suite, task, detector, convention, tokenizer, strategy="relevant"):
    profile = profile_for_task(suite, task)

    def factory(episode_id, **_):
        return RelationMemoryWriter(episode_id, profile, detector, convention)

    return factory, RelationMemoryRenderer(tokenizer, strategy=strategy)


def relation_source_hashes(root):
    root = Path(root)
    files = (
        "src/memory/relation_pipeline.py", "src/memory/relation_memory.py",
        "src/memory/libero_mem_profiles.py", "src/memory/models.py",
        "src/perception/relation_tracking.py", "src/perception/relation_detector.py",
        "src/perception/grounding_dino.py", "src/perception/visual_evidence.py",
        "src/perception/process_detector.py", "scripts/memory_detector_worker.py",
        "src/interfaces/relation_memory_text.py", "src/interfaces/libero_mem_text.py",
        "src/interfaces/text_memory.py", "configs/benchmarks/libero_mem_suite_source_v1.json",
    )
    return {name: sha256_file(root / name) for name in files}
