"""Select the relation recognizer while sharing the existing prompt renderer."""
from pathlib import Path

from src.memory.relation_cycle_writer import BowlCycleWriter
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_cycle_config import BowlCycleConfig, CONFIG_PATH
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_pipeline import relation_config, relation_pipeline, relation_source_hashes
from src.utils.file_hash import sha256_file


RELATION_WRITERS = {writer.VERSION: writer for writer in
                    (RelationMemoryWriter, BowlCycleWriter, BowlCycleWriterV3)}


def make_relation_pipeline(suite, task, detector, convention, tokenizer, strategy, version):
    if version == RelationMemoryWriter.VERSION:
        return relation_pipeline(suite, task, detector, convention, tokenizer, strategy)
    from src.interfaces.relation_memory_text import RelationMemoryRenderer
    from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
    from src.memory.libero_mem_profiles import profile_for_task

    writer_type = RELATION_WRITERS[version]
    profile = profile_for_task(suite, task)
    if version == BowlCycleWriterV3.VERSION and (profile.subject, profile.subject_count,
            profile.reference, profile.predicate, profile.count_cycles) != ("bowl", 1, "plate", "on", True):
        raise ValueError(f"{version} supports only single-bowl return profiles")
    config = BowlCycleConfig.load() if version == BowlCycleWriterV3.VERSION else None

    def factory(episode_id, **_):
        return writer_type(episode_id, profile, detector, convention, config=config)

    renderer_type = (RelationMemoryRendererV3 if version == BowlCycleWriterV3.VERSION
                     else RelationMemoryRenderer)
    return factory, renderer_type(tokenizer, strategy=strategy)


def versioned_relation_source_hashes(root, version):
    if version not in RELATION_WRITERS:
        raise ValueError(f"Unknown relation writer: {version}")
    hashes = relation_source_hashes(root)
    if version in (BowlCycleWriter.VERSION, BowlCycleWriterV3.VERSION):
        names = ["src/memory/relation_cycle_writer.py", "src/memory/relation_writer_versions.py",
                 "src/perception/relation_cycle_tracking.py", "src/memory/relation_cycle_config.py"]
        if version == BowlCycleWriterV3.VERSION:
            names.extend(("src/memory/relation_cycle_writer_v3.py",
                          "src/interfaces/relation_memory_text_v3.py",
                          "configs/memory/relation_cycle_v3.json"))
        for name in names:
            hashes[name] = sha256_file(Path(root) / name)
    return hashes
