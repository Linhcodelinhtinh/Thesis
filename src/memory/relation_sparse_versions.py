"""Registry for the sparse-detector writers, kept apart from the dense registry.

``relation_writer_versions.py`` and ``relation_pipeline.py`` are part of the
hash-pinned provenance of earlier runs, so the sparse variants are registered
here instead. The prompt projection of a sparse writer is the projection of the
dense writer it wraps; nothing in this module can render text.
"""
from pathlib import Path

from src.memory.libero_mem_profiles import profile_for_task
from src.memory.relation_pipeline import relation_config
from src.memory.relation_sparse_config import SparseWriterConfig
from src.memory.relation_sparse_writer import SparseBowlCycleWriterV3, SparseRelationWriterV1
from src.memory.relation_writer_versions import (RELATION_WRITERS, make_relation_pipeline,
                                                 versioned_relation_source_hashes)
from src.utils.file_hash import sha256_file

SPARSE_WRITERS = {writer.VERSION: writer for writer in (SparseRelationWriterV1, SparseBowlCycleWriterV3)}
ALL_RELATION_WRITERS = {**RELATION_WRITERS, **SPARSE_WRITERS}
SPARSE_SOURCE_FILES = ("src/memory/relation_sparse_writer.py", "src/memory/relation_sparse_config.py",
                       "src/memory/relation_sparse_versions.py", "configs/memory/relation_sparse_v1.json")


def core_version(version):
    """The dense writer version whose prompt contract a writer version follows."""
    return SPARSE_WRITERS[version].CORE_VERSION if version in SPARSE_WRITERS else version


def relation_config_for(strategy, version, sparse=None):
    config = relation_config(strategy, core_version(version))
    if version in SPARSE_WRITERS:
        config = dict(config, core_writer=core_version(version),
                      sparse=(sparse or SparseWriterConfig.load()).to_dict())
    return config


def source_hashes_for(root, version):
    if version not in ALL_RELATION_WRITERS:
        raise ValueError(f"Unknown relation writer: {version}")
    hashes = versioned_relation_source_hashes(root, core_version(version))
    if version in SPARSE_WRITERS:
        hashes.update({name: sha256_file(Path(root) / name) for name in SPARSE_SOURCE_FILES})
    return hashes


def make_pipeline_for(suite, task, detector, convention, tokenizer, strategy, version, sparse=None):
    """Return (writer_factory, renderer). The renderer is the wrapped writer's own."""
    if version not in SPARSE_WRITERS:
        return make_relation_pipeline(suite, task, detector, convention, tokenizer, strategy, version)
    _, renderer = make_relation_pipeline(suite, task, detector, convention, tokenizer, strategy,
                                         core_version(version))
    writer_type, profile = SPARSE_WRITERS[version], profile_for_task(suite, task)
    config = sparse or SparseWriterConfig.load()

    def factory(episode_id, **_):
        return writer_type(episode_id, profile, detector, convention, sparse=config)

    return factory, renderer
