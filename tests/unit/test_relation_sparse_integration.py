"""Registry, runtime-contract and backbone wiring of the sparse writers."""
from types import SimpleNamespace

import pytest

from scripts.evaluate_mem_adapter import parse_args
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory import libero_mem_profiles, relation_pipeline, relation_sparse_versions as registry
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_pipeline import relation_config, relation_source_hashes
from src.memory.relation_sparse_config import SparseWriterConfig
from src.memory.relation_sparse_versions import (ALL_RELATION_WRITERS, SPARSE_SOURCE_FILES, SPARSE_WRITERS,
                                                 core_version, make_pipeline_for, relation_config_for,
                                                 source_hashes_for)
from src.memory.relation_sparse_writer import SparseBowlCycleWriterV3, SparseRelationWriterV1
from src.memory.relation_writer_versions import RELATION_WRITERS, versioned_relation_source_hashes
from src.models.xvla_mem import XVLAMemPolicy
from src.training import mem_adapter_artifact as artifact
from tests.unit.test_bowl_cycle_writer import PROFILE

ROOT = artifact.ROOT
V1, V3 = "rgb_relations_sparse_v1", "rgb_relations_sparse_v3"


# -- registry -----------------------------------------------------------------------

def test_sparse_versions_extend_the_dense_registry_without_changing_it():
    assert set(SPARSE_WRITERS) == {V1, V3}
    assert set(RELATION_WRITERS) == {"rgb_relations_v1", "rgb_relations_v2", "rgb_relations_v3"}
    assert ALL_RELATION_WRITERS == {**RELATION_WRITERS, **SPARSE_WRITERS}
    assert core_version(V1) == "rgb_relations_v1" and core_version(V3) == "rgb_relations_v3"
    assert core_version("rgb_relations_v2") == "rgb_relations_v2"


def test_config_of_a_dense_version_is_unchanged_and_a_sparse_one_adds_its_schedule():
    for version in RELATION_WRITERS:
        assert relation_config_for("relevant", version) == relation_config("relevant", version)
    config = relation_config_for("relevant", V3)
    assert config["core_writer"] == "rgb_relations_v3"
    assert config["sparse"] == SparseWriterConfig.load().to_dict()
    assert {key: config[key] for key in relation_config("relevant", "rgb_relations_v3")} == \
        relation_config("relevant", "rgb_relations_v3")
    # The prompt projection is the dense one: same renderer version, same token budget.
    assert config["projection"] == RelationMemoryRendererV3.VERSION and config["max_total_tokens"] == 48
    assert relation_config_for("relevant", V1)["projection"] == RelationMemoryRenderer.VERSION


def test_source_hashes_keep_the_dense_provenance_and_add_the_sparse_files():
    dense = versioned_relation_source_hashes(ROOT, "rgb_relations_v1")
    assert source_hashes_for(ROOT, "rgb_relations_v1") == dense
    sparse = source_hashes_for(ROOT, V1)
    assert {name: sparse[name] for name in dense} == dense
    assert set(sparse) - set(dense) == set(SPARSE_SOURCE_FILES)
    assert not set(SPARSE_SOURCE_FILES) & set(relation_source_hashes(ROOT))
    with pytest.raises(ValueError, match="Unknown relation writer"):
        source_hashes_for(ROOT, "rgb_relations_sparse_v9")


@pytest.mark.parametrize("version, writer_type, renderer_type, core_type", [
    (V1, SparseRelationWriterV1, RelationMemoryRenderer, RelationMemoryWriter),
    (V3, SparseBowlCycleWriterV3, RelationMemoryRendererV3, BowlCycleWriterV3)])
def test_pipeline_builds_the_sparse_writer_with_the_dense_renderer(monkeypatch, version, writer_type,
                                                                   renderer_type, core_type):
    for module in (libero_mem_profiles, relation_pipeline, registry):
        monkeypatch.setattr(module, "profile_for_task", lambda suite, task: PROFILE)
    detector = SimpleNamespace(provenance={})
    factory, renderer = make_pipeline_for("suite", "T1", detector, "libero_raw", object(), "relevant", version)
    assert type(renderer) is renderer_type and renderer.strategy == "relevant"
    writer = factory("episode")
    assert type(writer) is writer_type and isinstance(writer, core_type)
    assert writer.detector is detector and writer.VERSION == version
    # The dense pipeline of the wrapped version is left alone.
    dense_factory, dense_renderer = make_pipeline_for("suite", "T1", detector, "libero_raw", object(), "relevant",
                                                      core_version(version))
    assert type(dense_renderer) is renderer_type and type(dense_factory("episode")) is core_type


# -- the explicit-override runtime contract -------------------------------------------

@pytest.fixture
def contract(monkeypatch):
    detector = {"model": "unit-detector"}
    value = dict(writer="rgb_relations_v1", detector=detector)
    monkeypatch.setattr(artifact, "verify_memory_code", lambda *args, **kwargs: value)
    monkeypatch.setattr(artifact, "adapter_training_condition", lambda manifest: "memory")
    return value


def runtime(version, contract, **changes):
    writer_type = SPARSE_WRITERS.get(version) or RELATION_WRITERS[version]
    writer = writer_type("episode", PROFILE, SimpleNamespace(provenance=contract["detector"]), "libero_raw")
    renderer_type = RelationMemoryRendererV3 if core_version(version) == "rgb_relations_v3" else RelationMemoryRenderer
    arguments = dict(writer_override=version, runtime_writer_source_sha256=source_hashes_for(ROOT, version),
                     runtime_writer_config=relation_config_for("relevant", version))
    arguments.update(changes)
    return dict(memory_provenance=[contract]), writer, renderer_type(object(), strategy="relevant"), arguments


@pytest.mark.parametrize("version", [V1, V3, "rgb_relations_v3"])
def test_override_accepts_the_matching_writer_and_renderer(contract, version):
    manifest, writer, renderer, arguments = runtime(version, contract)
    assert artifact.validate_runtime(manifest, writer, renderer, **arguments) is contract


def test_override_rejects_drift_after_preflight(contract):
    manifest, writer, renderer, arguments = runtime(V1, contract)
    stale = dict(arguments["runtime_writer_source_sha256"], **{SPARSE_SOURCE_FILES[0]: "0" * 64})
    with pytest.raises(ValueError, match="source changed"):
        artifact.validate_runtime(manifest, writer, renderer, **dict(arguments, runtime_writer_source_sha256=stale))
    tampered = dict(arguments["runtime_writer_config"], sparse=dict(arguments["runtime_writer_config"]["sparse"],
                                                                    keyframe_interval=16))
    with pytest.raises(ValueError, match="config changed"):
        artifact.validate_runtime(manifest, writer, renderer, **dict(arguments, runtime_writer_config=tampered))
    with pytest.raises(ValueError, match="differs from its preflight snapshot"):
        artifact.validate_runtime(manifest, writer, RelationMemoryRendererV3(object()), **arguments)
    dense = RelationMemoryWriter("episode", PROFILE, SimpleNamespace(provenance=contract["detector"]), "libero_raw")
    with pytest.raises(ValueError, match="differs from its preflight snapshot"):
        artifact.validate_runtime(manifest, dense, renderer, **arguments)


def test_a_sparse_writer_cannot_stand_in_for_the_training_writer_without_an_override(contract):
    manifest, writer, renderer, _ = runtime(V1, contract)
    with pytest.raises(ValueError, match="differs from training"):
        artifact.validate_runtime(manifest, writer, renderer)


# -- backbones and CLI ------------------------------------------------------------------

def test_xvla_binds_the_sparse_v1_writer_only():
    policy = XVLAMemPolicy("unused", None, lambda *args, **kwargs: dict(input_ids=[1]), dict(arm="F1"))
    renderer = RelationMemoryRenderer(object(), strategy="relevant")
    detector = SimpleNamespace(provenance={})
    policy.bind_memory_runtime(RelationMemoryWriter("e", PROFILE, detector, "libero_raw"), renderer)
    policy.bind_memory_runtime(SparseRelationWriterV1("e", PROFILE, detector, "libero_raw"), renderer)
    with pytest.raises(ValueError, match="differs from V2 shard contract"):
        policy.bind_memory_runtime(SparseBowlCycleWriterV3("e", PROFILE, detector, "libero_raw"), renderer)
    with pytest.raises(ValueError, match="differs from V2 shard contract"):
        policy.bind_memory_runtime(SparseRelationWriterV1("e", PROFILE, detector, "libero_raw"),
                                   RelationMemoryRendererV3(object()))


@pytest.mark.parametrize("version", [V1, V3, "rgb_relations_v1"])
def test_cli_accepts_the_sparse_override(version):
    arguments = parse_args(["--output-dir", "out", "--sim-python", "sim", "--detector-python", "det",
                            "--adapter-dir", "adapter", "--conditions", "F1", "--relation-writer", version])
    assert arguments.relation_writer == version
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "out", "--sim-python", "sim", "--detector-python", "det",
                    "--adapter-dir", "adapter", "--conditions", "F1", "--relation-writer", "rgb_relations_sparse_v9"])
