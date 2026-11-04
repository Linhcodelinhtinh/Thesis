"""Unmerged V2 adapters with verified base weights and causal-memory provenance."""
import json
import shutil
from pathlib import Path

from src.utils.file_hash import sha256_file

ROOT = Path(__file__).resolve().parents[2]
BASE_FILES = ("model.safetensors", "policy_preprocessor_step_5_normalizer_processor.safetensors",
              "policy_postprocessor_step_0_unnormalizer_processor.safetensors")


def verify_files(root, inventory):
    root = Path(root).resolve()
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if root not in path.parents or sha256_file(path) != digest:
            raise ValueError("Artifact hash/path mismatch: " + name)


def verify_memory_code(provenance):
    if not provenance:
        raise ValueError("A Mem adapter requires predicted-memory provenance")
    first = provenance[0]
    for item in provenance:
        if item != first:
            # Per-task index manifest hashes differ; the implementation must not.
            for key in ("writer", "writer_code_sha256", "writer_config", "detector", "tokenizer_revision"):
                if item[key] != first[key]:
                    raise ValueError("Inconsistent memory contract: " + key)
        verify_files(ROOT, item["writer_code_sha256"])
    return first


def export_adapter(policy, assets, destination, run_info):
    """Save only PEFT deltas; never merge or change the original base artifact."""
    destination, assets = Path(destination), Path(assets)
    if not hasattr(policy, "peft_config"):
        raise ValueError("Unmerged export requires a PEFT policy")
    provenance = run_info.get("memory_provenance", [])
    verify_memory_code(provenance)
    destination.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(destination, safe_serialization=True)
    shutil.copy2(assets / "backbone/tokenizer.json", destination / "tokenizer.json")
    if not (destination / "adapter_model.safetensors").is_file():
        raise ValueError("PEFT did not produce an adapter weight artifact")
    inventory = {p.relative_to(destination).as_posix(): sha256_file(p)
                 for p in destination.rglob("*") if p.is_file()}
    manifest = dict(schema_version=1, artifact="unmerged_mem_lora", run=run_info,
                    base_files={name: sha256_file(assets / "policy" / name) for name in BASE_FILES},
                    adapter_files=inventory, memory_provenance=provenance)
    (destination / "mem_adapter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_adapter(adapter_path, base_path):
    root = Path(adapter_path)
    manifest = json.loads((root / "mem_adapter_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != "unmerged_mem_lora" or manifest.get("schema_version") != 1:
        raise ValueError("Unknown Mem adapter artifact")
    if set(manifest["base_files"]) != set(BASE_FILES):
        raise ValueError("Incomplete base identity")
    if not {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}.issubset(manifest["adapter_files"]):
        raise ValueError("Incomplete adapter identity")
    verify_files(base_path, manifest["base_files"])
    verify_files(root, manifest["adapter_files"])
    verify_memory_code(manifest["memory_provenance"])
    return manifest


def validate_runtime(manifest, writer, renderer):
    from dataclasses import asdict
    contract = verify_memory_code(manifest["memory_provenance"])
    if writer.tracker.image_convention != "libero_raw":
        raise ValueError("Online tracker must consume raw simulator RGB")
    if asdict(writer.tracker.config) != contract["writer_config"]:
        raise ValueError("Online tracker config differs from training")
    if writer.tracker.detector.provenance != contract["detector"]:
        raise ValueError("Online detector differs from training")
    if renderer.max_total_tokens != 48 or not renderer.strict_tokenizer or renderer.allow_oracle:
        raise ValueError("Online prompt contract differs from training")
