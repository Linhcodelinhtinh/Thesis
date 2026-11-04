"""Artifact integrity and runtime contract tests; no benchmark execution."""
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from src.training.mem_adapter_artifact import verify_files, validate_runtime
from src.utils.file_hash import sha256_file


def test_hash_guard_rejects_changed_bytes_and_path_escape(tmp_path):
    f = tmp_path / "weights"
    f.write_bytes(b"first")
    inventory = {f.name: sha256_file(f)}
    verify_files(tmp_path, inventory)
    f.write_bytes(b"second")
    with pytest.raises(ValueError, match="mismatch"):
        verify_files(tmp_path, inventory)
    with pytest.raises(ValueError, match="mismatch"):
        verify_files(tmp_path, {"../outside": "irrelevant"})


def test_online_tracker_and_detector_must_match_training(monkeypatch):
    @dataclass
    class Config:
        dwell_steps: int = 3
    contract = dict(writer_config={"dwell_steps": 3}, detector={"revision": "pinned"})
    monkeypatch.setattr("src.training.mem_adapter_artifact.verify_memory_code", lambda _: contract)
    writer = SimpleNamespace(tracker=SimpleNamespace(config=Config(), image_convention="libero_raw", detector=SimpleNamespace(provenance={"revision": "pinned"})))
    renderer = SimpleNamespace(max_total_tokens=48, strict_tokenizer=True, allow_oracle=False)
    manifest = {"memory_provenance": []}
    validate_runtime(manifest, writer, renderer)
    writer.tracker.config.dwell_steps = 4
    with pytest.raises(ValueError, match="tracker"):
        validate_runtime(manifest, writer, renderer)
    writer.tracker.config.dwell_steps = 3
    writer.tracker.detector.provenance = {"revision": "changed"}
    with pytest.raises(ValueError, match="detector"):
        validate_runtime(manifest, writer, renderer)


def test_real_peft_export_stays_unmerged_and_reloads(tmp_path, monkeypatch):
    import torch
    peft = pytest.importorskip("peft")
    from src.training.mem_adapter_artifact import export_adapter, validate_adapter, BASE_FILES
    torch.manual_seed(8)
    base = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    original = base[0].weight.detach().clone()
    policy = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
    with torch.no_grad():
        policy.base_model.model[0].lora_B["default"].weight.fill_(0.25)
    x = torch.ones(1, 3)
    adapted = policy(x).detach()
    with policy.disable_adapter():
        torch.testing.assert_close(policy(x), x @ original.T)
    assets = tmp_path / "assets"
    (assets / "policy").mkdir(parents=True)
    (assets / "backbone").mkdir()
    for name in BASE_FILES:
        (assets / "policy" / name).write_bytes(b"unit-test identity, not benchmark weights")
    (assets / "backbone/tokenizer.json").write_text("{}")
    monkeypatch.setattr("src.training.mem_adapter_artifact.verify_memory_code", lambda _: {})
    out = tmp_path / "adapter"
    export_adapter(policy, assets, out, {"memory_provenance": [{}]})
    assert not (out / "model.safetensors").exists()
    validate_adapter(out, assets / "policy")
    fresh = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        fresh[0].weight.copy_(original)
    restored = peft.PeftModel.from_pretrained(fresh, out)
    torch.testing.assert_close(restored(x), adapted)
