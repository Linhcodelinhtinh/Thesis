"""Load only attested memory modules from an archived E2 source snapshot."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile


def install(root, shards_manifest):
    root = Path(root).absolute()
    contracts = json.loads(Path(shards_manifest).read_text(encoding="utf-8"))["memory_provenance"]
    inventory = contracts[0]["writer_code_sha256"]
    for name, digest in inventory.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Archived writer hash mismatch: " + name)
    modules = {name[:-3].replace("/", "."): root / name for name in inventory if name.startswith("src/") and name.endswith(".py")}
    # Reuse the exact causal runner used by the historical E2 reference.
    reference_runner = root / "src/evaluation/libero_mem_rollout.py"
    if not reference_runner.is_file():
        raise FileNotFoundError("E2 source snapshot is missing the historical rollout runner")
    modules["src.evaluation.libero_mem_rollout"] = reference_runner
    import sys
    if any(name in sys.modules for name in modules):
        raise RuntimeError("Install archived memory source before importing the evaluation stack")

    class AttestedMemorySource(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in modules:
                return importlib.util.spec_from_file_location(fullname, modules[fullname])
            return None

    sys.meta_path.insert(0, AttestedMemorySource())


def prepare(archive, destination, repository, shards_manifest):
    """Materialize Python source only, never overwrite the working tree."""
    destination, repository = Path(destination), Path(repository)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            path = Path(name)
            if (path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name):
                raise ValueError("Unsafe snapshot member")
            if name.startswith(("src/", "scripts/")) and name.endswith(".py"):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    # Worker code locates these audited data-only manifests relative to its source.
    for name in ("resources/manifests/memory_perception_source.json",
                 "configs/benchmarks/libero_mem_suite_source_v1.json"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    install(destination, shards_manifest)
    receipt = dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                   shards_manifest_sha256=hashlib.sha256(Path(shards_manifest).read_bytes()).hexdigest())
    (destination / "memory_source_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--shards-manifest", type=Path, default=Path(__file__).resolve().parents[2] / "experiments/training/relations-author-split-v2/shards/manifest.json")
    a = p.parse_args()
    prepare(a.archive, a.destination, a.repository, a.shards_manifest)
