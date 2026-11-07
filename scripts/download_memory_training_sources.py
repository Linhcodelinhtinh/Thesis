#!/usr/bin/env python
"""Download only pinned author demonstrations; verify their published SHA256."""
import argparse
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.file_hash import sha256_file


def source_inventory(base_root, mem_root, base_only=False):
    """Resolve only the reviewed, pinned source inventories (no network calls)."""
    inventory = json.loads((ROOT / "resources/manifests/libero_10_sft_sources.json").read_text(encoding="utf-8"))
    sources = [dict(repo="yifengzhu-hf/LIBERO-datasets",
                    revision=inventory["benchmark_provenance"]["pinned_commit"],
                    relative=r["task_suite"] + "/" + r["task_name"] + "_demo.hdf5",
                    sha256=r["sha256"], size_bytes=r["file_size_bytes"], root=str(base_root))
               for r in inventory["demonstrations"]]
    if not base_only:
        mem = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))["annotated_dataset"]
        sources += [dict(repo="libero-mem/LIBERO-Mem", revision=mem["revision"],
                         relative=r["path"], sha256=r["upstream_sha256"],
                         size_bytes=r["size_bytes"], root=str(mem_root))
                    for r in mem["selected_pilot_files"]]
    return sources


def plan_sources(sources, input_root=None):
    """Identify existing/attached/download sources; hash validation happens on fetch."""
    records = []
    for source in sources:
        destination = Path(source["root"]) / source["relative"]
        candidates = []
        if destination.is_file():
            candidates = [destination]
            origin = "existing"
        else:
            if destination.exists() or destination.is_symlink():
                raise ValueError("Invalid source destination: " + str(destination))
            if input_root is not None:
                candidates = sorted(p for p in Path(input_root).rglob(Path(source["relative"]).name) if p.is_file())
            if len(candidates) > 1:
                raise ValueError("Ambiguous attached source; attach only one copy: " + source["relative"])
            origin = "attached" if candidates else "download"
        if candidates and candidates[0].stat().st_size != source["size_bytes"]:
            raise ValueError("Source size mismatch (file preserved): " + str(candidates[0]))
        records.append(dict(source, destination=str(destination), origin=origin,
                            local_path=str(candidates[0].resolve()) if candidates else None))
    return dict(status="planned_not_hash_verified", sources=records,
                total_source_bytes=sum(s["size_bytes"] for s in records),
                download_bytes=sum(s["size_bytes"] for s in records if s["origin"] == "download"))


def materialize_source(source):
    path = Path(source["destination"])
    if source["origin"] == "download":
        from huggingface_hub import hf_hub_download
        print("Downloading " + source["relative"], flush=True)
        hf_hub_download(source["repo"], source["relative"], repo_type="dataset",
                        revision=source["revision"], local_dir=source["root"])
        local = path
    else:
        local = Path(source["local_path"])
    if local.stat().st_size != source["size_bytes"] or sha256_file(local) != source["sha256"]:
        raise ValueError("Source hash/size mismatch (file preserved for inspection): " + str(local))
    if source["origin"] == "attached":
        # Kaggle input is read-only. Link verified bytes without duplicating tens of GB.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(local)
    print("Verified " + source["relative"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", type=Path, default=ROOT / "resources/demonstrations")
    p.add_argument("--mem-root", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--input-root", type=Path, help="Reuse exact raw files attached to Kaggle via verified symlinks")
    p.add_argument("--plan-only", action="store_true", help="Identify sources and download volume without downloading")
    p.add_argument("--report", type=Path, help="Write the source plan / verified provenance receipt")
    a = p.parse_args()
    plan = plan_sources(source_inventory(a.base_root, a.mem_root, a.base_only), a.input_root)
    if not a.plan_only:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(materialize_source, plan["sources"]))
        plan["status"] = "verified"
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
