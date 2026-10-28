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


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", type=Path, default=ROOT / "resources/demonstrations")
    p.add_argument("--mem-root", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    inventory = json.loads((ROOT / "resources/manifests/libero_10_sft_sources.json").read_text(encoding="utf-8"))
    sources = [("yifengzhu-hf/LIBERO-datasets", inventory["benchmark_provenance"]["pinned_commit"],
                r["task_suite"] + "/" + r["task_name"] + "_demo.hdf5", r["sha256"], a.base_root)
               for r in inventory["demonstrations"]]
    if not a.base_only:
        mem = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))["annotated_dataset"]
        sources += [("libero-mem/LIBERO-Mem", mem["revision"], r["path"], r["upstream_sha256"], a.mem_root)
                    for r in mem["selected_pilot_files"]]
    def download(source):
        repo, revision, relative, expected, destination = source
        path = destination / relative
        if not path.is_file():
            print("Downloading " + relative, flush=True)
            hf_hub_download(repo, relative, repo_type="dataset", revision=revision, local_dir=destination)
        if sha256_file(path) != expected:
            raise ValueError("Source hash mismatch (file preserved for inspection): " + str(path))
        print("Verified " + relative, flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(download, sources))


if __name__ == "__main__":
    main()
