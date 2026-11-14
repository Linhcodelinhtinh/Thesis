#!/usr/bin/env python
"""Record source-verified T7 reset states for runtime development."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.libero_mem_protocol import load_suite_source, task_spec
from src.evaluation.mem_eval_io import write_json
from src.training.split_lock import seal_lock, state_digest
from src.utils.file_hash import sha256_file


def main():
    import h5py
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/libero_mem_raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new state lock output")
    inventory = ROOT / "resources/manifests/libero_mem_t7_refined_source.json"
    source = json.loads(inventory.read_text(encoding="utf-8"))["raw_source"]
    source_path = args.raw_root / source["path"]
    if source_path.stat().st_size != source["size_bytes"] or sha256_file(source_path) != source["sha256"]:
        raise ValueError("T7 raw source differs from its pinned release")
    suite = load_suite_source(ROOT / "external/libero-mem")
    task = task_spec(suite, "T7")
    if source_path.name != task["name"] + "_demo.hdf5" or task["goal_operator"] != "or":
        raise ValueError("T7 inventory differs from the locked suite")
    rows = []
    with h5py.File(source_path, "r") as stream:
        data = stream["data"]
        if set(data) != {f"demo_{i}" for i in range(1, 121)}:
            raise ValueError("Unexpected T7 raw episode names")
        for index in range(120):
            demo = f"demo_{index + 1}"
            group = data[demo]
            state = group["init_state"][()]
            if (state.shape != (90,) or not np.isfinite(state).all()
                    or group["states"].shape != (len(group["actions"]), 90)
                    or group["actions"].shape[1:] != (7,)):
                raise ValueError(f"Unexpected T7 state/action format at {demo}")
            rows.append(dict(demo=demo, raw_index=index, initial_state_sha256=state_digest(state),
                             initial_state=state.tolist(), actions=len(group["actions"])))
    if len({row["initial_state_sha256"] for row in rows}) != len(rows):
        raise ValueError("T7 raw source contains repeated reset states")
    write_json(args.output, seal_lock(dict(
        artifact="LIBERO_MEM_T7_RAW_SOURCE_INVENTORY", task=task["name"],
        classification="SOURCE_ONLY_RUNTIME_NOT_VALIDATED", source=source,
        source_inventory_sha256=sha256_file(inventory), upstream_revision=suite["upstream_revision"],
        bddl_sha256=suite["source_text_sha256"][task["bddl"]], states=rows)))
    print(f"Verified {len(rows)} official T7 reset states")


if __name__ == "__main__":
    main()
