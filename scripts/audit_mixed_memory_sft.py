#!/usr/bin/env python
"""Compare prepared targets and selected RGB frames with real pinned source data."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from src.training.mixed_sft import validate_mixed_dataset, canonical_raw_rgb
from src.utils.file_hash import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--base-root", type=Path, default=ROOT/"resources/demonstrations")
    p.add_argument("--mem-root", type=Path, default=ROOT/"data/libero_mem")
    p.add_argument("--policy-parity", action="store_true", help="Also compare collator against the real installed policy adapter")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = validate_mixed_dataset(a.data_dir)
    roots = dict(base=a.base_root, mem=a.mem_root)
    for source in m["sources"]:
        if sha256_file(roots[source["domain"]]/source["path"]) != source["sha256"]:
            raise ValueError("Original source hash mismatch")
    frames_checked = 0
    for e in m["episodes"]:
        with h5py.File(roots[e["domain"]]/e["source"], "r") as source, h5py.File(a.data_dir/e["shard"], "r") as prepared:
            old, new = source["data"][e["demo"]], prepared[e["demo"]]
            n = e["length"]; offset = int(e["domain"] == "base")
            np.testing.assert_array_equal(new["actions"][:], old["actions"][offset:].astype(np.float32))
            np.testing.assert_array_equal(new["robot_states"][:], old["robot_states"][:n].astype(np.float32))
            for i in sorted({0, n//2, n-1}):
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    expected = canonical_raw_rgb(old["obs/"+camera][i:i+1], e["domain"])[0]
                    np.testing.assert_array_equal(new[camera][i], expected)
                frames_checked += 1
    parity_samples = 0
    if a.policy_parity:
        import torch
        from src.training.memory_dataset import MemoryDataset, collate_memory
        from src.models.smolvla.adapter import SmolVLAAdapter
        adapter = SmolVLAAdapter()
        for split in ("train", "val", "test"):
            data = MemoryDataset(a.data_dir, split)
            try:
                cursor = 0
                for e in data.episodes:
                    for i in sorted({0, e["length"]-1}):
                        sample = data[cursor+i]; robot = sample["robot"]
                        obs = dict(agentview_image=sample["agentview"], robot0_eye_in_hand_image=sample["wrist"],
                                   robot0_gripper_qpos=robot[:2], robot0_eef_pos=robot[2:5], robot0_eef_quat=robot[5:9])
                        online = adapter.build_raw_features(obs, sample["task"])
                        offline = collate_memory([sample])
                        for key in ("observation.images.image", "observation.images.image2", "observation.state"):
                            torch.testing.assert_close(online[key], offline[key][0], rtol=0, atol=0)
                        if i == e["length"]-1:
                            assert sample["actions_id_pad"].tolist() == [False]+[True]*49
                        parity_samples += 1
                    cursor += e["length"]
            finally:
                data.close()
    report = dict(status="passed", classification="DATA_ALIGNMENT_QA_NOT_POLICY_SCORE",
                  split_lock_sha256=m["split_lock_sha256"], sample_selection_sha256=m["sample_selection_sha256"],
                  episodes=len(m["episodes"]), all_state_action_rows_compared=sum(e["length"] for e in m["episodes"]),
                  rgb_frames_compared_per_camera=frames_checked, policy_adapter_parity_samples=parity_samples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
