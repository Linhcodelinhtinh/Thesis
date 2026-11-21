#!/usr/bin/env python3
"""Build the X-VLA F0/F1 training view of the SmolVLA V2 shards (no image copy, no simulator).

Input : the V2 shard directory (manifest.json, split.lock.json, shards/*.hdf5) from
        libero_mem_t1_t3_shards_v2.zip  (split lock 8275e468..., 598 train episodes)
Output: <out>/xvla_abs_actions_v2.h5   abs_action_6d [n,10] per (shard, demo)
        <out>/metas_f0/{base,mem}.json  prompt_mode=instruction for both domains
        <out>/metas_f1/{base,mem}.json  base=instruction, mem=memory (per-frame shard prompts)
        <out>/build_receipt.json        counts, hashes, tracking diagnostics, token check

Two metas per arm -> upstream InfiniteDataReader samples base/mem 50/50
(DATA_WEIGHTS default 1.0 for both names).

Usage (Kaggle):
  python build_xvla_v2_dataset.py --shards-root /kaggle/input/<v2-dataset>/... \
      --out /kaggle/working/xvla_v2 --tokenizer /kaggle/working/x_vla_libero
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abs_actions import delta_to_abs6d  # noqa: E402
from shard_handler import HANDLER_NAME  # noqa: E402

EXPECTED_LOCK = "8275e468382835c06f9cc738361df26b7452965850878e7ca882f392c9ca4165"
EXPECTED_TRAIN = {"base": 400, "mem": 198}
XVLA_MAX_TOKENS = 50  # XVLAProcessor.language_max_length (truncation=True, padding=max_length)


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_root(root: Path) -> Path:
    if (root / "manifest.json").is_file() and (root / "shards").is_dir():
        return root
    for m in sorted(root.rglob("manifest.json")):
        if (m.parent / "shards").is_dir():
            return m.parent
    raise FileNotFoundError(f"No V2 shard root (manifest.json + shards/) under {root}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="X-VLA checkpoint dir (tokenizer.json) for the 50-token check")
    p.add_argument("--rot-frame", default="panda_rz_m90", choices=["panda_rz_m90", "identity"])
    p.add_argument("--verify-shard-hashes", action="store_true", help="sha256 every shard (~12 GB read)")
    p.add_argument("--allow-lock-mismatch", action="store_true")
    args = p.parse_args()

    root = find_root(args.shards_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("manifest status is not complete")
    if manifest.get("split_lock_sha256") != EXPECTED_LOCK and not args.allow_lock_mismatch:
        raise SystemExit(f"split lock {manifest.get('split_lock_sha256')} != V2 {EXPECTED_LOCK}")

    train = [e for e in manifest["episodes"] if e["split"] == "train"]
    counts = Counter(e["domain"] for e in train)
    if dict(counts) != EXPECTED_TRAIN and not args.allow_lock_mismatch:
        raise SystemExit(f"train episodes {dict(counts)} != {EXPECTED_TRAIN}")

    if args.verify_shard_hashes:
        for rel, digest in manifest["shards"].items():
            got = sha256(root / rel)
            if got != digest:
                raise SystemExit(f"shard hash mismatch {rel}")
            print("hash ok", rel, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    actions_h5 = args.out / "xvla_abs_actions_v2.h5"
    tmp = actions_h5.with_suffix(".partial.h5")
    t0 = time.time()
    per_ep, prompts_all = [], Counter()
    mem_frames = Counter()
    with h5py.File(tmp, "w") as fo:
        fo.attrs.update(split_lock_sha256=manifest["split_lock_sha256"], rot_frame=args.rot_frame,
                        layout="pos3 + rot6d(col0,col1 of controller goal_ori) + raw gripper action",
                        source="analytic robosuite-1.4 OSC_POSE goal from shard robot_states + actions")
        by_shard = {}
        for e in train:
            by_shard.setdefault(e["shard"], []).append(e)
        for rel, eps in sorted(by_shard.items()):
            with h5py.File(root / rel, "r") as f:
                for e in eps:
                    g = f[e["demo"]]
                    rs, ac = g["robot_states"][()], g["actions"][()]
                    n = len(ac)
                    if n != e["length"] or g["agentview_rgb"].shape[0] != n or g["eye_in_hand_rgb"].shape[0] != n:
                        raise SystemExit(f"length mismatch {rel}:{e['demo']}")
                    if g.attrs["split"] != "train" or g.attrs["domain"] != e["domain"]:
                        raise SystemExit(f"attr mismatch {rel}:{e['demo']}")
                    abs6d, info = delta_to_abs6d(rs, ac, args.rot_frame)
                    fo.create_dataset(f"{rel}/{e['demo']}/abs_action_6d", data=abs6d)
                    instr = g.attrs["instruction"]
                    instr = instr.decode() if isinstance(instr, bytes) else str(instr)
                    prompts = [x.decode() if isinstance(x, bytes) else str(x) for x in g["prompts"][()]]
                    flags = np.asarray(g["has_memory"][()], dtype=bool)
                    if any((pp != instr) != bool(fl) for pp, fl in zip(prompts, flags)):
                        raise SystemExit(f"has_memory/prompts disagree {rel}:{e['demo']}")
                    if e["domain"] == "base" and flags.any():
                        raise SystemExit(f"base episode carries memory prompts {rel}:{e['demo']}")
                    # frames the X-VLA sampler can actually show: image rows 1..n-10
                    shown = flags[1:max(1, n - 9)]
                    mem_frames["shown_with_memory"] += int(shown.sum())
                    mem_frames["shown_total_" + e["domain"]] += int(len(shown))
                    prompts_all.update(prompts if e["domain"] == "mem" else [instr])
                    per_ep.append(dict(domain=e["domain"], task=e["task"], demo=e["demo"], **info))
            print(json.dumps(dict(shard=rel, episodes=len(eps), sec=round(time.time() - t0, 1))), flush=True)
    tmp.replace(actions_h5)

    # Florence-2 (BART) token budget for every distinct prompt
    token_report = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
        except Exception:
            from transformers import BartTokenizerFast
            tok = BartTokenizerFast.from_pretrained(str(args.tokenizer))
        lens = {s: len(tok(s)["input_ids"]) for s in prompts_all}
        over = {s: n for s, n in lens.items() if n > XVLA_MAX_TOKENS}
        token_report = dict(distinct_prompts=len(lens), max_tokens=max(lens.values()), limit=XVLA_MAX_TOKENS,
                            over_limit=len(over), examples_over=list(over.items())[:5],
                            longest=max(lens, key=lens.get))
        if over:
            raise SystemExit(f"{len(over)} prompts exceed {XVLA_MAX_TOKENS} Florence tokens (would be truncated)")

    shards_root = str(root.resolve())
    for arm, mem_mode in (("f0", "instruction"), ("f1", "memory")):
        d = args.out / f"metas_{arm}"
        d.mkdir(exist_ok=True)
        for domain in ("base", "mem"):
            meta = dict(dataset_name=f"xvla_v2_{arm}_{domain}", robot_type=HANDLER_NAME,
                        prompt_mode=mem_mode if domain == "mem" else "instruction",
                        shards_root=shards_root, actions_h5=str(actions_h5.resolve()),
                        split_lock_sha256=manifest["split_lock_sha256"],
                        datalist=[dict(shard=e["shard"], demo=e["demo"], domain=e["domain"], task=e["task"],
                                       length=e["length"]) for e in train if e["domain"] == domain])
            (d / f"{domain}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def agg(key, dom):
        v = [r[key] for r in per_ep if r["domain"] == dom]
        return dict(median=float(np.median(v)), max=float(np.max(v)))

    receipt = dict(
        status="complete", split_lock_sha256=manifest["split_lock_sha256"],
        source_manifest_sha256=sha256(root / "manifest.json"), memory_source=manifest.get("memory_source"),
        rot_frame=args.rot_frame, train_episodes=dict(counts),
        frames={d: int(sum(r["frames"] for r in per_ep if r["domain"] == d)) for d in ("base", "mem")},
        sampler_frames=dict(mem_frames),
        zero_rotation_rows=int(sum(r["zero_rotation_rows"] for r in per_ep)),
        clipped_rows=int(sum(r["clipped_rows"] for r in per_ep)),
        tracking={d: {k: agg(k, d) for k in ("pos_track_median_m", "pos_track_p95_m",
                                              "rot_track_median_deg", "rot_track_p95_deg")} for d in ("base", "mem")},
        token_check=token_report, actions_h5_sha256=sha256(actions_h5),
        code_sha256={n: sha256(Path(__file__).parent / n) for n in
                     ("abs_actions.py", "shard_handler.py", "build_xvla_v2_dataset.py")},
        note="Analytic goals; confirm frame/offset with check_xvla_alignment.py (and optionally sim replay)")
    (args.out / "build_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("train_episodes", "frames", "sampler_frames", "tracking", "token_check")},
                     indent=1))


if __name__ == "__main__":
    main()
