#!/usr/bin/env python3
"""Checkpoint-agreement check before training (needs a GPU, ~5 min on a T4).

The frozen X-VLA-Libero checkpoint already solves LIBERO-10 (10/10 locally). If our
abs_action_6d, rotation frame, 6D layout, image flip and frame offset match its
training contract, its predicted chunk on V2 *base* frames must be close to our
targets. A wrong rotation frame shows up as a ~90 deg error, a wrong flip or
frame offset as a clearly larger position error than the reference variant.

Reports, per variant, median/p90 position error (cm) and rotation error (deg) of
the first predicted action and of the whole 30-step chunk:
  reference      : as built (flip agentview, site frame)
  body_frame     : targets re-expressed with identity offset (tests SITE_FROM_BODY)
  no_flip        : agentview NOT flipped (tests camera contract)
  frame_offset+1 : image one row later (tests the drop-first-frame pairing)

Usage:
  python check_xvla_alignment.py --xvla-root X-VLA --models x_vla_libero \
      --metas xvla_v2/metas_f0 --episodes 20 --per-episode 6 --out alignment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xvla-root", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--metas", required=True)
    p.add_argument("--domain", default="base", choices=["base", "mem"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--per-episode", type=int, default=6)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="alignment.json")
    args = p.parse_args()
    sys.path.insert(0, args.xvla_root); sys.path.insert(0, str(HERE))
    from abs_actions import SITE_FROM_BODY, rot6d_to_mat, rotation_angle_deg
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = XVLA.from_pretrained(args.models).to(dev).eval()
    proc = XVLAProcessor.from_pretrained(args.models)
    aug = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    meta = json.loads(Path(args.metas, f"{args.domain}.json").read_text())
    rng = np.random.default_rng(args.seed)
    eps = [meta["datalist"][i] for i in rng.choice(len(meta["datalist"]), args.episodes, replace=False)]
    off = SITE_FROM_BODY["panda_rz_m90"]
    H = model.num_actions
    res = {k: dict(pos0=[], posall=[], rot0=[], rotall=[]) for k in
           ("reference", "body_frame", "no_flip", "frame_offset+1")}

    def run(main_img, wrist_img, prompt, proprio10):
        img = torch.stack([aug(Image.fromarray(main_img)), aug(Image.fromarray(wrist_img)),
                           torch.zeros(3, 224, 224)], 0)[None].to(dev)
        mask = torch.tensor([[True, True, False]], device=dev)
        prop = torch.zeros(1, 20, device=dev); prop[0, :10] = torch.tensor(proprio10, device=dev)
        ids = proc.encode_language([prompt])["input_ids"].to(dev)
        with torch.no_grad():
            out = model.generate_actions(input_ids=ids, image_input=img, image_mask=mask,
                                         domain_id=torch.tensor([3], device=dev), proprio=prop, steps=args.steps)
        return out[0, :, :10].float().cpu().numpy()

    def score(key, pred, tgt_pos, tgt_rot):
        pe = np.linalg.norm(pred[:, :3] - tgt_pos, axis=1) * 100
        re = rotation_angle_deg(rot6d_to_mat(pred[:, 3:9]), tgt_rot)
        r = res[key]; r["pos0"].append(pe[0]); r["posall"].append(pe.mean()); r["rot0"].append(re[0]); r["rotall"].append(re.mean())

    with h5py.File(meta["actions_h5"], "r") as fa:
        for ep in eps:
            a = fa[ep["shard"]][ep["demo"]]["abs_action_6d"][()].astype(np.float64)
            n = len(a)
            with h5py.File(os.path.join(meta["shards_root"], ep["shard"]), "r") as f:
                g = f[ep["demo"]]
                instr = g.attrs["instruction"]; instr = instr.decode() if isinstance(instr, bytes) else str(instr)
                for idx in np.linspace(5, n - H - 3, args.per_episode).astype(int):
                    tgt = a[idx + 1: idx + 1 + H]
                    tpos, trot = tgt[:, :3], rot6d_to_mat(tgt[:, 3:9])
                    prop = a[idx].copy(); prop[9] = 0
                    av, wr = g["agentview_rgb"][idx + 1], g["eye_in_hand_rgb"][idx + 1]
                    flip = np.ascontiguousarray(av[::-1, ::-1])
                    pred = run(flip, wr, instr, prop)
                    score("reference", pred, tpos, trot)
                    score("body_frame", pred, tpos, trot @ off.T)
                    score("no_flip", run(np.ascontiguousarray(av), wr, instr, prop), tpos, trot)
                    av2, wr2 = g["agentview_rgb"][idx + 2], g["eye_in_hand_rgb"][idx + 2]
                    score("frame_offset+1", run(np.ascontiguousarray(av2[::-1, ::-1]), wr2, instr, prop), tpos, trot)
            print("done", ep["task"][:40], ep["demo"], flush=True)

    summary = {k: {m: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) for m, v in r.items()}
               for k, r in res.items()}
    ref = summary["reference"]
    verdict = dict(
        rotation_frame_ok=ref["rot0"]["median"] < summary["body_frame"]["rot0"]["median"] / 2,
        flip_ok=ref["posall"]["median"] <= summary["no_flip"]["posall"]["median"],
        reference_first_action_cm=ref["pos0"]["median"], reference_first_action_deg=ref["rot0"]["median"])
    Path(args.out).write_text(json.dumps(dict(summary=summary, verdict=verdict, samples=len(res["reference"]["pos0"])), indent=1))
    print(json.dumps(dict(summary=summary, verdict=verdict), indent=1))


if __name__ == "__main__":
    main()
