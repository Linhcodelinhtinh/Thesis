#!/usr/bin/env python3
"""Integrity check of downloaded X-VLA F0/F1 adapters (CPU, numpy + safetensors only).

  kaggle kernels output luquanglinh/x-vla-memory-adaptation-f0-f1 -p kaggle_out
  python scripts/xvla_v2/verify_adapter_export.py \
      --f0 kaggle_out/runs/f0/ckpt-5000 --f1 kaggle_out/runs/f1/ckpt-5000 [--zips kaggle_out]

Checks per adapter: export_manifest hashes match the files, state.json step = 5000,
all tensors finite and non-degenerate, LoRA B matrices moved away from the zero init,
expected key groups present, trainable-parameter count. Between arms: same tensor
names/shapes, different weights (F0 != F1), and relative distance of the weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from safetensors import safe_open


def sha(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    with safe_open(str(path), "numpy") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def check(name, d, steps):
    d = Path(d)
    rep, ok = {}, True
    man = json.loads((d / "export_manifest.json").read_text())
    for fn, want in man["files"].items():
        got = sha(d / fn)
        rep[f"sha_{fn}"] = got == want
        ok &= got == want
    state = json.loads((d / "state.json").read_text())
    rep["step"] = state["global_step"]; ok &= state["global_step"] == steps
    t = load(d / "adapter_model.safetensors")
    rep["tensors"] = len(t); rep["params"] = int(sum(v.size for v in t.values()))
    rep["finite"] = bool(all(np.isfinite(v).all() for v in t.values())); ok &= rep["finite"]
    lb = [v for k, v in t.items() if "lora_B" in k]
    rep["lora_B_tensors"] = len(lb)
    rep["lora_B_nonzero_frac"] = float(np.mean([bool(np.abs(v).max() > 0) for v in lb])) if lb else 0.0
    ok &= bool(lb) and rep["lora_B_nonzero_frac"] == 1.0
    for grp in ("soft_prompt_hub", "action_encoder", "action_decoder"):
        rep[f"has_{grp}"] = any(grp in k for k in t); ok &= rep[f"has_{grp}"]
    rep["arm"] = man["arm"]; rep["prompt_mode_mem"] = man["prompt_mode_mem"]
    print(name, json.dumps(rep)); return ok, t, man


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f0", required=True); p.add_argument("--f1", required=True)
    p.add_argument("--zips", help="dir with xvla_v2_f{0,1}_step5000.zip to cross-check")
    p.add_argument("--steps", type=int, default=5000)
    a = p.parse_args()
    ok0, t0, m0 = check("F0", a.f0, a.steps)
    ok1, t1, m1 = check("F1", a.f1, a.steps)
    ok = ok0 and ok1
    same_keys = t0.keys() == t1.keys() and all(t0[k].shape == t1[k].shape for k in t0)
    num = sum(float(np.sum((t0[k].astype(np.float64) - t1[k].astype(np.float64)) ** 2)) for k in t0) if same_keys else None
    den = sum(float(np.sum(t0[k].astype(np.float64) ** 2)) for k in t0) if same_keys else None
    identical = same_keys and all(np.array_equal(t0[k], t1[k]) for k in t0)
    print("same_keys_shapes", same_keys, "identical_weights", identical,
          "rel_l2_distance", None if num is None else (num / den) ** 0.5)
    ok &= same_keys and not identical
    for k in ("base_revision", "xvla_commit", "split_lock_sha256", "actions_h5_sha256", "seed", "lr", "warmup_steps"):
        eq = m0.get(k) == m1.get(k); print("manifest", k, "equal" if eq else f"DIFF {m0.get(k)} vs {m1.get(k)}"); ok &= eq
    print("prompt_mode_mem", m0["prompt_mode_mem"], m1["prompt_mode_mem"])
    ok &= (m0["prompt_mode_mem"], m1["prompt_mode_mem"]) == ("instruction", "memory")
    if a.zips:
        for arm, d in (("f0", a.f0), ("f1", a.f1)):
            z = Path(a.zips) / f"xvla_v2_{arm}_step5000.zip"
            if z.is_file():
                with zipfile.ZipFile(z) as zf:
                    names = zf.namelist()
                    inner = zf.read("adapter_model.safetensors")
                same = hashlib.sha256(inner).hexdigest() == sha(Path(d) / "adapter_model.safetensors")
                print(arm, "zip members", len(names), "zip adapter == ckpt adapter", same); ok &= same
    print("PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
