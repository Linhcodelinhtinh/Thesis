import json, sys, subprocess, re, numpy as np, h5py, torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
S = str(Path(__file__).resolve().parents[1]); X = sys.argv[1]; W = Path(sys.argv[2])
sys.path.insert(0, S)
root = W / "v2"; (root / "shards").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1); eps = []
for dom, task, demos in (("base", "TASKB", 2), ("mem", "TASKM", 2)):
    rel = f"shards/{dom}__{task}.hdf5"
    with h5py.File(root / rel, "w") as f:
        for k in range(demos):
            n = 60 + 7 * k; d = f"demo_{k}"; g = f.create_group(d)
            av = np.zeros((n, 16, 16, 3), np.uint8); wr = np.zeros((n, 16, 16, 3), np.uint8)
            for i in range(n):
                av[i, 0, 0] = 255; av[i, 5, 5, 0] = i; wr[i, 0, 0] = 200; wr[i, 5, 5, 0] = i
            g["agentview_rgb"] = av; g["eye_in_hand_rgb"] = wr
            q = R.from_euler("xyz", rng.normal(0, .3, (n, 3))).as_quat()
            rs = np.concatenate([np.zeros((n, 2)), rng.normal(0, .1, (n, 3)), q], 1).astype(np.float32)
            g["robot_states"] = rs; g["actions"] = np.concatenate([rng.uniform(-1, 1, (n, 6)), rng.choice([-1., 1.], (n, 1))], 1).astype(np.float32)
            instr = f"do {task}"
            prompts = [f"{instr} Memory: frame {i}" if (dom == "mem" and i >= 2) else instr for i in range(n)]
            g.create_dataset("prompts", data=prompts, dtype=h5py.string_dtype("utf-8"))
            g["has_memory"] = np.array([p != instr for p in prompts])
            g.attrs.update(instruction=instr, split="train", domain=dom)
            eps.append(dict(domain=dom, task=task, demo=d, split="train", length=n, shard=rel))
(root / "manifest.json").write_text(json.dumps(dict(status="complete", split_lock_sha256="x", episodes=eps, shards={})))
out = W / "out"
r = subprocess.run([sys.executable, f"{S}/build_xvla_v2_dataset.py", "--shards-root", str(W), "--out", str(out), "--allow-lock-mismatch"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-2000:]); assert r.returncode == 0
from shard_handler import register
H = register(X)
# exact checks with identity aug
for arm in ("f0", "f1"):
    for dom in ("base", "mem"):
        meta = json.loads((out / f"metas_{arm}" / f"{dom}.json").read_text())
        h = H(meta, 3); checked = 0
        with h5py.File(meta["actions_h5"]) as fa:
            for ti, ep in enumerate(meta["datalist"]):
                a = fa[ep["shard"]][ep["demo"]]["abs_action_6d"][()]
                for s in h.iter_episode(ti, num_actions=30, training=True, image_aug=lambda p: torch.from_numpy(np.array(p))):
                    img = s["image_input"]; main, wrist = img[0].numpy(), img[1].numpy()
                    assert main[15, 15, 0] == 255 and main[0, 0, 0] == 0, "agentview not flipped 180"
                    assert wrist[0, 0, 0] == 200, "wrist must stay raw"
                    shown = int(main[10, 10, 0])  # (5,5) -> (10,10) after 180 flip of 16x16
                    assert int(wrist[5, 5, 0]) == shown
                    prop = s["abs_trajectory"][0, :9].numpy()
                    idx = shown - 1
                    assert np.allclose(prop, a[idx, :9], atol=1e-5), "proprio must be abs[idx] for image idx+1"
                    if idx + 30 <= len(a) - 1:
                        assert np.allclose(s["abs_trajectory"][1:, :9].numpy(), a[idx + 1: idx + 31, :9], atol=1e-5)
                    m = re.search(r"frame (\d+)", s["language_instruction"])
                    if arm == "f1" and dom == "mem" and shown >= 2:
                        assert m and int(m.group(1)) == shown, "prompt must be the one of the shown frame"
                        assert bool(s["has_memory"]) == True
                    else:
                        assert m is None and not bool(s["has_memory"])
                    assert bool(s["is_mem"]) == (dom == "mem")
                    checked += 1
        print(arm, dom, "samples checked", checked)
# full upstream reader + loader
from datasets.dataset import InfiniteDataReader
from torch.utils.data import DataLoader
rd = InfiniteDataReader(str(out / "metas_f1"), num_actions=30, training=True, action_mode="ee6d")
dl = DataLoader(rd, batch_size=8, num_workers=2)
it = iter(dl); mem = n = 0
for _ in range(20):
    b = next(it); mem += int(b["is_mem"].sum()); n += len(b["is_mem"])
print({k: (tuple(v.shape) if hasattr(v, "shape") else len(v)) for k, v in b.items()}, "domain ids", b["domain_id"].unique().tolist(), "mem share", mem / n)
assert b["proprio"].shape[-1] == 20 and b["action"].shape[1:] == (30, 20) and b["image_input"].shape[1:] == (3, 3, 224, 224)
assert set(b["domain_id"].tolist()) == {3}
print(json.dumps(json.loads((out / "build_receipt.json").read_text())["sampler_frames"]))
print("ALL OK")
