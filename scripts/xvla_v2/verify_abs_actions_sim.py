#!/usr/bin/env python3
"""Optional receipt (local WSL, LIBERO reference venv): analytic abs targets vs simulator.

Replays recorded delta actions of a LIBERO HDF5 demo in the real LIBERO env and,
at every step, compares delta_to_abs6d(obs before step, action) with what the
OSC controller actually stores (controller.goal_pos / goal_ori) — the quantity the
upstream X-VLA rel2abs.py collects. Prints max/median errors per demo.

  python scripts/xvla_v2/verify_abs_actions_sim.py \
     --hdf5 resources/demonstrations/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 \
     --demos demo_0 demo_1 --out experiments/audits/xvla_abs_sim_receipt.json

Container check already done on robosuite 1.4.0 Panda OSC_POSE (Lift, 300 random steps):
max pos err 0.78 mm (median 0.28 mm), rot err median 0.08 deg / max 0.13 deg with the panda_rz_m90 frame,
~90 deg with the identity frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abs_actions import delta_to_abs6d, rot6d_to_mat, rotation_angle_deg  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demos", nargs="+", default=["demo_0"])
    p.add_argument("--out")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    report = {}
    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        pinfo = json.loads(data.attrs["problem_info"])
        bddl = data.attrs.get("bddl_file_name")
        bddl = bddl.decode() if isinstance(bddl, bytes) else bddl
        if not bddl or not Path(bddl).is_file():
            stem = Path(args.hdf5).name.replace("_demo.hdf5", ".bddl")
            hits = list(Path(get_libero_path("bddl_files")).rglob(stem))
            bddl = str(hits[0])
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        for d in args.demos:
            g = data[d]
            actions = g["actions"][()]
            env.reset()
            obs = env.set_init_state(g["states"][0])
            rs, gp, go = [], [], []
            for a in actions:
                rs.append(np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]))
                obs, *_ = env.step(a)
                c = env.env.robots[0].controller
                gp.append(c.goal_pos.copy()); go.append(c.goal_ori.copy())
            out, info = delta_to_abs6d(np.asarray(rs), actions)
            pe = np.linalg.norm(out[:, :3] - np.asarray(gp), axis=1)
            re = rotation_angle_deg(rot6d_to_mat(out[:, 3:9].astype(np.float64)), np.asarray(go))
            report[d] = dict(steps=len(actions), pos_err_mm_max=float(pe.max() * 1e3),
                             pos_err_mm_median=float(np.median(pe) * 1e3),
                             rot_err_deg_max=float(re.max()), rot_err_deg_median=float(np.median(re)), **info)
            print(d, json.dumps(report[d]))
        env.close()
    report = dict(hdf5=args.hdf5, task=pinfo.get("language_instruction"), demos=report,
                  passed=all(r["pos_err_mm_max"] < 2.0 and r["rot_err_deg_max"] < 1.0 for r in report.values()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    print("passed", report["passed"])


if __name__ == "__main__":
    main()
