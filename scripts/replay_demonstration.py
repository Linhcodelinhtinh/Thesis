"""Official LIBERO Demonstration Replay Script (Phase 3).

Replays an official demonstration episode using the exact environment reconstruction
per Phase 3 and SRS.md specifications:
  HDF5
   ├── env_args (or env_info)
   ├── problem_info
   ├── model_file / XML
   └── demo_0
        ├── actions
        ├── states
        └── init_state
            ↓
  construct EXACT demo environment
            ↓
  restore exact simulator state
            ↓
  env.step(recorded_action)
            ↓
  compare replayed_state ↔ recorded_state
            ↓
  check_success()
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def postprocess_demonstration_xml(xml_str: str) -> str:
    """Postprocess MuJoCo XML string to resolve both robosuite and LIBERO assets.

    Official libero.libero.utils.utils.postprocess_model_xml only replaces
    robosuite paths, leaving author's hardcoded `/Users/yifengz/.../chiliocosm/assets/`
    paths intact, which fails when replaying on any other system.
    This function dynamically re-routes:
      1. `robosuite/...` -> robosuite installation directory
      2. `chiliocosm/assets/...` or `libero/assets/...` -> local LIBERO assets cache/dir.
    """
    import xml.etree.ElementTree as ET
    import robosuite
    from libero.libero import get_libero_path

    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is not None:
        robosuite_dir = os.path.split(robosuite.__file__)[0].replace("\\", "/")
        libero_assets_dir = get_libero_path("assets")
        if not os.path.exists(libero_assets_dir):
            candidate = os.path.expanduser("~/.cache/libero/assets")
            if os.path.exists(candidate):
                libero_assets_dir = candidate
        libero_assets_dir = libero_assets_dir.replace("\\", "/")

        for elem in asset.findall("mesh") + asset.findall("texture"):
            old_path = elem.get("file")
            if not old_path:
                continue
            old_path = old_path.replace("\\", "/")
            if "robosuite" in old_path:
                parts = old_path.split("/")
                ind = max(i for i, v in enumerate(parts) if v == "robosuite")
                elem.set("file", robosuite_dir + "/" + "/".join(parts[ind + 1 :]))
            elif "assets" in old_path:
                parts = old_path.split("/")
                ind = max(i for i, v in enumerate(parts) if v == "assets")
                elem.set("file", libero_assets_dir + "/" + "/".join(parts[ind + 1 :]))

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def replay_demo(
    hdf5_path: str,
    demo_name: str = "demo_0",
    tolerance: float = 0.05,
    verbose: bool = True,
):
    """Replay a recorded demonstration episode inside an exactly reconstructed environment."""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found at: {hdf5_path}")

    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        if demo_name not in data_grp:
            raise KeyError(f"Episode '{demo_name}' not found. Available: {list(data_grp.keys())}")

        env_args = json.loads(data_grp.attrs["env_args"])
        problem_name = env_args["problem_name"]
        env_kwargs = env_args["env_kwargs"]

        # Resolve BDDL file path
        bddl_attr = data_grp.attrs.get("bddl_file_name", "")
        bddl_candidate = bddl_attr if bddl_attr else env_kwargs.get("bddl_file_name", "")
        bddl_base = os.path.basename(bddl_candidate)

        # Look in official libero bddl_files directory
        bddl_dir = get_libero_path("bddl_files")
        resolved_bddl = None
        for root, _, files in os.walk(bddl_dir):
            if bddl_base in files:
                resolved_bddl = os.path.join(root, bddl_base)
                break

        if resolved_bddl is not None:
            env_kwargs["bddl_file_name"] = resolved_bddl

        d0 = data_grp[demo_name]
        actions = np.array(d0["actions"][()])
        states = np.array(d0["states"][()])
        model_xml = d0.attrs.get("model_file", None)

    if verbose:
        print(f"Constructing exact environment: problem='{problem_name}'")
        print(f"BDDL file: {env_kwargs.get('bddl_file_name')}")
        print(f"Actions count: {len(actions)}, State dim: {states.shape[1]}")

    env = TASK_MAPPING[problem_name](**env_kwargs)

    try:
        if model_xml is not None:
            model_xml = postprocess_demonstration_xml(model_xml)
            env.reset_from_xml_string(model_xml)

        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        tracking_errors = []
        divergences = 0

        for t in range(len(actions)):
            obs, reward, done, info = env.step(actions[t])

            if t < len(states) - 1:
                cur_state = env.sim.get_state().flatten()
                exp_state = states[t + 1]
                err = float(np.linalg.norm(cur_state - exp_state))
                tracking_errors.append(err)
                if err > tolerance:
                    divergences += 1

        if hasattr(env, "check_success"):
            success = bool(env.check_success())
        elif hasattr(env, "_check_success"):
            success = bool(env._check_success())
        else:
            success = False
        max_err = float(np.max(tracking_errors)) if tracking_errors else 0.0
        mean_err = float(np.mean(tracking_errors)) if tracking_errors else 0.0

        if verbose:
            print(f"Replay completed: {len(actions)} steps.")
            print(f"Success Predicate: {success}")
            print(f"Mean Tracking Error: {mean_err:.6f}")
            print(f"Max Tracking Error: {max_err:.6f}")
            print(f"Divergence Steps (> {tolerance}): {divergences}")

        return {
            "demo_name": demo_name,
            "num_steps": len(actions),
            "success": success,
            "mean_tracking_error": mean_err,
            "max_tracking_error": max_err,
            "divergence_count": divergences,
        }

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay official LIBERO demonstration.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="resources/demonstrations/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
        help="Path to demonstration HDF5 file.",
    )
    parser.add_argument("--demo", type=str, default="demo_0", help="Demonstration episode key.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tracking tolerance threshold.")
    args = parser.parse_args()

    results = replay_demo(args.hdf5, args.demo, args.tolerance)
    if not results["success"]:
        print("ERROR: Demonstration replay failed the success predicate!", file=sys.stderr)
        sys.exit(1)
    print("SUCCESS: Demonstration replay passed official verification!")
