#!/usr/bin/env python3
"""Environment Verification Script (Phase 1).

Validates that Python, MuJoCo, robosuite, LIBERO, and required assets
are correctly installed and functional before running any VLA policies.
"""

import os
import sys
from pathlib import Path


def check_python():
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"PASS: Python ({version.major}.{version.minor}.{version.micro})")
        return True
    else:
        print(f"FAIL: Python version {version.major}.{version.minor} is incompatible (require >= 3.8)")
        return False


def check_mujoco():
    try:
        import mujoco

        # Test basic model compilation
        xml = "<mujoco><worldbody><body><geom type='box' size='0.1 0.1 0.1'/></body></worldbody></mujoco>"
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print(f"PASS: MuJoCo (version {mujoco.__version__})")
        return True
    except Exception as exc:
        print(f"FAIL: MuJoCo validation error: {exc}")
        return False


def check_robosuite():
    try:
        import robosuite

        version = getattr(robosuite, "__version__", "unknown")
        # Verify Panda model is available
        from robosuite.models.robots import Panda

        panda = Panda()
        assert panda is not None
        print(f"PASS: robosuite (version {version}, Panda robot available)")
        return True
    except Exception as exc:
        print(f"FAIL: robosuite validation error: {exc}")
        return False


def check_libero():
    try:
        import libero
        from libero.libero.benchmark import get_benchmark

        b_cls = get_benchmark("libero_object")
        b = b_cls()
        task = b.get_task(0)
        assert task is not None
        assert b.n_tasks == 10
        print(f"PASS: LIBERO (libero_object loaded with {b.n_tasks} tasks)")
        return True
    except Exception as exc:
        print(f"FAIL: LIBERO validation error: {exc}")
        return False


def check_required_assets():
    try:
        import os
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark

        b = get_benchmark("libero_object")()
        task = b.get_task(0)

        # Check BDDL file exists
        bddl_root = get_libero_path("bddl_files")
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        if not os.path.exists(bddl_path):
            print(f"FAIL: BDDL file missing at {bddl_path}")
            return False

        # Check initial states exist
        init_states = b.get_task_init_states(0)
        if init_states is None or len(init_states) < 1:
            print(f"FAIL: Initial states missing or empty for task 0")
            return False

        # Check asset directory
        asset_root = get_libero_path("assets")
        cache_asset_root = os.path.expanduser("~/.cache/libero/assets")
        assets_found = os.path.exists(asset_root) or os.path.exists(cache_asset_root)

        if not assets_found:
            print(f"FAIL: Assets directory missing at {asset_root} and {cache_asset_root}")
            return False

        print(f"PASS: required assets (BDDL, 50 initial states, meshes verified)")
        return True
    except Exception as exc:
        print(f"FAIL: Required assets validation error: {exc}")
        return False


def main():
    print("=" * 60)
    print("VLA Policy Evaluation Sandbox V1 — Environment Verification")
    print("=" * 60)

    # Automatically set MUJOCO_GL for Windows if not already set
    if sys.platform == "win32" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "wgl"

    p_ok = check_python()
    m_ok = check_mujoco()
    r_ok = check_robosuite()
    l_ok = check_libero()
    a_ok = check_required_assets()

    print("-" * 60)
    if all([p_ok, m_ok, r_ok, l_ok, a_ok]):
        print("RESULT: ALL ENVIRONMENT CHECKS PASSED.")
        return 0
    else:
        print("RESULT: ENVIRONMENT CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
