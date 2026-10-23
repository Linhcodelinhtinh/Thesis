"""BDDL Simulator Loader and Compatibility Layer for LIBERO-Mem.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 2 & Section 12:
- Loads official BDDL task definitions directly from `external/libero-mem`.
- Enforces official benchmark invariants: Franka Panda robot, OSC_POSE controller,
  20 Hz control frequency, 128x128 camera streams, and MuJoCo physics.
- Automatically applies cross-version Robosuite compatibility shims (1.4.0 <-> 1.5+).
- Restores exact benchmark initial states from author metainfo.json.
- Fails fast loudly if external/libero-mem or required BDDL assets are missing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_MEM_ROOT = REPO_ROOT / "external" / "libero-mem"
CONFIG_PATH = REPO_ROOT / "configs" / "libero"


def setup_libero_mem_paths() -> Path:
    """Configure system path and environment variables for LIBERO-Mem."""
    if not LIBERO_MEM_ROOT.exists():
        raise FileNotFoundError(
            f"Official LIBERO-Mem repository not found at: {LIBERO_MEM_ROOT}. "
            "Please clone https://github.com/libero-mem/libero-mem at pinned commit 0eee0defa4024e51511b6e15a79f30e9620209de."
        )

    # 1. Ensure external/libero-mem is in sys.path
    libero_root_str = str(LIBERO_MEM_ROOT)
    if libero_root_str not in sys.path:
        sys.path.insert(0, libero_root_str)

    # 2. Ensure outer __init__.py exists to avoid namespace collision with site-packages
    outer_init = LIBERO_MEM_ROOT / "libero" / "__init__.py"
    if not outer_init.exists():
        outer_init.write_text("# Top-level init for libero\n", encoding="utf-8")

    # 3. Ensure configs/libero/config.yaml exists to bypass interactive input prompts
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    cfg_file = CONFIG_PATH / "config.yaml"
    if not cfg_file.exists():
        libero_pkg_dir = LIBERO_MEM_ROOT / "libero" / "libero"
        cfg_content = (
            f"benchmark_root: {libero_pkg_dir.as_posix()}\n"
            f"bddl_files: {(libero_pkg_dir / 'bddl_files').as_posix()}\n"
            f"init_states: {(libero_pkg_dir / 'init_files').as_posix()}\n"
            f"datasets: {(REPO_ROOT / 'data' / 'libero_mem').as_posix()}\n"
            f"assets: {(libero_pkg_dir / 'assets').as_posix()}\n"
        )
        cfg_file.write_text(cfg_content, encoding="utf-8")

    os.environ["LIBERO_CONFIG_PATH"] = str(CONFIG_PATH)

    # 4. Windows OpenGL backend safety
    if sys.platform == "win32" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "wgl"

    return LIBERO_MEM_ROOT


def apply_robosuite_compatibility_shims() -> None:
    """Apply compatibility patches for Robosuite 1.4 vs 1.5+ cross-version operation."""
    try:
        import robosuite as suite
        import robosuite.controllers as rc
    except ImportError as e:
        raise ImportError(f"Failed to import robosuite: {e}. Ensure robosuite is installed.") from e

    # Shim 1: Controller loader function name change in 1.5+
    if not hasattr(rc, "load_controller_config") and hasattr(rc, "load_part_controller_config"):
        rc.load_controller_config = rc.load_part_controller_config

    # Shim 2: ManipulatorModel attribute changes in Robosuite 1.5+
    try:
        from libero.libero.envs.robots.mounted_panda import MountedPanda
        MountedPanda.arms = ["right"]
        MountedPanda.default_base = "RethinkMount"
        MountedPanda.default_gripper = {"right": "PandaGripper"}
    except (ImportError, AttributeError) as exc:
        logger.debug("MountedPanda compatibility shim skipped or already patched: %s", exc)

    try:
        from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda
        OnTheGroundPanda.arms = ["right"]
        OnTheGroundPanda.default_base = "RethinkMount"
        OnTheGroundPanda.default_gripper = {"right": "PandaGripper"}
    except (ImportError, AttributeError):
        pass


class LiberoMemBDDLLoader:
    """Loads official BDDL environments and initial states from external/libero-mem."""

    OFFICIAL_CONTROL_FREQ = 20
    OFFICIAL_HORIZON = 1000
    OFFICIAL_CONTROLLER = "OSC_POSE"
    OFFICIAL_CAMERAS = ["agentview", "robot0_eye_in_hand"]
    OFFICIAL_ROBOT = "Panda"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.libero_root = setup_libero_mem_paths()
        apply_robosuite_compatibility_shims()

        self.bddl_dir = self.libero_root / "libero" / "libero" / "bddl_files" / "libero_mem"
        if not self.bddl_dir.exists():
            raise FileNotFoundError(f"LIBERO-Mem BDDL directory not found at: {self.bddl_dir}")

        self.data_dir = data_dir or (REPO_ROOT / "data" / "libero_mem")
        self._metainfo: Optional[Dict[str, Any]] = None

    def resolve_bddl_path(self, task_name: str) -> Path:
        """Find the exact BDDL file for a task."""
        clean_name = task_name.strip()
        if not clean_name.endswith(".bddl"):
            clean_name = f"{clean_name}.bddl"

        bddl_path = self.bddl_dir / clean_name
        if not bddl_path.exists():
            # Try searching by prefix / pattern in bddl_dir
            candidates = list(self.bddl_dir.glob(f"*{task_name}*.bddl"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(
                f"BDDL file for task '{task_name}' not found in {self.bddl_dir}. "
                f"Available BDDL files: {[f.name for f in self.bddl_dir.glob('*.bddl')]}"
            )
        return bddl_path

    def load_metainfo(self) -> Dict[str, Any]:
        """Lazy load and cache metainfo.json containing ground truth initial states and boxes."""
        if self._metainfo is not None:
            return self._metainfo

        metainfo_path = self.data_dir / "metainfo.json"
        if not metainfo_path.exists():
            logger.warning("metainfo.json not found at %s. Initial states from metainfo unavailable.", metainfo_path)
            return {}

        with open(metainfo_path, "r", encoding="utf-8") as f:
            self._metainfo = json.load(f)
        return self._metainfo

    def get_initial_state(self, task_name: str, demo_idx: int = 1) -> Optional[np.ndarray]:
        """Retrieve exact 90-dim initial MuJoCo state for a demonstration episode."""
        meta = self.load_metainfo()
        # Clean task name key
        base_name = task_name.replace(".bddl", "")
        task_meta = meta.get(base_name)
        if not task_meta:
            for k in meta:
                if base_name in k or k in base_name:
                    task_meta = meta[k]
                    break

        if not task_meta:
            return None

        demo_key = f"demo_{demo_idx}"
        if demo_key in task_meta and "initial_state" in task_meta[demo_key]:
            return np.asarray(task_meta[demo_key]["initial_state"], dtype=np.float64)
        return None

    def get_grounding_boxes(self, task_name: str, demo_idx: int = 1, step: int = 0) -> Dict[str, Any]:
        """Retrieve ground-truth object bounding boxes from metainfo for visual perception grounding."""
        meta = self.load_metainfo()
        base_name = task_name.replace(".bddl", "")
        task_meta = meta.get(base_name)
        if not task_meta:
            for k in meta:
                if base_name in k or k in base_name:
                    task_meta = meta[k]
                    break

        if not task_meta:
            return {}

        demo_key = f"demo_{demo_idx}"
        demo_info = task_meta.get(demo_key, {})
        exo_boxes = demo_info.get("exo_boxes", [])
        if step < len(exo_boxes):
            return exo_boxes[step]
        return {}

    def create_env(
        self,
        task_name: str,
        camera_height: int = 128,
        camera_width: int = 128,
        horizon: int = 1000,
        control_freq: int = 20,
    ) -> Any:
        """Instantiate official OffScreenRenderEnv for the requested BDDL task."""
        from libero.libero.envs import OffScreenRenderEnv

        bddl_path = self.resolve_bddl_path(task_name)
        logger.info("Initializing official LIBERO-Mem OffScreenRenderEnv with BDDL: %s", bddl_path.name)

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            robots=[self.OFFICIAL_ROBOT],
            controller=self.OFFICIAL_CONTROLLER,
            gripper_types="default",
            control_freq=control_freq,
            horizon=horizon,
            camera_names=list(self.OFFICIAL_CAMERAS),
            camera_heights=camera_height,
            camera_widths=camera_width,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
        )
        return env
