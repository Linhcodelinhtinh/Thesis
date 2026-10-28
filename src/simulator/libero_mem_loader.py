"""Strict author-source simulator loader; no robot shims or fallback physics."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types
import subprocess

from src.evaluation.libero_mem_protocol import load_protocol, task_spec, ROOT

LIBERO_MEM_ROOT = ROOT / "external/libero-mem"


def setup_libero_mem_paths(source_root=LIBERO_MEM_ROOT):
    source_root = Path(source_root).resolve()
    load_protocol(source_root=source_root)
    existing = sys.modules.get("libero.libero")
    if existing is not None and not str(Path(existing.__file__).resolve()).startswith(str(source_root) + os.sep):
        raise RuntimeError("Another LIBERO is imported; use the isolated Mem simulator process")
    config = ROOT / "experiments/runtime/libero_mem_config"
    config.mkdir(parents=True, exist_ok=True)
    package = source_root / "libero/libero"
    # JSON is valid YAML, and avoids platform-specific YAML quoting of paths.
    paths = dict(benchmark_root=str(package), bddl_files=str(package / "bddl_files"),
                 init_states=str(package / "init_files"), datasets=str(ROOT / "data/libero_mem"),
                 assets=str(package / "assets"))
    (config / "config.yaml").write_text(json.dumps(paths), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    if "libero" not in sys.modules:
        namespace = types.ModuleType("libero")
        namespace.__path__ = [str(source_root / "libero")]
        sys.modules["libero"] = namespace
    return source_root


class LiberoMemBDDLLoader:
    def __init__(self, source_root=LIBERO_MEM_ROOT):
        self.libero_root = Path(source_root)
        self.protocol = load_protocol(source_root=self.libero_root)

    def resolve_bddl_path(self, task_name):
        return self.libero_root / task_spec(self.protocol, task_name)["bddl"]

    def create_env(self, task_name):
        revision = subprocess.check_output(["git", "-C", str(self.libero_root), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(self.libero_root), "diff", "--name-only", "HEAD", "--",
            "libero/libero", "scripts/mem_3_refine_demonstration.py"], text=True).strip()
        if revision != self.protocol["upstream_revision"] or changes:
            raise RuntimeError("Mem simulator requires the pinned, unmodified author source checkout")
        runtime = self.protocol["runtime"]
        for name in ("robosuite", "numpy", "bddl", "mujoco", "robomimic"):
            if importlib.metadata.version(name) != runtime[name]:
                raise RuntimeError("Use the isolated Mem simulator runtime; version mismatch for " + name)
        setup_libero_mem_paths(self.libero_root)
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.controllers import load_controller_config
        controller = load_controller_config(default_controller="OSC_POSE")
        if controller != self.protocol["simulation"]["controller"]:
            raise ValueError("OSC_POSE controller impedance/scales differ from the locked protocol")
        env = OffScreenRenderEnv(bddl_file_name=str(self.resolve_bddl_path(task_name)), robots=["Panda"],
            controller="OSC_POSE", gripper_types="default", control_freq=20, horizon=1020,
            camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
            camera_depths=False, camera_segmentations=None, ignore_done=True, reward_shaping=True)
        if (env.env.action_dim != 7 or env.env.control_freq != 20 or
            type(env.env.robots[0].robot_model).__name__ != self.protocol["simulation"]["actual_robot_class"]):
            env.close()
            raise ValueError("Unknown action dimension/control frequency")
        return env
