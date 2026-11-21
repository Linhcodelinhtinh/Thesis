"""Canonical LIBERO resources and the upstream X-VLA simulator setup."""

import os
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

from src.simulator.asset_validator import verify_task_asset_integrity


REFERENCE_VERSIONS = {
    "robosuite": "1.4.0",
    "bddl": "1.0.1",
    "robomimic": "0.2.0",
    "mujoco": "3.2.3",
    "numpy": "1.22.4",
    "json-numpy": "2.1.0",
}


def configure_reference(checkout, config_dir):
    if sys.platform != "linux" or sys.version_info[:3] != (3, 8, 13):
        raise RuntimeError("X-VLA LIBERO requires the Linux Python 3.8.13 simulator environment")
    versions = {name: version(name) for name in REFERENCE_VERSIONS}
    if versions != REFERENCE_VERSIONS:
        raise RuntimeError(f"Unexpected LIBERO runtime {versions}. Expected {REFERENCE_VERSIONS}")
    package = checkout / "libero" / "libero"
    paths = {
        "benchmark_root": package,
        "bddl_files": package / "bddl_files",
        "init_states": package / "init_files",
        "assets": package / "assets",
    }
    for path in paths.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump({key: str(path) for key, path in paths.items()}), encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(checkout))
    from libero.libero import __file__ as package_file, get_libero_path

    if Path(package_file).resolve().parent != package:
        raise RuntimeError(f"LIBERO imported from an unexpected package: {package_file}")
    for key, path in paths.items():
        if Path(get_libero_path(key)).resolve() != path:
            raise RuntimeError(f"LIBERO resolved the wrong {key}")
    return versions


def verify_task(suite, task_id):
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    receipt = verify_task_asset_integrity(bddl, init)
    if not receipt["init_verified"]:
        raise FileNotFoundError(init)
    states = suite.get_task_init_states(task_id)
    if states.ndim != 2 or states.shape[0] != 50:
        raise ValueError(f"Expected 50 official states for {task.name}, got shape {states.shape}")
    return receipt


def open_episode(evaluator, suite, task_id, initial_state_id):
    # The author creates a fresh env, restores the state and settles in delta mode
    # before enabling absolute OSC. Reuse the entire sequence from the pinned client.
    env, instruction, obs = evaluator._init_env(suite, task_id, initial_state_id)
    try:
        robot, = env.env.robots
        controller = robot.controller
        robot_name = type(robot.robot_model).__name__
        gripper_name = type(robot.gripper).__name__
        # LIBERO's task classes mount the default Panda differently per scene.
        if (robot_name not in {"Panda", "MountedPanda", "OnTheGroundPanda"} or gripper_name != "PandaGripper"
                or robot.action_dim != 7 or controller.name != "OSC_POSE" or controller.use_delta
                or controller.impedance_mode != "fixed" or env.env.control_freq != 20
                or env.env.horizon != 1000):
            raise RuntimeError(f"Unexpected X-VLA simulator runtime: {robot_name}/{gripper_name}, "
                               f"{robot.action_dim}D {controller.name}, use_delta={controller.use_delta}, "
                               f"impedance={controller.impedance_mode}, {env.env.control_freq} Hz, "
                               f"horizon={env.env.horizon}")
    except Exception:
        env.close()
        raise
    return env, instruction, obs


def render_metadata(env):
    from OpenGL import GL

    context = env.sim._render_context_offscreen.gl_ctx
    context.make_current()
    renderer = GL.glGetString(GL.GL_RENDERER)
    return {"gl_context": type(context).__module__,
            "gl_renderer": renderer.decode("utf-8", "replace") if renderer else None}
