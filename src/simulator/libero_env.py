"""Official LIBERO Simulation Environment Wrapper (Phase 2 Hardening).

Provides standardized interface for loading official LIBERO tasks,
restoring exact benchmark initial states, stepping 7-DoF robot actions,
rendering benchmark camera streams, enforcing strict-mode invariants,
and querying official success predicates per SRS.md and AGENTS.md.
"""

import os
import platform
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure Windows OpenGL uses WGL backend before importing mujoco/robosuite
if sys.platform == "win32" and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "wgl"


class BenchmarkMode(str, Enum):
    """Execution modes for benchmark evaluation per SRS.md Section 3.3."""
    STRICT_LIBERO = "STRICT-LIBERO"
    LIBERO_DERIVED = "LIBERO-DERIVED"


class StrictInvariantViolationError(Exception):
    """Raised when an environment configuration violates official STRICT-LIBERO benchmark invariants."""
    pass


class LiberoEnv:
    """Standardized wrapper around the official LIBERO simulation environment.

    Adheres strictly to the LIBERO evaluation benchmark protocol:
    - Mode A (STRICT-LIBERO): Locks Franka Panda robot, OSC_POSE controller,
      20 Hz control frequency, 1000 episode horizon, 128x128 resolution,
      official cameras ('agentview', 'robot0_eye_in_hand'), and asset hash verification.
    - Mode B (LIBERO-DERIVED): Permits custom exploratory parameters with
      explicit non-comparable provenance tagging.
    """

    OFFICIAL_CONTROL_FREQ = 20
    OFFICIAL_HORIZON = 1000
    OFFICIAL_CONTROLLER = "OSC_POSE"
    OFFICIAL_CAMERAS = ["agentview", "robot0_eye_in_hand"]
    OFFICIAL_IMAGE_SIZE = (128, 128)
    OFFICIAL_ROBOT = "Panda"

    def __init__(
        self,
        benchmark_name: str = "libero_object",
        task_id: int = 0,
        mode: Union[BenchmarkMode, str] = BenchmarkMode.STRICT_LIBERO,
        control_freq: int = 20,
        horizon: int = 1000,
        controller: str = "OSC_POSE",
        camera_names: Optional[List[str]] = None,
        camera_height: int = 128,
        camera_width: int = 128,
        verify_assets: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as err:
            raise ImportError(
                f"Failed to import official LIBERO package: {err}. "
                "Ensure LIBERO is installed in the current environment."
            ) from err

        if isinstance(mode, str):
            try:
                self.mode = BenchmarkMode(mode)
            except ValueError:
                raise ValueError(
                    f"Invalid benchmark mode '{mode}'. Must be one of: "
                    f"{[m.value for m in BenchmarkMode]}"
                )
        else:
            self.mode = mode

        if camera_names is None:
            camera_names = list(self.OFFICIAL_CAMERAS)

        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.control_freq = control_freq
        self.horizon = horizon
        self.controller_name = controller
        self.camera_names = list(camera_names)
        self.camera_height = camera_height
        self.camera_width = camera_width

        # Enforce strict benchmark invariants if in STRICT-LIBERO mode
        if self.mode == BenchmarkMode.STRICT_LIBERO:
            self._enforce_strict_invariants()

        # Retrieve benchmark registry programmatically
        benchmark_cls = get_benchmark(benchmark_name)
        self.benchmark = benchmark_cls()

        if task_id < 0 or task_id >= self.benchmark.n_tasks:
            raise ValueError(
                f"Invalid task_id {task_id} for benchmark '{benchmark_name}'. "
                f"Must be between 0 and {self.benchmark.n_tasks - 1}."
            )

        self.task = self.benchmark.get_task(task_id)
        self.language_instruction: str = self.task.language

        # Resolve full BDDL path
        bddl_root = get_libero_path("bddl_files")
        self.bddl_file_path = os.path.join(
            bddl_root, self.task.problem_folder, self.task.bddl_file
        )
        if not os.path.exists(self.bddl_file_path):
            raise FileNotFoundError(
                f"Official BDDL file not found at: {self.bddl_file_path}"
            )

        # Resolve initial states file candidate for provenance checking
        init_root = get_libero_path("init_states")
        init_file_candidate = os.path.join(
            init_root, self.task.problem_folder, f"{self.task.name}.init"
        )
        if not os.path.exists(init_file_candidate):
            init_file_candidate = os.path.join(
                init_root, self.task.problem_folder, f"{self.task.name}.pruned_init"
            )

        # Verify cryptographic asset integrity per Rule 6
        self.asset_integrity_info: Dict[str, Any] = {}
        if verify_assets:
            from src.simulator.asset_validator import verify_task_asset_integrity

            self.asset_integrity_info = verify_task_asset_integrity(
                bddl_file_path=self.bddl_file_path,
                init_file_path=init_file_candidate if os.path.exists(init_file_candidate) else None,
            )

        # Load official initial states array
        self.init_states: np.ndarray = self.benchmark.get_task_init_states(task_id)
        if self.init_states is None or len(self.init_states) == 0:
            raise ValueError(
                f"Failed to load official initial states for task {task_id} in {benchmark_name}."
            )

        # Instantiate official LIBERO OffScreenRenderEnv
        self._env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file_path,
            robots=[self.OFFICIAL_ROBOT],
            controller=controller,
            gripper_types="default",
            control_freq=control_freq,
            horizon=horizon,
            camera_names=self.camera_names,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            has_renderer=False,
            has_offscreen_renderer=True,
            **kwargs,
        )

        self.current_step = 0
        self.current_init_state_id: Optional[int] = None

    def _enforce_strict_invariants(self) -> None:
        """Lock and validate strict invariants required by STRICT-LIBERO mode."""
        if self.control_freq != self.OFFICIAL_CONTROL_FREQ:
            raise StrictInvariantViolationError(
                f"control_freq={self.control_freq} violates STRICT-LIBERO invariant "
                f"(expected {self.OFFICIAL_CONTROL_FREQ} Hz). Use mode='LIBERO-DERIVED' "
                "for altered frequencies per SRS.md Section 3.3."
            )

        if self.horizon != self.OFFICIAL_HORIZON:
            raise StrictInvariantViolationError(
                f"horizon={self.horizon} violates STRICT-LIBERO invariant "
                f"(expected {self.OFFICIAL_HORIZON} steps). Use mode='LIBERO-DERIVED' "
                "for modified horizon per SRS.md Section 8.4."
            )

        if self.controller_name != self.OFFICIAL_CONTROLLER:
            raise StrictInvariantViolationError(
                f"controller='{self.controller_name}' violates STRICT-LIBERO invariant "
                f"(expected '{self.OFFICIAL_CONTROLLER}'). Per SRS.md Section 7.1."
            )

        if sorted(self.camera_names) != sorted(self.OFFICIAL_CAMERAS):
            raise StrictInvariantViolationError(
                f"camera_names={self.camera_names} violates STRICT-LIBERO invariant "
                f"(expected {self.OFFICIAL_CAMERAS}). Per SRS.md Section 9.1."
            )

        if (self.camera_height, self.camera_width) != self.OFFICIAL_IMAGE_SIZE:
            raise StrictInvariantViolationError(
                f"image_size=({self.camera_height}, {self.camera_width}) violates STRICT-LIBERO "
                f"invariant (expected {self.OFFICIAL_IMAGE_SIZE}). Per SRS.md Section 9.1."
            )

    @property
    def is_strict_mode(self) -> bool:
        return self.mode == BenchmarkMode.STRICT_LIBERO

    @property
    def task_name(self) -> str:
        return self.task.name

    @property
    def num_initial_states(self) -> int:
        return len(self.init_states)

    def reset(self, initial_state_id: int = 0) -> Dict[str, np.ndarray]:
        """Reset the environment to an exact official benchmark initial state.

        Args:
            initial_state_id: Index of the initial state from official benchmark array (0 to 49).

        Returns:
            Dictionary of initial observations.
        """
        if initial_state_id < 0 or initial_state_id >= len(self.init_states):
            raise IndexError(
                f"initial_state_id {initial_state_id} out of bounds for task {self.task_name}. "
                f"Available states: 0 to {len(self.init_states) - 1}."
            )

        self.current_init_state_id = initial_state_id
        self.current_step = 0

        target_state = self.init_states[initial_state_id]
        obs = self._env.set_init_state(target_state)

        # Validate observation sanity
        self._validate_observation(obs)
        return obs

    def step(
        self, action: Union[np.ndarray, List[float]]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one 7-DoF control step in the environment.

        Args:
            action: 7-dimensional action vector:
                    [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    where gripper: +1 for open, -1 for close (or vice versa per model adapter).

        Returns:
            Tuple of (obs, reward, done, info).
        """
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (7,):
            raise ValueError(
                f"Action dimension mismatch: expected (7,), got {action_arr.shape}. "
                "Fail fast per ADR-0005."
            )
        if np.any(np.isnan(action_arr)) or np.any(np.isinf(action_arr)):
            raise ValueError(
                f"Invalid action containing NaN or Inf values: {action_arr}. "
                "Fail fast per ADR-0005."
            )

        obs, reward, done, info = self._env.step(action_arr)
        self.current_step += 1

        if info is None:
            info = {}

        success = self.check_success()
        info["success"] = success
        info["step"] = self.current_step
        info["task_name"] = self.task_name
        info["mode"] = self.mode.value

        if self.current_step >= self.horizon:
            done = True
            info["timeout"] = True

        return obs, float(reward), bool(done), info

    def check_success(self) -> bool:
        """Evaluate the official BDDL task success predicate."""
        return bool(self._env.check_success())

    def get_physics_state(self) -> np.ndarray:
        """Return the complete flattened physical state vector of the simulation."""
        return self._env.env.sim.get_state().flatten()

    def get_robot_kinematics_info(self) -> Dict[str, Any]:
        """Inspect robot kinematics, joints, actuators, and gripper configuration."""
        robot = self._env.env.robots[0]
        gripper = robot.gripper["right"] if isinstance(robot.gripper, dict) else robot.gripper
        m = self._env.env.sim.model
        return {
            "robot_name": robot.name,
            "arm_joints": list(robot.robot_joints),
            "num_arm_joints": len(robot.robot_joints),
            "gripper_type": type(gripper).__name__,
            "gripper_joints": list(gripper.joints),
            "num_gripper_joints": len(gripper.joints),
            "actuator_names": [m.actuator_id2name(i) for i in range(m.nu)],
            "num_actuators": int(m.nu),
            "controller_type": self.controller_name,
        }

    def get_camera_info(self, camera_name: str) -> Dict[str, Any]:
        """Inspect camera extrinsics, intrinsics, and parent body attachment."""
        sim = self._env.env.sim
        m = sim.model
        if camera_name not in self.camera_names and camera_name not in [
            m.camera_id2name(i) for i in range(m.ncam)
        ]:
            raise ValueError(f"Unknown camera '{camera_name}'.")

        cam_id = m.camera_name2id(camera_name)
        body_id = int(m.cam_bodyid[cam_id])
        body_name = m.body_id2name(body_id) if body_id >= 0 else "world"
        pos = m.cam_pos[cam_id].copy()
        quat = m.cam_quat[cam_id].copy()
        fovy = float(m.cam_fovy[cam_id])

        return {
            "camera_name": camera_name,
            "camera_id": int(cam_id),
            "parent_body_id": body_id,
            "parent_body_name": body_name,
            "pos": pos,
            "quat": quat,
            "fovy": fovy,
            "resolution": (self.camera_height, self.camera_width),
        }

    def get_provenance_metadata(self) -> Dict[str, Any]:
        """Inspect execution environment provenance and assign certification tier."""
        import robosuite

        py_version = sys.version_info
        is_py38 = (py_version.major == 3 and py_version.minor == 8)
        robosuite_ver = getattr(robosuite, "__version__", "unknown")
        is_rs_140 = (robosuite_ver == "1.4.0")
        is_linux = (sys.platform.startswith("linux"))

        is_official_reference_stack = (is_py38 and is_rs_140 and is_linux)

        if is_official_reference_stack and self.is_strict_mode:
            execution_tier = "STRICT-LIBERO-CERTIFIED"
            is_benchmark_comparable = True
        else:
            execution_tier = "HOST-SMOKE-ONLY (NON-COMPARABLE)"
            is_benchmark_comparable = False

        return {
            "mode": self.mode.value,
            "execution_tier": execution_tier,
            "is_benchmark_comparable": is_benchmark_comparable,
            "is_official_reference_stack": is_official_reference_stack,
            "python_version": f"{py_version.major}.{py_version.minor}.{py_version.micro}",
            "platform": platform.platform(),
            "robosuite_version": robosuite_ver,
            "backend": os.environ.get("MUJOCO_GL", "default"),
        }

    def render(self, camera_name: str = "agentview") -> np.ndarray:
        """Render and return current RGB camera frame."""
        if camera_name not in self.camera_names:
            raise ValueError(
                f"Unknown camera '{camera_name}'. Available cameras: {self.camera_names}"
            )
        sim = self._env.env.sim
        raw_img = sim.render(
            camera_name=camera_name,
            height=self.camera_height,
            width=self.camera_width,
            depth=False,
        )
        return raw_img[::-1]

    def _validate_observation(self, obs: Dict[str, Any]) -> None:
        """Ensure observation contains all required benchmark channels."""
        for cam in self.camera_names:
            key = f"{cam}_image"
            if key not in obs:
                raise KeyError(f"Expected camera observation '{key}' not in obs.")
            img = obs[key]
            expected_shape = (self.camera_height, self.camera_width, 3)
            if img.shape != expected_shape:
                raise ValueError(
                    f"Camera '{key}' shape mismatch: expected {expected_shape}, got {img.shape}"
                )

    def close(self) -> None:
        """Cleanly terminate the simulator and release OpenGL / MuJoCo resources."""
        if hasattr(self, "_env") and self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

    def __enter__(self) -> "LiberoEnv":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
