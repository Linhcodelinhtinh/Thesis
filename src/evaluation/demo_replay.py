"""Official LIBERO Demonstration Replay Engine (Phase 3).

Reconstructs the EXACT demonstration environment from HDF5 metadata (env_args,
problem_name, model_xml, init_state), replays recorded actions, tracks state
divergence (replayed_state vs recorded_state), and verifies the official
success predicate per Phase 3 specifications.
"""

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import yaml


@contextlib.contextmanager
def _legacy_model_compatibility():
    """Ensure cross-version robosuite compatibility when replaying demonstrations

    recorded with older robosuite versions (e.g. 1.4.0) on newer environments.
    """
    try:
        import robosuite.models.tasks.task as task_module
        import robosuite.robots.single_arm as sa_module

        orig_gen = task_module.Task.generate_id_mappings
        orig_setup = sa_module.SingleArm.setup_references

        def safe_generate_id_mappings(self, sim):
            models = [model for model in self.mujoco_objects]
            for robot in self.mujoco_robots:
                models += [robot] + robot.models

            worldbody = self.mujoco_arena.root.find("worldbody")
            exclude_bodies = ["table", "left_eef_target", "right_eef_target"]
            top_level_bodies = [
                body.attrib.get("name")
                for body in worldbody.findall("body")
                if body.attrib.get("name") not in exclude_bodies
            ]
            models.extend(top_level_bodies)

            self._instances_to_ids = {}
            self._geom_ids_to_instances = {}
            self._site_ids_to_instances = {}
            self._classes_to_ids = {}
            self._geom_ids_to_classes = {}
            self._site_ids_to_classes = {}

            for model in models:
                if isinstance(model, str):
                    body_name = model
                    visual_group_number = 1
                    body_id = sim.model.body_name2id(body_name)
                    inst, cls = body_name, body_name
                    geom_ids = task_module.get_subtree_geom_ids_by_group(sim.model, body_id, visual_group_number)
                    id_groups = [geom_ids, []]
                else:
                    cls = str(type(model)).split("'")[1].split(".")[-1]
                    inst = model.name
                    available_geoms = set(sim.model.geom_names)
                    available_sites = set(sim.model.site_names)
                    geoms = [g for g in model.visual_geoms + model.contact_geoms if g in available_geoms]
                    sites = [s for s in model.sites if s in available_sites]
                    id_groups = [
                        task_module.get_ids(sim=sim, elements=geoms, element_type="geom"),
                        task_module.get_ids(sim=sim, elements=sites, element_type="site"),
                    ]
                group_types = ("geom", "site")
                ids_to_instances = (self._geom_ids_to_instances, self._site_ids_to_instances)
                ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)

                assert inst not in self._instances_to_ids, f"Instance {inst} already registered; should be unique"
                self._instances_to_ids[inst] = {}
                if cls not in self._classes_to_ids:
                    self._classes_to_ids[cls] = {group_type: [] for group_type in group_types}

                for ids, group_type, ids_to_inst, ids_to_cls in zip(
                    id_groups, group_types, ids_to_instances, ids_to_classes
                ):
                    self._instances_to_ids[inst][group_type] = ids
                    self._classes_to_ids[cls][group_type] += ids
                    for idn in ids:
                        assert idn not in ids_to_inst, f"ID {idn} already registered; should be unique"
                        ids_to_inst[idn] = inst
                        ids_to_cls[idn] = cls

        def safe_setup_references(self):
            for arm in self.arms:
                if self.has_gripper[arm]:
                    if "gripper0_finger_joint1" in self.sim.model.joint_names:
                        self.gripper[arm].idn = self.idn
            if hasattr(self.robot_model, "_sites"):
                prefix = getattr(self.robot_model, "naming_prefix", "")
                self.robot_model._sites = [
                    s for s in self.robot_model._sites
                    if s in self.sim.model.site_names or f"{prefix}{s}" in self.sim.model.site_names
                ]
            orig_setup(self)

        task_module.Task.generate_id_mappings = safe_generate_id_mappings
        sa_module.SingleArm.setup_references = safe_setup_references
        try:
            yield
        finally:
            task_module.Task.generate_id_mappings = orig_gen
            sa_module.SingleArm.setup_references = orig_setup
    except (ImportError, AttributeError):
        yield


@dataclass
class ReplayMetrics:
    """Telemetry and tracking results for a demonstration replay episode."""
    demo_name: str
    num_steps: int
    success: bool
    env_success: bool
    tracking_success: bool
    final_reward: float
    max_tracking_error: float
    mean_tracking_error: float
    tracking_tolerance: float
    divergence_count: int
    tracking_errors: List[float] = field(default_factory=list)
    action_dim: int = 7
    action_range: Tuple[float, float] = (-1.0, 1.0)
    rendered_frames: List[np.ndarray] = field(default_factory=list)


class DemonstrationReplayer:
    """Engine for loading, reconstructing, and replaying official LIBERO demonstrations."""

    def __init__(self, hdf5_path: Union[str, Path]) -> None:
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Demonstration HDF5 file not found at: {self.hdf5_path}")

    def verify_hdf5_hash(
        self,
        manifest_path: Union[str, Path] = "resources/manifests/demonstrations_manifest.yaml",
    ) -> bool:
        """Verify the cryptographic SHA-256 hash of the demonstration HDF5 against manifest."""
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Demonstrations manifest not found at: {manifest_file}")

        with open(manifest_file, "r") as f:
            manifest_data = yaml.safe_load(f)

        expected_hash = None
        for entry in manifest_data.get("demonstrations", []):
            rel_path = entry.get("relative_path", "")
            if rel_path and self.hdf5_path.as_posix().endswith(rel_path.replace("\\", "/")):
                expected_hash = entry.get("sha256")
                break
            if entry.get("task_name") and entry["task_name"] in self.hdf5_path.name:
                expected_hash = entry.get("sha256")
                break

        if expected_hash is None:
            raise KeyError(
                f"No entry found in manifest {manifest_file} matching {self.hdf5_path.name}"
            )

        hasher = hashlib.sha256()
        with open(self.hdf5_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {self.hdf5_path.name}: "
                f"expected {expected_hash}, got {actual_hash}. Cryptographic integrity check failed!"
            )
        return True

    def list_demos(self) -> List[str]:
        """List all demonstration episode keys in the HDF5 file."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError("HDF5 file does not contain root group 'data'.")
            return sorted(list(f["data"].keys()))

    def get_env_metadata(self) -> Dict[str, Any]:
        """Extract exact environment arguments and problem metadata from HDF5 attributes."""
        with h5py.File(self.hdf5_path, "r") as f:
            data_grp = f["data"]
            if "env_args" in data_grp.attrs:
                raw_env_args = data_grp.attrs["env_args"]
                if isinstance(raw_env_args, (bytes, str)):
                    return json.loads(raw_env_args)
                return dict(raw_env_args)
            elif "problem_info" in data_grp.attrs:
                return dict(data_grp.attrs["problem_info"])
            return {}

    def replay_episode(
        self,
        demo_name: str = "demo_0",
        tracking_tolerance: float = 0.5,
        strict_tolerance: bool = False,
        render: bool = False,
    ) -> ReplayMetrics:
        """Replay a recorded demonstration episode inside an exactly reconstructed environment.

        Args:
            demo_name: Name of the demonstration key (e.g. 'demo_0').
            tracking_tolerance: Threshold for state divergence tolerance.
            strict_tolerance: If True, requires max_tracking_error <= tracking_tolerance for pass.
                              If False, requires mean_tracking_error <= tracking_tolerance.
            render: Whether to capture rendered frames during replay.

        Returns:
            ReplayMetrics with tracking errors, step count, and success status.
        """
        try:
            from libero.libero.envs import TASK_MAPPING
        except ImportError as err:
            raise ImportError(
                f"Failed to import official LIBERO: {err}. Ensure LIBERO is on sys.path."
            ) from err

        with h5py.File(self.hdf5_path, "r") as f:
            data_grp = f["data"]
            if demo_name not in data_grp:
                raise KeyError(
                    f"Episode '{demo_name}' not found. Available: {list(data_grp.keys())}"
                )

            ep_grp = data_grp[demo_name]
            actions = np.array(ep_grp["actions"][()])
            states = np.array(ep_grp["states"][()])
            model_xml = ep_grp.attrs.get("model_file", None)
            bddl_attr = data_grp.attrs.get("bddl_file_name", "")

            # Read env_args from dataset metadata
            env_metadata = self.get_env_metadata()
            problem_name = env_metadata.get("problem_name", None)
            env_kwargs = env_metadata.get("env_kwargs", {})

        # Resolve BDDL file path
        from libero.libero import get_libero_path
        bddl_candidate = bddl_attr or env_metadata.get("bddl_file", "") or env_kwargs.get("bddl_file_name", "")
        bddl_base = os.path.basename(bddl_candidate)
        bddl_dir = get_libero_path("bddl_files")
        resolved_bddl = None
        for root, _, files in os.walk(bddl_dir):
            if bddl_base in files:
                resolved_bddl = os.path.join(root, bddl_base)
                break
            # Also check alternative naming like pick_the_ -> pick_up_the_
            if bddl_base.replace("pick_the_", "pick_up_the_") in files:
                resolved_bddl = os.path.join(root, bddl_base.replace("pick_the_", "pick_up_the_"))
                break

        if resolved_bddl is not None:
            env_kwargs["bddl_file_name"] = resolved_bddl

        env = None
        try:
            with _legacy_model_compatibility():
                if problem_name is None or problem_name not in TASK_MAPPING:
                    from libero.libero.envs import OffScreenRenderEnv
                    env = OffScreenRenderEnv(**env_kwargs)
                else:
                    env = TASK_MAPPING[problem_name](**env_kwargs)

                # Reset environment using exact model_xml with dynamic asset resolution
                if model_xml is not None:
                    model_xml = self._postprocess_xml(model_xml)
                    env.reset_from_xml_string(model_xml)

                env.sim.reset()
                # Restore exact initial physical state
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()

            num_actions = len(actions)
            tracking_errors: List[float] = []
            rendered_frames: List[np.ndarray] = []
            divergence_count = 0

            for t, action in enumerate(actions):
                obs, reward, done, info = env.step(action)

                if render:
                    frame = obs.get("agentview_image")
                    if frame is not None:
                        # Robosuite OffScreenRenderEnv outputs raw OpenGL buffer which is vertically inverted.
                        # Flip vertically (frame[::-1]) to match upright video convention (consistent with LiberoEnv.render).
                        rendered_frames.append(frame[::-1])

                if t < len(states) - 1:
                    replayed_state = env.sim.get_state().flatten()
                    expected_state = states[t + 1]
                    err = float(np.linalg.norm(replayed_state - expected_state))
                    tracking_errors.append(err)
                    if err > tracking_tolerance:
                        divergence_count += 1

            if hasattr(env, "check_success"):
                env_success = bool(env.check_success())
            elif hasattr(env, "_check_success"):
                env_success = bool(env._check_success())
            else:
                env_success = False

            max_err = float(np.max(tracking_errors)) if tracking_errors else 0.0
            mean_err = float(np.mean(tracking_errors)) if tracking_errors else 0.0

            # Enforce tracking tolerance in pass/fail decision
            if strict_tolerance:
                tracking_success = (max_err <= tracking_tolerance)
            else:
                tracking_success = (mean_err <= tracking_tolerance)

            overall_success = env_success and tracking_success

            action_min = float(np.min(actions))
            action_max = float(np.max(actions))

            return ReplayMetrics(
                demo_name=demo_name,
                num_steps=num_actions,
                success=overall_success,
                env_success=env_success,
                tracking_success=tracking_success,
                final_reward=float(reward),
                max_tracking_error=max_err,
                mean_tracking_error=mean_err,
                tracking_tolerance=tracking_tolerance,
                divergence_count=divergence_count,
                tracking_errors=tracking_errors,
                action_dim=actions.shape[1],
                action_range=(action_min, action_max),
                rendered_frames=rendered_frames,
            )

        finally:
            if env is not None:
                env.close()

    def _postprocess_xml(self, xml_str: str) -> str:
        """Resolve mesh and texture paths for both robosuite and LIBERO assets."""
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

        # Ensure robot0_right_center exists on robot0_link0 if expected by newer composite controllers
        link0 = tree.find(".//body[@name='robot0_link0']")
        if link0 is not None and link0.find("./site[@name='robot0_right_center']") is None:
            ET.SubElement(
                link0,
                "site",
                name="robot0_right_center",
                pos="0 0 0",
                size="0.01",
                rgba="1 0.3 0.3 1",
                group="2",
            )

        return ET.tostring(tree, encoding="utf8").decode("utf8")
