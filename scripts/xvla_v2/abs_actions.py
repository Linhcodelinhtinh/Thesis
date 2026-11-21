"""Delta OSC_POSE -> absolute EEF-6D targets, computed from recorded robot state.

Why analytic instead of simulator replay
----------------------------------------
robosuite 1.4 OSC_POSE (control_delta=True, position/orientation limits null,
no interpolator) sets, at every policy step,

    goal_pos = ee_pos + clip(a[:3], -1, 1) * 0.05
    goal_ori = R(axisangle = clip(a[3:6], -1, 1) * 0.5) @ ee_ori_mat
               (goal_ori is left unchanged when all three rotation deltas are exactly 0)

where ee_pos / ee_ori_mat are read from the sim *at the moment the action is
applied* (OSC.set_goal calls self.update() first). This is exactly what the
upstream X-VLA rel2abs.py collects from controller.goal_pos / goal_ori.

The V2 shards store, per row i, robot_states[i] = [gripper_qpos(2), eef_pos(3),
eef_quat_xyzw(4)] observed right before shard action i (base: obs[i] -> actions[i+1],
Mem: obs[i] -> actions[i], both already re-indexed in the shard). So the goal of
row i is a pure function of (robot_states[i], actions[i]).

Frame pitfall: robosuite 1.4 `robot0_eef_quat` is the quaternion of the
`right_hand` BODY, while the controller uses the `grip_site` frame. For the
Panda gripper the site frame is the body frame rotated by -90 deg about z
(panda_gripper.xml: right_gripper quat="0.707107 0 0 -0.707107"). We apply that
offset, so the output is in the same frame as controller.ee_ori_mat, which is what
the X-VLA LIBERO client feeds as proprio and what the checkpoint was trained on.
Verify with verify_abs_actions_sim.py (simulator) or check_xvla_alignment.py
(checkpoint agreement) before training.

6D layout: X-VLA's LIBERO client uses Mat_to_Rotate6D = concat(R[:, 0], R[:, 1])
(first two COLUMNS). datasets/utils.quat_to_rotate6d uses a row-interleaved
layout, which is NOT what the LIBERO checkpoint/client use. We use the client layout.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

POS_SCALE = 0.05
ROT_SCALE = 0.5

# grip_site frame expressed in right_hand body frame (Rz(-90 deg)).
SITE_FROM_BODY = {
    "panda_rz_m90": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "identity": np.eye(3),
}


def mat_to_rot6d(mats: np.ndarray) -> np.ndarray:
    """[N,3,3] -> [N,6] = concat(first column, second column) (X-VLA LIBERO client layout)."""
    return np.concatenate([mats[:, :3, 0], mats[:, :3, 1]], axis=-1)


def rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    """Inverse of mat_to_rot6d with Gram-Schmidt, same as the client's Rotate6D_to_AxisAngle."""
    a1, a2 = r6d[..., 0:3], r6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotation matrices [...,3,3]."""
    rel = np.einsum("...ji,...jk->...ik", a, b)
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def site_rotations(robot_states: np.ndarray, frame: str = "panda_rz_m90") -> np.ndarray:
    quat_xyzw = np.asarray(robot_states[:, 5:9], dtype=np.float64)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    return R.from_quat(quat_xyzw).as_matrix() @ SITE_FROM_BODY[frame]


def delta_to_abs6d(robot_states: np.ndarray, actions: np.ndarray, frame: str = "panda_rz_m90"):
    """Return (abs_action_6d [N,10] float32, info dict).

    abs_action_6d[i] = [goal_pos(3), rot6d(goal_ori)(6), raw gripper action(1)],
    i.e. the same content upstream rel2abs.py + AxisAngle_to_Rotate6D produce.
    """
    robot_states = np.asarray(robot_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n = len(actions)
    if robot_states.shape != (n, 9) or actions.shape != (n, 7):
        raise ValueError(f"bad shapes robot_states={robot_states.shape} actions={actions.shape}")
    if not (np.isfinite(robot_states).all() and np.isfinite(actions).all()):
        raise ValueError("non-finite robot_states/actions")
    qn = np.linalg.norm(robot_states[:, 5:9], axis=1)
    if np.abs(qn - 1).max() > 1e-3:
        raise ValueError(f"quaternion norm off by {np.abs(qn - 1).max():.2e}")

    a = np.clip(actions[:, :6], -1.0, 1.0)
    ee_pos = robot_states[:, 2:5]
    ee_rot = site_rotations(robot_states, frame)

    goal_pos = ee_pos + a[:, :3] * POS_SCALE
    goal_rot = np.empty((n, 3, 3))
    zero_rot = 0
    prev = ee_rot[0]  # controller reset sets goal_ori to the current orientation
    for i in range(n):
        if np.all(a[i, 3:6] == 0.0):  # robosuite: math.isclose(x, 0) per element
            goal_rot[i] = prev
            zero_rot += 1
        else:
            goal_rot[i] = R.from_rotvec(a[i, 3:6] * ROT_SCALE).as_matrix() @ ee_rot[i]
        prev = goal_rot[i]

    out = np.concatenate([goal_pos, mat_to_rot6d(goal_rot), actions[:, 6:7]], axis=-1).astype(np.float32)

    # Tracking diagnostics: how far the next observed pose is from the commanded goal.
    pos_track = np.linalg.norm(goal_pos[:-1] - ee_pos[1:], axis=1) if n > 1 else np.zeros(0)
    rot_track = rotation_angle_deg(goal_rot[:-1], ee_rot[1:]) if n > 1 else np.zeros(0)
    info = dict(
        frames=n,
        zero_rotation_rows=int(zero_rot),
        clipped_rows=int((np.abs(actions[:, :6]) > 1.0).any(axis=1).sum()),
        pos_track_median_m=float(np.median(pos_track)) if n > 1 else 0.0,
        pos_track_p95_m=float(np.percentile(pos_track, 95)) if n > 1 else 0.0,
        rot_track_median_deg=float(np.median(rot_track)) if n > 1 else 0.0,
        rot_track_p95_deg=float(np.percentile(rot_track, 95)) if n > 1 else 0.0,
        gripper_values=sorted({float(x) for x in np.unique(np.round(actions[:, 6], 3))})[:6],
    )
    return out, info
