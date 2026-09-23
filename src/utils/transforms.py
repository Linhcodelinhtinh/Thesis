"""Pure NumPy spatial transform utilities for robot observations and states."""

from typing import Union
import numpy as np


def quat2axisangle(quat: Union[np.ndarray, list]) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle [ax, ay, az].

    Complies exactly with robosuite.utils.transform_utils.quat2axisangle
    without requiring MuJoCo or robosuite imports.

    Args:
        quat: 4-element array [x, y, z, w].

    Returns:
        3-element array [ax, ay, az] where norm is rotation angle in radians.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}")

    if q[3] < 0:
        q = -q

    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sqrt(np.maximum(1.0 - q[3] * q[3], 0.0))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return (q[:3] / s) * angle


def compute_eef_6d_state(eef_pos: np.ndarray, eef_quat: np.ndarray) -> np.ndarray:
    """Compute standard 6D EEF state [eef_pos (3,), axis_angle (3,)] for VLA input.

    Args:
        eef_pos: 3D end-effector Cartesian position [x, y, z].
        eef_quat: 4D end-effector quaternion [x, y, z, w].

    Returns:
        6D array [x, y, z, ax, ay, az].
    """
    pos = np.asarray(eef_pos, dtype=np.float32).flatten()
    if pos.shape != (3,):
        raise ValueError(f"Expected eef_pos of shape (3,), got {pos.shape}")
    axis_angle = quat2axisangle(eef_quat).astype(np.float32)
    return np.concatenate([pos, axis_angle])
