"""Strict array/observation transport shared by Mem worker and policy proxy."""
import base64
import numpy as np

OBSERVATIONS = {"agentview_image": (256, 256, 3), "robot0_eye_in_hand_image": (256, 256, 3),
                "robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,)}


def encode_array(array):
    array = np.asarray(array)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("IPC requires finite numeric arrays")
    return dict(dtype=array.dtype.str, shape=list(array.shape), data=base64.b64encode(array.tobytes()).decode("ascii"))


def decode_array(value):
    dtype, shape = np.dtype(value["dtype"]), value["shape"]
    if dtype.kind not in "fiu" or not isinstance(shape, list) or len(shape) > 3 or any(type(n) is not int or n < 0 or n > 4096 for n in shape):
        raise ValueError("Invalid IPC array dtype/shape")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("IPC byte count differs from shape")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite IPC array")
    return result


def validate_observation(obs):
    if set(obs) != set(OBSERVATIONS):
        raise ValueError("Unexpected observation keys; only policy-visible allowlist is permitted")
    for key, shape in OBSERVATIONS.items():
        a = obs[key]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Invalid observation: " + key)
        if "image" in key and a.dtype != np.uint8:
            raise ValueError("Camera must remain uint8 RGB")
        if "image" not in key and a.dtype.kind != "f":
            raise ValueError("Proprioception must remain floating point")
    return obs


def serialize_obs(obs):
    validate_observation(obs)
    return {key: encode_array(value) for key, value in obs.items()}


def deserialize_obs(obs):
    return validate_observation({key: decode_array(value) for key, value in obs.items()})
