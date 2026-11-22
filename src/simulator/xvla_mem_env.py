"""Explicit absolute-control derivative of the locked Mem evaluation protocol.

Reset, warmup, state restoration and evaluator remain owned by LiberoMemEnv.
Only the post-warmup controller mode and native policy chunk contract differ.
"""
from copy import deepcopy
import numpy as np
from src.simulator.libero_mem_env import LiberoMemEnv
from src.simulator.process_libero_mem_env import ProcessLiberoMemEnv


class XVLAMemEnv(LiberoMemEnv):
    def decode_xvla(self, chunk):
        from scripts.run_x_vla_libero import load_upstream, ROOT
        from src.models.x_vla import validate_chunk
        if not hasattr(self, "_xvla_processor"):
            self._xvla_processor = load_upstream(ROOT / "external/X-VLA").LiberoAbsActionProcessor()
        chunk = validate_chunk(chunk)[:, :10]
        decoded = self._xvla_processor.action_6d_to_axisangle(chunk).astype(np.float32)
        decoded[:, -1] = np.where(chunk[:, 9] > .5, 1., -1.)
        return decoded

    def reset(self, initial_state_id=0):
        # Every reset must settle in delta mode, including repeated episodes.
        self.upstream.robots[0].controller.use_delta = True
        obs = super().reset(initial_state_id)
        controller = self.upstream.robots[0].controller
        if (controller.name != "OSC_POSE" or controller.impedance_mode != "fixed"
                or self.upstream.control_freq != 20 or self.upstream.horizon != 1020):
            raise RuntimeError("Unexpected XVLA Mem controller contract")
        controller.use_delta = False
        self._controller_info()
        return obs

    def _controller_info(self):
        controller = self.upstream.robots[0].controller
        rotation = controller.ee_ori_mat
        self.last_info["xvla_controller"] = dict(
            position=np.asarray(controller.ee_pos).tolist(),
            rotation6d=np.concatenate((rotation[:, 0], rotation[:, 1])).tolist(),
            use_delta=bool(controller.use_delta))

    def step(self, action):
        if self.upstream.robots[0].controller.use_delta:
            raise RuntimeError("XVLA requires absolute OSC after warmup")
        obs, reward, done, _ = super().step(action)
        self._controller_info()
        return obs, reward, done, dict(self.last_info)


class ProcessXVLAMemEnv(ProcessLiberoMemEnv):
    worker_script = "scripts/xvla_v2/mem_sim_worker.py"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # An explicit derivative, never mutate the canonical protocol on disk.
        self.protocol = deepcopy(self.protocol)
        self.protocol["simulation"]["controller"]["control_delta"] = False
        self.protocol["rollout"]["action_chunk"] = 30

    def reset(self, initial_state_id=0):
        obs = super().reset(initial_state_id)
        if self.last_info.get("xvla_controller", {}).get("use_delta") is not False:
            raise RuntimeError("Worker did not attest absolute XVLA control")
        return obs

    def decode_xvla(self, chunk):
        from src.simulator.mem_ipc import encode_array
        return np.asarray(self._send_command(dict(command="decode_xvla", chunk=encode_array(chunk)))["actions"],
                          dtype=np.float32)
