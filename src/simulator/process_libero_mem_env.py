"""Sequential, checked IPC to the real isolated LIBERO-Mem simulator."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json
import queue
import subprocess
import threading

import numpy as np
from src.evaluation.libero_mem_metrics import SubgoalEvaluationResult
from src.evaluation.libero_mem_protocol import load_protocol, task_spec, ROOT
from src.training.split_lock import load_split_lock, mem_episode_rows
from src.simulator.mem_ipc import encode_array, deserialize_obs


def wsl_path(path):
    path = Path(path).absolute()
    if not path.drive or len(path.drive) != 2 or path.drive[1] != ":":
        raise ValueError("--use-wsl requires an absolute Windows drive path")
    return "/mnt/" + path.drive[0].lower() + "/" + path.relative_to(path.anchor).as_posix()


class ProcessLiberoMemEnv:
    use_real_sim = True

    def __init__(self, task_name="T1", *, split_lock, split="val", purpose="evaluation",
                 python_executable=None, use_wsl=False, wsl_distro="Ubuntu-24.04", timeout_seconds=300):
        if not python_executable:
            if use_wsl:
                python_executable = f"{wsl_path(ROOT)}/.venv_libero_reference/bin/python"
            else:
                raise ValueError("Explicit isolated simulator Python required")
        self.protocol = load_protocol()
        self.task_name = task_spec(self.protocol, task_name)["name"]
        self.split_lock = load_split_lock(split_lock)
        self.episodes = mem_episode_rows(self.split_lock, self.task_name, split)
        self.timeout = timeout_seconds
        self._responses, self._stderr = queue.Queue(), deque(maxlen=60)
        self._lock = threading.Lock()
        self._closed, self._request_id, self._step = False, 0, None
        self.last_info = {}
        if use_wsl:
            command = ["wsl", "-d", wsl_distro, "--cd", wsl_path(ROOT), "--",
                       "bash", "-c", f"export MUJOCO_GL=osmesa && exec '{python_executable}' scripts/libero_mem_sim_worker.py"]
            worker_lock = wsl_path(split_lock)
        else:
            command = [str(python_executable), str(ROOT / "scripts/libero_mem_sim_worker.py")]
            worker_lock = str(Path(split_lock).absolute())
        self._process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._readers = [threading.Thread(target=reader, daemon=True)
                         for reader in (self._read_stdout, self._read_stderr)]
        for reader in self._readers:
            reader.start()
        try:
            ready = self._receive()
            if ready != dict(status="ready", protocol_version=1):
                raise RuntimeError("Unexpected simulator handshake")
            meta = self._send_command(dict(command="init", task_name=task_name, split_lock=worker_lock,
                                            split=split, purpose=purpose))
            if meta["protocol"] != self.protocol or meta["task_name"] != self.task_name or meta["num_episodes"] != len(self.episodes):
                raise ValueError("Worker protocol/task/split differs from caller")
            spec = task_spec(self.protocol, self.task_name)
            if meta["target_cycles"] != spec["target_cycles"] or meta["horizon"] != self.protocol["rollout"]["horizon_policy_actions"]:
                raise ValueError("Worker cycles/horizon differ from locked protocol")
            self.target_cycles, self.horizon = meta["target_cycles"], meta["horizon"]
            self.num_episodes = meta["num_episodes"]
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        for line in self._process.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _read_stderr(self):
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def _receive(self):
        try:
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError("Simulator response timeout: " + "\n".join(self._stderr)) from exc
        if line is None:
            raise RuntimeError("Simulator worker exited: " + "\n".join(self._stderr))
        return json.loads(line)

    def _send_command(self, command):
        with self._lock:
            if self._closed:
                raise RuntimeError("Simulator worker is closed")
            self._request_id += 1
            self._process.stdin.write(json.dumps(dict(command, id=self._request_id), allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = self._receive()
            if response.get("id") != self._request_id:
                raise RuntimeError("Stale/mismatched simulator response ID")
            if response.get("status") != "ok":
                raise RuntimeError("Simulator failure: " + response.get("traceback", str(response)))
            return response

    def _observation(self, response, expected_step):
        info = dict(response["info"])
        row = self.selected_episode
        if info["step"] != expected_step or info["initial_state_sha256"] != row["initial_state_sha256"] or info["source_demo"] != row["demo"]:
            raise ValueError("Stale observation or wrong initial state")
        if info["split_lock_sha256"] != self.split_lock["lock_sha256"] or info["protocol_id"] != self.protocol["protocol_id"]:
            raise ValueError("Worker episode provenance mismatch")
        info["eval_result"] = SubgoalEvaluationResult(**info["eval_result"])
        self.last_info = info
        return deserialize_obs(response["observation"])

    def reset(self, initial_state_id=0):
        if type(initial_state_id) is not int or not 0 <= initial_state_id < len(self.episodes):
            raise ValueError("State index outside selected split")
        self.selected_episode = self.episodes[initial_state_id]
        response = self._send_command(dict(command="reset", initial_state_id=initial_state_id))
        obs = self._observation(response, 0)
        self._step = 0
        return obs

    def step(self, action):
        action = np.asarray(action)
        if self._step is None or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Reset required and action must be finite native action7")
        response = self._send_command(dict(command="step", action=encode_array(action)))
        obs = self._observation(response, self._step + 1)
        self._step += 1
        return obs, response["reward"], response["done"], dict(self.last_info)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for reader in self._readers:
            reader.join(timeout=1)
        for stream in (self._process.stdout, self._process.stderr):
            stream.close()
