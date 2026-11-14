#!/usr/bin/env python
"""Real Mem simulator in its pinned environment; stdout is reserved for IPC."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import asdict


def main():
    # Redirect native-library prints too, not just Python's stdout.
    transport = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.simulator.libero_mem_env import LiberoMemEnv
    from src.simulator.mem_ipc import decode_array, serialize_obs

    def send(value):
        transport.write(json.dumps(value, allow_nan=False) + "\n")
        transport.flush()

    def info_dict(info):
        return {k: asdict(v) if k == "eval_result" else v for k, v in info.items()}

    def policy_observation(obs):
        # Joint positions are recorded by the replay generator in-process.
        # The policy IPC contract predates that field and does not consume it.
        return serialize_obs({key: value for key, value in obs.items() if key != "robot0_joint_pos"})

    env, previous_id = None, 0
    send(dict(status="ready", protocol_version=1))
    try:
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                request_id = request["id"]
                if type(request_id) is not int or request_id != previous_id + 1:
                    raise ValueError("Nonsequential request ID")
                previous_id = request_id
                command = request["command"]
                if command == "init":
                    if env is not None:
                        raise RuntimeError("Environment already initialized")
                    env = LiberoMemEnv(task_name=request["task_name"], split_lock=request["split_lock"],
                                       split=request["split"], purpose=request["purpose"])
                    out = dict(task_name=env.task_name, protocol=env.protocol,
                        target_cycles=env.target_cycles, horizon=env.horizon, num_episodes=len(env.episodes))
                elif command == "reset":
                    if env is None:
                        raise RuntimeError("Environment not initialized")
                    obs = env.reset(initial_state_id=request["initial_state_id"])
                    out = dict(observation=policy_observation(obs), info=info_dict(env.last_info))
                elif command == "step":
                    if env is None:
                        raise RuntimeError("Environment not initialized")
                    obs, reward, done, info = env.step(decode_array(request["action"]))
                    out = dict(observation=policy_observation(obs), reward=float(reward), done=bool(done), info=info_dict(info))
                elif command == "close":
                    send(dict(id=request_id, status="ok"))
                    break
                else:
                    raise ValueError("Unknown worker command")
                send(dict(id=request_id, status="ok", **out))
            except Exception as exc:
                send(dict(id=request_id, status="error", error=str(exc), traceback=traceback.format_exc()))
                return 1
    finally:
        if env is not None:
            env.close()
        transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
