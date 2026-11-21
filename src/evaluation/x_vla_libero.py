"""Episode loop and artifacts for the X-VLA authors' LIBERO protocol."""

import json
import time

import imageio.v2 as imageio
import numpy as np

from src.simulator.x_vla_libero import open_episode, render_metadata


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_episode(upstream, evaluator, suite, task_id, state_id, policy, output):
    env, instruction, obs = open_episode(evaluator, suite, task_id, state_id)
    try:
        output.mkdir(parents=True)
        policy.reset()
        renderer = render_metadata(env)
        controller = env.env.robots[0].controller
        actions, goals, measured, simulation_ms = [], [], [], []
        total_reward = 0.0
        with imageio.get_writer(str(output / "video.mp4"), fps=30) as video:
            for _ in range(evaluator.eval_horizon):
                obs["robo_pos"] = controller.ee_pos
                obs["robo_ori"] = policy.processor.Mat_to_Rotate6D(controller.ee_ori_mat)
                action = policy.step(obs, instruction)
                video.append_data(upstream._flip_agentview(obs["agentview_image"]))
                start = time.perf_counter()
                obs, reward, done, _ = env.step(action)
                simulation_ms.append((time.perf_counter() - start) * 1000)
                total_reward += float(reward)
                actions.append(action)
                goals.append(np.concatenate((controller.goal_pos, controller.goal_ori.ravel())))
                measured.append(np.concatenate((controller.ee_pos, controller.ee_ori_mat.ravel())))
                if done:
                    break

        success = bool(env.check_success())
        if success != bool(done):
            raise RuntimeError("LIBERO done disagrees with the official success predicate")
        raw = np.stack(policy.raw_chunks)
        with (output / "model_output.jsonl").open("w", encoding="utf-8") as stream:
            for chunk_id, (chunk, latency) in enumerate(zip(raw, policy.inference_ms)):
                stream.write(json.dumps({"chunk_id": chunk_id, "start_step": chunk_id * policy.chunk_size,
                                         "inference_ms": latency, "raw_actions": chunk.tolist()}) + "\n")
        np.savez_compressed(
            output / "trajectory.npz",
            raw_action_chunks=raw,
            policy_actions=raw[:, :, :10].reshape(-1, 10)[:len(actions)],
            actions=np.stack(actions),
            controller_goal_xyz_matrix=np.stack(goals),
            controller_ee_xyz_matrix=np.stack(measured),
        )
        timing = {"inference_ms": policy.inference_ms, "simulation_ms": simulation_ms}
        write_json(output / "timing.json", timing)
        result = {
            "task_id": task_id,
            "task_name": suite.get_task(task_id).name,
            "instruction": instruction,
            "initial_state_id": state_id,
            "simulator_seed": evaluator.init_seed + state_id + 100,
            "success": success,
            "termination_reason": "success" if success else "timeout",
            "num_steps": len(actions),
            "total_reward": total_reward,
            "mean_inference_ms": float(np.mean(policy.inference_ms)),
            "p95_inference_ms": float(np.percentile(policy.inference_ms, 95)),
            "video": "video.mp4",
            **renderer,
        }
        write_json(output / "episode.json", result)
        return result
    finally:
        env.close()
