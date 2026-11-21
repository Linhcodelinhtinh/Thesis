"""Run the pinned authors' X-VLA LIBERO client with local provenance and telemetry."""

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.x_vla_libero import run_episode, write_json
from src.models.x_vla import make_client
from src.simulator.x_vla_libero import configure_reference, open_episode, render_metadata, verify_task
from src.utils.file_hash import sha256_file


LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def verify_checkout(path, revision):
    # Windows and WSL share this checkout. Ignore executable bits and normalize
    # CRLF for the source comparison. Resource payloads are hashed separately.
    command = ["git", "-C", str(path), "-c", "core.filemode=false", "-c", "core.autocrlf=true"]
    head = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=no"], text=True)
    if head != revision or status.strip():
        raise RuntimeError(f"Expected a clean checkout at {revision}: {path}. "
                           f"Got {head}, changes: {status.splitlines()[:5]}")


def load_upstream(path):
    spec = importlib.util.spec_from_file_location(
        "x_vla_author_libero_client", path / "evaluation" / "libero" / "libero_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def start_server(python, checkout, checkpoint, output, port, timeout):
    import requests

    with socket.socket() as sock:
        # Reject an existing service before the managed inference process starts.
        sock.bind(("127.0.0.1", port))
    server_env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    server_env.pop("PYTHONPATH", None)
    launch = (
        "import runpy, sys, torch, transformers; "
        "assert sys.version_info[:2] == (3, 10), 'Expected Python 3.10 for X-VLA'; "
        "assert torch.cuda.is_available(), 'CUDA unavailable in the X-VLA environment'; "
        "assert transformers.__version__ == '4.51.3', 'Expected Transformers 4.51.3'; "
        "print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); "
        "runpy.run_module('deploy', run_name='__main__', alter_sys=True)"
    )
    with (output / "server.log").open("wb") as log:
        proc = subprocess.Popen(
            [str(python), "-c", launch, "--model_path", str(checkpoint), "--device", "cuda",
             "--host", "127.0.0.1", "--port", str(port), "--disable_slurm",
             "--output_dir", str(output / "server")],
            cwd=checkout, env=server_env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None and time.monotonic() < deadline:
                if (output / "server" / "info.json").is_file():
                    try:
                        response = requests.get(f"http://127.0.0.1:{port}/openapi.json", timeout=1)
                        if response.ok and "/act" in response.json()["paths"]:
                            startup = (output / "server.log").read_text(encoding="utf-8", errors="replace")
                            if "Processor loaded successfully" not in startup or "newly initialized" in startup:
                                raise RuntimeError(f"X-VLA failed model or processor verification. See {output / 'server.log'}")
                            break
                    except requests.RequestException:
                        pass
                time.sleep(0.25)
            else:
                raise RuntimeError(f"X-VLA server did not start. See {output / 'server.log'}")
            yield
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--initial-state-ids", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42, help="Author's simulator seed base")
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the pinned X-VLA snapshot")
    parser.add_argument("--inference-python", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="Validate resources and render one reset")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=600, help="HTTP and server startup timeout in seconds")
    args = parser.parse_args()
    if not args.preflight_only and (args.inference_python is None or not args.inference_python.is_file()):
        parser.error("--inference-python must point to the Python executable in the X-VLA env")
    for ids, bound in ((args.task_ids, 10), (args.initial_state_ids, 50)):
        if ids is not None and (len(set(ids)) != len(ids) or any(i < 0 or i >= bound for i in ids)):
            parser.error(f"IDs must be unique and belong to 0..{bound - 1}")
    if not 1 <= args.port <= 65535 or not 0 < args.timeout < float("inf"):
        parser.error("Expected a valid TCP port and a finite positive timeout")
    return args


def main():
    args = parse_args()
    manifest_path = ROOT / "resources" / "manifests" / "models" / "x_vla_libero.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else ROOT / manifest["checkpoint"]
    for name, expected in manifest["sha256"].items():
        if sha256_file(checkpoint / name) != expected:
            raise RuntimeError(f"X-VLA checkpoint SHA-256 mismatch for {checkpoint / name}")

    upstream_path = ROOT / "external" / "X-VLA"
    libero_path = ROOT / "external" / "LIBERO"
    verify_checkout(upstream_path, manifest["code_revision"])
    verify_checkout(libero_path, LIBERO_REVISION)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    versions = configure_reference(libero_path, output / "libero_config")
    upstream = load_upstream(upstream_path)
    horizon = upstream.LIBERO_DATASETS_HORIZON[args.suite]
    evaluator = upstream.LIBEROEval(args.suite, eval_horizon=horizon, act_type="abs", init_seed=args.seed)
    suite = evaluator.task_suite_list[0]
    task_ids = args.task_ids or list(range(len(suite.tasks)))
    state_ids = args.initial_state_ids or list(range(50))
    assets = {task_id: verify_task(suite, task_id) for task_id in task_ids}
    write_json(output / "run.json", {
        "protocol": "X-VLA authors' LIBERO client",
        "reporting_label": "NON-COMPARABLE_OFFICIAL_PAPER",
        "evaluation_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "evaluation_files_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "scripts/run_x_vla_libero.py", "src/models/x_vla.py", "src/simulator/x_vla_libero.py",
                "src/evaluation/x_vla_libero.py", str(manifest_path.relative_to(ROOT)),
            )
        },
        "libero_revision": LIBERO_REVISION,
        "model": manifest,
        "checkpoint_path": str(checkpoint),
        "suite": args.suite,
        "task_ids": task_ids,
        "initial_state_ids": state_ids,
        "max_policy_steps": horizon,
        "env_horizon": 1000,
        "settling_steps": 10,
        "controller_use_delta": False,
        "simulator_seed_base": args.seed,
        "inference_seed": None,
        "inference_python": str(args.inference_python.absolute()) if args.inference_python else None,
        "simulator_python": sys.version,
        "simulator_packages": versions,
        "platform": platform.platform(),
        "mujoco_gl_requested": os.environ.get("MUJOCO_GL"),
        "asset_verification": assets,
        "preflight_only": args.preflight_only,
    })
    if args.preflight_only:
        env, instruction, obs = open_episode(evaluator, suite, task_ids[0], state_ids[0])
        try:
            frames = {key: list(obs[key].shape) for key in ("agentview_image", "robot0_eye_in_hand_image")}
            if any(shape != [256, 256, 3] for shape in frames.values()):
                raise RuntimeError(f"Unexpected LIBERO frame shape: {frames}")
            write_json(output / "preflight.json", {"instruction": instruction, "camera_shapes": frames,
                                                   **render_metadata(env)})
        finally:
            env.close()
        print(f"LIBERO preflight passed: {output}")
        return

    policy = make_client(upstream, args.port, args.timeout)
    summaries = []
    with start_server(args.inference_python.absolute(), upstream_path, checkpoint, output, args.port, args.timeout):
        for task_id in task_ids:
            successes = 0
            for state_id in state_ids:
                episode_dir = output / f"task_{task_id:02d}" / f"init_{state_id:02d}"
                print(f"{args.suite} task {task_id} state {state_id}", flush=True)
                result = run_episode(upstream, evaluator, suite, task_id, state_id, policy, episode_dir)
                successes += int(result["success"])
            summaries.append({"task_id": task_id, "task_name": suite.get_task(task_id).name,
                              "episodes": len(state_ids), "successes": successes,
                              "success_rate": successes / len(state_ids)})
    total = sum(row["episodes"] for row in summaries)
    successes = sum(row["successes"] for row in summaries)
    write_json(output / "summary.json", {"suite": args.suite, "episodes": total,
               "successes": successes, "success_rate": successes / total, "tasks": summaries})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
