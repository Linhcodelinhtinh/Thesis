#!/usr/bin/env python
"""Evaluate the versioned V2 SmolVLA pilot treatments."""
from __future__ import annotations
import argparse
import csv
import faulthandler
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mem_eval_io import EpisodeRecorder, evaluation_roster, write_json
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.simulator.process_libero_mem_env import ProcessLiberoMemEnv, wsl_path
from src.training.mem_adapter_artifact import validate_adapter, verify_files
from src.training.split_lock import AUTHOR_EVAL_SPLITS, AUTHOR_MEM_SPLIT, DEFAULT_AUTHOR_MEM_LOCK
from src.utils.file_hash import sha256_file
from src.evaluation.mem_conditions import CLI_CONDITIONS, condition_spec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-dir", type=Path)
    p.add_argument("--f0-adapter-dir", type=Path)
    p.add_argument("--base-checkpoint", type=Path, default=ROOT / "resources/checkpoints/smolvla_libero")
    p.add_argument("--split-lock", type=Path, default=DEFAULT_AUTHOR_MEM_LOCK)
    p.add_argument("--split", choices=[*AUTHOR_EVAL_SPLITS, "val", "test"], default="seen")
    p.add_argument("--tasks", nargs="+", choices=["T1", "T3"], default=["T1", "T3"])
    p.add_argument("--conditions", nargs="+", choices=CLI_CONDITIONS, default=["B0"])
    p.add_argument("--states", default="0", help="Comma-separated indices within each locked split, or all")
    p.add_argument("--policy-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True, help="New, nonexistent run directory")
    p.add_argument("--detector-model", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    p.add_argument("--detector-python", help="Isolated perception Python; required for F1")
    p.add_argument("--sim-python", required=True, help="Isolated simulator Python; Linux path when --use-wsl")
    p.add_argument("--use-wsl", action="store_true", help="Windows policy -> WSL simulator; omit when already in WSL")
    p.add_argument("--wsl-distro", default="Ubuntu-24.04")
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stream-memory-trace", action="store_true",
                   help="Keep the per-step trace in JSONL only, with a final snapshot in the summary")
    p.add_argument("--preflight-only", action="store_true", help="Verify files, source and environments; no policy rollout")
    p.add_argument("--runtime-check", action="store_true", help="Real load/reset and two actions per condition/task; diagnostic only, no score")
    p.add_argument("--resume", action="store_true", help="Resume interrupted evaluation in output-dir, skipping completed episodes")
    p.add_argument("--worker-timeout", type=int, default=300)
    p.add_argument("--git-executable", default=os.environ.get("MEM_EVAL_GIT", "git"),
                   help="Git used for read-only provenance; git.exe can speed up WSL /mnt source checks")
    a = p.parse_args(argv)
    if "F0" in a.conditions and not a.f0_adapter_dir:
        p.error("F0 requires --f0-adapter-dir")
    if any(condition_spec(name).weights == "memory" for name in a.conditions) and not a.adapter_dir:
        p.error("Memory-trained conditions require --adapter-dir")
    if "F1" in a.conditions and not a.detector_python:
        p.error("F1 requires --detector-python in a separate environment")
    if len(set(a.conditions)) != len(a.conditions) or a.worker_timeout < 1:
        p.error("Conditions must be distinct and timeout positive")
    if a.runtime_check and (a.preflight_only or a.states != "0"):
        p.error("Runtime check requires --states 0 and cannot combine with --preflight-only")
    if a.runtime_check and a.split == "unseen":
        p.error("Use seen states for runtime checks")
    return a


def runtime_identity(python, use_wsl=False, distro="Ubuntu-24.04"):
    script = ROOT / "scripts/mem_eval_runtime.py"
    command = (["wsl", "-d", distro, "--exec", str(python), wsl_path(script)] if use_wsl
               else [str(python), str(script)])
    return json.loads(subprocess.check_output(command, text=True, encoding="utf-8", timeout=120))


def check_environments(args, protocol):
    envs = dict(policy=runtime_identity(sys.executable),
                simulator=runtime_identity(args.sim_python, args.use_wsl, args.wsl_distro))
    if "F1" in args.conditions:
        envs["detector"] = runtime_identity(args.detector_python)
    identities = [(v["platform"], v["prefix"]) for v in envs.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("Policy, simulator and detector must use separate environments")
    sim = envs["simulator"]
    if not sim["python"].startswith("3.8."):
        raise ValueError("Pinned Mem simulator requires Python 3.8.x")
    for package in ("robosuite", "numpy", "bddl", "mujoco", "robomimic"):
        if sim["packages"][package] != protocol["runtime"][package]:
            raise ValueError(f"Simulator {package}: expected {protocol['runtime'][package]}, found {sim['packages'][package]}")
    required = ["torch", "lerobot", "transformers", "tokenizers"]
    if any(condition_spec(name).weights != "base" for name in args.conditions):
        required.append("peft")
    if args.save_video:
        required += ["imageio", "imageio-ffmpeg"]
    for package in required:
        if not envs["policy"]["packages"][package]:
            raise ValueError("Policy environment missing " + package)
    if "detector" in envs:
        for package in ("torch", "torchvision", "transformers", "tokenizers"):
            if not envs["detector"]["packages"][package]:
                raise ValueError("Detector environment missing " + package)
    return envs


def detector_launcher(python, output):
    """Keep the pinned ProcessVisualDetector unchanged, including its source hash.

    Its Path.resolve() dereferences Linux venv Python symlinks, losing sys.prefix.
    A recorded exec wrapper preserves the explicitly selected venv executable.
    """
    executable = Path(python).absolute()
    if os.name != "nt":
        wrapper = output / "detector-python.sh"
        wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(str(executable)) + ' "$@"\n', encoding="utf-8")
        wrapper.chmod(0o700)
        return str(wrapper)
    return str(executable)


def git_read(executable, directory, *arguments, text=True):
    directory = str(directory)
    if os.name != "nt" and str(executable).lower().endswith(".exe"):
        directory = subprocess.check_output(["wslpath", "-w", directory], text=True, timeout=30).strip()
    return subprocess.check_output([executable, "-C", directory, *arguments], text=text, timeout=600)


def code_identity(output, git_executable="git"):
    tracked = git_read(git_executable, ROOT, "rev-parse", "HEAD").strip()
    diff = git_read(git_executable, ROOT, "diff", "--binary", "HEAD", "--", "src", "scripts", "configs", "envs", text=False)
    (output / "code.patch").write_bytes(diff)
    files = [p for folder in ("src", "scripts") for p in (ROOT / folder).rglob("*")
             if p.suffix in (".py", ".sh", ".ps1")]
    with zipfile.ZipFile(output / "code_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    return dict(git_revision=tracked, dirty_diff_sha256=sha256_file(output / "code.patch"),
                code_snapshot_sha256=sha256_file(output / "code_snapshot.zip"),
                source_sha256={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(files)})


def report(output, rows, expected, status):
    fields = ["episode_id", "task", "condition", "split", "state_idx", "source_demo", "initial_state_sha256",
              "success", "scr", "overshot", "steps", "policy_ms", "wall_time_s"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Custom V2 LIBERO-Mem evaluation", "", "NON-COMPARABLE_OFFICIAL_PAPER", "",
             f"Status: {status}; completed {len(rows)}/{expected} episodes.", "",
             "| Task | Condition | N | SR | Mean SCR | Overshot |", "|---|---|---:|---:|---:|---:|"]
    for task, condition in sorted({(r["task"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["task"], r["condition"]) == (task, condition)]
        n = len(group)
        lines.append(f"| {task} | {condition} | {n} | {sum(r['success'] for r in group)/n:.3f} | {sum(r['scr'] for r in group)/n:.3f} | {sum(r['overshot'] for r in group)/n:.3f} |")
    pairs = {}
    for r in rows:
        pairs.setdefault((r["task"], r["initial_state_sha256"]), {})[r["condition"]] = r
    matched = [p for p in pairs.values() if {"B0", "F1"} <= p.keys()]
    lines += ["", f"Complete B0/F1 pairs: {len(matched)}.",
              "T1/T3 can share initial states; task-episodes are not independent state samples."]
    if matched:
        delta = sum(int(p["F1"]["success"]) - int(p["B0"]["success"]) for p in matched) / len(matched)
        lines.append(f"Descriptive paired delta SR: {delta:+.3f}. No confidence interval claimed.")
    primary = [p for p in pairs.values() if {"F0", "F1"} <= p.keys()]
    if primary:
        by_task = {}
        for pair in primary:
            by_task.setdefault(pair["F1"]["task"], []).append(
                int(pair["F1"]["success"]) - int(pair["F0"]["success"]))
        macro = sum(sum(deltas) / len(deltas) for deltas in by_task.values()) / len(by_task)
        lines.append(f"Observed F1/F0 pairs: {len(primary)}. Task-macro paired delta SR: {macro:+.3f}.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_runtime_episode(env, policy, renderer, factory, recorder, row, condition):
    """Two real actions solely to probe interfaces; never a shortened score."""
    import numpy as np
    from src.evaluation.libero_mem_rollout import action_shape, bind_instruction_only
    expected_shape = action_shape(policy, env)
    policy.reset()
    bind_instruction_only(policy, condition, row["instruction"], renderer)
    obs = env.reset(initial_state_id=row["state_idx"])
    writer = factory(episode_id="runtime_check", target_object_id="bowl_A", support_id="plate_A") if factory else None
    if writer is not None:
        policy.bind_memory_runtime(writer, renderer)
    last_action = None
    for step in range(3):
        recorder("observation", step=step, observation=obs, info=env.last_info)
        snapshot = writer.step(step, obs, last_action) if writer else None
        if writer:
            recorder("memory", step=step, snapshot=snapshot.to_dict(), signals=writer.last_signals)
        if step == 2:
            break
        rendered = renderer.render(row["instruction"], snapshot, enabled=writer is not None)
        if rendered.tokenizer_truncated:
            raise ValueError("Runtime probe instruction exceeds token budget")
        chunk = policy.predict_action_chunk(obs, rendered.text)
        if not isinstance(chunk, np.ndarray) or chunk.shape != expected_shape or not np.isfinite(chunk).all():
            raise ValueError("Runtime probe returned invalid action chunk")
        recorder("query", step=step, prompt=rendered.text, chunk=chunk, render=rendered)
        last_action = chunk[0]
        obs, reward, done, info = env.step(last_action)
        recorder("action", step=step, action=last_action, chunk_index=0, reward=reward, done=done)
        if done:
            raise RuntimeError("Environment terminated unexpectedly during two-action diagnostic")
    recorder.finish(diagnostic=True)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.absolute()
    rows, expected, runtime_checks = [], 0, 0
    if args.resume and output.exists():
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else dict(
            classification="CUSTOM_V2_LIBERO_MEM", comparability="NON-COMPARABLE_OFFICIAL_PAPER")
        manifest.update(status="resumed", timestamp=datetime.now(timezone.utc).isoformat(), arguments=vars(args))
        summary_path = output / "summary.csv"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["success"] = r["success"].lower() == "true"
                    r["overshot"] = r["overshot"].lower() == "true"
                    r["scr"] = float(r["scr"])
                    r["steps"] = int(r["steps"])
                    r["policy_ms"] = float(r["policy_ms"])
                    r["wall_time_s"] = float(r["wall_time_s"])
                    r["state_idx"] = int(r["state_idx"])
                    rows.append(r)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(classification="CUSTOM_V2_LIBERO_MEM", comparability="NON-COMPARABLE_OFFICIAL_PAPER",
            status="preflight", timestamp=datetime.now(timezone.utc).isoformat(), arguments=vars(args))
    def phase(name):
        manifest["phase"] = name
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_manifest.json", manifest)
        print(name, flush=True)

    try:
        faulthandler.enable()
        write_json(output / "run_manifest.json", manifest)
        print("Preflight: source, checkpoint, split and environments", flush=True)
        protocol, lock, roster = evaluation_roster(args.split_lock, args.split, args.tasks, args.states)
        source = ROOT / "external/libero-mem"
        revision = git_read(args.git_executable, source, "rev-parse", "HEAD").strip()
        changes = git_read(args.git_executable, source, "diff", "--name-only", "HEAD", "--",
            "libero/libero", "scripts/mem_3_refine_demonstration.py").strip()
        if revision != protocol["upstream_revision"] or changes:
            raise ValueError("Mem source checkout must match pinned revision without library modifications")
        expected = len(roster) * len(args.conditions)
        manifest.update(code_identity(output, args.git_executable), roster=roster, expected_episodes=expected,
                        protocol=protocol, protocol_sha256=source_digest(DEFAULT_PROTOCOL),
                        split_lock_sha256=lock["lock_sha256"], split_lock_file_sha256=sha256_file(args.split_lock))
        if lock.get("mem_split_policy") == AUTHOR_MEM_SPLIT:
            manifest.update(mem_split_policy=AUTHOR_MEM_SPLIT,
                            evaluation_state_source="LIBERO-Mem-Raw", raw_revision=lock["raw_revision"])
        write_json(output / "split.lock.json", lock)
        import yaml
        base_spec = yaml.safe_load((ROOT / "resources/manifests/models/smolvla_libero.yaml").read_text())
        base_files = {name: item["sha256"] for name, item in base_spec["files"].items()}
        verify_files(args.base_checkpoint, base_files)
        manifest["base"] = dict(revision=base_spec["revision"], files=base_files)
        # The adapter's existing offline load path uses this local backbone.
        # Audit it explicitly rather than trusting a directory merely because it exists.
        assets = ROOT / "experiments/training/assets_mem_adapter_v1"
        asset_lock = json.loads((assets / "assets.json").read_text(encoding="utf-8"))
        from src.memory.visual_pipeline import TOKENIZER_REPO, TOKENIZER_REVISION, load_prompt_tokenizer
        if (asset_lock["model_revision"] != base_spec["revision"] or asset_lock["backbone_repo"] != TOKENIZER_REPO
                or asset_lock["backbone_revision"] != TOKENIZER_REVISION):
            raise ValueError("Local backbone assets are not from pinned revisions")
        verify_files(assets, {name.replace("\\", "/"): digest for name, digest in asset_lock["files"].items()})
        load_prompt_tokenizer(assets / "backbone")
        manifest["local_assets_lock_sha256"] = sha256_file(assets / "assets.json")
        from dataclasses import asdict
        from src.training.mem_adapter_artifact import adapter_training_condition, validate_matched_adapters
        adapters = {}
        manifest["treatments"] = {}
        for condition in args.conditions:
            spec = condition_spec(condition)
            treatment = manifest["treatments"][condition] = asdict(spec)
            if spec.weights == "base":
                treatment["weight_identity"] = base_files
                continue
            directory = args.f0_adapter_dir if condition == "F0" else args.adapter_dir
            adapter = validate_adapter(directory, args.base_checkpoint)
            if adapter_training_condition(adapter) != spec.weights:
                raise ValueError(f"{condition} received the wrong adapter training condition")
            author_lock = lock.get("mem_split_policy") == AUTHOR_MEM_SPLIT
            author_adapter = adapter["run"].get("mem_split_policy") == AUTHOR_MEM_SPLIT
            if author_lock != author_adapter or (
                    author_lock and adapter["run"].get("evaluation_split_lock_sha256") != lock["lock_sha256"]):
                raise ValueError(f"{condition} adapter was trained under a different evaluation lock")
            adapters[condition] = (directory, adapter)
            treatment["weight_identity"] = adapter["adapter_files"]["adapter_model.safetensors"]
            copied_manifest = output / f"adapter_manifest_{condition}.json"
            shutil.copyfile(directory / "mem_adapter_manifest.json", copied_manifest)
            if json.loads(copied_manifest.read_text(encoding="utf-8")) != adapter:
                raise ValueError(f"{condition} adapter manifest changed during preflight")
            treatment["adapter_manifest_sha256"] = sha256_file(copied_manifest)
            if condition == "F1":
                memory_contract = adapter["memory_provenance"][0]
                if memory_contract["writer"] == "rgb_relations_v1":
                    treatment.update(memory_source=memory_contract["writer"],
                                     projection=memory_contract["writer_config"]["projection"],
                                     projection_strategy=memory_contract["writer_config"]["strategy"])
                manifest["adapter_manifest_sha256"] = treatment["adapter_manifest_sha256"]
                write_json(output / "adapter_manifest.json", adapter)
                verify_files(args.detector_model, adapter["memory_provenance"][0]["detector"]["files"])
        if "F0" in adapters and "F1" in adapters:
            validate_matched_adapters(adapters["F0"][1], adapters["F1"][1])
        manifest["environments"] = check_environments(args, protocol)
        manifest["status"] = "preflight_passed"
        write_json(output / "run_manifest.json", manifest)
        if args.preflight_only:
            report(output, rows, expected, "preflight_passed_no_rollout")
            print(f"Preflight passed; no rollout executed. {output}", flush=True)
            return

        phase("Loading policy runtime libraries")
        import numpy as np
        import torch
        from src.models.smolvla.adapter import SmolVLAAdapter
        from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
        from src.interfaces.libero_mem_text import LiberoMemTextRenderer
        from src.memory.visual_pipeline import visual_writer_factory, load_prompt_tokenizer
        from src.perception.object_tracker import TrackerConfig
        from src.perception.process_detector import ProcessVisualDetector
        from src.perception.relation_detector import ProcessRelationDetector
        from src.evaluation.libero_mem_protocol import load_suite_source
        from src.memory.relation_pipeline import relation_pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA policy runtime required; no CPU fallback")
        manifest["cuda"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        manifest["status"] = "running"
        write_json(output / "run_manifest.json", manifest)
        # Only one policy resident at a time, leaving memory for the detector.
        for condition in args.conditions:
            policy = SmolVLAAdapter()
            detector = None
            try:
                phase(f"Loading {condition} policy")
                if condition in adapters:
                    directory, adapter = adapters[condition]
                    policy.load(str(directory), base_checkpoint=str(args.base_checkpoint), device="cuda", strict=True)
                    factory, tokenizer = None, policy.tokenizer
                else:
                    policy.load(str(args.base_checkpoint), device="cuda", strict=True)
                    factory = None
                    tokenizer = load_prompt_tokenizer(ROOT / "experiments/training/assets_mem_adapter_v1/backbone")
                if condition == "F1":
                    contract = adapter["memory_provenance"][0]
                    launcher = detector_launcher(args.detector_python, output)
                    manifest["detector_launcher_sha256"] = sha256_file(launcher) if launcher.endswith(".sh") else None
                    write_json(output / "run_manifest.json", manifest)
                    phase("Loading F1 detector")
                    detector_type = (ProcessRelationDetector if contract["writer"] == "rgb_relations_v1"
                                     else ProcessVisualDetector)
                    detector = detector_type(launcher, args.detector_model, device="cuda", timeout_seconds=args.worker_timeout)
                    if contract["writer"] == "rgb_cycle_v3":
                        factory = visual_writer_factory(detector, "libero_raw", TrackerConfig(**contract["writer_config"]))
                import inspect
                base_policy = policy.policy.get_base_model() if condition in adapters else policy.policy
                policy_source = Path(inspect.getfile(type(base_policy)))
                processor_source = Path(inspect.getfile(type(policy.env_preprocessor)))
                manifest.setdefault("loaded_policies", {})[condition] = dict(
                    official_policy_source_sha256=sha256_file(policy_source),
                    official_env_processor_source_sha256=sha256_file(processor_source),
                    parameter_dtype=str(next(base_policy.parameters()).dtype),
                    numpy_imported_version=np.__version__, torch_imported_version=torch.__version__)
                write_json(output / "run_manifest.json", manifest)
                del base_policy
                renderer = LiberoMemTextRenderer(tokenizer=tokenizer, strict_tokenizer=True, max_total_tokens=48, allow_oracle=False)
                for task in args.tasks:
                    if condition == "F1" and contract["writer"] == "rgb_relations_v1":
                        suite = load_suite_source(source)
                        factory, renderer = relation_pipeline(suite, task, detector, "libero_raw", tokenizer,
                                                              contract["writer_config"]["strategy"])
                    phase(f"Starting {condition} {task} simulator")
                    env = ProcessLiberoMemEnv(task_name=task, split_lock=args.split_lock, split=args.split,
                        python_executable=args.sim_python, use_wsl=args.use_wsl,
                        wsl_distro=args.wsl_distro, timeout_seconds=args.worker_timeout)
                    try:
                        for row in [r for r in roster if r["task"] == task]:
                            ep_id = f"{task}_{condition}_{args.split}_state{row['state_idx']}"
                            ep_dir = output / "episodes" / ep_id
                            status_file = ep_dir / "status.json"
                            if args.resume and any(r["episode_id"] == ep_id for r in rows) and status_file.exists():
                                try:
                                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                                    if status_data.get("status") == "complete":
                                        print(f"{ep_id}: already completed (success={status_data.get('success')}), skipping...", flush=True)
                                        continue
                                except Exception:
                                    pass
                            if ep_dir.exists():
                                shutil.rmtree(ep_dir)
                            phase(f"Running {ep_id}")
                            metadata = dict(row, episode_id=ep_id, condition=condition, policy_seed=args.policy_seed)
                            recorder = EpisodeRecorder(output / "episodes" / ep_id, metadata, args.save_video,
                                                       fps=protocol["simulation"]["control_freq"])
                            try:
                                random.seed(args.policy_seed)
                                np.random.seed(args.policy_seed)
                                torch.manual_seed(args.policy_seed)
                                torch.cuda.manual_seed_all(args.policy_seed)
                                if args.runtime_check:
                                    check_runtime_episode(env, policy, renderer, factory, recorder, row, condition)
                                    runtime_checks += 1
                                    print(f"{ep_id}: runtime check passed (two actions; no score)", flush=True)
                                    continue
                                runner = LiberoMemRolloutRunner(env, policy, condition=condition,
                                    execution_horizon=protocol["rollout"]["execution_horizon"],
                                    max_steps=protocol["rollout"]["horizon_policy_actions"],
                                    renderer=renderer, writer_factory=factory, observer=recorder,
                                    stream_memory_trace=args.stream_memory_trace)
                                result = runner.run_episode(ep_id, row["instruction"], row["state_idx"])
                                recorder.finish(result=result)
                            except BaseException as exc:
                                recorder.finish(error=exc)
                                raise
                            rows.append({k: metadata[k] for k in ("episode_id", "task", "condition", "split", "state_idx", "source_demo", "initial_state_sha256")})
                            rows[-1].update(success=result.success, scr=result.scr, overshot=result.overshot,
                                steps=result.total_steps, policy_ms=result.avg_latency_ms, wall_time_s=result.total_wall_time_s)
                            report(output, rows, expected, "running")
                            print(f"{ep_id}: success={result.success} SCR={result.scr:.3f} overshot={result.overshot}", flush=True)
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    finally:
                        env.close()
            finally:
                if detector is not None:
                    detector.close()
                del policy
                # The runner owns a policy reference; release it before loading
                # the next condition so B0 cannot remain resident beside F1.
                if "runner" in locals():
                    del runner
                gc.collect()
                torch.cuda.empty_cache()
        manifest["status"] = "runtime_checked" if args.runtime_check else "complete"
        manifest["phase"] = "finished"
        manifest["completed_episodes"] = len(rows)
        manifest["completed_runtime_checks"] = runtime_checks
        if args.runtime_check:
            (output / "report.md").write_text("# Runtime diagnostic\n\nReal model/reset/IPC/two actions passed per selected condition/task.\nNo benchmark success rate measured.\n", encoding="utf-8")
        else:
            report(output, rows, expected, "complete")
        write_json(output / "run_manifest.json", manifest)
        print(f"{manifest['status']}: {output}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc(), completed_episodes=len(rows),
                        completed_runtime_checks=runtime_checks)
        write_json(output / "run_manifest.json", manifest)
        report(output, rows, expected, "failed")
        raise


if __name__ == "__main__":
    main()
