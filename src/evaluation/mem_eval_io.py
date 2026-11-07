"""Auditable LIBERO-Mem roster and streamed episode artifacts (custom V2)."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from src.evaluation.libero_mem_protocol import load_protocol, task_spec, source_digest, DEFAULT_PROTOCOL, ROOT
from src.training.split_lock import load_split_lock
from src.utils.file_hash import sha256_file


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluation_roster(split_lock, split, tasks, states, source_root=None):
    source_root = Path(source_root or ROOT / "external/libero-mem")
    protocol = load_protocol(source_root=source_root)
    lock = load_split_lock(split_lock)
    if lock["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL):
        raise ValueError("Split lock references another protocol")
    if split not in ("val", "test") or len(tasks) != len(set(tasks)):
        raise ValueError("Use distinct tasks and a held-out split")
    roster = []
    for alias in tasks:
        spec = task_spec(protocol, alias)
        bddl = source_root / spec["bddl"]
        # The complete source is hash-checked above. Only accept the single
        # plain-text language clause used by these two pinned official files.
        clauses = re.findall(r"\(:language\s+([^()]*)\)", bddl.read_text(encoding="utf-8"))
        if len(clauses) != 1:
            raise ValueError("Ambiguous/missing BDDL language: " + str(bddl))
        instruction = " ".join(clauses[0].split())
        if not instruction:
            raise ValueError("Empty BDDL instruction")
        episodes = [e for e in lock["episodes"] if e["domain"] == "mem" and
                    e["task"] == spec["name"] and e["split"] == split]
        if not episodes:
            raise ValueError("No held-out episodes for " + alias)
        indices = list(range(len(episodes))) if states == "all" else [int(s) for s in states.split(",")]
        if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(episodes) for i in indices):
            raise ValueError("Invalid or repeated split state indices for " + alias)
        for index in indices:
            row = episodes[index]
            roster.append(dict(task=spec["alias"], task_name=spec["name"], state_idx=index,
                split=split, source_demo=row["demo"], initial_state_sha256=row["initial_state_sha256"],
                instruction=instruction, bddl=spec["bddl"], bddl_text_sha256=source_digest(bddl)))
    return protocol, lock, roster


class EpisodeRecorder:
    """Stream evidence as it happens; only mark complete after all files close.

    RGB is stored as video (visualization), with raw-frame hashes in JSONL.
    Initial/final raw observations, raw proprioception, executed actions and
    entire unnormalized action chunks are saved losslessly in NPZ files.
    """
    def __init__(self, directory, metadata, save_video=True, fps=20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.save_video, self.fps = save_video, fps
        self.streams, self.videos = {}, {}
        self.actions, self.chunks, self.query_steps, self.states = [], [], [], []
        self.last_observation = None
        self.closed = False
        write_json(self.directory / "status.json", dict(status="running", **metadata))

    def _line(self, name, value):
        if name not in self.streams:
            self.streams[name] = (self.directory / name).open("x", encoding="utf-8")
        self.streams[name].write(json.dumps(value, default=json_default, allow_nan=False) + "\n")
        self.streams[name].flush()

    def __call__(self, event, **payload):
        if event == "observation":
            obs = payload.pop("observation")
            self.last_observation = {k: np.asarray(v).copy() for k, v in obs.items()}
            if payload["step"] == 0:
                np.savez_compressed(self.directory / "initial_observation.npz", **obs)
            raw_state = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]])
            self.states.append(raw_state.copy())
            payload["raw_proprioception_pos_quatxyzw_gripper"] = raw_state
            payload["rgb_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(obs[k]).tobytes()).hexdigest()
                                      for k in ("agentview_image", "robot0_eye_in_hand_image")}
            self._line("observations.jsonl", payload)
            if self.save_video:
                import imageio.v2 as imageio
                for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                    if camera not in self.videos:
                        self.videos[camera] = imageio.get_writer(str(self.directory / (camera + ".mp4")),
                            fps=self.fps, codec="libx264", macro_block_size=1)
                    # Display raw simulator frames; never feed encoded frames back to policy.
                    self.videos[camera].append_data(obs[camera])
        elif event == "query":
            self.chunks.append(np.asarray(payload["chunk"]).copy())
            self.query_steps.append(payload["step"])
            self._line("queries.jsonl", payload)
        elif event == "action":
            self.actions.append(np.asarray(payload["action"]).copy())
            self._line("actions.jsonl", payload)
        elif event == "memory":
            self._line("memory_trace.jsonl", payload)
        else:
            raise ValueError("Unknown recorder event " + event)

    def finish(self, result=None, error=None, diagnostic=False):
        if self.closed:
            return
        self.closed = True
        close_errors = []
        for stream in [*self.streams.values(), *self.videos.values()]:
            try:
                stream.close()
            except Exception as exc:
                close_errors.append(repr(exc))
        np.savez_compressed(self.directory / "trajectory.npz",
            actions=np.stack(self.actions) if self.actions else np.empty((0, 7)),
            action_chunks=np.stack(self.chunks) if self.chunks else np.empty((0, 50, 7)),
            query_steps=np.asarray(self.query_steps, dtype=np.int64),
            raw_proprioception_pos_quatxyzw_gripper=np.stack(self.states) if self.states else np.empty((0, 9)))
        if self.last_observation is not None:
            np.savez_compressed(self.directory / "final_observation.npz", **self.last_observation)
        if result is not None:
            if len(self.actions) != result.total_steps or len(self.states) != result.total_steps + 1:
                close_errors.append("Trajectory length differs from rollout")
            write_json(self.directory / "summary.json", dict(**self.metadata, result=asdict(result)))
        failure = str(error) if error is not None else "; ".join(close_errors)
        inventory = {p.name: sha256_file(p) for p in self.directory.iterdir() if p.is_file() and p.name != "status.json"}
        write_json(self.directory / "status.json", dict(**self.metadata,
            status="failed" if failure or (result is None and not diagnostic) else "diagnostic_complete" if diagnostic else "complete", error=failure or None,
            logging_errors=close_errors, files_sha256=inventory))
        if close_errors:
            raise RuntimeError("Episode artifact finalization failed: " + "; ".join(close_errors))
