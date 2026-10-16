"""Causal memory annotations for the author-released LIBERO HDF5 format.

IMPORTANT: the pinned LIBERO create_dataset.py records obs AFTER action[i].
Prepared observation[i] therefore supervises action[i+1], not action[i].
No simulator, reward inference, generated images, or fallback policy is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation.diagnostics import extract_all_targets_and_goal
from src.interfaces.text_memory import TextMemoryRenderer
from src.memory.models import MemoryQuery
from src.memory.observation_writer import ObservationMemoryWriter
from src.memory.retriever import DeterministicMemoryRetriever
from src.memory.store import EpisodeMemoryStore
from src.memory.updater import MemoryUpdater

DATA_REPO = "yifengzhu-hf/LIBERO-datasets"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
MODEL_REPO = "lerobot/smolvla_libero"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
BACKBONE_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MEMORY_FILES = (
    "src/memory/observation_writer.py", "src/memory/models.py",
    "src/memory/store.py", "src/memory/updater.py", "src/memory/retriever.py",
    "src/interfaces/text_memory.py", "src/evaluation/diagnostics.py",
    "src/training/memory_data.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def code_hashes(root):
    return {name: sha256(Path(root) / name) for name in MEMORY_FILES}


def validate_episode(demo):
    """Validate the source contract, rather than guessing missing features."""
    actions = np.asarray(demo["actions"], dtype=np.float32)
    robot = np.asarray(demo["robot_states"], dtype=np.float32)
    n = len(actions)
    if n < 2 or actions.shape != (n, 7) or robot.shape != (n, 9):
        raise ValueError(f"Invalid actions/robot_states: {actions.shape}, {robot.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(robot).all():
        raise ValueError("Non-finite state/action")
    if np.abs(actions).max() > 1.00001:
        raise ValueError("Actions outside native OSC_POSE [-1,1]; refusing clipping")
    # Official get_robot_state_vector: [gripper qpos(2), eef pos(3), quat xyzw(4)].
    np.testing.assert_allclose(robot[:, :2], demo["obs/gripper_states"][:], atol=1e-6)
    np.testing.assert_allclose(robot[:, 2:5], demo["obs/ee_pos"][:], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(robot[:, 5:9], axis=1), 1, atol=1e-4)
    resolutions = {}
    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
        images = demo[f"obs/{camera}"]
        if images.ndim != 4 or images.shape[0] != n or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"Invalid raw RGB camera {camera}: {images.shape}, {images.dtype}")
        resolutions[camera] = list(images.shape[1:])
    return actions, robot, resolutions


def annotate_episode(instruction, robot, actions, tokenizer, episode_id, max_tokens=48, max_chars=512):
    """Emit prompts for obs[0:N-1], each BEFORE its target actions[i+1:].

    The first retained observation is the episode reset for this derived dataset.
    At obs[i], the writer has seen only actions[1:i+1] and obs[1:i+1].
    Event step indexing follows the benchmark writer (transition index i-1).
    """
    targets, goal = extract_all_targets_and_goal(None, instruction)
    targets = tuple(t for t in targets if t != goal)
    entities = tuple(dict.fromkeys(targets + ((goal,) if goal else ()))) or (episode_id,)
    store = EpisodeMemoryStore(episode_id)
    writer = ObservationMemoryWriter(store, MemoryUpdater(store), targets, goal)
    retriever = DeterministicMemoryRetriever()
    renderer = TextMemoryRenderer(max_context_chars=max_chars, max_total_tokens=max_tokens,
                                  compact_mode=True, tokenizer=tokenizer, strict_tokenizer=True)
    if renderer.count_tokens(instruction) > max_tokens:
        raise ValueError(f"Original instruction exceeds {max_tokens} tokens: {instruction}")

    def obs(row):
        return {"robot0_gripper_qpos": row[:2], "robot0_eef_pos": row[2:5]}

    writer.on_episode_start(episode_id, None, obs(robot[0]))
    records = []
    for i in range(len(actions) - 1):
        if i:
            writer.on_step(step=i - 1, timestamp=i * 0.05, action=actions[i], next_obs=obs(robot[i]))
        query = MemoryQuery(episode_id=episode_id, as_of_step=i, entity_ids=entities, minimum_confidence=0.5)
        result = renderer.render_with_audit(instruction, retriever.retrieve(store.snapshot(i), query))
        if result.total_tokens_with_instruction > max_tokens or result.tokenizer_truncated:
            raise ValueError("Tokenizer would truncate the prepared prompt")
        records.append({"observation_index": i, "action_start_index": i + 1,
                        "max_observed_source_index": i, "prompt": result.text,
                        "has_memory": result.text != instruction,
                        "tokens": result.total_tokens_with_instruction,
                        "facts": list(result.facts_included), "skipped": result.memory_skipped_reason})
    return records


def assign_splits(episode_groups, seed=42, val_fraction=0.1, test_fraction=0.1):
    """Split groups, not frames. Exact duplicate initial states remain together."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Require positive train/validation/test fractions")
    unique = sorted(set(episode_groups.values()),
                    key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    if len(unique) < 3:
        raise ValueError("Need at least three distinct initial-state groups per task")
    nv, nt = max(1, round(len(unique) * val_fraction)), max(1, round(len(unique) * test_fraction))
    if nv + nt >= len(unique):
        raise ValueError("Split leaves no training groups")
    mapping = {g: "val" if i < nv else "test" if i < nv + nt else "train" for i, g in enumerate(unique)}
    return {episode: mapping[group] for episode, group in episode_groups.items()}


def action_window(actions, start, horizon):
    if not 0 <= start < len(actions) or horizon < 1:
        raise ValueError("Invalid action window")
    indices = np.arange(start, start + horizon)
    # Repeat final action for padding, with official loss mask; never cross episodes.
    return np.asarray(actions[np.minimum(indices, len(actions) - 1)], dtype=np.float32), indices >= len(actions)


def drop_memory(seed, epoch, sample_id, probability):
    if not 0 <= probability <= 1:
        raise ValueError("Memory dropout must lie in [0,1]")
    value = int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()[:8], "big")
    return value / 2**64 < probability
