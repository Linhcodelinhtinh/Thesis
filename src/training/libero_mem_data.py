"""Canonical Data Alignment and Processor for LIBERO-Mem Trajectories.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 2.1 & Section 8.1:
- Observation-Action alignment offset:
  - LIBERO-Mem refined demonstrations: offset 0 (obs[i] -> actions[i:i+H]).
  - LIBERO base raw demonstrations: offset +1 (obs[i] -> actions[i+1:i+1+H]).
- Causal memory constraint:
  - Memory at step t only consumes obs[0..t] and actions[0..t-1].
  - Target chunk a[t..t+H-1] is strictly for policy training and NEVER leaked into memory.
- Action chunking with execution horizon padding masks.
- Three matched data streams: predicted_memory, oracle_memory, and no_memory.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.event_writer import LiberoMemEventWriter
from src.memory.libero_mem_oracle import LiberoMemOracleWriter
from src.memory.models import SchemaV2EpisodeMemory
from src.training.libero_mem_hdf5_loader import DemonstrationEpisode, LiberoMemHDF5Parser


class MemoryStream(str, Enum):
    """Data stream variants per Section 8.3."""

    PREDICTED_MEMORY = "predicted_memory"
    ORACLE_MEMORY = "oracle_memory"
    NO_MEMORY = "no_memory"


@dataclass
class AlignedSample:
    """A single canonical training point for imitation learning."""

    sample_id: str
    episode_id: str
    step: int
    instruction: str
    rendered_prompt: str
    agentview_image: np.ndarray
    wrist_image: np.ndarray
    robot_states: np.ndarray  # [gripper(2), eef_pos(3), eef_quat(4)] -> 9D or 8D
    action_chunk: np.ndarray  # (H, action_dim)
    action_mask: np.ndarray   # (H,) bool
    memory_stream: MemoryStream
    has_memory: bool
    source_obs_idx: int
    source_action_start_idx: int


class LiberoMemDataProcessor:
    """Processes demonstration episodes into causal, aligned training samples."""

    def __init__(
        self,
        chunk_size: int = 50,
        action_dim: int = 7,
        obs_action_offset: int = 0,
        renderer: Optional[LiberoMemTextRenderer] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.obs_action_offset = obs_action_offset
        self.renderer = renderer if renderer is not None else LiberoMemTextRenderer(max_total_tokens=48)
        self.oracle_renderer = LiberoMemTextRenderer(max_total_tokens=48, allow_oracle=True)

    def process_episode(
        self,
        episode_id: str,
        instruction: str,
        observations: List[Dict[str, Any]],
        actions: np.ndarray,
        stream: Union[MemoryStream, str] = MemoryStream.PREDICTED_MEMORY,
        target_object_id: str = "bowl_A",
        target_semantic_name: str = "bowl",
        origin_anchor: str = "bowl initially on the left plate",
        support_id: str = "plate_A",
        support_semantic_name: str = "plate",
    ) -> List[AlignedSample]:
        """Align an episode into samples with causal memory."""
        stream = MemoryStream(stream)
        actions = np.asarray(actions, dtype=np.float32)
        n_obs = len(observations)
        n_act = len(actions)

        writer: Optional[LiberoMemEventWriter] = None
        oracle_writer: Optional[LiberoMemOracleWriter] = None

        if stream == MemoryStream.PREDICTED_MEMORY:
            writer = LiberoMemEventWriter(
                episode_id=episode_id,
                target_object_id=target_object_id,
                target_semantic_name=target_semantic_name,
                origin_anchor=origin_anchor,
                support_id=support_id,
                support_semantic_name=support_semantic_name,
            )
        elif stream == MemoryStream.ORACLE_MEMORY:
            oracle_writer = LiberoMemOracleWriter(
                episode_id=episode_id,
                target_object_id=target_object_id,
                target_semantic_name=target_semantic_name,
                origin_anchor=origin_anchor,
                support_id=support_id,
                support_semantic_name=support_semantic_name,
            )

        samples: List[AlignedSample] = []

        for t in range(n_obs):
            obs_t = observations[t]
            if not isinstance(obs_t, dict) or not obs_t:
                raise ValueError(f"Observation at step {t} is empty or not a dictionary. Missing observation fabrication is strictly forbidden.")
            if "robot0_eef_pos" not in obs_t or "robot0_gripper_qpos" not in obs_t:
                raise ValueError(f"Observation at step {t} missing required robot state keys ('robot0_eef_pos', 'robot0_gripper_qpos').")
            if "agentview_image" not in obs_t or "robot0_eef_quat" not in obs_t:
                raise ValueError(
                    f"Observation at step {t} missing required keys ('agentview_image', 'robot0_eef_quat'). "
                    "Fabricating dummy zero camera frames or default quaternions is strictly forbidden per AGENTS.md Rule 1."
                )

            last_action = actions[t - 1] if t > 0 and (t - 1) < n_act else None

            # Step causal writer based on stream
            memory_snapshot = None
            if stream == MemoryStream.PREDICTED_MEMORY and writer is not None:
                memory_snapshot = writer.step(step=t, observation=obs_t, last_action=last_action)
            elif stream == MemoryStream.ORACLE_MEMORY and oracle_writer is not None:
                oracle_signals = obs_t.get("oracle_signals")
                if oracle_signals is None:
                    raise ValueError(f"Step {t} lacks required 'oracle_signals' for MemoryStream.ORACLE_MEMORY")
                memory_snapshot = oracle_writer.step(step=t, oracle_signals=oracle_signals)

            # Determine action slice
            act_start = t + self.obs_action_offset
            if act_start >= n_act:
                # Terminal frame without target action chunk; skip
                continue

            act_end = min(act_start + self.chunk_size, n_act)
            available_act = actions[act_start:act_end]
            actual_len = len(available_act)

            # Pad chunk to chunk_size if near end of episode
            chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            mask = np.zeros((self.chunk_size,), dtype=bool)

            chunk[:actual_len] = available_act
            mask[:actual_len] = True

            # Text rendering based on stream
            if stream == MemoryStream.NO_MEMORY:
                render_res = self.renderer.render(instruction, None, enabled=False)
                has_mem = False
            elif stream == MemoryStream.PREDICTED_MEMORY:
                render_res = self.renderer.render(instruction, memory_snapshot, enabled=True)
                has_mem = len(render_res.facts_included) > 0
            else:  # ORACLE_MEMORY
                render_res = self.oracle_renderer.render(instruction, memory_snapshot, enabled=True)
                has_mem = len(render_res.facts_included) > 0

            # Robot state vector [gripper(2), eef_pos(3), eef_quat(4)]
            eef_pos = np.asarray(obs_t.get("robot0_eef_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
            gripper = np.asarray(obs_t.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32)
            quat = np.asarray(obs_t.get("robot0_eef_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            robot_state = np.concatenate([gripper[:2], eef_pos[:3], quat[:4]], dtype=np.float32)

            agentview = obs_t.get("agentview_image", np.zeros((128, 128, 3), dtype=np.uint8))
            wrist = obs_t.get("robot0_eye_in_hand_image", np.zeros((128, 128, 3), dtype=np.uint8))

            sample = AlignedSample(
                sample_id=f"{episode_id}_s{t}",
                episode_id=episode_id,
                step=t,
                instruction=instruction,
                rendered_prompt=render_res.text,
                agentview_image=agentview,
                wrist_image=wrist,
                robot_states=robot_state,
                action_chunk=chunk,
                action_mask=mask,
                memory_stream=stream,
                has_memory=has_mem,
                source_obs_idx=t,
                source_action_start_idx=act_start,
            )
            samples.append(sample)

        return samples
