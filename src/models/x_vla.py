"""Validated transport for the pinned X-VLA LIBERO client."""

import time

import numpy as np


def validate_chunk(raw):
    """Return the author's wire output and validate the active Panda block."""
    chunk = np.asarray(raw)
    if chunk.shape != (30, 20) or chunk.dtype.kind not in "fiu":
        raise ValueError(f"Expected a numeric X-VLA action chunk (30, 20), got {chunk.shape} {chunk.dtype}")
    if not np.isfinite(chunk).all():
        raise ValueError("X-VLA returned a non-finite action")
    if not ((chunk[:, 9] >= 0) & (chunk[:, 9] <= 1)).all():
        raise ValueError("X-VLA gripper must be a probability after sigmoid")
    if np.any(np.linalg.norm(np.cross(chunk[:, 3:6], chunk[:, 6:9]), axis=1) <= 1e-6):
        raise ValueError("X-VLA returned a degenerate EEF6D rotation")
    return chunk


def make_client(upstream, port, timeout):
    """Reuse the authors' camera formatting, proprio, decoder and full-chunk FIFO."""
    import requests

    class AuditedClient(upstream.ClientModel):
        action_dim = 10
        chunk_size = 30

        def reset(self):
            super().reset()
            self.raw_chunks = []
            self.inference_ms = []

        def _format_query(self, obs, goal):
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                frame = obs[key]
                if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Invalid camera frame for {key}: {frame.shape} {frame.dtype}")
            for key, width in (("robo_pos", 3), ("robo_ori", 6)):
                if obs[key].shape != (width,) or not np.isfinite(obs[key]).all():
                    raise ValueError(f"Invalid controller observation: {key}")
            return super()._format_query(obs, goal)

        def _post(self, payload):
            start = time.perf_counter()
            response = requests.post(self.url, json=payload, timeout=timeout)
            response.raise_for_status()
            raw = validate_chunk(response.json()["action"])
            self.raw_chunks.append(raw)
            self.inference_ms.append((time.perf_counter() - start) * 1000)
            return raw[:, :self.action_dim]

    return AuditedClient("127.0.0.1", port)
