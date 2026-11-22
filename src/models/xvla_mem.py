"""Chunk interface for the shared E2 runner, using the author XVLA client.

Native chunk = 30; execute its first 10 actions. Proprio follows the author's
previous-chunk-final-pose convention, including the unexecuted suffix.
"""
import numpy as np
from src.models.x_vla import validate_chunk


class XVLAMemPolicy:
    chunk_size, action_dim = 30, 7

    def __init__(self, url, env, tokenizer, export, timeout=300):
        self.url, self.env, self.tokenizer, self.timeout = url, env, tokenizer, timeout
        self.mem_adapter_manifest = export
        self.export = export

    def reset(self):
        self.proprio = None
        self.raw_chunks = []

    def bind_instruction_only_runtime(self, condition, instruction, renderer):
        expected = "F0" if condition == "F0" else "F1"
        if self.export["arm"] != expected:
            raise ValueError("Wrong XVLA adapter arm")

    def bind_memory_runtime(self, writer, renderer, **kwargs):
        if (self.export["arm"] not in ("F0", "F1")
                or writer.VERSION not in ("rgb_relations_v1", "rgb_relations_sparse_v1")
                or renderer.VERSION != "relation_text_v1"
                or renderer.strategy != "relevant"):
            raise ValueError("XVLA memory runtime differs from V2 shard contract")

    def predict_action_chunk(self, obs, prompt):
        # Match XVLAProcessor.encode_language special tokens and length limit.
        ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) > 50:
            raise ValueError("Memory instruction exceeds XVLA's 50-token limit")
        controller = self.env.last_info["xvla_controller"]
        if controller["use_delta"]:
            raise ValueError("XVLA policy cannot execute delta control")
        import json_numpy
        import requests
        if self.proprio is None:
            self.proprio = np.concatenate((controller["position"], controller["rotation6d"], [0.], np.zeros(10)))
        payload = dict(proprio=json_numpy.dumps(self.proprio), language_instruction=prompt,
            image0=json_numpy.dumps(np.flip(np.flip(obs["agentview_image"], 0), 1)),
            image1=json_numpy.dumps(obs["robot0_eye_in_hand_image"]), domain_id=3, steps=10)
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw = validate_chunk(response.json()["action"])
        self.raw_chunks.append(raw.copy())
        self.proprio[:9] = raw[-1, :9].copy()
        return self.env.decode_xvla(raw)
