"""Unit transport/protocol checks only; fixtures never generate benchmark scores."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.xvla_v2.evaluate_memory import comparison
from src.models.xvla_mem import XVLAMemPolicy


def test_payload_matches_pinned_author_client_and_keeps_native_proprio(monkeypatch):
    # Codec is opaque here; compare both clients using the same deterministic
    # codec so this check needs neither inference nor simulator dependencies.
    import json
    import sys
    json_numpy = SimpleNamespace(dumps=lambda x: json.dumps(np.asarray(x).tolist()),
                                 loads=lambda x: np.asarray(json.loads(x)))
    monkeypatch.setitem(sys.modules, "json_numpy", json_numpy)
    import requests
    path = Path(__file__).resolve().parents[2] / "external/X-VLA/evaluation/libero/libero_client.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Execute the author's payload code without importing simulator/model stacks.
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ("_flip_agentview", "ClientModel")]
    ns = dict(np=np, json_numpy=json_numpy, collections=__import__("collections"),
              LiberoAbsActionProcessor=lambda: None)
    exec(compile("from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)
    author = ns["ClientModel"]("127.0.0.1", 8010)
    obs = dict(agentview_image=np.arange(256*256*3, dtype=np.uint8).reshape(256,256,3),
               robot0_eye_in_hand_image=np.zeros((256,256,3), dtype=np.uint8))
    pose = dict(position=[.1,.2,.3], rotation6d=[1.,0.,0.,0.,1.,0.], use_delta=False)
    raw = np.zeros((30,20)); raw[:,3] = raw[:,7] = 1.; raw[:,9] = .5
    raw[-1,:3] = [.6,.7,.8]
    sent = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return dict(action=raw.tolist())
    monkeypatch.setattr(requests, "post", lambda url, **kw: (sent.append(kw["json"]) or Response()))
    env = SimpleNamespace(last_info=dict(xvla_controller=pose), decode_xvla=lambda x: x[:, :7])
    policy = XVLAMemPolicy("http://127.0.0.1:8010/act", env, lambda *a, **k: dict(input_ids=[1]), dict(arm="F1"))
    policy.reset()
    prompt = "lift the bowl Memory: bowl: 1 observed returns."
    expected = author._format_query(dict(obs, robo_pos=np.array(pose["position"]), robo_ori=np.array(pose["rotation6d"])), prompt)
    policy.predict_action_chunk(obs, prompt)
    assert sent[0] == expected
    policy.predict_action_chunk(obs, prompt)
    np.testing.assert_array_equal(json_numpy.loads(sent[1]["proprio"])[:9], raw[-1,:9])
    policy.reset()
    assert policy.proprio is None and policy.raw_chunks == []


def test_prompt_overflow_fails_before_request():
    policy = XVLAMemPolicy("unused", None, lambda *a, **kw: dict(input_ids=list(range(51))), dict(arm="F1"))
    with pytest.raises(ValueError, match="50-token"):
        policy.predict_action_chunk({}, "too long")


def test_absolute_mode_is_enabled_only_after_every_delta_warmup(monkeypatch):
    from src.simulator.xvla_mem_env import XVLAMemEnv
    from src.simulator.libero_mem_env import LiberoMemEnv
    controller = SimpleNamespace(use_delta=False, name="OSC_POSE", impedance_mode="fixed",
                                 ee_pos=np.zeros(3), ee_ori_mat=np.eye(3))
    env = object.__new__(XVLAMemEnv)
    env._real_env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)],
                                                       control_freq=20, horizon=1020))
    resets = []
    def reset(self, index):
        assert controller.use_delta is True
        resets.append(index)
        self.last_info = {}
        return {"unit_observation": index}
    monkeypatch.setattr(LiberoMemEnv, "reset", reset)
    for state in (0, 1):
        assert env.reset(state) == {"unit_observation": state}
        assert controller.use_delta is False
        assert env.last_info["xvla_controller"]["use_delta"] is False
    assert resets == [0, 1]


def test_comparison_refuses_unverified_or_changed_roster():
    assert comparison(None, {}, [], 42, {}, {}, {})["status"] == "UNVERIFIED_NO_REFERENCE"
    cfg = dict(projection="relation_text_v1", strategy="relevant")
    contract = dict(writer="rgb_relations_v1", writer_config=cfg, writer_code_sha256={})
    row = dict(task="T1", state_idx=0, source_demo="demo_101", initial_state_sha256="abc", instruction="lift")
    shared = ("src/simulator/libero_mem_env.py", "src/simulator/libero_mem_loader.py",
              "src/evaluation/libero_mem_rollout.py", "src/evaluation/libero_mem_metrics.py")
    hashes = dict.fromkeys(shared, "hash")
    reference = dict(status="complete", arguments=dict(policy_seed=42, decision_mode=None), protocol={},
        roster=[row], source_sha256=hashes, treatments=dict(F1=dict(memory_source="rgb_relations_v1",
        projection="relation_text_v1", projection_strategy="relevant")), environments={})
    assert comparison(reference, {}, [row], 42, hashes, {}, contract)["status"] == "MATCHED_TASK_PROTOCOL_CROSS_BACKBONE"
    assert comparison(reference, {}, [], 42, hashes, {}, contract)["status"] == "DESCRIPTIVE_ONLY"
