"""Software fixtures only; these tests never produce benchmark scores."""
import io
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from scripts.evaluate_mem_adapter import parse_args, report, detector_launcher
from src.evaluation.mem_eval_io import EpisodeRecorder, evaluation_roster
from src.evaluation.libero_mem_protocol import ROOT, load_protocol, source_digest
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.evaluation.libero_mem_metrics import SubgoalEvaluationResult
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.simulator.mem_ipc import encode_array, decode_array, deserialize_obs, serialize_obs
from src.simulator.process_libero_mem_env import ProcessLiberoMemEnv
from src.training.split_lock import seal_lock, state_digest


def observation(step=0):
    return dict(agentview_image=np.full((256, 256, 3), step, dtype=np.uint8),
                robot0_eye_in_hand_image=np.full((256, 256, 3), step + 2, dtype=np.uint8),
                robot0_eef_pos=np.array([step, .2, .3], dtype=np.float64),
                robot0_eef_quat=np.array([0., 0., 0., 1.], dtype=np.float64),
                robot0_gripper_qpos=np.array([.03, -.03], dtype=np.float64))


def fixture_lock(tmp_path):
    p = load_protocol()
    rows = []
    for task in p["tasks"]:
        for i in range(2):
            state = [float(i)]
            rows.append(dict(domain="mem", task=task["name"], demo=f"demo_{i}", split="val",
                             initial_state=state, initial_state_sha256=state_digest(state)))
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(seal_lock(dict(episodes=rows,
        protocol_sha256=source_digest(ROOT / "configs/benchmarks/libero_mem_t1_t3_protocol.json")))))
    return path


def test_instructions_from_pinned_bddl_and_all_from_split(tmp_path):
    _, _, roster = evaluation_roster(fixture_lock(tmp_path), "val", ["T1", "T3"], "all")
    assert len(roster) == 4  # not hard-coded ten per task
    assert roster[0]["instruction"] == "pick up the bowl and place it back on the plate"
    assert roster[2]["instruction"] == "lift the bowl and place it back on the plate 3 times"
    with pytest.raises(ValueError, match="Invalid or repeated"):
        evaluation_roster(fixture_lock(tmp_path), "val", ["T3"], "0,0")


def test_roster_rejects_modified_bddl(tmp_path, monkeypatch):
    import src.evaluation.libero_mem_protocol as protocol_module
    original = protocol_module.source_digest
    monkeypatch.setattr(protocol_module, "source_digest", lambda p: "wrong" if str(p).endswith(".bddl") else original(p))
    with pytest.raises(ValueError, match="source differs"):
        evaluation_roster(fixture_lock(tmp_path), "val", ["T3"], "0")


def test_ipc_preserves_bytes_rejects_extra_and_corruption():
    obs = observation()
    decoded = deserialize_obs(serialize_obs(obs))
    for key in obs:
        assert decoded[key].dtype == obs[key].dtype
        assert decoded[key].tobytes() == obs[key].tobytes()
    with pytest.raises(ValueError, match="allowlist"):
        serialize_obs(dict(obs, oracle_signals={}))
    value = encode_array(np.ones(7, dtype=np.float32))
    value["shape"] = [8]
    with pytest.raises(ValueError, match="byte count"):
        decode_array(value)
    with pytest.raises(ValueError, match="finite"):
        encode_array(np.array([np.nan]))


def test_proxy_rejects_stale_reply_and_invalid_action():
    proxy = ProcessLiberoMemEnv.__new__(ProcessLiberoMemEnv)
    proxy._lock, proxy._closed, proxy._request_id, proxy._step = threading.Lock(), False, 0, 0
    from types import SimpleNamespace
    proxy._process = SimpleNamespace(stdin=io.StringIO())
    proxy._receive = lambda: dict(status="ok", id=0)
    with pytest.raises(RuntimeError, match="response ID"):
        proxy._send_command(dict(command="reset"))
    with pytest.raises(ValueError, match="action7"):
        proxy.step(np.zeros(6))


class FixtureEnv:
    task_name = "UNIT_FIXTURE_ONLY"
    def reset(self, initial_state_id=0):
        self.step_id = 0
        return observation()
    def step(self, action):
        self.step_id += 1
        result = SubgoalEvaluationResult(self.task_name, 1, 0, 0., False, False, 0, 1)
        return observation(self.step_id), 0., self.step_id == 3, dict(eval_result=result)


class FixturePolicy:
    chunk_size, action_dim = 50, 7
    def reset(self):
        pass
    def predict_action_chunk(self, obs, prompt):
        return np.arange(350, dtype=np.float32).reshape(50, 7)


def test_recorded_actions_match_execution_and_queries(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episode", dict(classification="UNIT_FIXTURE"), save_video=False)
    result = LiberoMemRolloutRunner(FixtureEnv(), FixturePolicy(), condition="B0", execution_horizon=2,
        max_steps=5, renderer=LiberoMemTextRenderer(), observer=recorder).run_episode("unit", "lift bowl")
    recorder.finish(result)
    with np.load(tmp_path / "episode/trajectory.npz", allow_pickle=False) as data:
        assert data["actions"].shape == (3, 7)
        assert data["raw_proprioception_pos_quatxyzw_gripper"].shape == (4, 9)
        assert data["query_steps"].tolist() == [0, 2]
        np.testing.assert_array_equal(data["actions"][2], data["action_chunks"][1, 0])
    status = json.loads((tmp_path / "episode/status.json").read_text())
    assert status["status"] == "complete"
    assert "queries.jsonl" in status["files_sha256"]
    with pytest.raises(FileExistsError):
        EpisodeRecorder(tmp_path / "episode", {})


def test_logger_failure_propagates_and_marks_failed(tmp_path, monkeypatch):
    recorder = EpisodeRecorder(tmp_path / "episode", {}, save_video=False)
    monkeypatch.setattr(recorder, "_line", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full") as exc:
        LiberoMemRolloutRunner(FixtureEnv(), FixturePolicy(), condition="B0",
            renderer=LiberoMemTextRenderer(), observer=recorder).run_episode("unit", "lift bowl")
    recorder.finish(error=exc.value)
    assert json.loads((tmp_path / "episode/status.json").read_text())["status"] == "failed"


def test_video_can_be_disabled_and_worker_env_required():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "out"])
    args = parse_args(["--output-dir", "out", "--sim-python", "sim", "--detector-python", "det", "--no-save-video"])
    assert args.save_video is False


def test_preflight_failure_is_recorded_without_score(tmp_path):
    from scripts.evaluate_mem_adapter import main
    with pytest.raises(FileNotFoundError):
        main(["--output-dir", str(tmp_path / "run"), "--split-lock", str(tmp_path / "missing.json"),
              "--conditions", "B0", "--sim-python", sys.executable, "--preflight-only"])
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_episodes"] == 0
    assert "failed" in (tmp_path / "run/report.md").read_text()


def test_video_artifact_decodes_with_expected_frames(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    recorder = EpisodeRecorder(tmp_path / "video", {}, save_video=True)
    recorder("observation", step=0, observation=observation(), info={})
    recorder("observation", step=1, observation=observation(1), info={})
    recorder.finish(diagnostic=True)
    with imageio.get_reader(str(tmp_path / "video/agentview_image.mp4")) as video:
        frames = list(video.iter_data())
    assert len(frames) == 2 and frames[0].shape == (256, 256, 3)
    assert json.loads((tmp_path / "video/status.json").read_text())["status"] == "diagnostic_complete"


def test_worker_bad_request_returns_error_and_exits():
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/libero_mem_sim_worker.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = proc.communicate(json.dumps(dict(id=7, command="unknown")) + "\n", timeout=120)
    messages = [json.loads(line) for line in output.splitlines()]
    assert messages[0] == dict(status="ready", protocol_version=1)
    assert messages[1]["status"] == "error" and "Nonsequential" in messages[1]["error"]
    assert proc.returncode != 0


def test_reports_pair_exact_state_hashes(tmp_path):
    rows = []
    for i in [1, 10]:
        for cond in ["B0", "F1"]:
            rows.append(dict(episode_id=f"state{i}", task="T3", condition=cond, split="val", state_idx=i,
                source_demo=f"demo_{i}", initial_state_sha256=str(i), success=cond == "F1", scr=.5,
                overshot=False, steps=3, policy_ms=1., wall_time_s=1.))
    report(tmp_path, rows, 4, "complete")
    text = (tmp_path / "report.md").read_text()
    assert "Complete B0/F1 pairs: 2" in text
    assert "+1.000" in text
    assert "NON-COMPARABLE_OFFICIAL_PAPER" in text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec wrapper")
def test_detector_launcher_preserves_venv_executable(tmp_path):
    link = tmp_path / "python-link"
    link.symlink_to(sys.executable)
    wrapper = detector_launcher(link, tmp_path)
    assert str(link) in Path(wrapper).read_text()
    result = subprocess.check_output([wrapper, "-c", "print('wrapper-ok')"], text=True)
    assert result.strip() == "wrapper-ok"
