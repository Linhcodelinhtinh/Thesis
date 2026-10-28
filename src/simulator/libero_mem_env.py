"""T1/T3 real simulator with locked split, upstream progress and explicit timing."""
from __future__ import annotations
import numpy as np
from src.evaluation.libero_mem_metrics import LiberoMemEvaluator
from src.evaluation.libero_mem_protocol import load_protocol, task_spec, source_digest
from src.simulator.libero_mem_loader import LiberoMemBDDLLoader
from src.training.split_lock import load_split_lock


class LiberoMemEnv:
    use_real_sim = True

    def __init__(self, task_name="T1", *, split_lock, split="test", purpose="evaluation",
                 privileged_oracle=False, use_real_sim=True):
        if use_real_sim is not True:
            raise ValueError("Fixture/fallback simulation is not supported by the benchmark adapter")
        if privileged_oracle:
            raise NotImplementedError("This protocol exposes F0/F1 observations only; no fabricated oracle signals")
        if split not in ("train", "val", "test") or purpose not in ("evaluation", "demo_replay"):
            raise ValueError("Invalid split/purpose")
        if split == "train" and purpose != "demo_replay":
            raise ValueError("Training initial states cannot be used for held-out evaluation")
        self.protocol = load_protocol()
        self.spec = task_spec(self.protocol, task_name)
        self.task_name = self.spec["name"]
        self.split_lock = load_split_lock(split_lock)
        from src.utils.file_hash import sha256_file
        from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
        if self.split_lock["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL):
            raise ValueError("Evaluation protocol differs from split lock")
        self.episodes = [e for e in self.split_lock["episodes"] if e["domain"] == "mem" and
                         e["task"] == self.task_name and e["split"] == split]
        if not self.episodes:
            raise ValueError("No locked initial states in requested split")
        self.target_cycles = self.spec["target_cycles"]
        self.horizon = self.protocol["rollout"]["horizon_policy_actions"]
        self.evaluator = LiberoMemEvaluator(self.task_name, self.target_cycles)
        self._loader = LiberoMemBDDLLoader()
        self._real_env = self._loader.create_env(self.task_name)
        self._is_done = True

    @property
    def upstream(self):
        return self._real_env.env

    def _clear_progress(self):
        self.upstream.reset_subgoal_progress()
        self.upstream._overshot = False  # upstream reset_subgoal_progress omits this field

    def _observe(self):
        # Exactly one increment for this policy-visible observation. Internal
        # reward/step inc=False calls remain upstream's original behavior.
        upstream_return = bool(self.upstream._check_success(inc=True))
        predicate = bool(self.upstream._eval_predicate(self.spec["predicate"]))
        self._is_done = self._step_count >= self.horizon or bool(self.upstream._overshot)
        result = self.evaluator.observe(self._step_count, self.upstream, predicate, self._is_done)
        self.last_info = dict(eval_result=result, upstream_return=upstream_return,
            upstream_live_time=self.upstream._sub_goal_live_time,
            upstream_nonlive_time=self.upstream._sub_goal_nonlive_time,
            final_predicate=predicate, step=self._step_count, protocol_id=self.protocol["protocol_id"],
            split_lock_sha256=self.split_lock["lock_sha256"], source_demo=self.selected_episode["demo"],
            initial_state_sha256=self.selected_episode["initial_state_sha256"])

    def reset(self, initial_state_id=0):
        if not isinstance(initial_state_id, int) or not 0 <= initial_state_id < len(self.episodes):
            raise ValueError("Initial state ID indexes the selected locked split, not an arbitrary demo number")
        self.selected_episode = self.episodes[initial_state_id]
        self._real_env.seed(self.protocol["reset"]["seed"])
        self._real_env.reset()
        self._clear_progress()
        self._real_env.set_init_state(np.asarray(self.selected_episode["initial_state"], dtype=np.float64))
        # Check state restoration before advancing dynamics.
        np.testing.assert_allclose(self._real_env.get_sim_state(), self.selected_episode["initial_state"], rtol=0, atol=1e-12)
        self._clear_progress()
        for _ in range(self.protocol["reset"]["warmup_steps"]):
            obs, _, _, _ = self._real_env.step(self.protocol["reset"]["warmup_action"])
        self._clear_progress()
        self.evaluator.reset()
        self._step_count = 0
        self._observe()
        return self._format_obs(obs)

    def step(self, action):
        if self._is_done:
            raise RuntimeError("Reset before stepping a new or terminated episode")
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected finite native OSC_POSE action7")
        # Preserve the official controller's input saturation. Unnormalized
        # continuous policy predictions may exceed [-1,1]; do not alter them here.
        obs, _, _, _ = self._real_env.step(action)
        self._step_count += 1
        self._observe()
        return self._format_obs(obs), float(self.last_info["eval_result"].full_success), self._is_done, dict(self.last_info)

    @staticmethod
    def _format_obs(obs):
        result = {}
        for key, shape in (("agentview_image", (256, 256, 3)), ("robot0_eye_in_hand_image", (256, 256, 3)),
                           ("robot0_eef_pos", (3,)), ("robot0_eef_quat", (4,)), ("robot0_gripper_qpos", (2,))):
            value = np.asarray(obs[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Unexpected raw observation contract: " + key)
            if "image" in key and value.dtype != np.uint8:
                raise ValueError("Camera must return uint8 RGB")
            result[key] = value
        return result

    def close(self):
        self._real_env.close()
