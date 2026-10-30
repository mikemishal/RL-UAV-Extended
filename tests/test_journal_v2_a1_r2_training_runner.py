"""Power-loss-safe rollout checkpointing for `a1_v2_training_v2`: atomic commits, recovery after simulated power loss,
corruption fallback, exact-rollout accounting, isolation. Uses a TINY dummy environment (not A1 training)."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import experiments.journal_v2_a1_r2 as R
import experiments.journal_v2_a1_r2_training as T

SPEC = T.Spec(n_envs=2, n_steps=8, n_rollouts=3)
PPO_KW = dict(n_steps=8, batch_size=16, n_epochs=1, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)
DESC = {"training_protocol": R.TRAINING_PROTOCOL, "reward_version": R.REWARD_VERSION, "action_version": R.ACTION_VERSION,
        "observation_version": R.OBSERVATION_VERSION, "theta_max_deg": 15.0}


class Dummy(gym.Env):
    """8-step episodes aligned with rollouts; deterministic observations."""

    observation_space = spaces.Box(-1.0, 1.0, (37,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.full(37, np.sin(self.t), dtype=np.float32)

    def step(self, action):
        self.t += 1
        done = self.t >= 8
        info = {}
        if done:
            info["a1_episode"] = dict(outcome="timeout", return_undiscounted=-1.0, return_discounted=-1.0, path_excess=1.0,
                                      residual_angle_mean_deg=0.0, ceiling_occupancy=0.0, sustained_hover=False, ceiling_hover=False,
                                      components=dict(terminal=-10.0, progress=0.0, path=0.0, step=-0.8))
        return self._obs(), float(-np.abs(action).sum()), done, False, info


def factory(resume_index: int):
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([Dummy for _ in range(SPEC.n_envs)])


def run(tmp_path, hook=None, seed=201, scratch=None):
    return T.train_seed(seed, scratch_root=scratch or tmp_path / "scratch", results_root=tmp_path / "results", spec=SPEC,
                        env_factory=factory, ppo_kwargs=PPO_KW, description=DESC, fault_hook=hook)


def crash_at(stage, k):
    def hook(s, kk):
        if s == stage and kk == k:
            raise T.SimulatedPowerLoss(f"{s}@{k}")

    return hook


def params(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path).removesuffix(".zip"), device="cpu")
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def test_canonical_budget_accounting():
    assert T.CANONICAL_SPEC.steps_per_rollout == 32_768 and T.N_ROLLOUTS == 13
    assert T.CANONICAL_SPEC.total_timesteps == 425_984 == 13 * 16 * 2048


def test_clean_run_commits_exactly_n_rollouts_and_final_artifacts(tmp_path):
    final = run(tmp_path)
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    assert final["num_timesteps"] == SPEC.total_timesteps == 48 and final["rollouts_complete"] == 3
    for k in (1, 2, 3):
        for directory in (layout.scratch_rollouts, layout.canonical_rollouts):
            manifest = T.verify_rollout(directory, k, seed=201, spec=SPEC, description=DESC)
            assert manifest and manifest["num_timesteps"] == k * 16
    assert not list(layout.scratch_rollouts.glob("rollout_04*"))
    assert T.sha256_file(layout.final_model) == final["model_sha256"] == T.verify_rollout(layout.canonical_rollouts, 3, seed=201, spec=SPEC, description=DESC)["model_sha256"]
    assert final["canonical"] is False and final["reward_version"] == "a1_v2_reward_v2" and final["action_version"] == "a1_v2_action_v2"
    assert (layout.logs / "episodes_all_rollouts.jsonl").exists()


def test_a_completed_seed_is_never_retrained_or_overwritten(tmp_path):
    first = run(tmp_path)
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    before = T.sha256_file(layout.final_model)
    again = run(tmp_path)
    assert again["model_sha256"] == first["model_sha256"] == before == T.sha256_file(layout.final_model)


@pytest.mark.parametrize("stage,k,expected_resume_from", [("after_learn", 2, 1), ("after_model_written", 2, 1), ("before_manifest", 2, 1),
                                                          ("after_manifest", 2, 2), ("after_commit", 2, 2), ("before_learn", 3, 2)])
def test_power_loss_at_every_stage_resumes_from_latest_complete_rollout(tmp_path, stage, k, expected_resume_from):
    with pytest.raises(T.SimulatedPowerLoss):
        run(tmp_path, crash_at(stage, k))
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    assert not layout.final_model.exists()
    final = run(tmp_path)
    assert final["num_timesteps"] == 48 and final["rollouts_complete"] == 3
    manifests = [T.verify_rollout(layout.canonical_rollouts, i, seed=201, spec=SPEC, description=DESC) for i in (1, 2, 3)]
    assert all(manifests) and not list(layout.scratch_rollouts.glob("rollout_04*"))
    segments = [m["resume_index"] for m in manifests]
    assert segments[expected_resume_from - 1] == 0 and max(segments) == 1  # exactly one resume, no restart from scratch
    assert not list(layout.scratch_rollouts.glob(".*.tmp*"))


def test_partial_rollout_files_are_discarded_not_trusted(tmp_path):
    with pytest.raises(T.SimulatedPowerLoss):
        run(tmp_path, crash_at("before_manifest", 2))
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    names = T.rollout_names(2)
    assert (layout.scratch_rollouts / names["model"]).exists() and not (layout.scratch_rollouts / names["manifest"]).exists()
    assert T.verify_rollout(layout.scratch_rollouts, 2, seed=201, spec=SPEC, description=DESC) is None
    completed, _ = T.recover(layout, spec=SPEC, description=DESC)
    assert completed == 1 and not (layout.scratch_rollouts / names["model"]).exists()


def test_corrupted_latest_checkpoint_falls_back_to_the_previous_one(tmp_path):
    with pytest.raises(T.SimulatedPowerLoss):
        run(tmp_path, crash_at("after_commit", 2))
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    for directory in (layout.scratch_rollouts, layout.canonical_rollouts):  # damage both copies of rollout 2
        model = directory / T.rollout_names(2)["model"]
        model.write_bytes(model.read_bytes()[:50])
    completed, manifest = T.recover(layout, spec=SPEC, description=DESC)
    assert completed == 1 and manifest["rollout"] == 1
    assert run(tmp_path)["num_timesteps"] == 48


def test_lost_scratch_is_restored_from_the_verified_canonical_mirror(tmp_path):
    import shutil

    with pytest.raises(T.SimulatedPowerLoss):
        run(tmp_path, crash_at("after_commit", 2))
    shutil.rmtree(tmp_path / "scratch")
    final = run(tmp_path)
    assert final["num_timesteps"] == 48 and final["rollouts_complete"] == 3


def test_resumed_training_matches_uninterrupted_training_exactly(tmp_path):
    """RNG state is restored, so a power loss changes nothing about the trained weights (episodes align with rollouts)."""
    clean = run(tmp_path / "a", seed=201)
    with pytest.raises(T.SimulatedPowerLoss):
        run(tmp_path / "b", crash_at("after_learn", 3), seed=201)
    resumed = run(tmp_path / "b", seed=201)
    pa = params(T.make_layout(201, tmp_path / "a" / "scratch", tmp_path / "a" / "results").final_model)
    pb = params(T.make_layout(201, tmp_path / "b" / "scratch", tmp_path / "b" / "results").final_model)
    assert resumed["num_timesteps"] == clean["num_timesteps"] == 48
    assert all(bool((pa[k] == pb[k]).all()) for k in pa)


def test_a_non_finite_update_is_never_committed(tmp_path, monkeypatch):
    import torch

    original = T.assert_model_finite

    def poison(model):
        with torch.no_grad():
            next(model.policy.parameters()).fill_(float("nan"))
        original(model)

    monkeypatch.setattr(T, "assert_model_finite", poison)
    with pytest.raises(FloatingPointError):
        run(tmp_path)
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    assert not list(layout.scratch_rollouts.glob("rollout_01.manifest.json"))


def test_manifest_with_a_different_reward_action_or_seed_is_not_complete(tmp_path):
    run(tmp_path)
    layout = T.make_layout(201, tmp_path / "scratch", tmp_path / "results")
    assert T.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=SPEC, description=DESC)
    assert T.verify_rollout(layout.scratch_rollouts, 1, seed=202, spec=SPEC, description=DESC) is None
    assert T.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=SPEC, description={**DESC, "reward_version": "a1_v2_reward_v1"}) is None
    assert T.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=SPEC, description={**DESC, "action_version": "other"}) is None
    assert T.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=T.Spec(2, 8, 3).__class__(2, 4, 3), description=DESC) is None


def test_failed_v1_namespace_and_other_v1_trees_are_never_writable(tmp_path):
    with pytest.raises(T.RunConflictError):
        T.make_layout(201, tmp_path, R.FAILED_V1_RESULTS_ROOT)
    with pytest.raises(T.RunConflictError):
        T.assert_isolated(R.FAILED_V1_RESULTS_ROOT / "seed_201" / "checkpoints" / "final_model.zip")
    with pytest.raises(T.RunConflictError):
        T.assert_isolated(T.PROJECT_ROOT / "results" / "adaptive_attacker_400k" / "seed_201")
    with pytest.raises(T.RunConflictError):
        T.assert_isolated(T.PROJECT_ROOT / "results" / "elsewhere")
    assert R.RESULTS_ROOT != R.FAILED_V1_RESULTS_ROOT and R.RESULTS_ROOT.name == "a1_v2_training_v2"
    with pytest.raises(ValueError):
        T.make_layout(999, tmp_path, tmp_path)


def test_scratch_is_local_non_synced_and_atomic(tmp_path):
    scratch = T.select_scratch_root()
    assert not T._is_synced(scratch) and scratch.exists()
    assert T._is_synced(Path(r"C:\Users\x\OneDrive\Documents\y")) and not T._is_synced(Path(r"C:\Users\Public\a"))
    target = tmp_path / "f.bin"
    T.atomic_write_bytes(target, b"abc")
    T.atomic_write_bytes(target, b"defg")
    assert target.read_bytes() == b"defg" and [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_rng_capture_restore_roundtrip():
    import random

    import torch

    torch.manual_seed(3), np.random.seed(3), random.seed(3)
    blob = T.capture_rng()
    expected = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    torch.manual_seed(99), np.random.seed(99), random.seed(99)
    T.restore_rng(blob)
    assert (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random()) == expected
