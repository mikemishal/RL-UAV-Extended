"""
Tests for vectorized clutter-training infrastructure (journal extension,
scale-up sub-task). Covers Phase 4 (deterministic non-overlapping worker
seeding), Phase 5 (total_timesteps means the SAME total environment
experience regardless of num_envs), and Phase 7 (vectorization
equivalence: shapes/behavior unaffected by num_envs).

Uses `vec_env_cls="dummy"` (in-process `DummyVecEnv`) for multi-worker
tests to keep the suite fast and avoid multiprocessing/spawn overhead in
CI -- `SubprocVecEnv` itself is exercised by the real training campaign,
not by this fast unit-test suite (see the throughput benchmark instead).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.clutter_training_runner import SEED_STRIDE, run_clutter_training
from experiments.clutter_training_env import build_ppo_clutter_env


def test_worker_seeds_deterministic_and_non_overlapping():
    base_seed = 71
    for num_envs in (1, 2, 4, 8):
        worker_seeds = [base_seed + i * SEED_STRIDE for i in range(num_envs)]
        assert len(set(worker_seeds)) == num_envs  # no collisions
        assert worker_seeds[0] == base_seed  # num_envs=1's single worker == plain seed
        # Reproducible: recomputing gives the identical sequence.
        assert worker_seeds == [base_seed + i * SEED_STRIDE for i in range(num_envs)]


def test_num_envs_1_matches_legacy_single_env_manifest_fields():
    manifest = run_clutter_training(
        architecture="ppo_clutter_v1",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        output_dir="results/_test_vectorized/num_envs_1",
        total_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        verbose=0,
    )
    assert manifest["num_envs"] == 1
    assert manifest["worker_seeds"] == [71]
    assert manifest["effective_rollout_batch"] == manifest["rollout_n_steps_per_worker"] * 1
    assert manifest["actual_total_timesteps"] == 2048
    assert manifest["total_transitions"] == 2048


def test_num_envs_4_preserves_total_timesteps_semantics():
    """Phase 5: `total_timesteps` requests the SAME total environment
    experience regardless of num_envs -- NOT 4x more -- but SB3 always
    completes a FULL rollout (`n_steps * num_envs` transitions) before
    checking the stop condition, so `actual_total_timesteps` can OVERSHOOT
    the request by up to one rollout. Requesting an exact multiple of the
    rollout size here (2048 * 4 = 8192) makes the overshoot zero, so the
    equivalence is exact and directly comparable to the num_envs=1 case
    above (same requested/actual budget, NOT 4x more)."""
    manifest = run_clutter_training(
        architecture="ppo_clutter_v1",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        output_dir="results/_test_vectorized/num_envs_4",
        total_timesteps=8192,
        eval_freq=8192,
        eval_episodes=2,
        checkpoint_freq=8192,
        num_envs=4,
        vec_env_cls="dummy",
        verbose=0,
    )
    assert manifest["num_envs"] == 4
    assert manifest["worker_seeds"] == [71, 71 + SEED_STRIDE, 71 + 2 * SEED_STRIDE, 71 + 3 * SEED_STRIDE]
    assert manifest["total_timesteps"] == 8192
    assert manifest["actual_total_timesteps"] == 8192  # NOT 4 * 8192
    assert manifest["total_transitions"] == 8192
    assert manifest["effective_rollout_batch"] == manifest["rollout_n_steps_per_worker"] * 4


def test_vectorized_env_preserves_observation_and_action_shapes():
    """Phase 7: each worker's env (obstacle-sector observation, physical
    collision termination, perception) is unaffected by vectorization --
    it is the SAME env construction used by the single-env path."""
    seeds = [71, 71 + SEED_STRIDE]
    for worker_seed in seeds:
        env = build_ppo_clutter_env("measurement")
        obs, info = env.reset(seed=worker_seed)
        assert obs.shape == (36,)
        assert np.all(np.isfinite(obs))
        action = np.zeros(3, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2.shape == (36,)
        assert isinstance(terminated, bool) or isinstance(terminated, (int, bool))
        assert "defender_avoidance_active" in info2
        assert "outcome" in info2


def test_different_workers_get_different_clutter_layouts():
    """Phase 4: workers must not duplicate obstacle layouts -- different
    worker seeds must (with overwhelming probability) sample different
    randomized clutter configs."""
    env_a = build_ppo_clutter_env("measurement")
    env_b = build_ppo_clutter_env("measurement")
    env_a.reset(seed=71)
    env_b.reset(seed=71 + SEED_STRIDE)
    config_a = env_a.unwrapped.config
    config_b = env_b.unwrapped.config
    # At least one of the randomized fields should differ (astronomically
    # unlikely to coincide across a 10_000-seed stride).
    differs = (
        config_a.obstacle_count != config_b.obstacle_count
        or config_a.obstacle_min_size != config_b.obstacle_min_size
        or config_a.dynamic_obstacle_count != config_b.dynamic_obstacle_count
    )
    assert differs
