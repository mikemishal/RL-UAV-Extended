"""Tests for the FA-LAPG supervised trajectory-prediction dataset
(Phase 27-28): no future leakage, episode-level splitting, deterministic
seed generation, horizon alignment, unavailable-target omission."""

import numpy as np
import pytest

from experiments.fa_lapg_foundation import seed_config
from experiments.fa_lapg_foundation.trajectory_dataset import (
    FEATURE_DIM,
    EpisodeRecord,
    EpisodeStep,
    build_samples,
    rollout_episode,
)
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv


def _synthetic_episode(n_steps: int, dt: float = 0.5) -> EpisodeRecord:
    """Constant-velocity synthetic hostile trajectory, always detected."""
    steps = []
    for i in range(n_steps):
        true_pos = np.array([10.0 + i * 1.0, 0.0, 5.0])
        features = np.zeros(FEATURE_DIM)
        features[0:3] = true_pos  # est_pos == true_pos (noiseless synthetic case)
        features[3:6] = np.array([1.0, 0.0, 0.0])  # est_vel (m/s per dt=0.5 -> 2 m/step, but here 1 m/s * dt=0.5 = 0.5/step)
        steps.append(EpisodeStep(i, i * dt, True, features, true_pos.copy()))
    return EpisodeRecord(seed=123, scenario="synthetic", dt=dt, steps=steps)


def test_horizon_alignment_rejects_non_dt_multiple():
    episode = _synthetic_episode(20)
    with pytest.raises(ValueError):
        build_samples(episode, horizons=(0.3,))


def test_unavailable_future_targets_omitted_not_fabricated():
    episode = _synthetic_episode(5)  # only 5 steps -> horizon=4s (8 steps) never available
    samples = build_samples(episode, horizons=(0.5, 4.0))
    last_sample = samples[-1]
    assert last_sample["targets"][4.0] is None
    # a small, always-reachable horizon near the end is still None if it
    # exceeds the recorded episode, otherwise must be a real ndarray
    for s in samples:
        for h, tgt in s["targets"].items():
            if tgt is not None:
                assert isinstance(tgt, np.ndarray) and tgt.shape == (3,)


def test_no_future_leakage_history_and_cv_pred_use_only_past_steps():
    episode = _synthetic_episode(20)
    samples = build_samples(episode, horizons=(0.5, 1.0))
    sample = samples[10]
    # history's last row must equal the CURRENT step's features, never a later one
    current_features = episode.steps[10].features
    assert np.allclose(sample["history"][-1], current_features)
    # cv_pred computed from current est_pos/vel only, not from any future truth
    expected_cv_half = sample["est_pos_now"] + sample["est_vel_now"] * 0.5
    assert np.allclose(sample["cv_pred"][0.5], expected_cv_half)
    # none of the history rows contain a step index beyond the current one
    for offset in range(1, 8):
        j = 10 - offset
        if j >= 0:
            assert np.allclose(sample["history"][-1 - offset], episode.steps[j].features) or (
                sample["history_mask"][-1 - offset] == 0
            )


def test_history_padding_mask_at_episode_start():
    episode = _synthetic_episode(20)
    samples = build_samples(episode, history_length=8, horizons=(0.5,))
    first_sample = samples[0]  # step 0: only 1 real history row, 7 padded
    assert first_sample["history_mask"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_deterministic_seed_generation_same_seed_same_rollout():
    config = EnvConfig()
    env1 = SoldierEnv(config=config)
    env2 = SoldierEnv(config=config)
    ep1 = rollout_episode(env1, seed=70000, scenario="nominal_open")
    ep2 = rollout_episode(env2, seed=70000, scenario="nominal_open")
    assert len(ep1.steps) == len(ep2.steps)
    for s1, s2 in zip(ep1.steps, ep2.steps):
        assert np.allclose(s1.true_position, s2.true_position)
        assert np.allclose(s1.features, s2.features)


def test_predictor_seed_blocks_disjoint_from_all_locked_ranges():
    seed_config.assert_disjoint_from_locked_ranges()


def test_predictor_seed_blocks_do_not_include_diagnostic_pilot_range():
    pilot_range = set(range(60000, 60200))
    assert not (set(seed_config.train_seeds()) & pilot_range)
    assert not (set(seed_config.val_seeds()) & pilot_range)
    assert not (set(seed_config.test_seeds()) & pilot_range)
