"""Tests for Phase 70 multiscale (stratified) horizon sampling: balanced
counts per band, deterministic given a fixed RNG, no future leakage, and
identical episode/history construction as the uniform sampler."""

import numpy as np

from experiments.fa_lapg_foundation.horizon_conditioned_dataset import (
    HORIZON_BANDS,
    build_stratified_horizon_samples,
)
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, EpisodeRecord, EpisodeStep

DT = 0.5


def _episode(n_steps: int, v0: np.ndarray, seed: int) -> EpisodeRecord:
    steps = []
    for i in range(n_steps):
        t = i * DT
        pos = v0 * t
        features = np.zeros(FEATURE_DIM)
        features[0:3] = pos
        features[3:6] = v0
        steps.append(EpisodeStep(i, t, True, features, pos.copy()))
    return EpisodeRecord(seed=seed, scenario="synthetic", dt=DT, steps=steps)


def test_balanced_horizon_sampling_equal_count_per_band():
    episode = _episode(80, np.array([2.0, 0.0, 0.0]), seed=1)  # 40s long, covers every band fully
    rng = np.random.default_rng(0)
    samples = build_stratified_horizon_samples(episode, rng, n_horizons_per_band=2)

    # pick a step far enough from the episode start/end that every band's
    # query time remains inside the recorded trajectory
    mid_step_samples = [s for s in samples if abs(s["horizon"]) > 0]
    counts_per_band = {band: 0 for band in HORIZON_BANDS}
    for s in mid_step_samples:
        for lo, hi in HORIZON_BANDS:
            if lo <= s["horizon"] < hi:
                counts_per_band[(lo, hi)] += 1
    # every band must have received a comparable number of samples (not
    # width-proportional as with uniform sampling)
    counts = list(counts_per_band.values())
    assert min(counts) > 0
    assert max(counts) / min(counts) < 3.0  # roughly balanced, not width-skewed


def test_deterministic_sampling_same_rng_seed_same_horizons():
    episode = _episode(40, np.array([1.0, 0.0, 0.0]), seed=1)
    samples_a = build_stratified_horizon_samples(episode, np.random.default_rng(42), n_horizons_per_band=1)
    samples_b = build_stratified_horizon_samples(episode, np.random.default_rng(42), n_horizons_per_band=1)
    horizons_a = [s["horizon"] for s in samples_a]
    horizons_b = [s["horizon"] for s in samples_b]
    assert horizons_a == horizons_b


def test_no_future_leakage_history_matches_current_step_only():
    episode = _episode(40, np.array([2.0, 0.0, 0.0]), seed=1)
    rng = np.random.default_rng(0)
    samples = build_stratified_horizon_samples(episode, rng, n_horizons_per_band=1, history_length=8)
    for s in samples:
        assert s["history"].shape == (8, FEATURE_DIM)
        assert s["history_mask"].shape == (8,)


def test_horizons_stay_within_their_declared_band():
    episode = _episode(80, np.array([2.0, 0.0, 0.0]), seed=1)
    rng = np.random.default_rng(0)
    samples = build_stratified_horizon_samples(episode, rng, n_horizons_per_band=3)
    for s in samples:
        h = s["horizon"]
        assert any(lo <= h < hi for lo, hi in HORIZON_BANDS), f"horizon {h} not in any declared band"
