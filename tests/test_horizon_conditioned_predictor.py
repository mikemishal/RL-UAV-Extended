"""Tests for the horizon-conditioned predictor (Phase 48-50): arbitrary
horizon query, zero-correction behavior, deterministic inference, horizon
normalization, and synthetic constant-acceleration motion prediction."""

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_dataset import (
    HorizonTrainConfig,
    build_horizon_samples,
    train_horizon_predictor,
)
from experiments.fa_lapg_foundation.horizon_conditioned_predictor import (
    HORIZON_NORMALIZATION_SCALE,
    HorizonConditionedPredictor,
)
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, EpisodeRecord, EpisodeStep

DT = 0.5


def test_arbitrary_horizon_query_accepts_non_grid_aligned_values():
    model = HorizonConditionedPredictor()
    history = torch.zeros(3, 8, FEATURE_DIM)
    mask = torch.ones(3, 8)
    horizon = torch.tensor([0.37, 5.123, 11.9])  # NOT multiples of dt=0.5
    out = model(history, mask, horizon)
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()


def test_zero_correction_when_decoder_output_layer_zeroed():
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    history = torch.randn(2, 8, FEATURE_DIM)
    mask = torch.ones(2, 8)
    horizon = torch.tensor([1.0, 8.0])
    out = model(history, mask, horizon)
    assert torch.allclose(out, torch.zeros_like(out))


def test_deterministic_inference_same_input_same_output():
    torch.manual_seed(0)
    model = HorizonConditionedPredictor()
    model.eval()
    history = torch.randn(2, 8, FEATURE_DIM)
    mask = torch.ones(2, 8)
    horizon = torch.tensor([3.0, 3.0])
    with torch.no_grad():
        out1 = model(history, mask, horizon)
        out2 = model(history, mask, horizon)
    assert torch.allclose(out1, out2)


def test_horizon_normalization_scale_matches_module_constant():
    assert HORIZON_NORMALIZATION_SCALE == 16.0


def _episode_with_acceleration(n_steps: int, v0: np.ndarray, a: np.ndarray, seed: int) -> EpisodeRecord:
    steps = []
    p0 = np.zeros(3)
    for i in range(n_steps):
        t = i * DT
        pos = p0 + v0 * t + 0.5 * a * t * t
        vel = v0 + a * t
        features = np.zeros(FEATURE_DIM)
        features[0:3] = pos
        features[3:6] = vel
        steps.append(EpisodeStep(i, t, True, features, pos.copy()))
    return EpisodeRecord(seed=seed, scenario="synthetic", dt=DT, steps=steps)


def test_build_horizon_samples_targets_match_analytic_trajectory():
    v0 = np.array([2.0, 0.0, 0.0])
    a = np.array([1.0, -0.5, 0.0])
    episode = _episode_with_acceleration(60, v0=v0, a=a, seed=1)
    rng = np.random.default_rng(0)
    samples = build_horizon_samples(episode, rng, n_horizons_per_step=2, horizon_min=0.5, horizon_max=10.0)
    assert len(samples) > 0
    for s in samples:
        # target_delta = true_future_pos - cv_pred; true motion here is
        # constant-acceleration, so target_delta should closely match the
        # analytic 0.5*a*h^2 term (interpolation is linear between coarse
        # dt=0.5 samples, so allow modest tolerance).
        h = s["horizon"]
        expected_quadratic_term = 0.5 * a * h * h
        assert np.linalg.norm(s["target_delta"] - expected_quadratic_term) < 2.0


def test_overfit_small_synthetic_dataset_reduces_loss():
    samples = []
    rng = np.random.default_rng(0)
    for seed in range(5):
        a = np.array([1.0 + 0.1 * seed, -0.5, 0.0])
        episode = _episode_with_acceleration(40, v0=np.array([2.0, 0.0, 0.0]), a=a, seed=seed)
        samples.extend(build_horizon_samples(episode, rng, n_horizons_per_step=3))

    torch.manual_seed(0)
    model = HorizonConditionedPredictor(hidden_size=16, decoder_hidden=16)
    losses = train_horizon_predictor(model, samples, HorizonTrainConfig(epochs=40, batch_size=32, learning_rate=5e-3))
    assert losses[-1] < losses[0] * 0.5
