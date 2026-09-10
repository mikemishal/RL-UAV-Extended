"""Tests for the GRU residual predictor (Phase 29C/30/31): tensor shapes,
deterministic evaluation, and an overfit-small-synthetic sanity check."""

import numpy as np
import torch

from experiments.fa_lapg_foundation.baseline_predictors import cv_predict
from experiments.fa_lapg_foundation.gru_predictor import GRUResidualPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, HORIZONS, EpisodeRecord, EpisodeStep, build_samples
from experiments.fa_lapg_foundation.train_predictor import TrainConfig, train_predictor

DT = 0.5


def test_forward_pass_output_shape():
    model = GRUResidualPredictor()
    batch = 4
    history = torch.zeros(batch, 8, FEATURE_DIM)
    mask = torch.ones(batch, 8)
    out = model(history, mask)
    assert out.shape == (batch, len(HORIZONS), 3)


def test_deterministic_evaluation_same_input_same_output():
    torch.manual_seed(0)
    model = GRUResidualPredictor()
    model.eval()
    history = torch.randn(2, 8, FEATURE_DIM)
    mask = torch.ones(2, 8)
    with torch.no_grad():
        out1 = model(history, mask)
        out2 = model(history, mask)
    assert torch.allclose(out1, out2)


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


def test_overfit_small_synthetic_dataset_reduces_loss_substantially():
    """Sanity check: training on a tiny fixed synthetic accelerating
    trajectory (which CV mispredicts) should let the GRU-residual model
    reduce its own training loss by a large factor within a modest number
    of epochs -- proof the wiring (forward/backward/masking) is correct."""
    samples = []
    for seed in range(5):
        a = np.array([1.0 + 0.1 * seed, -0.5, 0.0])
        episode = _episode_with_acceleration(30, v0=np.array([2.0, 0.0, 0.0]), a=a, seed=seed)
        samples.extend(build_samples(episode))

    torch.manual_seed(0)
    model = GRUResidualPredictor(hidden_size=16)

    from experiments.fa_lapg_foundation.train_predictor import ResidualPredictorDataset, masked_mse_loss

    dataset = ResidualPredictorDataset(samples)
    first_batch = dataset[0]
    with torch.no_grad():
        initial_pred = model(first_batch["history"].unsqueeze(0), first_batch["history_mask"].unsqueeze(0))
        initial_loss = masked_mse_loss(
            initial_pred, first_batch["target_delta"].unsqueeze(0), first_batch["target_valid"].unsqueeze(0)
        ).item()

    losses = train_predictor(model, samples, TrainConfig(epochs=40, batch_size=32, learning_rate=5e-3, seed=0))

    assert losses[-1] < losses[0] * 0.5, f"expected substantial loss reduction, got {losses[0]} -> {losses[-1]}"
