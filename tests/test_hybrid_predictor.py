"""Tests for the Phase 73 hybrid CV/learned predictor: zero correction
below h_switch, learned correction above it, and deterministic behavior."""

import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.hybrid_predictor import DEFAULT_H_SWITCH, HybridPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM


def _nonzero_predictor() -> HorizonConditionedPredictor:
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].bias.fill_(1.0)  # guarantee a nonzero output regardless of input
    model.eval()
    return model


def test_zero_correction_below_switch_horizon():
    hybrid = HybridPredictor(_nonzero_predictor(), h_switch=2.0)
    history = torch.zeros(1, 8, FEATURE_DIM)
    mask = torch.ones(1, 8)
    horizon = torch.tensor([0.5])
    out = hybrid(history, mask, horizon)
    assert torch.allclose(out, torch.zeros_like(out))


def test_learned_correction_above_switch_horizon():
    hybrid = HybridPredictor(_nonzero_predictor(), h_switch=2.0)
    history = torch.zeros(1, 8, FEATURE_DIM)
    mask = torch.ones(1, 8)
    horizon = torch.tensor([4.0])
    out = hybrid(history, mask, horizon)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_exactly_at_switch_horizon_uses_cv():
    hybrid = HybridPredictor(_nonzero_predictor(), h_switch=2.0)
    history = torch.zeros(1, 8, FEATURE_DIM)
    mask = torch.ones(1, 8)
    horizon = torch.tensor([2.0])
    out = hybrid(history, mask, horizon)
    assert torch.allclose(out, torch.zeros_like(out))  # h <= h_switch -> CV


def test_mixed_batch_applies_per_sample_switch():
    hybrid = HybridPredictor(_nonzero_predictor(), h_switch=2.0)
    history = torch.zeros(2, 8, FEATURE_DIM)
    mask = torch.ones(2, 8)
    horizon = torch.tensor([0.5, 4.0])
    out = hybrid(history, mask, horizon)
    assert torch.allclose(out[0], torch.zeros(3))
    assert not torch.allclose(out[1], torch.zeros(3))


def test_default_switch_matches_documented_crossover():
    assert DEFAULT_H_SWITCH == 2.0


def test_deterministic_output():
    hybrid = HybridPredictor(_nonzero_predictor(), h_switch=2.0)
    history = torch.randn(1, 8, FEATURE_DIM)
    mask = torch.ones(1, 8)
    horizon = torch.tensor([5.0])
    out1 = hybrid(history, mask, horizon)
    out2 = hybrid(history, mask, horizon)
    assert torch.allclose(out1, out2)
