"""Test for Phase 66 policy-shift analysis: predictor error correctly
grouped and summarized by which controller generated the rollout."""

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.policy_shift_analysis import (
    predictor_error_under_policy,
    summarize_by_rollout_policy,
)
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, EpisodeRecord, EpisodeStep

DT = 0.5


def _zeroed_predictor() -> HorizonConditionedPredictor:
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    model.eval()
    return model


def _episode(seed: int, scenario: str, v0: np.ndarray) -> EpisodeRecord:
    steps = []
    for i in range(30):
        t = i * DT
        pos = v0 * t
        features = np.zeros(FEATURE_DIM)
        features[0:3] = pos
        features[3:6] = v0
        steps.append(EpisodeStep(i, t, True, features, pos.copy()))
    return EpisodeRecord(seed=seed, scenario=scenario, dt=DT, steps=steps)


def test_predictor_error_under_policy_groups_by_scenario_and_horizon():
    model = _zeroed_predictor()  # zero correction -> error reduces to CV error
    episodes = [_episode(1, "nominal_open", np.array([2.0, 0.0, 0.0]))]
    df = predictor_error_under_policy(model, episodes, eval_horizons=(1.0, 4.0))
    assert set(df["horizon"].unique()) == {1.0, 4.0}
    assert (df["error"] < 1e-6).all()  # constant velocity, zero correction -> exact CV match


def test_summarize_by_rollout_policy_distinguishes_controllers():
    model = _zeroed_predictor()
    lead_episodes = [_episode(1, "nominal_open", np.array([2.0, 0.0, 0.0]))]
    ra_lapg_episodes = [_episode(2, "nominal_open", np.array([2.0, 1.0, 0.0]))]

    per_policy = {
        "lead": predictor_error_under_policy(model, lead_episodes, eval_horizons=(1.0,)),
        "ra_lapg": predictor_error_under_policy(model, ra_lapg_episodes, eval_horizons=(1.0,)),
    }
    summary = summarize_by_rollout_policy(per_policy)
    assert set(summary["rollout_policy"]) == {"lead", "ra_lapg"}
    assert len(summary) == 2  # one row per (policy, scenario, horizon) combination here
