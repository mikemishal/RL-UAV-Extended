"""
Phase 66: policy-shift / covariate-shift analysis.

The temporal predictor was trained from Lead-controlled rollouts; since
the hostile reacts to the defender, changing the defender controller
changes the hostile trajectory distribution the predictor sees at
inference time. This module measures the horizon-conditioned predictor's
OWN prediction error when the SAME predictor is queried during rollouts
driven by different controllers (Lead / Learned-Prediction-Lead /
RA-LAPG), to check whether that covariate shift meaningfully degrades
predictor accuracy in deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.fa_lapg_foundation.trajectory_dataset import build_samples
from experiments.learning_benefit.prediction_error import TrajectorySample, interpolate_true_position


def predictor_error_under_policy(model, episodes, eval_horizons=(1.0, 4.0, 8.0)) -> pd.DataFrame:
    """For each recorded episode (rolled out under SOME controlling
    policy), compute the horizon-conditioned predictor's error against
    that episode's OWN realized future hostile trajectory -- i.e. under
    the covariate distribution that specific controller induces."""
    rows = []
    for episode in episodes:
        traj_samples = [TrajectorySample(s.step_index, s.sim_time, s.true_position) for s in episode.steps]
        fixed_samples = build_samples(episode, horizons=(0.5,))
        for sample in fixed_samples:
            for h in eval_horizons:
                query_time = sample["sim_time"] + h
                target = interpolate_true_position(traj_samples, query_time)
                if target is None:
                    continue
                with torch.no_grad():
                    history = torch.as_tensor(sample["history"], dtype=torch.float32).unsqueeze(0)
                    mask = torch.as_tensor(sample["history_mask"], dtype=torch.float32).unsqueeze(0)
                    horizon_t = torch.tensor([h], dtype=torch.float32)
                    delta = model(history, mask, horizon_t)[0].numpy()
                pred = sample["est_pos_now"] + sample["est_vel_now"] * h + delta
                rows.append({
                    "scenario": episode.scenario, "seed": episode.seed, "horizon": h,
                    "error": float(np.linalg.norm(target - pred)),
                })
    return pd.DataFrame(rows)


def summarize_by_rollout_policy(per_policy_errors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Phase 66: mean/median predictor error, grouped by which policy
    generated the rollout (the covariate-shift comparison)."""
    rows = []
    for policy_name, df in per_policy_errors.items():
        for (scenario, horizon), sub in df.groupby(["scenario", "horizon"]):
            rows.append({
                "rollout_policy": policy_name, "scenario": scenario, "horizon": horizon,
                "n": len(sub), "mean_error": sub["error"].mean(), "median_error": sub["error"].median(),
            })
    return pd.DataFrame(rows)
