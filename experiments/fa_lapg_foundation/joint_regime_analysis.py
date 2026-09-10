"""
Phase 40-41: joint prediction-error x dynamic-feasibility regime analysis,
plus obstacle/clutter as a third context variable.

Classifies EPISODES (not raw steps, to avoid pseudoreplication) into four
regimes by median-split on (a) mean Lead CV prediction error and (b) mean
D_turn (dynamic-feasibility stress), then reports Lead/LR-PPO success and
LR-PPO benefit within each regime. Also splits `clutter`/`clutter_strong_
maneuver` episodes by whether the defender's own obstacle-avoidance layer
was active, to test whether the negative LR-PPO transfer found in the
diagnostic study concentrates in a particular regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_joint_regime(
    episode_features: pd.DataFrame,
    error_column: str,
    stress_column: str,
    error_threshold: float | None = None,
    stress_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Phase 40: label each episode row with one of four regimes:
      1. low_error_low_stress
      2. high_error_low_stress
      3. low_error_high_stress
      4. high_error_high_stress

    Thresholds default to the median of each column (exploratory,
    data-informed) if not supplied explicitly.
    """
    df = episode_features.copy()
    error_threshold = df[error_column].median() if error_threshold is None else error_threshold
    stress_threshold = df[stress_column].median() if stress_threshold is None else stress_threshold

    high_error = df[error_column] > error_threshold
    high_stress = df[stress_column] > stress_threshold

    regime = np.where(
        high_error & high_stress, "high_error_high_stress",
        np.where(high_error & ~high_stress, "high_error_low_stress",
                 np.where(~high_error & high_stress, "low_error_high_stress", "low_error_low_stress")),
    )
    df["regime"] = regime
    df["error_threshold"] = error_threshold
    df["stress_threshold"] = stress_threshold
    return df


def summarize_regime_outcomes(
    regime_df: pd.DataFrame,
    lead_success: pd.Series,
    lr_ppo_success: pd.Series,
) -> pd.DataFrame:
    """Phase 40: Lead success, LR-PPO success, and paired benefit within
    each of the four regimes. `lead_success`/`lr_ppo_success` must be
    indexed identically to `regime_df` (e.g. matched by (scenario, seed))."""
    df = regime_df.copy()
    df["lead_success"] = lead_success.values
    df["lr_ppo_success"] = lr_ppo_success.values
    df["lr_ppo_benefit"] = df["lr_ppo_success"] - df["lead_success"]

    return df.groupby("regime").agg(
        n=("lead_success", "size"),
        lead_success_rate=("lead_success", "mean"),
        lr_ppo_success_rate=("lr_ppo_success", "mean"),
        mean_lr_ppo_benefit=("lr_ppo_benefit", "mean"),
    ).reset_index()


def summarize_by_obstacle_context(
    episode_features: pd.DataFrame,
    obstacle_active_column: str,
    lead_success: pd.Series,
    lr_ppo_success: pd.Series,
) -> pd.DataFrame:
    """Phase 41: split episodes by whether obstacle-avoidance was active
    during the episode, reporting Lead/LR-PPO success and benefit
    separately -- used to test whether negative LR-PPO transfer in
    clutter+maneuver concentrates specifically in obstacle-active
    episodes."""
    df = episode_features.copy()
    df["lead_success"] = lead_success.values
    df["lr_ppo_success"] = lr_ppo_success.values
    df["lr_ppo_benefit"] = df["lr_ppo_success"] - df["lead_success"]

    return df.groupby(obstacle_active_column).agg(
        n=("lead_success", "size"),
        lead_success_rate=("lead_success", "mean"),
        lr_ppo_success_rate=("lr_ppo_success", "mean"),
        mean_lr_ppo_benefit=("lr_ppo_benefit", "mean"),
    ).reset_index()
