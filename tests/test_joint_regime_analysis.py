"""Tests for Phase 40-41 joint-regime classification and outcome
summarization."""

import pandas as pd

from experiments.fa_lapg_foundation.joint_regime_analysis import (
    classify_joint_regime,
    summarize_by_obstacle_context,
    summarize_regime_outcomes,
)


def test_classify_joint_regime_four_quadrants_via_median_split():
    df = pd.DataFrame({
        "scenario": ["s"] * 4,
        "seed": [1, 2, 3, 4],
        "error": [10.0, 10.0, 100.0, 100.0],
        "stress": [0.5, 5.0, 0.5, 5.0],
    })
    classified = classify_joint_regime(df, error_column="error", stress_column="stress")
    # median error=55 -> {10,10} low, {100,100} high; median stress=2.75 -> {0.5,0.5} low, {5,5} high
    regimes = dict(zip(classified["seed"], classified["regime"]))
    assert regimes[1] == "low_error_low_stress"
    assert regimes[2] == "low_error_high_stress"
    assert regimes[3] == "high_error_low_stress"
    assert regimes[4] == "high_error_high_stress"


def test_classify_joint_regime_explicit_thresholds():
    df = pd.DataFrame({"error": [1.0, 100.0], "stress": [0.1, 10.0]})
    classified = classify_joint_regime(df, "error", "stress", error_threshold=50.0, stress_threshold=1.0)
    assert classified.loc[0, "regime"] == "low_error_low_stress"
    assert classified.loc[1, "regime"] == "high_error_high_stress"


def test_summarize_regime_outcomes_computes_benefit_per_regime():
    df = pd.DataFrame({
        "scenario": ["s"] * 4, "seed": [1, 2, 3, 4],
        "error": [10.0, 10.0, 100.0, 100.0], "stress": [0.5, 5.0, 0.5, 5.0],
    })
    classified = classify_joint_regime(df, "error", "stress")
    lead_success = pd.Series([1, 0, 1, 0])
    lr_ppo_success = pd.Series([1, 1, 0, 1])
    summary = summarize_regime_outcomes(classified, lead_success, lr_ppo_success)
    assert set(summary["regime"]) == {
        "low_error_low_stress", "low_error_high_stress",
        "high_error_low_stress", "high_error_high_stress",
    }
    row = summary[summary.regime == "low_error_high_stress"].iloc[0]
    assert row["mean_lr_ppo_benefit"] == 1.0  # lead=0, lr_ppo=1


def test_summarize_by_obstacle_context_splits_active_vs_inactive():
    df = pd.DataFrame({"obstacle_active": [True, True, False, False]})
    lead_success = pd.Series([1, 0, 1, 1])
    lr_ppo_success = pd.Series([0, 0, 1, 1])
    summary = summarize_by_obstacle_context(df, "obstacle_active", lead_success, lr_ppo_success)
    active_row = summary[summary.obstacle_active == True].iloc[0]  # noqa: E712
    assert active_row["mean_lr_ppo_benefit"] == -0.5  # (0-1 + 0-0)/2
    inactive_row = summary[summary.obstacle_active == False].iloc[0]  # noqa: E712
    assert inactive_row["mean_lr_ppo_benefit"] == 0.0
