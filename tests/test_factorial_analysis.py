"""Tests for Phase 82 factorial analysis: main effects, interaction
calculation, and paired bootstrap usage."""

import numpy as np
import pandas as pd

from experiments.fa_lapg_foundation.factorial_analysis import factorial_effects


def _build_df(scenario, lead, ra_lead, lp_lead, ra_lapg):
    n = len(lead)
    rows = []
    for seed in range(n):
        rows.append({"scenario": scenario, "seed": seed, "controller": "lead", "success": lead[seed]})
        rows.append({"scenario": scenario, "seed": seed, "controller": "ra_lead", "success": ra_lead[seed]})
        rows.append({"scenario": scenario, "seed": seed, "controller": "learned_prediction_lead", "success": lp_lead[seed]})
        rows.append({"scenario": scenario, "seed": seed, "controller": "ra_lapg", "success": ra_lapg[seed]})
    return pd.DataFrame(rows)


def test_prediction_main_effect_under_kinematic_guidance():
    n = 40
    lead = np.zeros(n)
    lp_lead = np.ones(n)  # always improves
    ra_lead = np.zeros(n)
    ra_lapg = np.zeros(n)
    df = _build_df("s", lead, ra_lead, lp_lead, ra_lapg)
    effects = factorial_effects(df, "s")
    assert abs(effects["delta_prediction_kinematic"] - 1.0) < 1e-9


def test_reachability_main_effect_under_cv_prediction():
    n = 40
    lead = np.zeros(n)
    ra_lead = np.ones(n)
    lp_lead = np.zeros(n)
    ra_lapg = np.zeros(n)
    df = _build_df("s", lead, ra_lead, lp_lead, ra_lapg)
    effects = factorial_effects(df, "s")
    assert abs(effects["delta_reachability_cv"] - 1.0) < 1e-9


def test_interaction_zero_when_effects_purely_additive():
    n = 40
    rng = np.random.default_rng(0)
    lead = (rng.random(n) < 0.5).astype(float)
    pred_effect = 0.2
    ra_effect = 0.1
    lp_lead = np.clip(lead + pred_effect, 0, 1)
    ra_lead = np.clip(lead + ra_effect, 0, 1)
    ra_lapg = np.clip(lead + pred_effect + ra_effect, 0, 1)
    df = _build_df("s", lead, ra_lead, lp_lead, ra_lapg)
    effects = factorial_effects(df, "s")
    assert abs(effects["interaction"]) < 1e-9


def test_interaction_nonzero_when_effects_are_synergistic():
    n = 40
    lead = np.zeros(n)
    lp_lead = np.full(n, 0.2)
    ra_lead = np.full(n, 0.1)
    ra_lapg = np.ones(n)  # combined effect far exceeds additive sum
    df = _build_df("s", lead, ra_lead, lp_lead, ra_lapg)
    effects = factorial_effects(df, "s")
    assert effects["interaction"] > 0.5


def test_confidence_intervals_present_and_ordered():
    n = 30
    rng = np.random.default_rng(1)
    lead = (rng.random(n) < 0.6).astype(float)
    ra_lead = (rng.random(n) < 0.6).astype(float)
    lp_lead = (rng.random(n) < 0.6).astype(float)
    ra_lapg = (rng.random(n) < 0.6).astype(float)
    df = _build_df("s", lead, ra_lead, lp_lead, ra_lapg)
    effects = factorial_effects(df, "s")
    for key in ("delta_prediction_kinematic_ci", "delta_reachability_cv_ci"):
        lo, hi = effects[key]
        assert lo <= hi
