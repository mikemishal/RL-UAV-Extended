"""
Phase 82: 2x2 factorial analysis (prediction x reachability) with paired
episode-level bootstrap confidence intervals.

             KINEMATIC        REACHABILITY
CV           Lead             RA-Lead
Learned      LP-Lead          RA-LAPG
"""

from __future__ import annotations

import pandas as pd

from experiments.learning_benefit.statistics_utils import paired_bootstrap_difference


def factorial_effects(success_df: pd.DataFrame, scenario: str) -> dict:
    """`success_df` must have columns seed, controller, success, with
    controller values 'lead', 'ra_lead', 'learned_prediction_lead',
    'ra_lapg' all sharing the SAME seed set for `scenario`."""
    sub = success_df[success_df.scenario == scenario].set_index("seed")
    lead = sub[sub.controller == "lead"].sort_index()["success"].values
    ra_lead = sub[sub.controller == "ra_lead"].sort_index()["success"].values
    lp_lead = sub[sub.controller == "learned_prediction_lead"].sort_index()["success"].values
    ra_lapg = sub[sub.controller == "ra_lapg"].sort_index()["success"].values

    def _p(x):
        return float(x.mean())

    delta_pred_kin = paired_bootstrap_difference(lead, lp_lead)
    delta_pred_ra = paired_bootstrap_difference(ra_lead, ra_lapg)
    delta_ra_cv = paired_bootstrap_difference(lead, ra_lead)
    delta_ra_learned = paired_bootstrap_difference(lp_lead, ra_lapg)

    interaction_point = (_p(ra_lapg) - _p(ra_lead)) - (_p(lp_lead) - _p(lead))

    return {
        "scenario": scenario,
        "P_Lead": _p(lead), "P_RA_Lead": _p(ra_lead), "P_LP_Lead": _p(lp_lead), "P_RA_LAPG": _p(ra_lapg),
        "delta_prediction_kinematic": delta_pred_kin["mean_diff"],
        "delta_prediction_kinematic_ci": (delta_pred_kin["ci_lower"], delta_pred_kin["ci_upper"]),
        "delta_prediction_reachability": delta_pred_ra["mean_diff"],
        "delta_prediction_reachability_ci": (delta_pred_ra["ci_lower"], delta_pred_ra["ci_upper"]),
        "delta_reachability_cv": delta_ra_cv["mean_diff"],
        "delta_reachability_cv_ci": (delta_ra_cv["ci_lower"], delta_ra_cv["ci_upper"]),
        "delta_reachability_learned": delta_ra_learned["mean_diff"],
        "delta_reachability_learned_ci": (delta_ra_learned["ci_lower"], delta_ra_learned["ci_upper"]),
        "interaction": interaction_point,
    }


def factorial_effects_all_scenarios(success_df: pd.DataFrame) -> pd.DataFrame:
    rows = [factorial_effects(success_df, scenario) for scenario in success_df["scenario"].unique()]
    return pd.DataFrame(rows)
