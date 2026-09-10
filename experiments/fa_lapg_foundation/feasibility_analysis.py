"""
Phase 38-41: apply the dynamic-feasibility module retrospectively to the
EXISTING learning-benefit diagnostic step logs (Lead-controller rows),
and analyze feasibility features against Lead success/failure and
LR-PPO benefit. Episode-level aggregation throughout (Phase 38's explicit
anti-pseudoreplication requirement).

Reuses `results/learning_benefit_analysis/pilot/logs/step_diagnostics.csv`
and `.../summaries/scenario_summary.csv` from the prior diagnostic study
(same repo, `analysis/learning-benefit` branch) rather than re-simulating,
since those logs already contain everything `compute_feasibility_features`
needs (defender position/velocity, Lead's commanded direction, intercept
point/time) for the Lead controller's own rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.learning_benefit.scenarios import build_scenario_battery
from uav_defend.guidance.dynamic_feasibility import compute_feasibility_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_ROOT = PROJECT_ROOT / "results" / "learning_benefit_analysis" / "pilot"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "feasibility"


def compute_feasibility_for_lead_rows(step_diagnostics: pd.DataFrame, scenario_configs: dict) -> pd.DataFrame:
    """Row-wise feasibility features for every Lead-controller step with a
    valid intercept solution. Pure post-hoc application of
    `compute_feasibility_features` to already-logged fields -- no
    environment re-simulation, no obstacle layout (not logged per-episode
    in the original diagnostic pass; see `run_feasibility_clutter_pass`
    for the obstacle-aware follow-up used in Phase 41)."""
    lead_rows = step_diagnostics[step_diagnostics.controller == "lead"]
    records = []
    for row in lead_rows.itertuples():
        if not row.lead_solution_valid or pd.isna(row.lead_intercept_time):
            continue
        config = scenario_configs[row.scenario]
        features = compute_feasibility_features(
            defender_position=np.array([row.defender_pos_x, row.defender_pos_y, row.defender_pos_z]),
            defender_velocity=np.array([row.defender_vel_x, row.defender_vel_y, row.defender_vel_z]),
            lead_direction=np.array([row.lead_command_x, row.lead_command_y, row.lead_command_z]),
            intercept_point=np.array([row.lead_intercept_point_x, row.lead_intercept_point_y, row.lead_intercept_point_z]),
            intercept_time=row.lead_intercept_time,
            config=config,
        )
        records.append({
            "seed": row.seed, "scenario": row.scenario, "step_index": row.step_index,
            "theta_req_deg": features.theta_req_deg,
            "theta_available_deg": features.theta_available_deg,
            "D_turn": features.D_turn,
            "required_speed": features.required_speed,
            "closing_speed_ratio": features.closing_speed_ratio,
            "predicted_accel_saturated_fraction": features.predicted_accel_saturated_fraction,
            "predicted_turn_saturated_fraction": features.predicted_turn_saturated_fraction,
            "predicted_climb_saturated_fraction": features.predicted_climb_saturated_fraction,
            "E_reach": features.E_reach,
            "enemy_turn_rate": row.enemy_turn_rate,
        })
    return pd.DataFrame(records)


def episode_level_feasibility(feasibility_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 38 anti-pseudoreplication: collapse per-step feasibility rows
    to ONE row per (scenario, seed) episode BEFORE any correlation with
    outcome/benefit."""
    agg_cols = [
        "D_turn", "closing_speed_ratio", "predicted_accel_saturated_fraction",
        "predicted_turn_saturated_fraction", "predicted_climb_saturated_fraction",
        "E_reach", "enemy_turn_rate",
    ]
    return feasibility_df.groupby(["scenario", "seed"])[agg_cols].mean().reset_index()


def build_feasibility_strata(episode_feasibility: pd.DataFrame, column: str, n_strata: int = 3) -> pd.Series:
    """Phase 39: quantile-based strata (exploratory) for a feasibility
    column, labeled LOW/MEDIUM/HIGH stress."""
    labels = ["LOW", "MEDIUM", "HIGH"][:n_strata]
    return pd.qcut(episode_feasibility[column], q=n_strata, labels=labels, duplicates="drop")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    step_diagnostics = pd.read_csv(DIAGNOSTIC_ROOT / "logs" / "step_diagnostics.csv")
    scenario_configs = {s.name: s.config for s in build_scenario_battery()}

    feasibility_df = compute_feasibility_for_lead_rows(step_diagnostics, scenario_configs)
    feasibility_df.to_csv(OUTPUT_ROOT / "lead_step_feasibility.csv", index=False)
    print(f"computed feasibility for {len(feasibility_df)} Lead steps")

    episode_feasibility = episode_level_feasibility(feasibility_df)
    episode_feasibility.to_csv(OUTPUT_ROOT / "episode_feasibility.csv", index=False)
    print(f"episode-level feasibility rows: {len(episode_feasibility)}")


if __name__ == "__main__":
    main()
