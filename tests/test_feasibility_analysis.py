"""Tests for Phase 38 episode-level feasibility aggregation (the
anti-pseudoreplication step before any correlation/outcome analysis)."""

import pandas as pd

from experiments.fa_lapg_foundation.feasibility_analysis import episode_level_feasibility


def test_episode_level_feasibility_collapses_steps_to_one_row_per_episode():
    step_rows = pd.DataFrame({
        "scenario": ["nominal_open"] * 4 + ["strong_maneuver"] * 2,
        "seed": [1, 1, 2, 2, 1, 1],
        "D_turn": [0.5, 1.5, 0.2, 0.4, 2.0, 3.0],
        "closing_speed_ratio": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "predicted_accel_saturated_fraction": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "predicted_turn_saturated_fraction": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "predicted_climb_saturated_fraction": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "E_reach": [1.0, 3.0, 2.0, 2.0, 5.0, 5.0],
        "enemy_turn_rate": [10.0, 20.0, 5.0, 5.0, 30.0, 30.0],
    })
    result = episode_level_feasibility(step_rows)
    assert len(result) == 3  # (nominal_open,1), (nominal_open,2), (strong_maneuver,1)
    row = result[(result.scenario == "nominal_open") & (result.seed == 1)].iloc[0]
    assert row["D_turn"] == 1.0  # mean(0.5, 1.5)
    assert row["E_reach"] == 2.0  # mean(1.0, 3.0)
