"""Targeted tests for the architecture-mechanism diagnostic's pure-function
logic (no model loading / environment rollouts -- those are covered by the
smoke-tested collection scripts themselves, consistent with this
project's convention of targeted unit tests for diagnostic infrastructure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uav_defend.policies.residual.lead_residual_composition import angle_between_deg

from experiments.mechanism_diagnostic import (
    N_DIAGNOSTIC_SEEDS, DIAG_SEEDS_C0, DIAG_SEEDS_C3, _maneuver_intensity, _diag_seeds_for,
)
from experiments.defender_adaptation_canonical_c0_evaluation import EVAL_SEEDS as CANONICAL_EVAL_SEEDS
from experiments.defender_adaptation_evaluation import EVAL_SEEDS as C1C3_EVAL_SEEDS
from experiments.mechanism_diagnostic_authority import AUTHORITY_ANGLES_DEG, N_AUTHORITY_SEEDS
from experiments.mechanism_diagnostic_stats import (
    _path_curvature, _action_reversals, _parse_tuple_col, select_matched_cases, RESIDUAL_BIN_LABELS,
)


def test_diagnostic_seeds_are_subsets_of_existing_matched_ranges():
    assert set(DIAG_SEEDS_C0).issubset(set(CANONICAL_EVAL_SEEDS))
    assert set(DIAG_SEEDS_C3).issubset(set(C1C3_EVAL_SEEDS))


def test_diagnostic_seed_count_matches_constant():
    assert len(DIAG_SEEDS_C0) == N_DIAGNOSTIC_SEEDS
    assert len(DIAG_SEEDS_C3) == N_DIAGNOSTIC_SEEDS


def test_diag_seeds_for_dispatches_by_level():
    assert _diag_seeds_for("C0") == DIAG_SEEDS_C0
    assert _diag_seeds_for("C3") == DIAG_SEEDS_C3


def test_authority_angles_include_canonical_training_bound():
    assert 30.0 in AUTHORITY_ANGLES_DEG


def test_authority_diagnostic_uses_n100_per_task_instruction():
    assert N_AUTHORITY_SEEDS == 100


def test_maneuver_intensity_zero_at_episode_start():
    vel_history = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    assert _maneuver_intensity(vel_history, 0, dt=0.5) == 0.0


def test_maneuver_intensity_matches_finite_difference():
    vel_history = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    dt = 0.5
    expected = float(np.linalg.norm(vel_history[1] - vel_history[0]) / dt)
    assert _maneuver_intensity(vel_history, 1, dt) == pytest.approx(expected)


def test_path_curvature_zero_for_straight_line():
    xs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert _path_curvature(xs) == pytest.approx(0.0, abs=1e-6)


def test_path_curvature_nonzero_for_right_angle_turn():
    xs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert _path_curvature(xs) == pytest.approx(90.0, abs=1e-3)


def test_action_reversals_counts_large_direction_changes():
    actions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert _action_reversals(actions, threshold_deg=90.0) == 1


def test_action_reversals_zero_for_consistent_direction():
    actions = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.2, 0.0]])
    assert _action_reversals(actions, threshold_deg=90.0) == 0


def test_parse_tuple_col_roundtrips_stringified_tuples():
    s = pd.Series(["(1.0, 2.0, 3.0)", "(4.0, 5.0, 6.0)"])
    arr = _parse_tuple_col(s)
    assert arr.shape == (2, 3)
    assert arr[0].tolist() == [1.0, 2.0, 3.0]


def test_parse_tuple_col_passthrough_for_non_string():
    arr = _parse_tuple_col(pd.Series([(1.0, 2.0, 3.0)]))
    assert arr.shape == (1, 3)


def test_select_matched_cases_categorizes_by_rule_not_by_hand():
    df = pd.DataFrame([
        {"architecture": "x", "attacker_type": "A1", "level": "C0", "generation": "d0_plus",
         "defender_seed": 1, "eval_seed": 10, "defender_safe_intercept": 1, "episode_length": 50},
        {"architecture": "x", "attacker_type": "A1", "level": "C0", "generation": "d1",
         "defender_seed": 1, "eval_seed": 10, "defender_safe_intercept": 0, "episode_length": 100},
        {"architecture": "x", "attacker_type": "A1", "level": "C0", "generation": "d0_plus",
         "defender_seed": 2, "eval_seed": 11, "defender_safe_intercept": 0, "episode_length": 60},
        {"architecture": "x", "attacker_type": "A1", "level": "C0", "generation": "d1",
         "defender_seed": 2, "eval_seed": 11, "defender_safe_intercept": 1, "episode_length": 70},
    ])
    cases = select_matched_cases(df, "x", "A1", "C0")
    assert cases["B_gen_a_succeeds_gen_b_fails"] == {"defender_seed": 1, "eval_seed": 10, "n_in_category": 1}
    assert cases["C_gen_a_fails_gen_b_succeeds"] == {"defender_seed": 2, "eval_seed": 11, "n_in_category": 1}
    assert cases["A_both_succeed"] is None
    assert cases["D_both_fail"] is None


def test_residual_bin_labels_cover_zero_to_thirty():
    assert RESIDUAL_BIN_LABELS == ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30"]


def test_angle_between_deg_zero_for_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert angle_between_deg(v, v) == pytest.approx(0.0, abs=1e-6)
