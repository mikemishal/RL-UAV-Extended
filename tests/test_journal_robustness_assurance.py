"""Targeted tests for the final journal robustness/assurance evaluation
harness (Phase 22: validate before full launch)."""

from __future__ import annotations

import pytest

from experiments.defender_adaptation_canonical_c0_evaluation import EVAL_SEEDS as CANONICAL_C0_SEEDS
from experiments.defender_adaptation_evaluation import EVAL_SEEDS as C1C3_SEEDS
from experiments.robustness_measurement_noise_protocol import MEASUREMENT_VAR_VALUES
from experiments.robustness_detection_protocol import DETECTION_RADIUS_VALUES
from experiments.robustness_mobility_protocol import DEFENDER_SPEED_VALUES, HOSTILE_SPEED_VALUES

import experiments.journal_robustness_assurance as jra


@pytest.fixture(scope="module")
def registry():
    return jra.load_model_registry()


def test_seed_range_is_100_contiguous_seeds():
    assert jra.EVAL_SEEDS == tuple(range(200_000, 200_100))


def test_seed_range_disjoint_from_canonical_c0_and_c1c3():
    assert set(jra.EVAL_SEEDS).isdisjoint(set(CANONICAL_C0_SEEDS))
    assert set(jra.EVAL_SEEDS).isdisjoint(set(C1C3_SEEDS))


def test_seed_range_disjoint_from_training_and_attacker_namespaces():
    training_like = set(range(42, 45)) | set(range(71, 80))
    attacker_heldout = set(range(190_000, 200_000))
    assert set(jra.EVAL_SEEDS).isdisjoint(training_like)
    assert set(jra.EVAL_SEEDS).isdisjoint(attacker_heldout)


def test_off_nominal_cells_exactly_six():
    assert len(jra.OFF_NOMINAL_CELLS) == 6
    assert len(set(jra.OFF_NOMINAL_CELLS)) == 6


def test_r1_values_are_members_of_original_sweep_grid():
    assert jra.R1_SETTINGS["low"] in MEASUREMENT_VAR_VALUES
    assert jra.R1_SETTINGS["high"] in MEASUREMENT_VAR_VALUES
    assert jra.R1_SETTINGS["low"] == min(MEASUREMENT_VAR_VALUES)
    assert jra.R1_SETTINGS["high"] == max(MEASUREMENT_VAR_VALUES)


def test_r2_values_are_members_of_original_sweep_grid():
    assert jra.R2_SETTINGS["short"] in DETECTION_RADIUS_VALUES
    assert jra.R2_SETTINGS["long"] in DETECTION_RADIUS_VALUES
    assert jra.R2_SETTINGS["short"] == min(DETECTION_RADIUS_VALUES)
    assert jra.R2_SETTINGS["long"] == max(DETECTION_RADIUS_VALUES)


def test_r3_values_are_members_of_original_sweep_grid():
    for v_d, v_e in jra.R3_SETTINGS.values():
        assert v_d in DEFENDER_SPEED_VALUES
        assert v_e in HOSTILE_SPEED_VALUES


def test_r3_defender_advantaged_has_highest_ratio():
    ratios = {label: vd / ve for label, (vd, ve) in jra.R3_SETTINGS.items()}
    assert ratios["defender_advantaged"] > jra.NOMINAL_POINT["v_d"] / jra.NOMINAL_POINT["v_e"]
    assert ratios["defender_disadvantaged"] < jra.NOMINAL_POINT["v_d"] / jra.NOMINAL_POINT["v_e"]


def test_primary_attacker_is_a1_unseen_not_seen():
    assert jra.PRIMARY_ATTACKER == "A1-unseen-203"
    assert "seen" not in jra.PRIMARY_ATTACKER or "unseen" in jra.PRIMARY_ATTACKER


def test_clutter_levels_are_c0_and_c2_only():
    assert jra.CLUTTER_LEVELS == ("C0", "C2")


def test_generations_are_d0plus_and_d1_only():
    assert jra.GENERATIONS == ("d0_plus", "d1")


def test_nominal_c0_config_matches_nominal_point():
    cfg = jra.build_canonical_env_config("C0", "measurement", extra_overrides=jra.CANONICAL_EXTRA_OVERRIDES)
    assert cfg.measurement_var == jra.NOMINAL_POINT["measurement_var"]
    assert cfg.detection_radius == jra.NOMINAL_POINT["detection_radius"]
    assert cfg.v_d == jra.NOMINAL_POINT["v_d"] and cfg.v_e == jra.NOMINAL_POINT["v_e"]


def test_assert_robustness_config_passes_for_correct_off_nominal_c0():
    cfg = jra._env_config_for("C0", "measurement", "R1_measurement_noise", "high")
    resolved = jra.assert_robustness_config(cfg, "C0", "R1_measurement_noise", "high")
    assert resolved["measurement_var"] == jra.R1_SETTINGS["high"]
    assert resolved["obstacles_enabled"] is False


def test_assert_robustness_config_passes_for_correct_off_nominal_c2():
    cfg = jra._env_config_for("C2", "measurement", "R2_detection_radius", "short")
    resolved = jra.assert_robustness_config(cfg, "C2", "R2_detection_radius", "short")
    assert resolved["detection_radius"] == jra.R2_SETTINGS["short"]
    assert resolved["obstacles_enabled"] is True


def test_assert_robustness_config_fails_loudly_on_wrong_measurement_var():
    cfg = jra._env_config_for("C0", "measurement", "R1_measurement_noise", "high")
    with pytest.raises(AssertionError):
        jra.assert_robustness_config(cfg, "C0", "R1_measurement_noise", "low")


def test_assert_robustness_config_fails_loudly_on_wrong_level_obstacles():
    cfg = jra._env_config_for("C0", "measurement", "R1_measurement_noise", "low")
    with pytest.raises(AssertionError):
        jra.assert_robustness_config(cfg, "C2", "R1_measurement_noise", "low")


def test_manifest_has_exactly_21600_keys(registry):
    manifest = jra.build_evaluation_manifest(registry)
    assert manifest["total_expected_keys"] == 18 * 6 * 2 * 100
    assert manifest["n_defenders"] == 18


def test_checkpoint_integrity_passes(registry):
    assert jra.verify_checkpoint_integrity(registry) == []
