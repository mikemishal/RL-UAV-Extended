"""
Tests for the canonical no-obstacle (C0) LR-PPO-semantics evaluation
harness (journal "first publication evaluation" sub-task).
"""

from __future__ import annotations

import dataclasses

import pytest

from uav_defend.config.env_config import EnvConfig

from experiments.defender_adaptation_canonical_c0_evaluation import (
    CANONICAL_C0_EVAL_SEED_START,
    CANONICAL_EXTRA_OVERRIDES,
    EVAL_SEEDS,
    N_EVAL_SEEDS,
    assert_canonical_lrppo_config,
    build_evaluation_manifest,
    build_model_registry,
)
from experiments.defender_adaptation_c0_evaluation import EVAL_SEEDS as LEGACY_EVAL_SEEDS


@pytest.fixture(scope="module")
def registry():
    return build_model_registry()


@pytest.fixture(scope="module")
def manifest(registry):
    return build_evaluation_manifest(registry)


def test_seed_range_disjoint_from_legacy_c0():
    assert set(EVAL_SEEDS).isdisjoint(set(LEGACY_EVAL_SEEDS))


def test_seed_range_is_100_contiguous_seeds():
    assert N_EVAL_SEEDS == 100
    assert EVAL_SEEDS == tuple(range(CANONICAL_C0_EVAL_SEED_START, CANONICAL_C0_EVAL_SEED_START + 100))


def test_assert_canonical_config_passes_for_bare_default():
    cfg = EnvConfig(use_kalman_tracking=False)
    resolved = assert_canonical_lrppo_config(cfg)
    assert resolved["defender_standby_until_detection"] is True
    assert resolved["obstacles_enabled"] is False
    assert resolved["dynamic_obstacles_enabled"] is False


def test_assert_canonical_config_fails_loudly_on_standby_false():
    cfg = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=False)
    with pytest.raises(AssertionError, match="defender_standby_until_detection"):
        assert_canonical_lrppo_config(cfg)


def test_assert_canonical_config_fails_loudly_on_obstacles_enabled():
    cfg = EnvConfig(use_kalman_tracking=False, obstacles_enabled=True)
    with pytest.raises(AssertionError, match="obstacles_enabled"):
        assert_canonical_lrppo_config(cfg)


def test_manifest_has_exactly_10800_keys(manifest):
    assert manifest["total_expected_keys"] == 10_800
    assert manifest["n_defenders"] == 27
    assert manifest["n_attacker_conditions"] == 4
    assert manifest["n_eval_seeds"] == 100


def test_manifest_resolved_config_has_canonical_standby(manifest):
    for estimator_cfg in manifest["resolved_env_config"].values():
        assert estimator_cfg["defender_standby_until_detection"] is True
        assert estimator_cfg["obstacles_enabled"] is False
        assert estimator_cfg["dynamic_obstacles_enabled"] is False
        for field, expected in CANONICAL_EXTRA_OVERRIDES.items():
            assert estimator_cfg[field] == expected
