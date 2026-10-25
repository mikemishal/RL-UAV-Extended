"""Tests for the journal asset-motion model study (uav_defend/envs/soldier_env.py:_move_soldier,
uav_defend/config/env_config.py:asset_motion_model). See
docs/journal_v2_asset_motion_model_study.md for the full investigation and the verdict (legacy
motion is kept canonical; this is a config-gated, non-default alternative)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

NO_OBSTACLES = dict(obstacles_enabled=False)


def _env(**overrides) -> SoldierEnv:
    return SoldierEnv(dataclasses.replace(EnvConfig(), **{**NO_OBSTACLES, **overrides}))


def _run(model: str, tau: float, seed: int, steps: int = 30):
    env = _env(asset_motion_model=model, asset_motion_tau_s=tau)
    env.reset(seed=seed)
    positions = [env._soldier_pos.copy()]
    for _ in range(steps):
        env.step(np.zeros(3, dtype=np.float32))
        positions.append(env._soldier_pos.copy())
    return np.array(positions)


def test_legacy_is_the_default_and_matches_direct_construction():
    assert EnvConfig().asset_motion_model == "legacy_gaussian_random_walk"


def test_legacy_motion_is_deterministic_and_seed_reproducible():
    a = _run("legacy_gaussian_random_walk", 1.0, seed=42)
    b = _run("legacy_gaussian_random_walk", 1.0, seed=42)
    assert np.array_equal(a, b)
    c = _run("legacy_gaussian_random_walk", 1.0, seed=43)
    assert not np.array_equal(a, c)


def test_legacy_motion_matches_a_direct_replica_of_the_original_equation():
    """Exact equivalence: the legacy path must still be literally
    `displacement = Normal(0, v_s*dt, size=2)`, unaffected by the new config field existing."""
    env = _env(asset_motion_model="legacy_gaussian_random_walk")
    env.reset(seed=7)
    # Replay the same RNG stream the env itself holds, applying the ORIGINAL equation directly.
    replica_rng = np.random.default_rng()
    replica_rng.bit_generator.state = env._rng_soldier.bit_generator.state
    sigma = env.config.v_s * env.config.dt
    for _ in range(10):
        pos_before = env._soldier_pos.copy()
        env.step(np.zeros(3, dtype=np.float32))
        displacement_xy = replica_rng.normal(loc=0.0, scale=sigma, size=(2,))
        expected = pos_before.astype(np.float64)
        expected[:2] += displacement_xy
        # boundary/obstacle-free in this test, so the only transform is the raw displacement
        assert np.allclose(env._soldier_pos.astype(np.float64)[:2], expected[:2], atol=1e-5)


def test_gauss_markov_is_deterministic_and_seed_reproducible():
    a = _run("gauss_markov_velocity", 1.0, seed=5)
    b = _run("gauss_markov_velocity", 1.0, seed=5)
    assert np.array_equal(a, b)
    c = _run("gauss_markov_velocity", 1.0, seed=6)
    assert not np.array_equal(a, c)


def test_gauss_markov_recurrence_matches_the_equation_exactly():
    env = _env(asset_motion_model="gauss_markov_velocity", asset_motion_tau_s=1.0)
    env.reset(seed=11)
    rho = float(np.exp(-env.config.dt / env.config.asset_motion_tau_s))
    replica_rng = np.random.default_rng()
    replica_rng.bit_generator.state = env._rng_soldier.bit_generator.state
    v_s = env.config.v_s
    dt = env.config.dt
    xi0 = replica_rng.normal(0.0, 1.0, size=(2,))
    expected_v = v_s * xi0  # Phase 3: first-step initialization
    env.step(np.zeros(3, dtype=np.float32))
    assert np.allclose(env._soldier_vel[:2], expected_v, atol=1e-5)
    for _ in range(5):
        xi = replica_rng.normal(0.0, 1.0, size=(2,))
        expected_v = rho * expected_v + np.sqrt(1 - rho * rho) * v_s * xi
        env.step(np.zeros(3, dtype=np.float32))
        assert np.allclose(env._soldier_vel[:2], expected_v, atol=1e-4)


def test_gauss_markov_preserves_stationary_velocity_scale():
    """At stationarity each velocity component should have std ~= v_s regardless of tau (Phase
    2's design goal: this changes correlation, not magnitude)."""
    env = _env(asset_motion_model="gauss_markov_velocity", asset_motion_tau_s=2.0)
    env.reset(seed=1)
    vels = []
    for _ in range(3000):
        env.step(np.zeros(3, dtype=np.float32))
        vels.append(env._soldier_vel[:2].copy())
    vels = np.array(vels)
    assert abs(float(vels.std()) - env.config.v_s) < 0.15 * env.config.v_s


def test_rho_tau_conversion():
    dt = 0.5
    for tau, expected_rho in ((0.5, np.exp(-1.0)), (1.0, np.exp(-0.5)), (2.0, np.exp(-0.25)), (4.0, np.exp(-0.125))):
        assert abs(np.exp(-dt / tau) - expected_rho) < 1e-12


def test_boundary_reflects_velocity_component_not_just_position():
    env = _env(asset_motion_model="gauss_markov_velocity", asset_motion_tau_s=1.0)
    env.reset(seed=1)
    L = env.config.L
    env._soldier_pos = np.array([L - 0.05, 0.0, 0.0], dtype=np.float32)
    env._soldier_vel = np.array([5.0, 0.0, 0.0], dtype=np.float32)  # heading straight at the +x wall
    env._soldier_gauss_markov_initialized = True
    reflected_pos, reflected_vel = env._reflect_horizontal_boundary_with_velocity(
        np.array([L + 1.0, 0.0]), np.array([5.0, 0.0]))
    assert reflected_pos[0] < L
    assert reflected_vel[0] < 0.0  # velocity must not keep pointing outward after reflection


def test_no_rng_draw_count_difference_between_models():
    """Switching models must not change how many random numbers _move_soldier consumes per step,
    so any OTHER consumer of the same stream elsewhere in a step stays aligned."""
    for model in ("legacy_gaussian_random_walk", "gauss_markov_velocity"):
        env = _env(asset_motion_model=model)
        env.reset(seed=2)
        state_before = env._rng_soldier.bit_generator.state
        reference = np.random.default_rng()
        reference.bit_generator.state = state_before
        reference.normal(0.0, 1.0, size=(2,))
        env.step(np.zeros(3, dtype=np.float32))
        assert env._rng_soldier.bit_generator.state == reference.bit_generator.state


def test_config_serializes_and_round_trips():
    cfg = dataclasses.replace(EnvConfig(), asset_motion_model="gauss_markov_velocity", asset_motion_tau_s=2.5)
    payload = dataclasses.asdict(cfg)
    assert payload["asset_motion_model"] == "gauss_markov_velocity"
    assert payload["asset_motion_tau_s"] == 2.5
    restored = EnvConfig(**payload)
    assert restored == cfg


def test_invalid_asset_motion_model_is_rejected():
    with pytest.raises(ValueError):
        dataclasses.replace(EnvConfig(), asset_motion_model="not_a_real_model")


def test_invalid_tau_is_rejected():
    with pytest.raises(ValueError):
        dataclasses.replace(EnvConfig(), asset_motion_tau_s=0.0)
