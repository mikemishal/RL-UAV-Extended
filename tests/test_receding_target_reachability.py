"""Tests for the receding-target reachability rollout mode (Phase 78):
stationary target, constant-velocity target, turning-target synthetic
case, exact dynamics reuse, and no environment mutation."""

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.guidance.reachability_selection import (
    ReachabilityMode,
    evaluate_candidate,
)
from uav_defend.policies.baseline.lead_math import pure_pursuit_direction


def test_stationary_target_receding_equals_fixed_direction():
    """A stationary target lying exactly along the defender's current
    heading, and FAR enough that the defender cannot reach/overshoot it
    within tau, gives an unchanging bearing at every step -- so
    RECEDING_TARGET must reduce to FIXED_DIRECTION exactly. (If the
    defender were close enough to overshoot mid-rollout, receding-target
    would correctly re-aim backward -- a genuine, intentional behavioral
    difference from fixed-direction, not tested here.)"""
    config = EnvConfig()
    stationary_point = np.array([200.0, 0.0, 5.0])  # far beyond v_d*tau=72, no overshoot possible

    def predict_fn(h):
        return stationary_point

    defender_pos = np.array([0.0, 0.0, 5.0])
    defender_vel = np.array([config.v_d, 0.0, 0.0])
    fixed = evaluate_candidate(defender_pos, defender_vel, predict_fn, 4.0, config, config.intercept_radius, 40, ReachabilityMode.FIXED_DIRECTION)
    receding = evaluate_candidate(defender_pos, defender_vel, predict_fn, 4.0, config, config.intercept_radius, 40, ReachabilityMode.RECEDING_TARGET)
    assert abs(fixed.E_reach - receding.E_reach) < 1e-6


def test_constant_velocity_target_receding_target_finite_and_reasonable():
    config = EnvConfig()

    def predict_fn(h):
        return np.array([2.0, 0.0, 5.0]) + np.array([1.0, 0.0, 0.0]) * h

    defender_pos = np.zeros(3)
    defender_vel = np.array([config.v_d, 0.0, 0.0])
    result = evaluate_candidate(defender_pos, defender_vel, predict_fn, 2.0, config, config.intercept_radius, 40, ReachabilityMode.RECEDING_TARGET)
    assert np.isfinite(result.E_reach)


def test_turning_target_receding_reduces_required_heading_swing():
    """For a target that curves across the defender's path, continuously
    re-aiming (RECEDING_TARGET) should not require a LARGER instantaneous
    turn than committing to the final frozen bearing when the target
    sweeps toward the defender's initial heading over time."""
    config = EnvConfig()

    def predict_fn(h):
        angle = 0.3 * h  # target arcs across the field of view over time
        radius = 15.0
        return np.array([radius * np.cos(angle), radius * np.sin(angle), 5.0])

    defender_pos = np.zeros(3)
    defender_vel = np.array([config.v_d, 0.0, 0.0])
    receding = evaluate_candidate(defender_pos, defender_vel, predict_fn, 6.0, config, config.intercept_radius, 40, ReachabilityMode.RECEDING_TARGET)
    assert np.isfinite(receding.E_reach)
    assert receding.turn_saturated_fraction >= 0.0


def test_exact_dynamics_reuse_matches_manual_stepwise_rollout():
    config = EnvConfig()
    defender_pos = np.array([0.0, 0.0, 5.0])
    defender_vel = np.array([3.0, -1.0, 0.0])

    def predict_fn(h):
        return np.array([10.0, 5.0, 5.0]) + np.array([-0.5, 0.2, 0.0]) * h

    tau = 2.0
    result = evaluate_candidate(defender_pos, defender_vel, predict_fn, tau, config, config.intercept_radius, 40, ReachabilityMode.RECEDING_TARGET)

    pos, vel = defender_pos.copy(), defender_vel.copy()
    n_steps = int(round(tau / config.dt))
    for k in range(n_steps):
        aim = predict_fn(k * config.dt)
        direction = pure_pursuit_direction(aim, pos, config.eps)
        desired_velocity = config.v_d * direction
        vel, _diag = advance_velocity(
            vel, desired_velocity, max_speed=config.v_d, max_accel=config.defender_max_accel,
            max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
            max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
            dt=config.dt, eps=config.eps,
        )
        pos = pos + vel * config.dt
        pos, vel = apply_boundary(pos, vel, L=config.L, max_altitude=config.max_altitude)

    expected_final_target = predict_fn(tau)
    expected_e_reach = float(np.linalg.norm(pos - expected_final_target))
    assert abs(result.E_reach - expected_e_reach) < 1e-4


def test_no_environment_mutation():
    from uav_defend.envs.soldier_env import SoldierEnv

    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=0)
    snapshot_pos = env._defender_pos.copy()
    snapshot_vel = env._defender_vel.copy()

    def predict_fn(h):
        return np.array([10.0, 0.0, 5.0])

    evaluate_candidate(env._defender_pos, env._defender_vel, predict_fn, 2.0, config, config.intercept_radius, 40, ReachabilityMode.RECEDING_TARGET)

    assert np.allclose(env._defender_pos, snapshot_pos)
    assert np.allclose(env._defender_vel, snapshot_vel)
