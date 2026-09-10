"""Tests for the policy-independent dynamic-feasibility module
(Phase 35-37): zero-velocity defender, aligned command, large heading
reversal, saturation-limited case, analytically feasible case, obstacle
path conflict, and exact reuse of the environment's own dynamics."""

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.dynamics.constrained_point_mass import advance_velocity
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.guidance.dynamic_feasibility import (
    compute_feasibility_features,
    simulate_feasibility_rollout,
)
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout


def test_zero_velocity_defender_heading_undefined():
    config = EnvConfig()
    features = compute_feasibility_features(
        defender_position=np.array([0.0, 0.0, 10.0]),
        defender_velocity=np.zeros(3),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=np.array([10.0, 0.0, 10.0]),
        intercept_time=2.0,
        config=config,
    )
    assert features.theta_req_deg is None  # heading undefined at rest
    assert features.D_turn is None
    assert features.theta_available_deg is not None


def test_aligned_command_requires_zero_heading_change():
    config = EnvConfig()
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=np.array([50.0, 0.0, 0.0]),
        intercept_time=3.0,
        config=config,
    )
    assert features.theta_req_deg == 0.0
    assert features.D_turn == 0.0


def test_large_heading_reversal_requires_180_degrees():
    config = EnvConfig()
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([-1.0, 0.0, 0.0]),
        intercept_point=np.array([-50.0, 0.0, 0.0]),
        intercept_time=1.0,
        config=config,
    )
    assert abs(features.theta_req_deg - 180.0) < 1e-6
    # tight tau -> reversal likely exceeds available turn authority
    assert features.D_turn > 1.0


def test_saturation_limited_case_reports_high_saturation_fractions():
    config = EnvConfig(defender_max_accel=1.0, defender_max_turn_rate_deg=5.0)
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([0.0, 1.0, 0.0]),  # 90-degree turn demanded
        intercept_point=np.array([0.0, 50.0, 0.0]),
        intercept_time=2.0,
        config=config,
    )
    assert features.predicted_turn_saturated_fraction > 0.5
    assert features.E_reach > 0.0  # cannot reach the nominal intercept point


def test_analytically_feasible_case_reaches_intercept_point_closely():
    config = EnvConfig()
    tau = 1.0  # stay well within the domain boundary (L=50)
    intercept_point = np.array([config.v_d * tau, 0.0, 0.0])
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=intercept_point,
        intercept_time=tau,
        config=config,
    )
    assert features.E_reach < 1e-3
    assert features.predicted_accel_saturated_fraction == 0.0
    assert features.predicted_turn_saturated_fraction == 0.0


def test_obstacle_path_conflict_detected_when_layout_provided():
    config = EnvConfig()
    obstacle = AABBObstacle(min_corner=np.array([5.0, -2.0, 0.0]), max_corner=np.array([7.0, 2.0, 20.0]))
    layout = ObstacleLayout(obstacles=(obstacle,))
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=np.array([20.0, 0.0, 0.0]),
        intercept_time=2.0,
        config=config,
        obstacle_layout=layout,
    )
    assert features.obstacle_path_conflict is True


def test_no_obstacle_conflict_when_path_clear():
    config = EnvConfig()
    obstacle = AABBObstacle(min_corner=np.array([5.0, 10.0, 0.0]), max_corner=np.array([7.0, 14.0, 20.0]))
    layout = ObstacleLayout(obstacles=(obstacle,))
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=np.array([20.0, 0.0, 0.0]),
        intercept_time=2.0,
        config=config,
        obstacle_layout=layout,
    )
    assert features.obstacle_path_conflict is False


def test_none_intercept_time_yields_zero_rollout_steps_and_none_features():
    config = EnvConfig()
    features = compute_feasibility_features(
        defender_position=np.zeros(3),
        defender_velocity=np.array([config.v_d, 0.0, 0.0]),
        lead_direction=np.array([1.0, 0.0, 0.0]),
        intercept_point=None,
        intercept_time=None,
        config=config,
    )
    assert features.n_rollout_steps == 0
    assert features.theta_available_deg is None
    assert features.E_reach is None


def test_exact_reuse_of_environment_dynamics_single_step():
    """simulate_feasibility_rollout(n_steps=1) must produce the identical
    next position/velocity SoldierEnv._move_defender would, for the SAME
    starting state and commanded direction (no re-implementation)."""
    config = EnvConfig(obstacles_enabled=False)
    env = SoldierEnv(config=config)
    env.reset(seed=0)

    defender_pos = env._defender_pos.copy()
    defender_vel = np.array([3.0, -1.0, 0.5])
    env._defender_vel = defender_vel.copy()
    direction = np.array([0.6, 0.8, 0.0])
    action = direction / np.linalg.norm(direction)

    env._move_defender(action)
    env_next_pos = env._defender_pos.copy()
    env_next_vel = env._defender_vel.copy()

    rollout = simulate_feasibility_rollout(
        defender_position=defender_pos, defender_velocity=defender_vel,
        lead_direction=direction, config=config, tau=config.dt, max_steps_cap=1,
    )
    assert np.allclose(rollout.terminal_position, env_next_pos, atol=1e-5)

    # cross-check velocity via the pure dynamics function directly
    next_vel, _diag = advance_velocity(
        current_velocity=defender_vel, desired_velocity=config.v_d * (direction / np.linalg.norm(direction)),
        max_speed=config.v_d, max_accel=config.defender_max_accel,
        max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
        max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
        dt=config.dt, eps=config.eps,
    )
    assert np.allclose(next_vel, env_next_vel, atol=1e-5)
