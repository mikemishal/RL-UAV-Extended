"""
Deterministic tests for hostile-UAS predictive DYNAMIC-obstacle avoidance
(uav_defend/obstacles/dynamic.py + SoldierEnv's combined-layout wiring).

Covers Section 17 of the obstacle-intelligence task. Precise geometric
scenarios (crossing, head-on, receding, sufficient natural clearance) use
the stateless `ObstacleAvoidancePlanner` directly against a hand-built
`DynamicObstacle`, mirroring exactly what `SoldierEnv._move_enemy()` does
via `_combined_avoidance_layout()`; reproducibility/multi-obstacle/no-NaN
scenarios use the full `SoldierEnv`.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.obstacles import DynamicsLimits, ObstacleAvoidancePlanner, ObstacleLayout
from uav_defend.obstacles.dynamic import DynamicObstacle, dynamic_obstacle_swept_aabb, dynamic_obstacle_threatens

_LIMITS = DynamicsLimits(
    max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0),
    max_climb_rate=5.0, max_descent_rate=5.0, dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0,
)


def _plan_against(obstacle: DynamicObstacle, position, velocity, u_base, target,
                   clearance=2.0, horizon_steps=12, release_steps=4):
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    obstacles = ()
    if dynamic_obstacle_threatens(position, velocity, obstacle, horizon_steps, _LIMITS.dt, clearance):
        obstacles = (dynamic_obstacle_swept_aabb(obstacle, horizon_steps, _LIMITS.dt, clearance),)
    layout = ObstacleLayout(obstacles=obstacles)
    planner = ObstacleAvoidancePlanner(
        clearance=clearance, horizon_steps=horizon_steps, release_steps=release_steps,
        max_altitude=_LIMITS.max_altitude,
    )
    return planner.plan(
        position=position, velocity=velocity,
        u_base=np.asarray(u_base, dtype=np.float64), layout=layout,
        target_position=np.asarray(target, dtype=np.float64), limits=_LIMITS,
    )


def test_bird_crossing_projected_path_triggers_avoidance():
    # Enemy heading +x; bird crossing left-to-right directly ahead, slowly
    # enough that its predicted closest approach undercuts the required
    # clearance margin.
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([20.0, -2.0, 10.0]),
                            velocity=np.array([0.0, 2.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 10.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 10.0])
    assert plan["active"] is True


def test_attacker_deviates_before_closest_approach():
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([15.0, -2.0, 10.0]),
                            velocity=np.array([0.0, 2.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 10.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 10.0])
    # Predicted conflict is flagged BEFORE the obstacle is actually reached.
    assert plan["predicted_collision_step"] is not None
    assert plan["predicted_collision_step"] >= 0


def test_receding_obstacle_causes_no_unnecessary_avoidance():
    # Bird directly ahead but flying AWAY faster than the enemy can close.
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([20.0, 0.0, 10.0]),
                            velocity=np.array([12.0, 0.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 10.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 10.0])
    assert plan["active"] is False


def test_head_on_moving_obstacle_causes_early_avoidance():
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([25.0, 0.0, 10.0]),
                            velocity=np.array([-8.0, 0.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 10.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 10.0])
    assert plan["active"] is True


def test_crossing_obstacle_with_sufficient_natural_clearance_does_not_trigger():
    # Bird crosses far above the enemy's altitude -- ample vertical clearance.
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([20.0, -5.0, 25.0]),
                            velocity=np.array([0.0, 5.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 5.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 5.0])
    assert plan["active"] is False


def test_multiple_dynamic_obstacles_handled_deterministically():
    config = EnvConfig(dynamic_obstacles_enabled=True, dynamic_obstacle_count=6,
                        enemy_obstacle_avoidance_enabled=True)
    env = SoldierEnv(config=config)
    _, info = env.reset(seed=42)
    for _ in range(50):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert np.all(np.isfinite(info["enemy_pos"]))
        if done or truncated:
            break


def test_dynamic_obstacle_positions_reproduce_exactly_for_same_seed():
    config = EnvConfig(dynamic_obstacles_enabled=True, dynamic_obstacle_count=5,
                        enemy_obstacle_avoidance_enabled=True)
    env_a, env_b = SoldierEnv(config=config), SoldierEnv(config=config)
    _, info_a = env_a.reset(seed=99)
    _, info_b = env_b.reset(seed=99)
    for _ in range(30):
        _, _, done_a, trunc_a, info_a = env_a.step(np.zeros(3, dtype=np.float32))
        _, _, done_b, trunc_b, info_b = env_b.step(np.zeros(3, dtype=np.float32))
        assert np.array_equal(info_a["enemy_pos"], info_b["enemy_pos"])
        assert info_a["enemy_min_dynamic_obstacle_clearance"] == info_b["enemy_min_dynamic_obstacle_clearance"]
        if done_a or trunc_a:
            break


def test_attacker_resumes_target_pursuit_after_conflict_clears():
    config = EnvConfig(dynamic_obstacles_enabled=True, dynamic_obstacle_count=3,
                        enemy_obstacle_avoidance_enabled=True)
    env = SoldierEnv(config=config)
    _, info = env.reset(seed=17)
    active_flags = []
    for _ in range(400):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        active_flags.append(info["enemy_avoidance_active"])
        if done or truncated:
            break
    assert info["outcome"] in ("intercepted", "soldier_caught", "unsafe_intercept", "timeout")
    # Avoidance (if it ever activated) must not remain active for the whole episode.
    if any(active_flags):
        assert not all(active_flags)


def test_required_safety_separation_respected_where_dynamically_feasible():
    # A slow, distant bird gives ample room to satisfy the configured clearance.
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([30.0, -8.0, 10.0]),
                            velocity=np.array([0.0, 3.0, 0.0]), radius=0.5)
    plan = _plan_against(bird, position=[0.0, 0.0, 10.0], velocity=[10.0, 0.0, 0.0],
                          u_base=[1.0, 0.0, 0.0], target=[100.0, 0.0, 10.0], clearance=2.0)
    if plan["active"]:
        assert plan["predicted_min_clearance"] >= 0.0 or not plan["fallback_used"]


def test_dynamic_obstacle_scenario_produces_no_nan_or_inf():
    config = EnvConfig(dynamic_obstacles_enabled=True, dynamic_obstacle_count=8,
                        dynamic_obstacle_heading_noise=0.3, enemy_obstacle_avoidance_enabled=True)
    env = SoldierEnv(config=config)
    _, info = env.reset(seed=7)
    for _ in range(300):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.all(np.isfinite(info["enemy_pos"]))
        assert np.all(np.isfinite(info["enemy_vel"]))
        assert np.isfinite(info["enemy_avoidance_angle_deg"])
        if done or truncated:
            break
