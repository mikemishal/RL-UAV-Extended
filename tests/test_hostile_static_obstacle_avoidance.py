"""
Deterministic tests for hostile-UAS predictive STATIC-obstacle avoidance
(uav_defend/obstacles/avoidance.py wired into SoldierEnv._move_enemy()).

Covers Section 16 of the obstacle-intelligence task: (1) no-obstacle
behavior unchanged, (2) no unnecessary avoidance far from obstacles,
(3-4) building/tree-like obstacles force a detour, (5) obstacles off the
flight path never trigger avoidance, (6) high-speed swept collision cannot
tunnel, (7) resumes pursuit after clearing, (8) hysteresis prevents
oscillation, (9) two nearby obstacles don't cause a second collision,
(10) pathological geometry never produces NaN/inf.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv


def _run(config: EnvConfig, seed: int, max_steps: int = 400):
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    history = [info]
    for _ in range(max_steps):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        history.append(info)
        if done or truncated:
            break
    return history


def _safe_fixed_spec(center: np.ndarray, half_xy: float, half_z: float, config: EnvConfig) -> tuple:
    """Clip an obstacle center so the box stays fully inside the domain,
    regardless of where the enemy happened to spawn this seed."""
    margin = 0.5
    cx = float(np.clip(center[0], -config.L + half_xy + margin, config.L - half_xy - margin))
    cy = float(np.clip(center[1], -config.L + half_xy + margin, config.L - half_xy - margin))
    cz = float(np.clip(center[2], half_z + margin, config.max_altitude - half_z - margin))
    return (cx, cy, cz, half_xy, half_xy, half_z)


def test_no_obstacles_behavior_unchanged_with_avoidance_enabled():
    """Enabling avoidance with an EMPTY layout must be a bit-for-bit no-op."""
    baseline_cfg = EnvConfig()
    avoidance_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=0,
        enemy_obstacle_avoidance_enabled=True,
    )
    baseline = _run(baseline_cfg, seed=1)
    avoided = _run(avoidance_cfg, seed=1)
    assert len(baseline) == len(avoided)
    for h_base, h_avoid in zip(baseline, avoided):
        assert np.allclose(h_base["enemy_pos"], h_avoid["enemy_pos"])
        assert h_avoid["enemy_avoidance_active"] is False


def test_far_away_obstacle_causes_no_unnecessary_avoidance():
    # Obstacle placed in a corner far from the enemy-to-soldier path.
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((45.0, 45.0, 5.0, 2.0, 2.0, 5.0),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=2)
    assert not any(h["enemy_avoidance_active"] for h in history)


def test_building_directly_between_attacker_and_target_forces_detour():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=3)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    midpoint[2] = 5.0
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(midpoint, 6.0, 10.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=3)
    assert any(h["enemy_avoidance_active"] for h in history), "expected avoidance to activate for a direct blocker"
    assert not any(h["enemy_obstacle_collision"] for h in history), "attacker should go around, not through"


def test_tree_like_vertical_obstacle_blocks_straight_path_and_causes_detour():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=4)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    midpoint[2] = 15.0
    # Thin (tree-like), tall obstacle: small horizontal footprint, full height.
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(midpoint, 1.0, 10.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=4)
    assert any(h["enemy_avoidance_active"] for h in history)
    assert not any(h["enemy_obstacle_collision"] for h in history)


def test_obstacle_off_flight_path_does_not_trigger_avoidance():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=5)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    path = soldier_pos - enemy_start
    lateral = np.array([-path[1], path[0], 0.0])
    lateral_norm = np.linalg.norm(lateral)
    lateral = lateral / lateral_norm if lateral_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    # Perpendicular offset from the midpoint -- well off the direct line.
    far_point = (enemy_start + soldier_pos) / 2.0 + lateral * 25.0
    far_point[2] = 5.0
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(far_point, 2.0, 5.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=5)
    assert not any(h["enemy_avoidance_active"] for h in history)


def test_high_speed_swept_collision_cannot_tunnel_through_thin_obstacle():
    from uav_defend.obstacles import AABBObstacle, ObstacleLayout

    thin_wall = AABBObstacle.from_center_half_extents((10.0, 0.0, 5.0), (0.05, 5.0, 5.0))
    layout = ObstacleLayout(obstacles=(thin_wall,))
    # A large single-step displacement straight through the wall.
    result = layout.first_segment_intersection(
        np.array([5.0, 0.0, 5.0]), np.array([15.0, 0.0, 5.0])
    )
    assert result.intersects, "a fast entity crossing a thin obstacle in one step must still be detected"


def test_attacker_clears_obstacle_and_resumes_pursuit():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=6)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    midpoint[2] = 5.0
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(midpoint, 8.0, 8.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=6, max_steps=500)
    active_flags = [h["enemy_avoidance_active"] for h in history]
    assert any(active_flags), "expected avoidance to activate at some point"
    assert not all(active_flags), "avoidance must release and resume normal pursuit, not stay active forever"
    assert history[-1]["outcome"] in ("intercepted", "soldier_caught", "unsafe_intercept", "timeout")


def test_avoidance_hysteresis_prevents_left_right_oscillation():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=7)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    midpoint[2] = 5.0
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(midpoint, 5.0, 8.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=7, max_steps=400)
    modes = [h["enemy_obstacle_avoidance_mode"] for h in history if h["enemy_avoidance_active"]]
    # Count mode FLIPS between "left" and "right" while continuously active
    # (a "climb" step or a release/reselect is not itself an oscillation).
    flips = sum(
        1 for a, b in zip(modes, modes[1:])
        if {a, b} == {"left", "right"}
    )
    assert flips <= 1, f"expected at most one left/right transition (hysteresis), got {flips} in {modes}"


def test_two_nearby_obstacles_do_not_cause_immediate_second_collision():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=8)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    midpoint[2] = 5.0
    lateral = np.array([-(soldier_pos[1] - enemy_start[1]), soldier_pos[0] - enemy_start[0], 0.0])
    lateral_norm = np.linalg.norm(lateral)
    lateral = lateral / lateral_norm if lateral_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    obs_a = midpoint + lateral * 8.0
    obs_b = midpoint - lateral * 8.0
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(
            _safe_fixed_spec(obs_a, 3.0, 8.0, base_config),
            _safe_fixed_spec(obs_b, 3.0, 8.0, base_config),
        ),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=8, max_steps=400)
    assert not any(h["enemy_obstacle_collision"] for h in history)


def test_pathological_geometry_produces_finite_safe_action_not_nan():
    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=9)
    enemy_start = info["enemy_pos"].copy()
    # Obstacle placed almost exactly at the enemy's own starting position
    # (as close as the clearance validation for FIXED layouts allows: not
    # containing the point, but leaving essentially no maneuvering room).
    near = enemy_start + np.array([2.5, 0.0, 0.0])
    near[2] = np.clip(near[2], 3.0, base_config.max_altitude - 3.0)
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(near, 2.0, 2.0, base_config),),
        enemy_obstacle_avoidance_enabled=True,
    )
    history = _run(config, seed=9, max_steps=100)
    for h in history:
        assert np.all(np.isfinite(h["enemy_pos"]))
        assert np.all(np.isfinite(h["enemy_vel"]))
        assert np.isfinite(h["enemy_avoidance_angle_deg"])
