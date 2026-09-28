"""
Tests for the shared, controller-agnostic defender obstacle-awareness/
avoidance layer (journal extension): perception generalized from the
hostile's EnemyObstaclePerception (mover-agnostic despite the name), and
the SAME ObstacleAvoidancePlanner reused via a separate defender instance,
applied once in SoldierEnv._move_defender() -- upstream of EVERY defender
controller family (Greedy/PN/Lead/RMPC/PPO/PPO-Kalman/LR-PPO), which never
need to know it exists.

Covers: no-op when disabled/no-conflict (Phase 10), obstacle-ahead
triggers a collision-free modification, Level B/Level C perception limits
(Phase 5/6), dynamic-obstacle avoidance (Phase 13), the hostile UAV is
NEVER treated as environmental clutter (Phase 14), diagnostics (Phase 15),
controller-agnosticism across multiple controller families (Phase 17),
C0 regression (Phase 27), determinism, and config validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info


def _safe_fixed_spec(center: np.ndarray, half_xy: float, half_z: float, config: EnvConfig) -> tuple:
    margin = 0.5
    cx = float(np.clip(center[0], -config.L + half_xy + margin, config.L - half_xy - margin))
    cy = float(np.clip(center[1], -config.L + half_xy + margin, config.L - half_xy - margin))
    cz = float(np.clip(center[2], half_z + margin, config.max_altitude - half_z - margin))
    return (cx, cy, cz, half_xy, half_xy, half_z)


def _run_with_policy(config: EnvConfig, policy, seed: int, max_steps: int = 150):
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    policy.reset()
    history = [info]
    for _ in range(max_steps):
        action = np.asarray(policy.act(obs, build_policy_info(info, "measurement")), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        history.append(info)
        if done or truncated:
            break
    return history


def test_no_op_when_avoidance_disabled():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=10,
        defender_obstacle_avoidance_enabled=False,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=1)
    for h in history:
        assert h["defender_avoidance_active"] is False
        np.testing.assert_allclose(h["defender_desired_command"], h["defender_safe_command"])


def test_no_op_when_no_obstacle_conflict():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(np.array([-40.0, -40.0, 5.0]), 2.0, 5.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=2, max_steps=60)
    # A far-away, irrelevant obstacle must never trigger a modification.
    assert not any(h["defender_avoidance_active"] for h in history)


def test_obstacle_ahead_triggers_collision_free_modification():
    base = EnvConfig()
    obstacle_center = np.array([10.0, 0.0, 5.0])
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(obstacle_center, 5.0, 5.0, base),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        defender_standby_until_detection=False,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=3, max_steps=200)
    assert any(h["defender_avoidance_active"] for h in history)
    for h in history:
        pos = h["defender_pos"]
        inside = (5.0 <= pos[0] <= 15.0) and (-5.0 <= pos[1] <= 5.0) and (0.0 <= pos[2] <= 10.0)
        assert not inside


def test_level_b_defender_ignores_unperceived_obstacle():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(np.array([45.0, 45.0, 5.0]), 2.0, 5.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        defender_obstacle_sensor_range=5.0,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=4, max_steps=30)
    # Defender starts at origin, far from the obstacle -- must not be perceived yet.
    assert history[1]["defender_total_perceived_obstacles"] == 0


def test_level_c_defender_perceives_mapped_obstacle_immediately():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(np.array([45.0, 45.0, 5.0]), 2.0, 5.0, EnvConfig()),),
        obstacle_fixed_map_known=(True,),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="map_local",
        defender_obstacle_sensor_range=5.0, defender_standby_until_detection=False,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=5, max_steps=10)
    assert history[1]["defender_num_mapped_obstacles_known"] == 1
    assert history[1]["defender_total_perceived_obstacles"] == 1


def test_hostile_uav_never_treated_as_obstacle():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=0,
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=6, max_steps=100)
    # With zero environmental obstacles, avoidance must never activate no
    # matter how close the defender gets to the hostile (its mission target).
    assert not any(h["defender_avoidance_active"] for h in history)
    # The pursuit must still make progress (never permanently deflected).
    dists = [np.linalg.norm(h["defender_pos"] - h["enemy_pos"]) for h in history]
    assert min(dists) < dists[0]


def test_dynamic_obstacle_avoidance_active_and_transient():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=0,
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=3,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=7, max_steps=200)
    for h in history:
        assert np.all(np.isfinite(h["defender_pos"]))
    # A transient dynamic conflict must not permanently latch active.
    active_flags = [h["defender_avoidance_active"] for h in history]
    if any(active_flags):
        assert not all(active_flags)


def test_diagnostics_present_in_info():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(np.array([10.0, 0.0, 5.0]), 4.0, 5.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=8, max_steps=60)
    required = {
        "defender_awareness_mode", "defender_num_mapped_obstacles_known",
        "defender_num_locally_sensed_static", "defender_num_locally_sensed_dynamic",
        "defender_total_perceived_obstacles", "defender_avoidance_active",
        "defender_avoidance_obstacle_id", "defender_avoidance_obstacle_type",
        "defender_avoidance_angle_deg", "defender_desired_command", "defender_safe_command",
        "defender_predicted_min_clearance", "defender_actual_min_clearance",
    }
    for h in history:
        assert required.issubset(h.keys())


def test_controller_agnostic_greedy_and_lead_both_use_same_safety_layer():
    obstacle_center = np.array([10.0, 0.0, 5.0])
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(obstacle_center, 5.0, 5.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
    )
    for policy in (GreedyInterceptPolicy(), LeadInterceptPolicy(state_source="measurement", config=config)):
        history = _run_with_policy(config, policy, seed=9, max_steps=200)
        for h in history:
            pos = h["defender_pos"]
            inside = (5.0 <= pos[0] <= 15.0) and (-5.0 <= pos[1] <= 5.0) and (0.0 <= pos[2] <= 10.0)
            assert not inside, f"{type(policy).__name__} penetrated the obstacle"


def test_rmpc_also_uses_the_same_safety_layer():
    """Phase 17: RMPC is a fundamentally different (optimization-based,
    not reactive-analytical) controller family -- confirm the shared
    safety layer applies to it identically, with no special-casing."""
    from uav_defend.policies.mpc.rmpc_policy import RMPCConfig, RMPCPolicy

    obstacle_center = np.array([10.0, 0.0, 5.0])
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(obstacle_center, 5.0, 5.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        defender_standby_until_detection=False,
    )
    policy = RMPCPolicy(config=config, rmpc_config=RMPCConfig(enable_timing=False))
    history = _run_with_policy(config, policy, seed=3, max_steps=15)
    assert any(h["defender_avoidance_active"] for h in history)


def test_c0_defender_default_disabled_matches_legacy_behavior():
    config_a = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=0)
    config_b = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=0,
        defender_obstacle_avoidance_enabled=False,  # explicit == default
    )
    history_a = _run_with_policy(config_a, GreedyInterceptPolicy(), seed=10, max_steps=100)
    history_b = _run_with_policy(config_b, GreedyInterceptPolicy(), seed=10, max_steps=100)
    assert len(history_a) == len(history_b)
    for ha, hb in zip(history_a, history_b):
        np.testing.assert_allclose(ha["defender_pos"], hb["defender_pos"], atol=1e-9)


def test_obstacle_collisions_remain_diagnostic_only_never_terminal():
    """Phase 21: ground-truth obstacle collisions must never affect
    episode outcome/termination -- only interception geometry does."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(_safe_fixed_spec(np.array([0.0, 0.0, 10.0]), 20.0, 9.0, EnvConfig()),),
        defender_obstacle_avoidance_enabled=False,  # force straight-line collisions
        defender_standby_until_detection=False,
    )
    history = _run_with_policy(config, GreedyInterceptPolicy(), seed=11, max_steps=300)
    outcomes = {h["outcome"] for h in history[:-1]}
    assert outcomes <= {"ongoing"}  # never terminated early just from a collision


def test_seed_reproducibility():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=10,
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=2,
    )
    history_a = _run_with_policy(config, GreedyInterceptPolicy(), seed=12, max_steps=150)
    history_b = _run_with_policy(config, GreedyInterceptPolicy(), seed=12, max_steps=150)
    assert len(history_a) == len(history_b)
    for ha, hb in zip(history_a, history_b):
        np.testing.assert_allclose(ha["defender_pos"], hb["defender_pos"], atol=1e-12)


def test_invalid_defender_config_rejected():
    with pytest.raises(ValueError):
        EnvConfig(defender_obstacle_awareness_mode="bogus")
    with pytest.raises(ValueError):
        EnvConfig(defender_obstacle_sensor_range=-1.0)
    with pytest.raises(ValueError):
        EnvConfig(defender_obstacle_clearance=-1.0)
    with pytest.raises(ValueError):
        EnvConfig(defender_obstacle_prediction_horizon_steps=0)
    with pytest.raises(ValueError):
        EnvConfig(defender_obstacle_release_steps=0)


def test_matched_scenario_independent_of_defender_policy_choice():
    """Phase 26 fairness: obstacle layout and hostile/soldier trajectories
    are determined ENTIRELY by env.reset(seed=...), independent of which
    defender controller family is used -- i.e. every controller in a
    benchmark run sees the identical scenario for a given seed."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=6,
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="map_local",
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=2,
    )
    env_a = SoldierEnv(config=config)
    obs_a, info_a = env_a.reset(seed=42)
    env_b = SoldierEnv(config=config)
    obs_b, info_b = env_b.reset(seed=42)
    # Same seed -> identical obstacle layout and initial enemy/soldier state,
    # regardless of which policy will later drive the defender.
    np.testing.assert_allclose(info_a["enemy_pos"], info_b["enemy_pos"])
    np.testing.assert_allclose(info_a["soldier_pos"], info_b["soldier_pos"])
    np.testing.assert_allclose(info_a["defender_pos"], info_b["defender_pos"])
    assert info_a["defender_num_mapped_obstacles_known"] == info_b["defender_num_mapped_obstacles_known"]

    # Advance the hostile/soldier one step under two DIFFERENT defender
    # policies -- the enemy's own trajectory must not depend on the
    # defender's controller family (only on the shared seeded RNG state,
    # which is identical since env_a/env_b started from the same seed).
    policy_greedy = GreedyInterceptPolicy()
    policy_lead = LeadInterceptPolicy(state_source="measurement", config=config)
    action_a = np.asarray(policy_greedy.act(obs_a, build_policy_info(info_a, "measurement")), dtype=np.float32)
    action_b = np.asarray(policy_lead.act(obs_b, build_policy_info(info_b, "measurement")), dtype=np.float32)
    _, _, _, _, info_a2 = env_a.step(action_a)
    _, _, _, _, info_b2 = env_b.step(action_b)
    np.testing.assert_allclose(info_a2["enemy_pos"], info_b2["enemy_pos"], atol=1e-12)
    np.testing.assert_allclose(info_a2["soldier_pos"], info_b2["soldier_pos"], atol=1e-12)
