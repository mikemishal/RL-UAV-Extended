"""Tests for the A1-CBF viability-conditioned clutter benchmark (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3):
initial-state viability test, attacker-only resampling, and route-status (route_ok / route_temporarily_unavailable
/ route_error) handling. No RL/training here. The full 1,000-episode Clutter Monte Carlo and 100-case near-boundary
stress suite are run by experiments/journal_cbf_attacker_clutter_v2_run.py and recorded under
docs/journal_cbf_attacker/obstacle_cbf_v2/.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.attackers.cbf_obstacle import (
    ROUTE_ERROR,
    ROUTE_OK,
    ROUTE_TEMPORARILY_UNAVAILABLE,
    CBFRouteNavigator,
    initial_state_dynamically_viable,
)
from uav_defend.obstacles.geometry import AABBObstacle

from experiments import journal_v2_validation_core as VC
from uav_defend.envs.soldier_env import SoldierEnv


# ----------------------------------------------------------------------------- 1: dynamic viability calculation
def test_viability_pass_when_ample_stopping_room():
    obstacle = AABBObstacle.from_center_half_extents((0.0, 30.0, 2.0), (1.0, 1.0, 2.0))
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.array([0.0, 12.0, 0.0])
    result = initial_state_dynamically_viable(position, velocity, (obstacle,), max_accel=6.0)
    assert result["viable"] and result["min_margin"] > 0


def test_viability_fail_when_too_close_at_full_speed():
    obstacle = AABBObstacle.from_center_half_extents((0.0, 8.0, 2.0), (1.0, 1.0, 2.0))
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.array([0.0, 12.0, 0.0])   # full speed directly at an obstacle only ~5m of inflated clearance away
    result = initial_state_dynamically_viable(position, velocity, (obstacle,), max_accel=6.0)
    assert not result["viable"] and result["min_margin"] < 0


# ----------------------------------------------------------------------------- 2: tangential velocity needs no margin
def test_tangential_velocity_requires_no_stopping_margin():
    obstacle = AABBObstacle.from_center_half_extents((0.0, 3.0, 2.0), (1.0, 1.0, 2.0))  # very close
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.array([12.0, 0.0, 0.0])   # purely TANGENTIAL (perpendicular to the obstacle direction)
    result = initial_state_dynamically_viable(position, velocity, (obstacle,), max_accel=6.0)
    hz = result["hazards"][0]
    assert hz.inward_speed == pytest.approx(0.0, abs=1e-6) and hz.stopping_margin == pytest.approx(0.0, abs=1e-6)
    assert result["viable"]   # the obstacle is still "far enough" once there is zero inward speed


# ----------------------------------------------------------------------------- 3: inward velocity stopping distance
def test_inward_velocity_stopping_distance_matches_closed_form():
    obstacle = AABBObstacle.from_center_half_extents((0.0, 50.0, 2.0), (1.0, 1.0, 2.0))
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.array([0.0, 10.0, 0.0])
    result = initial_state_dynamically_viable(position, velocity, (obstacle,), max_accel=6.0, dt_sub=0.1)
    hz = result["hazards"][0]
    expected = (10.0 ** 2) / (2 * 6.0) + 10.0 * 0.1
    assert hz.stopping_margin == pytest.approx(expected)


# ----------------------------------------------------------------------------- 4: multiple obstacles
def test_viability_checks_every_obstacle_not_just_the_nearest():
    far_safe = AABBObstacle.from_center_half_extents((0.0, 50.0, 2.0), (1.0, 1.0, 2.0))
    near_unsafe = AABBObstacle.from_center_half_extents((10.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.array([12.0, 0.0, 0.0])  # aimed at the near, unsafe obstacle
    result = initial_state_dynamically_viable(position, velocity, (far_safe, near_unsafe), max_accel=6.0)
    assert not result["viable"] and len(result["hazards"]) == 2


# ----------------------------------------------------------------------------- 5-6: resample only attacker state; obstacle field unchanged
def test_resample_only_attacker_state_obstacle_field_unchanged():
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, dynamic_obstacles_enabled=False)
    env_off = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    env_on = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    # Use a seed known (from the Prompt-2 campaign) to require resampling.
    _, info_off = env_off.reset(seed=500100)
    _, info_on = env_on.reset(seed=500100)
    l1, l2 = env_off._obstacle_layout.obstacles, env_on._obstacle_layout.obstacles
    assert len(l1) == len(l2) and all(np.allclose(a.min_corner, b.min_corner) and np.allclose(a.max_corner, b.max_corner) for a, b in zip(l1, l2))
    assert np.allclose(info_off["soldier_pos"], info_on["soldier_pos"])   # asset initialization also unchanged
    assert info_on["viability_resample_count"] >= 1


# ----------------------------------------------------------------------------- 7: deterministic resampling
def test_resampling_is_deterministic():
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, dynamic_obstacles_enabled=False)
    env_a = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    env_b = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    _, info_a = env_a.reset(seed=500100)
    _, info_b = env_b.reset(seed=500100)
    assert np.allclose(info_a["enemy_pos"], info_b["enemy_pos"]) and info_a["viability_resample_count"] == info_b["viability_resample_count"]


# ----------------------------------------------------------------------------- 8: 100-attempt failure behavior
def test_scenario_generation_failure_after_100_attempts(monkeypatch):
    """Forces every candidate spawn to be non-viable (patch initial_state_dynamically_viable to always report
    infeasible) and confirms the env marks scenario_generation_failure rather than looping forever or silently
    relaxing the viability condition."""
    import uav_defend.envs.soldier_env as se

    def _always_infeasible(position, velocity, obstacles, **kwargs):
        return {"viable": False, "min_margin": -1.0, "hazards": []}

    monkeypatch.setattr(se, "initial_state_dynamically_viable", _always_infeasible)
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, dynamic_obstacles_enabled=False)
    env = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    _, info = env.reset(seed=500000)
    assert info["scenario_generation_failure"] is True and info["viability_resample_count"] == 99


# ----------------------------------------------------------------------------- 9: route status classification
def test_route_status_ok_direct():
    nav = CBFRouteNavigator(max_altitude=30.0)
    target, diag = nav.update(np.array([-20.0, 0.0, 2.0]), np.array([20.0, 0.0, 2.0]), (), elapsed_time_s=0.0)
    assert diag["route_status"] == ROUTE_OK and np.allclose(target, [20.0, 0.0, 2.0])


def test_route_status_ok_replanned():
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    target, diag = nav.update(np.array([-20.0, 0.0, 2.0]), np.array([20.0, 0.0, 2.0]), obstacles, elapsed_time_s=0.0)
    assert diag["route_status"] == ROUTE_OK and target is not None


# ----------------------------------------------------------------------------- 10: temporarily unavailable + previous waypoint retained
def test_route_temporarily_unavailable_retains_previous_waypoint():
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0, replan_period_s=1000.0)
    position = np.array([-20.0, 0.0, 2.0])
    goal = np.array([20.0, 0.0, 2.0])
    target1, diag1 = nav.update(position, goal, obstacles, elapsed_time_s=0.0)
    assert diag1["route_status"] == ROUTE_OK
    position_inside = np.array([0.0, 0.0, 2.0])
    goal_moved = goal + np.array([5.0, 0.0, 0.0])
    target2, diag2 = nav.update(position_inside, goal_moved, obstacles, elapsed_time_s=0.1)
    assert diag2["route_status"] == ROUTE_TEMPORARILY_UNAVAILABLE
    assert diag2["route_waypoint_retained"] is True and not diag2["route_hover_fallback"]
    assert np.allclose(target2, target1)


# ----------------------------------------------------------------------------- 11: temporarily unavailable + no waypoint -> hover
def test_route_temporarily_unavailable_no_waypoint_hovers():
    """A goal embedded inside an obstacle's inflated zone with NO previous route -- must hover (target=None),
    never fall back to the known-blocked direct target."""
    obstacle = AABBObstacle.from_center_half_extents((10.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    nav = CBFRouteNavigator(max_altitude=30.0)
    position = np.array([0.0, 0.0, 2.0])
    goal_inside = np.array([10.0, 0.0, 2.0])  # exactly the obstacle's center -- inside the inflated zone
    target, diag = nav.update(position, goal_inside, (obstacle,), elapsed_time_s=0.0)
    assert diag["route_status"] == ROUTE_TEMPORARILY_UNAVAILABLE
    assert target is None and diag["route_hover_fallback"] is True and diag["route_waypoint_retained"] is False


# ----------------------------------------------------------------------------- 12: route_error handling
def test_route_error_on_exception(monkeypatch):
    import uav_defend.attackers.cbf_obstacle as co

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated planner crash")

    monkeypatch.setattr(co, "build_and_search_route", _raise)
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    target, diag = nav.update(np.array([-20.0, 0.0, 2.0]), np.array([20.0, 0.0, 2.0]), obstacles, elapsed_time_s=0.0)
    assert diag["route_status"] == ROUTE_ERROR and diag["route_errors"] == 1


def test_route_error_on_malformed_waypoint_output(monkeypatch):
    import uav_defend.attackers.cbf_obstacle as co

    def _nan_waypoints(*args, **kwargs):
        return (np.array([np.nan, 0.0, 0.0]),)

    monkeypatch.setattr(co, "build_and_search_route", _nan_waypoints)
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    target, diag = nav.update(np.array([-20.0, 0.0, 2.0]), np.array([20.0, 0.0, 2.0]), obstacles, elapsed_time_s=0.0)
    assert diag["route_status"] == ROUTE_ERROR


# ----------------------------------------------------------------------------- 13: reproduction of Prompt-2 collision seed
def test_prompt2_collision_seed_resolved_by_viability_conditioning():
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)
    env_off = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    obs, info = env_off.reset(seed=500100)
    unconditioned_check = initial_state_dynamically_viable(
        np.asarray(info["enemy_pos"], dtype=np.float64), np.asarray(info["enemy_vel"], dtype=np.float64),
        env_off._obstacle_layout.obstacles, max_accel=6.0,
    )
    assert not unconditioned_check["viable"]   # confirms the ORIGINAL (unconditioned) spawn is infeasible

    env_on = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    obs, info = env_on.reset(seed=500100)
    assert info["initial_viability_margin"] >= 0 and not info["scenario_generation_failure"]
    done = False
    steps = 0
    while not done:
        obs, _r, te, tr, info = env_on.step(np.zeros(3, dtype=np.float32))
        done = te or tr
        steps += 1
        if steps > 3000:
            break
    assert info["outcome"] != "enemy_static_obstacle_collision"


# ----------------------------------------------------------------------------- 14: reproduction of Prompt-2 planner-failure cases
@pytest.mark.parametrize("seed", [500036, 500100, 500114, 500152, 500227, 500289, 500340, 500441, 500530, 500885, 500901])
def test_prompt2_planner_failure_seeds_produce_no_route_errors(seed):
    """Every one of the 11 Prompt-2 planner-failure seeds must now classify as route_ok/route_temporarily_unavailable
    only -- never route_error -- confirming the audit's conclusion (no planner bug) under the corrected handling."""
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)
    env = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
    obs, info = env.reset(seed=seed)
    if info["scenario_generation_failure"]:
        return
    done = False
    steps = 0
    route_errors_seen = 0
    while not done:
        obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        route_errors_seen = max(route_errors_seen, info["route_errors"])
        done = te or tr
        steps += 1
        if steps > 3000:
            break
    assert route_errors_seen == 0
    assert info["outcome"] != "enemy_static_obstacle_collision"


# ----------------------------------------------------------------------------- 15: near-viability-boundary safety
def test_near_viability_boundary_state_still_avoids_collision():
    """A state engineered to be LEGAL but difficult: 0 <= margin <= 0.5 m for one obstacle (brief section 15)."""
    obstacle = AABBObstacle.from_center_half_extents((0.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    # Solve for a distance that gives a margin in [0, 0.5]: stopping_margin(12, 6, 0.1) = 13.2; pick d = 13.1
    # (margin = d - stopping_margin = -0.1 -- adjust to a genuinely small POSITIVE margin instead).
    stopping = (12.0 ** 2) / (2 * 6.0) + 12.0 * 0.1
    d_target = stopping + 0.3   # margin ~ 0.3 m, within [0, 0.5]
    position = np.array([0.0, -(d_target + 2.0 + 1.0), 2.0])  # 2.0 clearance inflation + 1.0 obstacle half-extent
    velocity = np.array([0.0, 12.0, 0.0])
    result = initial_state_dynamically_viable(position, velocity, (obstacle,), max_accel=6.0)
    assert result["viable"] and 0.0 <= result["min_margin"] <= 0.5
