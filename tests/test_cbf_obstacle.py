"""Tests for the A1-CBF static-obstacle navigation stage (uav_defend/attackers/cbf_obstacle.py +
uav_defend/envs/soldier_env.py's obstacle-aware CBF path). No RL/training here -- everything is a deterministic
geometry/QP/route-planner unit test, or a small direct-simulation check. The full 1,000-episode C2 Monte Carlo
acceptance evaluation is run by experiments/journal_cbf_attacker_obstacle_run.py and recorded under
docs/journal_cbf_attacker/obstacle_cbf/.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.attackers.cbf_attacker import CBFAttackerController, CBFAttackerLimits
from uav_defend.attackers.cbf_obstacle import (
    CBF_ACTIVE_OBSTACLE_RANGE,
    CBF_ALPHA,
    CBF_OBSTACLE_CLEARANCE,
    CBFRouteNavigator,
    aabb_signed_distance_and_normal,
    build_static_hazards,
    inflate_aabb,
    integrate_obstacle_cbf_substeps,
    solve_velocity_cbf_qp,
    velocity_polygon_halfplanes,
    velocity_polygon_max_conservatism,
)
from uav_defend.obstacles.geometry import AABBObstacle

LIMITS = CBFAttackerLimits()


# ----------------------------------------------------------------------------- 1: AABB inflation
def test_aabb_inflation_expands_every_face_by_clearance():
    obstacle = AABBObstacle.from_center_half_extents((0.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    min_c, max_c = inflate_aabb(obstacle, 2.0)
    assert np.allclose(min_c, [-3.0, -3.0, -2.0]) and np.allclose(max_c, [3.0, 3.0, 6.0])


# ----------------------------------------------------------------------------- 2: signed AABB distance
def test_signed_aabb_distance_outside_matches_euclidean_distance():
    d, n, closest = aabb_signed_distance_and_normal(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([5.0, 1.0, 1.0]))
    assert d == pytest.approx(3.0) and np.allclose(n, [1.0, 0.0, 0.0]) and np.allclose(closest, [2.0, 1.0, 1.0])


def test_signed_aabb_distance_inside_is_negative():
    d, n, _ = aabb_signed_distance_and_normal(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
    assert d < 0 and d == pytest.approx(-1.0)


# ----------------------------------------------------------------------------- 3-5: face / edge / corner normal
def test_face_normal_is_single_axis_aligned():
    d, n, _ = aabb_signed_distance_and_normal(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([1.0, 3.0, 1.0]))
    assert d == pytest.approx(1.0) and np.allclose(n, [0.0, 1.0, 0.0])


def test_edge_normal_is_diagonal_between_two_axes():
    d, n, _ = aabb_signed_distance_and_normal(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 1.0]))
    assert d == pytest.approx(np.sqrt(2.0)) and np.allclose(n, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])


def test_corner_normal_is_diagonal_between_three_axes():
    d, n, _ = aabb_signed_distance_and_normal(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]))
    assert d == pytest.approx(np.sqrt(3.0)) and np.allclose(n, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))


# ----------------------------------------------------------------------------- 6-7: stopping-distance barrier / static CBF inequality
def test_stopping_distance_barrier_accounts_for_inward_speed():
    obstacle = AABBObstacle.from_center_half_extents((10.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    position = np.array([0.0, 0.0, 2.0])
    velocity_fast = np.array([12.0, 0.0, 0.0])
    velocity_slow = np.array([1.0, 0.0, 0.0])
    hz_fast = build_static_hazards(position, velocity_fast, (obstacle,), max_accel=6.0, dt_sub=0.1)[0]
    hz_slow = build_static_hazards(position, velocity_slow, (obstacle,), max_accel=6.0, dt_sub=0.1)[0]
    assert hz_fast.stopping_margin > hz_slow.stopping_margin        # faster closing speed -> larger required margin
    assert hz_fast.h < hz_slow.h                                     # -> smaller (more urgent) barrier value
    expected_stop = (12.0 ** 2) / (2 * 6.0) + 12.0 * 0.1
    assert hz_fast.stopping_margin == pytest.approx(expected_stop)


def test_static_cbf_inequality_limits_inward_velocity_component():
    from uav_defend.attackers.cbf_obstacle import StaticHazard

    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    # v_current=0, max_accel=1e6 (effectively infinite acceleration-reachable ball): this legacy Prompt-2/3 test
    # exercises the OBSTACLE-CBF inequality specifically, not acceleration-reachability (which has its own
    # dedicated tests in test_cbf_defender_v3.py, journal_cbf_attacker Prompt 6).
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([12.0, 0.0, 0.0]), [hz], v_current=np.zeros(3), max_accel=1e6,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "hard"
    assert float(np.dot(hz.normal, v_safe)) >= -CBF_ALPHA * hz.h - 1e-4   # n^T v_safe >= -alpha*h, satisfied
    assert v_safe[0] <= 1.0 + 1e-6                                        # forward speed is correctly capped


# ----------------------------------------------------------------------------- 8: multiple simultaneous obstacles
def test_multiple_obstacles_within_range_are_all_simultaneously_active():
    position = np.array([0.0, 0.0, 2.0])
    velocity = np.zeros(3)
    obstacles = (
        AABBObstacle.from_center_half_extents((5.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
        AABBObstacle.from_center_half_extents((-5.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
        AABBObstacle.from_center_half_extents((0.0, 100.0, 2.0), (1.0, 1.0, 2.0)),  # far outside active range
    )
    hz = build_static_hazards(position, velocity, obstacles, active_range=20.0, max_accel=6.0, dt_sub=0.1)
    assert len(hz) == 2 and {h.obstacle_id for h in hz} == {0, 1}            # never only the single nearest


# ----------------------------------------------------------------------------- 9: 32-sided velocity polygon
def test_velocity_polygon_is_inscribed_and_conservatism_is_small():
    normals, apothem = velocity_polygon_halfplanes(12.0, 32)
    assert normals.shape == (32, 2)
    assert apothem < 12.0                                                    # strictly inside the true circle
    conservatism = velocity_polygon_max_conservatism(12.0, 32)
    assert 0 < conservatism < 0.1                                            # documented: ~0.058 m/s for n=32, R=12
    # every polygon vertex itself lies exactly ON the true circle (by construction)
    vertex_radius = 12.0
    for nx, ny in normals:
        assert np.hypot(nx, ny) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- 10: hard QP
def test_hard_qp_succeeds_when_feasible():
    from uav_defend.attackers.cbf_obstacle import StaticHazard

    hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=5.0, h=3.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=7.0)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 12.0, 0.0]), [hz], v_current=np.zeros(3), max_accel=1e6,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "hard" and diag["qp_max_constraint_residual"] < 1e-4
    assert v_safe[1] <= 3.0 + 1e-4                                            # vy capped to satisfy -alpha*h = -3


# ----------------------------------------------------------------------------- 11: obstacle-hard infeasibility (Prompt 6: no more obstacle soft fallback)
def test_obstacle_hard_infeasibility_goes_straight_to_reachable_emergency_brake():
    """Prompt-6 behavior change (journal_cbf_attacker, defender_cbf_v3, brief section 11): obstacle safety may
    NEVER receive slack any more, not even in this no-defender, obstacle-only-infeasible case -- there is no
    longer an obstacle-slack 'soft_fallback' stage at all. When the hard obstacle+bounds(+acceleration)
    constraints are infeasible together, the controller goes straight to the REACHABLE emergency brake and
    classifies `physical_safety_infeasible`, honestly reporting the TRUE (un-softened) obstacle residual rather
    than a slack-absorbed near-zero one."""
    from uav_defend.attackers.cbf_obstacle import StaticHazard

    hz1 = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=-5.0, h=-10.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=-5.0)
    hz2 = StaticHazard(obstacle_id=1, normal=np.array([1.0, 0.0, 0.0]), d=-5.0, h=-10.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=-5.0)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([5.0, 0.0, 0.0]), [hz1, hz2], v_current=np.zeros(3), max_accel=1e6,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "emergency_brake"
    assert diag["priority_classification"] == "physical_safety_infeasible"
    assert diag["soft_slack"] == 0.0                      # no obstacle slack was ever introduced
    assert np.allclose(v_safe, 0.0)                       # v_current=0 -> the reachable brake is exactly zero


# ----------------------------------------------------------------------------- 12: emergency brake
def test_emergency_brake_triggers_when_both_qps_fail(monkeypatch):
    class _FakeResult:
        success = False
        status = 99
        x = np.array([20.0, 20.0, 20.0, 999.0])   # deliberately violates every constraint (speed bound, CBF, etc.)
        nit = 1

    def _always_fail(*args, **kwargs):
        return _FakeResult()

    monkeypatch.setattr("uav_defend.attackers.cbf_obstacle.minimize", _always_fail)
    from uav_defend.attackers.cbf_obstacle import StaticHazard

    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([12.0, 0.0, 0.0]), [hz], v_current=np.zeros(3), max_accel=1e6,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "emergency_brake" and np.allclose(v_safe, 0.0)


# ----------------------------------------------------------------------------- 13: identity / no-constraint behavior
def test_identity_behavior_when_no_hazards():
    controller = CBFAttackerController()
    v_nom = controller.nominal_velocity(np.array([5.0, 0.0, 0.0]))
    v_safe, diag = controller.filter_velocity(v_nom, hazards=())
    assert np.array_equal(v_safe, v_nom) and diag["cbf_qp_status"] == "identity_no_constraints"


# ----------------------------------------------------------------------------- 14: route planner integration
def test_route_planner_integration_routes_around_a_blocking_obstacle():
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    position = np.array([-20.0, 0.0, 2.0])
    goal = np.array([20.0, 0.0, 2.0])
    target, diag = nav.update(position, goal, obstacles, elapsed_time_s=0.0)
    assert diag["route_planner_status"] == "replanned" and not np.allclose(target, goal)
    assert diag["route_replan_count"] == 1


def test_route_planner_direct_shot_when_unblocked():
    nav = CBFRouteNavigator(max_altitude=30.0)
    position = np.array([-20.0, 0.0, 2.0])
    goal = np.array([20.0, 0.0, 2.0])
    target, diag = nav.update(position, goal, (), elapsed_time_s=0.0)
    assert diag["route_planner_status"] == "direct" and np.allclose(target, goal)


# ----------------------------------------------------------------------------- 15: moving-goal replanning
def test_route_replans_when_asset_moves_past_trigger_distance():
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    position = np.array([-20.0, 0.0, 2.0])
    goal1 = np.array([20.0, 0.0, 2.0])
    _, _ = nav.update(position, goal1, obstacles, elapsed_time_s=0.0)
    count_after_first = nav.replan_count
    goal2 = goal1 + np.array([5.0, 0.0, 0.0])   # moved > 2.0 m trigger
    _, diag2 = nav.update(position, goal2, obstacles, elapsed_time_s=0.2)
    assert diag2["route_planner_status"] == "replanned" and nav.replan_count == count_after_first + 1


def test_route_does_not_replan_before_period_or_trigger():
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0)
    position = np.array([-20.0, 0.0, 2.0])
    goal = np.array([20.0, 0.0, 2.0])
    nav.update(position, goal, obstacles, elapsed_time_s=0.0)
    count = nav.replan_count
    _, diag = nav.update(position, goal, obstacles, elapsed_time_s=0.5)   # < replan_period_s=1.0, no triggers
    assert diag["route_planner_status"] == "following" and nav.replan_count == count


def test_route_planner_failure_keeps_previous_waypoint_not_the_blocked_direct_target():
    """Regression test for a bug found during development: when a replan attempt fails (the planner returns None,
    e.g. because the attacker is already inside the inflated clearance zone -- see
    build_and_search_route's 'Phase 24' docstring), the navigator must NOT fall back to the raw (still-blocked)
    asset position -- that would aim guidance straight back through the obstacle. It must keep following the
    PREVIOUS route's current waypoint instead."""
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),)
    nav = CBFRouteNavigator(max_altitude=30.0, replan_period_s=1000.0)  # disable periodic replanning for this test
    position = np.array([-20.0, 0.0, 2.0])
    goal = np.array([20.0, 0.0, 2.0])
    target1, diag1 = nav.update(position, goal, obstacles, elapsed_time_s=0.0)
    assert diag1["route_planner_status"] == "replanned"
    # Force the NEXT replan to fail by moving the attacker INSIDE the inflated clearance zone (a position from
    # which build_and_search_route cannot find any unblocked edge -- see the module docstring).
    position_inside = np.array([0.0, 0.0, 2.0])  # center of the (inflated) obstacle
    goal_moved = goal + np.array([5.0, 0.0, 0.0])  # > 2.0 m trigger -> forces a replan attempt
    target2, diag2 = nav.update(position_inside, goal_moved, obstacles, elapsed_time_s=0.1)
    assert diag2["route_planner_status"] == "planner_failed"
    assert np.allclose(target2, target1)            # kept the PREVIOUS waypoint, not the (still-blocked) new goal
    assert not np.allclose(target2, goal_moved)


# ----------------------------------------------------------------------------- 16: no future target information
def test_nominal_velocity_uses_only_current_relative_position():
    controller = CBFAttackerController()
    r_now = np.array([6.0, 0.0, 0.0])
    v1 = controller.nominal_velocity(r_now)
    v2 = controller.nominal_velocity(r_now)       # calling again with the SAME current r must reproduce the SAME v
    assert np.allclose(v1, v2)                    # no hidden internal memory of "where the target used to be"


# ----------------------------------------------------------------------------- 17: C2 static-obstacle mapping
def test_c2_static_obstacles_are_all_map_known_and_static_geometry_unaffected_by_dynamic_toggle():
    import dataclasses

    from experiments import journal_v2_validation_core as VC
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg_nodyn = dataclasses.replace(cfg2, dynamic_obstacles_enabled=False)
    env_full, env_nodyn = SoldierEnv(cfg2), SoldierEnv(cfg_nodyn)
    _, info_full = env_full.reset(seed=500000)
    _, info_nodyn = env_nodyn.reset(seed=500000)
    assert all(o.map_known for o in env_full._obstacle_layout.obstacles)
    l1, l2 = env_full._obstacle_layout.obstacles, env_nodyn._obstacle_layout.obstacles
    assert len(l1) == len(l2) and all(np.allclose(a.min_corner, b.min_corner) and np.allclose(a.max_corner, b.max_corner) for a, b in zip(l1, l2))


# ----------------------------------------------------------------------------- 18: collision-free substep execution
def test_collision_free_execution_with_ample_stopping_room():
    """A scenario engineered with AMPLE stopping room (obstacle encountered from rest, well within the active
    range but far beyond the full-speed stopping distance) -- confirms the executor can actually avoid a static
    obstacle when physically possible, not merely that the QP reports 'hard' success."""
    controller = CBFAttackerController()
    obstacles = (AABBObstacle.from_center_half_extents((0.0, 15.0, 2.0), (2.0, 2.0, 2.0)),)
    pos = np.array([0.0, 0.0, 2.0])
    vel = np.zeros(3)                              # starts at REST -- no pre-existing inward speed
    route_target = np.array([0.0, 30.0, 2.0])       # straight through/past the obstacle
    min_physical_clearance = float("inf")
    for _ in range(40):
        pos, vel, exec_diag, cbf_diag = integrate_obstacle_cbf_substeps(
            pos, vel, route_target=route_target, controller=controller, obstacles=obstacles,
            dt=0.5, L=50.0, max_altitude=30.0,
        )
        min_physical_clearance = min(min_physical_clearance, obstacles[0].clearance_to_point(pos))
        if pos[1] >= 29.0:
            break
    assert min_physical_clearance >= 0.0            # never penetrates the raw obstacle


def test_known_acceleration_lag_collision_seed_is_a_spawn_time_physical_infeasibility():
    """Documents a diagnosed, NOT-a-bug finding from the acceptance campaign: seed 500100 (one of 1000 C2 Monte
    Carlo episodes) collides with a static obstacle because the attacker SPAWNS already closer to it than its own
    full-speed stopping distance allows (v_e=12 m/s, a_max=6 m/s^2 -> ~12.0-13.2 m minimum stopping distance,
    including the dt_sub reaction term; this obstacle's inflated boundary is only ~7.96 m away at spawn, with the
    spawn velocity pointed almost directly at it). No CBF-QP formulation bound by the SAME frozen a_max can avoid
    this: canonical C2's obstacle_clearance_from_spawn=3.0 m (designed for the turn-rate-limited legacy attacker)
    is far smaller than a velocity-commanded, acceleration-limited quadrotor's physical stopping envelope. This
    test asserts the diagnosis (pre-existing negative h at reset, for the specific colliding obstacle), NOT that
    the collision is avoided -- see docs/journal_cbf_attacker/obstacle_cbf/README.md's failure-analysis section."""
    import dataclasses

    from experiments import journal_v2_validation_core as VC
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)
    env = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity")
    _, info = env.reset(seed=500100)
    pos = np.asarray(info["enemy_pos"], dtype=np.float64)
    vel = np.asarray(info["enemy_vel"], dtype=np.float64)
    hazards = build_static_hazards(pos, vel, env._obstacle_layout.obstacles, max_accel=6.0, dt_sub=0.1)
    assert any(h.h < 0 for h in hazards)              # ALREADY past the safe-stopping threshold at spawn, frame 0
    full_speed_stop_distance = (12.0 ** 2) / (2 * 6.0)
    assert full_speed_stop_distance > 2.0 + CBF_OBSTACLE_CLEARANCE  # exceeds canonical obstacle_clearance_from_spawn


# ----------------------------------------------------------------------------- 19: diagnostic fields
def test_env_diagnostic_fields_present_for_obstacle_cbf():
    import dataclasses

    from experiments import journal_v2_validation_core as VC
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg2, _ = VC.build_config("C2", "V2", "passive")
    cfg2 = dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)
    env = SoldierEnv(cfg2, enemy_control_mode="cbf_velocity")
    obs, info = env.reset(seed=500000)
    obs, _r, _te, _tr, info = env.step(np.zeros(3, dtype=np.float32))
    for key in (
        "route_planner_status", "route_waypoint", "route_replan_count", "active_obstacle_count",
        "min_inflated_obstacle_clearance", "min_physical_obstacle_clearance", "cbf_h_min", "qp_success",
        "qp_iterations", "qp_max_constraint_residual", "soft_slack", "emergency_brake_count",
        # Prompt-1 fields preserved
        "cbf_enabled", "cbf_constraint_count", "cbf_qp_status", "attacker_velocity_command",
        "attacker_actual_velocity", "attacker_speed", "asset_relative_position", "asset_distance",
        "velocity_tracking_error", "cbf_intervention_norm",
    ):
        assert key in info, key


# ----------------------------------------------------------------------------- C0 identity regression (section 2)
def test_c0_obstacle_cbf_path_is_bit_identical_to_foundation_prompt1():
    import dataclasses

    from experiments import journal_cbf_attacker_foundation as F1
    from experiments import journal_v2_validation_core as VC
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg = F1.foundation_config()
    env_now = SoldierEnv(cfg, enemy_control_mode="cbf_velocity")
    for seed in range(500000, 500010):
        episode = F1.run_episode(env_now, seed)
        assert episode["outcome"] in ("soldier_caught", "enemy_endurance_exhausted")
