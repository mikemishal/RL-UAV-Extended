"""Tests for the acceleration-reachable CBF velocity-constrained optimization (journal_cbf_attacker,
defender_cbf_v3 / Prompt 6, brief section 24). Replaces Prompt 5's velocity-space-only hierarchical QP, which had
no notion of the vehicle's own one-substep acceleration-reachable set -- traced to 2/200 residual stress
collisions (seeds 630123, 630203) where the optimizer certified a velocity the executor could not actually reach
within one 0.1 s substep. See uav_defend/attackers/cbf_obstacle.py:solve_velocity_cbf_qp for the implementation
and docs/journal_cbf_attacker/defender_cbf_v3/README.md for the full Monte Carlo reproduction/retest results.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.attackers.cbf_attacker import CBFAttackerController, CBFAttackerLimits
from uav_defend.attackers.cbf_defender import DEFENDER_SAFETY_RADIUS, DefenderHazard
from uav_defend.attackers.cbf_obstacle import (
    StaticHazard,
    _accel_reachable_func,
    _reachable_emergency_brake,
    _reachable_initial_guess,
    solve_velocity_cbf_qp,
)

LIMITS = CBFAttackerLimits()
MAX_ACCEL = 6.0
DT_SUB = 0.1
ACCEL_LIMIT = MAX_ACCEL * DT_SUB  # 0.6 m/s


# ----------------------------------------------------------------------------- 1: exact acceleration ball
def test_accel_reachable_func_is_the_exact_euclidean_ball():
    v_current = np.array([1.0, -2.0, 0.5])
    f = _accel_reachable_func(v_current, MAX_ACCEL, DT_SUB)
    # Exactly on the boundary (||v - v_current|| == ACCEL_LIMIT): residual ~0.
    on_boundary = v_current + np.array([ACCEL_LIMIT, 0.0, 0.0])
    assert f(on_boundary) == pytest.approx(0.0, abs=1e-9)
    # Strictly inside: positive residual.
    assert f(v_current + np.array([ACCEL_LIMIT * 0.5, 0.0, 0.0])) > 0
    # Strictly outside: negative residual.
    assert f(v_current + np.array([ACCEL_LIMIT * 2.0, 0.0, 0.0])) < 0
    # At v_current itself: residual == ACCEL_LIMIT^2 (maximum slack).
    assert f(v_current) == pytest.approx(ACCEL_LIMIT ** 2)


# ----------------------------------------------------------------------------- 2: reachable vs unreachable candidates
def test_solve_never_returns_a_velocity_outside_the_accel_ball():
    v_current = np.array([5.0, 0.0, 0.0])
    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    # A nominal velocity FAR outside the reachable ball (would need an enormous, unreachable jump).
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([-12.0, 0.0, 0.0]), [hz], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert np.linalg.norm(v_safe - v_current) <= ACCEL_LIMIT + 1e-6


# ----------------------------------------------------------------------------- 3: reachable initial guess
def test_reachable_initial_guess_is_itself_within_the_accel_ball_and_velocity_bounds():
    v_current = np.array([0.0, 0.0, 0.0])
    v_nom = np.array([12.0, 0.0, 0.0])  # far outside the 0.6 m/s ball
    x0 = _reachable_initial_guess(v_nom, v_current, MAX_ACCEL, DT_SUB, 12.0, 5.0, 5.0)
    assert np.linalg.norm(x0 - v_current) <= ACCEL_LIMIT + 1e-9
    assert np.hypot(x0[0], x0[1]) <= 12.0 + 1e-6


def test_reachable_initial_guess_is_unclipped_when_v_nom_is_already_reachable():
    v_current = np.array([1.0, 1.0, 0.0])
    v_nom = v_current + np.array([0.1, 0.0, 0.0])  # well within the ball
    x0 = _reachable_initial_guess(v_nom, v_current, MAX_ACCEL, DT_SUB, 12.0, 5.0, 5.0)
    assert np.allclose(x0, v_nom)


# ----------------------------------------------------------------------------- 4: executor equality (optimizer/executor contract)
def test_optimized_velocity_equals_executed_velocity_within_tolerance():
    from uav_defend.attackers.cbf_attacker import apply_one_substep

    v_current = np.array([4.0, 0.0, 0.0])
    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.5, h=0.5, inward_speed=4.0, stopping_margin=1.0, physical_clearance=3.5)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([4.0, 2.0, 0.0]), [hz], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    pos = np.zeros(3)
    _, vel_after, _ = apply_one_substep(pos, v_current, v_safe, limits=LIMITS, dt_sub=DT_SUB, L=200.0, max_altitude=60.0)
    assert np.linalg.norm(vel_after - v_safe) <= 1e-4  # documented equivalent numerical tolerance (SLSQP residual)


# ----------------------------------------------------------------------------- 5: reachable emergency brake
def test_reachable_emergency_brake_formula():
    v_current = np.array([3.0, 0.0, 4.0])  # speed 5.0
    brake = _reachable_emergency_brake(v_current, MAX_ACCEL, DT_SUB)
    expected = v_current - ACCEL_LIMIT * (v_current / 5.0)
    assert np.allclose(brake, expected)
    assert np.linalg.norm(brake - v_current) == pytest.approx(ACCEL_LIMIT)


def test_reachable_emergency_brake_zero_velocity_case():
    assert np.allclose(_reachable_emergency_brake(np.zeros(3), MAX_ACCEL, DT_SUB), 0.0)


def test_reachable_emergency_brake_does_not_overshoot_past_zero_for_slow_velocity():
    v_current = np.array([0.1, 0.0, 0.0])  # speed < ACCEL_LIMIT (0.6)
    brake = _reachable_emergency_brake(v_current, MAX_ACCEL, DT_SUB)
    assert np.allclose(brake, 0.0)  # min(ACCEL_LIMIT, 0.1) == 0.1 -> fully stops, never overshoots


# ----------------------------------------------------------------------------- 6: obstacle hard + accel hard + defender soft
def test_obstacle_and_accel_stay_hard_while_defender_absorbs_conflict_via_slack():
    v_current = np.array([0.0, 0.0, 0.0])
    # Severity kept mild (h=-0.2) so the obstacle escape speed alone (0.2 m/s) is well within the 0.6 m/s
    # accel-reachable ball from v_current=0 -- isolating the DEFENDER-priority conflict this test targets from
    # the separate physical_safety_infeasible case (covered by its own dedicated test above).
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=0.3, h=-0.2, inward_speed=0.5, stopping_margin=0.1, physical_clearance=0.3)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=0.3, h=-0.2, closing_speed=0.5, stopping_margin=0.1, distance=0.3 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        defender_hazard=defender_hz, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["obstacle_constraints_hard"] is True
    assert np.linalg.norm(v_safe - v_current) <= ACCEL_LIMIT + 1e-6
    obstacle_lhs = float(np.dot(obstacle_hz.normal, v_safe)) + obstacle_hz.h
    assert obstacle_lhs >= -1e-4
    assert diag["priority_conflict"] == (diag["defender_slack"] > 1e-4)


# ----------------------------------------------------------------------------- 7: physical infeasibility classification
def test_physical_safety_infeasible_when_accel_ball_cannot_satisfy_obstacle_hazard():
    """Starting at rest (v_current=0) right next to a deeply-penetrated obstacle hazard: the 0.6 m/s
    acceleration-reachable ball cannot supply the escape speed the obstacle inequality demands, regardless of the
    defender -- this must classify `physical_safety_infeasible` and command the REACHABLE emergency brake
    (v_current=0 -> brake=0), never an invented unreachable escape velocity, and never relax obstacle safety."""
    v_current = np.zeros(3)
    hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=-1.0, h=-5.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=1.0)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 5.0, 0.0]), [hz], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["priority_classification"] == "physical_safety_infeasible"
    assert diag["cbf_qp_status"] == "emergency_brake"
    assert diag["reachable_emergency_brake"] is True
    assert np.allclose(v_safe, 0.0)


# ----------------------------------------------------------------------------- 8: seeds 630123/630203 regression (fixed)
def _run_stress_seed(seed: int, defender: str = "greedy") -> dict:
    import dataclasses

    import experiments.journal_cbf_attacker_defender_foundation as F
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator = F.defender_config(defender)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    assert F.build_stress_scenario(env, seed)
    policy = F.VC.make_policy(defender, cfg)
    policy.reset()
    done, info, step = False, env._get_info(), 0
    max_err = 0.0
    while not done and step < 60:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        step += 1
        max_err = max(max_err, info.get("executor_velocity_error", 0.0))
        done = te or tr
    return {"outcome": info["outcome"], "max_executor_velocity_error": max_err}


def test_seeds_630123_and_630203_no_longer_collide_with_accel_reachable_controller():
    for seed in (630123, 630203):
        result = _run_stress_seed(seed)
        assert result["outcome"] != "enemy_static_obstacle_collision", f"seed {seed} still collides"
        assert result["max_executor_velocity_error"] <= 1e-4


# ----------------------------------------------------------------------------- 9: seed 630069 retention
def test_seed_630069_remains_collision_free():
    result = _run_stress_seed(630069)
    assert result["outcome"] != "enemy_static_obstacle_collision"


# ----------------------------------------------------------------------------- 10: zero-velocity behavior
def test_zero_current_velocity_with_no_hazards_resolves_to_nominal_within_reach():
    v_current = np.zeros(3)
    v_nom = np.array([12.0, 0.0, 0.0])
    v_safe, diag = solve_velocity_cbf_qp(
        v_nom, [], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "hard"
    assert np.linalg.norm(v_safe - v_current) <= ACCEL_LIMIT + 1e-6
    # With no hazards, the QP should use the FULL reachable budget to move toward v_nom (closest feasible point).
    assert v_safe[0] == pytest.approx(ACCEL_LIMIT, abs=1e-4)


# ----------------------------------------------------------------------------- 11: velocity-limit interaction
def test_accel_ball_and_velocity_bounds_combine_correctly_near_the_speed_limit():
    # Already at the 12 m/s horizontal cap; nominal continues to push outward -- the accel ball (0.6 m/s wide)
    # keeps the solver from requesting anything that would exceed the velocity-bound polygon either.
    v_current = np.array([11.8, 0.0, 0.0])
    v_nom = np.array([20.0, 0.0, 0.0])
    v_safe, diag = solve_velocity_cbf_qp(
        v_nom, [], v_current=v_current, max_accel=MAX_ACCEL, dt_sub=DT_SUB,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert np.hypot(v_safe[0], v_safe[1]) <= 12.0 + 1e-3
    assert np.linalg.norm(v_safe - v_current) <= ACCEL_LIMIT + 1e-6


# ----------------------------------------------------------------------------- 12: executor contract diagnostics present
def test_controller_filter_velocity_requires_state_velocity_when_constraints_present():
    controller = CBFAttackerController(limits=LIMITS)
    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    with pytest.raises(ValueError):
        controller.filter_velocity(np.array([5.0, 0.0, 0.0]), hazards=(hz,))  # no state -> must raise, never silently default


def test_controller_filter_velocity_passes_accel_diagnostics_through():
    controller = CBFAttackerController(limits=LIMITS)
    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    v_safe, diag = controller.filter_velocity(
        np.array([5.0, 0.0, 0.0]), state={"position": np.zeros(3), "velocity": np.array([1.0, 0.0, 0.0])}, hazards=(hz,),
    )
    for key in ("optimizer_accel_request", "optimizer_accel_norm", "accel_constraint_active", "accel_reachable_margin"):
        assert key in diag
    assert np.linalg.norm(v_safe - np.array([1.0, 0.0, 0.0])) <= controller.limits.max_accel * 0.1 + 1e-6
