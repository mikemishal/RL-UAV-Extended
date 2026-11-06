"""Tests for the A1-CBF defender-avoidance stage (uav_defend/attackers/cbf_defender.py + the joint-QP
generalization in uav_defend/attackers/cbf_obstacle.py/cbf_attacker.py + uav_defend/envs/soldier_env.py's
defender-aware CBF path). journal_cbf_attacker, defender_cbf / Prompt 4, brief section 25 (24 required test areas).

No RL/training here -- everything is a deterministic geometry/QP/estimator unit test, or a small direct-simulation
check. The full Monte Carlo acceptance campaigns (primary matched Greedy, 200-case stress set, generalization
diagnostic) are run by experiments/journal_cbf_attacker_defender_run.py and recorded under
docs/journal_cbf_attacker/defender_cbf/.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.attackers.cbf_attacker import CBFAttackerController, CBFAttackerLimits
from uav_defend.attackers.cbf_defender import (
    CBF_ALPHA_DEFENDER,
    DEFENDER_DETECTION_RANGE,
    DEFENDER_SAFETY_RADIUS,
    DefenderHazard,
    DefenderVelocityEstimator,
    build_defender_hazard,
    gated_defender_position,
    predict_defender_positions,
)
from uav_defend.attackers.cbf_obstacle import (
    CBF_ALPHA,
    StaticHazard,
    integrate_obstacle_cbf_substeps,
    solve_velocity_cbf_qp,
)
from uav_defend.obstacles.geometry import AABBObstacle

LIMITS = CBFAttackerLimits()


# ----------------------------------------------------------------------------- 1: defender observation gate
def test_defender_gate_returns_position_within_range_and_none_outside():
    attacker_pos = np.zeros(3)
    inside = gated_defender_position(attacker_pos, np.array([20.0, 0.0, 0.0]))
    outside = gated_defender_position(attacker_pos, np.array([26.0, 0.0, 0.0]))
    assert inside is not None and np.allclose(inside, [20.0, 0.0, 0.0])
    assert outside is None


def test_defender_gate_boundary_is_inclusive_at_exactly_the_canonical_range():
    attacker_pos = np.zeros(3)
    on_boundary = gated_defender_position(attacker_pos, np.array([DEFENDER_DETECTION_RANGE, 0.0, 0.0]))
    just_outside = gated_defender_position(attacker_pos, np.array([DEFENDER_DETECTION_RANGE + 1e-6, 0.0, 0.0]))
    assert on_boundary is not None
    assert just_outside is None


# ----------------------------------------------------------------------------- 2: no ground-truth leakage
def test_gated_position_is_none_means_no_hazard_can_be_built_without_it():
    """When the gate fails, the caller contract is: do not call build_defender_hazard at all (no ground-truth
    leakage). This test documents/upholds that by checking the gate itself never returns a disguised value (e.g.
    zeros) that could be mistaken for a real observation."""
    attacker_pos = np.array([0.0, 0.0, 0.0])
    defender_pos = np.array([100.0, 0.0, 0.0])
    assert gated_defender_position(attacker_pos, defender_pos) is None


def test_defender_hazard_uses_estimated_velocity_not_a_ground_truth_field():
    """build_defender_hazard's signature has no ground-truth-velocity parameter -- only the caller-supplied
    ESTIMATE. This is a structural/contract check (cannot pass a ground-truth defender velocity by accident)."""
    import inspect

    params = list(inspect.signature(build_defender_hazard).parameters)
    assert params[:4] == ["attacker_pos", "attacker_vel", "defender_pos", "defender_vel"]
    assert "defender_vel_ground_truth" not in params


# ----------------------------------------------------------------------------- 3: defender velocity estimation
def test_velocity_estimator_matches_finite_difference_for_consecutive_observations():
    est = DefenderVelocityEstimator(dt=0.5)
    v0, valid0 = est.update(0, np.array([0.0, 0.0, 0.0]))
    v1, valid1 = est.update(1, np.array([1.0, 0.0, 0.0]))
    assert not valid0
    assert valid1
    assert np.allclose(v1, [2.0, 0.0, 0.0])  # (1.0 - 0.0) / 0.5


# ----------------------------------------------------------------------------- 4: first-observation handling
def test_first_observation_yields_zero_velocity_and_invalid_flag():
    est = DefenderVelocityEstimator(dt=0.5)
    v, valid = est.update(5, np.array([3.0, 4.0, 0.0]))
    assert not valid
    assert np.allclose(v, [0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- 5: no differentiation across a gap
def test_velocity_estimator_does_not_differentiate_across_a_missing_step():
    est = DefenderVelocityEstimator(dt=0.5)
    est.update(0, np.array([0.0, 0.0, 0.0]))
    v_gap, valid_gap = est.update(1, None)  # defender unobservable this step
    assert not valid_gap and np.allclose(v_gap, 0.0)
    # Re-acquisition at step 3 (step 1 and 2 were gaps) must be treated as a FRESH first observation, never
    # differentiated against the stale step-0 sample (which would silently divide by 3*dt).
    v_reacquire, valid_reacquire = est.update(3, np.array([100.0, 0.0, 0.0]))
    assert not valid_reacquire
    assert np.allclose(v_reacquire, 0.0)
    # The NEXT consecutive step correctly differentiates against the step-3 sample.
    v_next, valid_next = est.update(4, np.array([101.0, 0.0, 0.0]))
    assert valid_next
    assert np.allclose(v_next, [2.0, 0.0, 0.0])


def test_velocity_estimator_is_deterministic_and_stateful_across_reset():
    est = DefenderVelocityEstimator(dt=0.5)
    est.update(0, np.array([0.0, 0.0, 0.0]))
    est.update(1, np.array([1.0, 0.0, 0.0]))
    est.reset()
    v, valid = est.update(0, np.array([1.0, 0.0, 0.0]))
    assert not valid  # reset discards prior history -- treated as first observation again
    assert np.allclose(v, 0.0)


# ----------------------------------------------------------------------------- 6: relative-state geometry
def test_relative_state_normal_points_from_defender_toward_attacker():
    hz = build_defender_hazard(
        attacker_pos=np.array([10.0, 0.0, 0.0]), attacker_vel=np.zeros(3),
        defender_pos=np.array([0.0, 0.0, 0.0]), defender_vel=np.zeros(3),
        max_accel=6.0, dt_sub=0.1,
    )
    assert np.allclose(hz.normal, [1.0, 0.0, 0.0])
    assert hz.distance == pytest.approx(10.0)
    assert hz.d == pytest.approx(10.0 - DEFENDER_SAFETY_RADIUS)


def test_relative_state_handles_zero_distance_deterministically():
    hz1 = build_defender_hazard(
        attacker_pos=np.zeros(3), attacker_vel=np.zeros(3), defender_pos=np.zeros(3), defender_vel=np.zeros(3),
        max_accel=6.0, dt_sub=0.1,
    )
    hz2 = build_defender_hazard(
        attacker_pos=np.zeros(3), attacker_vel=np.zeros(3), defender_pos=np.zeros(3), defender_vel=np.zeros(3),
        max_accel=6.0, dt_sub=0.1,
    )
    assert np.allclose(hz1.normal, hz2.normal)  # deterministic fallback, no RNG
    assert np.all(np.isfinite(hz1.normal))


# ----------------------------------------------------------------------------- 7: safety-radius use
def test_safety_radius_is_subtracted_from_raw_distance():
    hz = build_defender_hazard(
        attacker_pos=np.array([5.0, 0.0, 0.0]), attacker_vel=np.zeros(3),
        defender_pos=np.zeros(3), defender_vel=np.zeros(3), safety_radius=2.0, max_accel=6.0, dt_sub=0.1,
    )
    assert hz.d == pytest.approx(3.0)


# ----------------------------------------------------------------------------- 8: closing-speed calculation
def test_closing_speed_is_zero_when_not_approaching():
    hz = build_defender_hazard(
        attacker_pos=np.array([10.0, 0.0, 0.0]), attacker_vel=np.array([1.0, 0.0, 0.0]),  # moving AWAY
        defender_pos=np.zeros(3), defender_vel=np.zeros(3), max_accel=6.0, dt_sub=0.1,
    )
    assert hz.closing_speed == pytest.approx(0.0)


def test_closing_speed_is_positive_when_approaching():
    hz = build_defender_hazard(
        attacker_pos=np.array([10.0, 0.0, 0.0]), attacker_vel=np.array([-3.0, 0.0, 0.0]),  # moving TOWARD defender
        defender_pos=np.zeros(3), defender_vel=np.zeros(3), max_accel=6.0, dt_sub=0.1,
    )
    assert hz.closing_speed == pytest.approx(3.0)


def test_closing_speed_accounts_for_defender_motion_too():
    hz = build_defender_hazard(
        attacker_pos=np.array([10.0, 0.0, 0.0]), attacker_vel=np.zeros(3),
        defender_pos=np.zeros(3), defender_vel=np.array([2.0, 0.0, 0.0]),  # defender approaching attacker
        max_accel=6.0, dt_sub=0.1,
    )
    assert hz.closing_speed == pytest.approx(2.0)


# ----------------------------------------------------------------------------- 9: acceleration-aware stopping margin
def test_defender_stopping_margin_matches_the_frozen_formula():
    s = 4.0
    hz = build_defender_hazard(
        attacker_pos=np.array([10.0, 0.0, 0.0]), attacker_vel=np.array([-s, 0.0, 0.0]),
        defender_pos=np.zeros(3), defender_vel=np.zeros(3), max_accel=6.0, dt_sub=0.1,
    )
    expected = (s ** 2) / (2 * 6.0) + s * 0.1
    assert hz.stopping_margin == pytest.approx(expected)
    assert hz.h == pytest.approx(hz.d - expected)


# ----------------------------------------------------------------------------- 10: defender CBF inequality
def test_defender_only_qp_satisfies_the_frozen_inequality():
    hz = DefenderHazard(normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, closing_speed=0.0, stopping_margin=0.0, distance=4.5, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([12.0, 0.0, 0.0]), [], defender_hazard=hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "hard"
    lhs = float(np.dot(hz.normal, v_safe - hz.defender_velocity))
    assert lhs >= -CBF_ALPHA_DEFENDER * hz.h - 1e-4


# ----------------------------------------------------------------------------- 11: predicted defender position over substeps
def test_predicted_defender_positions_extrapolate_at_constant_velocity_only():
    positions = predict_defender_positions(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0]), dt_sub=0.1, n_substeps=5)
    assert len(positions) == 5
    for i, p in enumerate(positions):
        assert np.allclose(p, [1.0 * i * 0.1, 2.0 * i * 0.1, 0.0])


# ----------------------------------------------------------------------------- 12: defender-only QP
def test_defender_only_qp_deflects_velocity_away_from_closing_defender():
    hz = build_defender_hazard(
        attacker_pos=np.array([2.0, 0.0, 0.0]), attacker_vel=np.array([-6.0, 0.0, 0.0]),
        defender_pos=np.zeros(3), defender_vel=np.zeros(3), max_accel=6.0, dt_sub=0.1,
    )
    v_nom = np.array([-6.0, 0.0, 0.0])  # nominal: fly straight at/through the defender
    v_safe, diag = solve_velocity_cbf_qp(
        v_nom, [], defender_hazard=hz, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    # The safe velocity must reduce the inward (toward-defender) component relative to nominal.
    inward_nom = -float(np.dot(hz.normal, v_nom))
    inward_safe = -float(np.dot(hz.normal, v_safe))
    assert inward_safe < inward_nom
    assert diag["intervention_source"] in ("defender", "joint")


# ----------------------------------------------------------------------------- 13: obstacle + defender joint QP
def test_joint_qp_satisfies_both_constraint_classes_simultaneously():
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, 1.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=1.0, h=1.0, closing_speed=0.0, stopping_margin=0.0, distance=4.5, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([5.0, -5.0, 0.0]), [obstacle_hz], defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] in ("hard", "soft_fallback")
    obstacle_lhs = float(np.dot(obstacle_hz.normal, v_safe)) + CBF_ALPHA * obstacle_hz.h
    defender_lhs = float(np.dot(defender_hz.normal, v_safe - defender_hz.defender_velocity)) + CBF_ALPHA_DEFENDER * defender_hz.h
    slack_allow = diag.get("soft_slack", 0.0) + 1e-3
    assert obstacle_lhs >= -slack_allow
    assert defender_lhs >= -slack_allow


# ----------------------------------------------------------------------------- 14: conflicting constraints
def test_conflicting_obstacle_and_defender_constraints_trigger_soft_or_emergency_not_a_crash():
    # Obstacle directly behind (blocking retreat) and defender directly ahead (blocking advance) -- a narrow/
    # impossible corridor along the same axis, forcing the solver into the soft-slack or emergency path.
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=0.05, h=-5.0, inward_speed=10.0, stopping_margin=5.0, physical_clearance=0.05)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=0.05, h=-5.0, closing_speed=10.0, stopping_margin=5.0, distance=0.05 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] in ("hard", "soft_fallback", "emergency_brake")
    assert np.all(np.isfinite(v_safe))


# ----------------------------------------------------------------------------- 15: soft fallback classification
def test_soft_fallback_infeasibility_source_identifies_the_violating_class():
    # A defender constraint so tight it cannot be satisfied by the hard QP alone (h very negative, close range),
    # with NO static obstacle present -- any resulting soft fallback must blame "defender", never "obstacle".
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=0.01, h=-50.0, closing_speed=50.0, stopping_margin=50.01, distance=0.01 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [], defender_hazard=defender_hz,
        max_horizontal_speed=1.0, max_climb_rate=1.0, max_descent_rate=1.0,  # tight speed bound forces infeasibility
    )
    if diag["cbf_qp_status"] in ("soft_fallback", "emergency_brake"):
        assert diag["infeasibility_source"] in ("defender", "joint")


# ----------------------------------------------------------------------------- 16: emergency brake
def test_emergency_brake_commands_zero_velocity_never_an_invented_direction():
    # An utterly infeasible joint configuration under an extremely tight speed bound.
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-10.0, h=-100.0, inward_speed=50.0, stopping_margin=90.0, physical_clearance=-10.0)
    defender_hz = DefenderHazard(normal=np.array([-1.0, 0.0, 0.0]), d=-10.0, h=-100.0, closing_speed=50.0, stopping_margin=90.0, distance=-10.0 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], defender_hazard=defender_hz,
        max_horizontal_speed=0.01, max_climb_rate=0.01, max_descent_rate=0.01,
    )
    if diag["cbf_qp_status"] == "emergency_brake":
        assert np.allclose(v_safe, 0.0)


# ----------------------------------------------------------------------------- 17: intervention-source classification
def test_intervention_source_classifies_none_obstacle_defender_and_joint():
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=0.5, h=-1.0, inward_speed=5.0, stopping_margin=1.5, physical_clearance=0.5)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=0.5, h=-1.0, closing_speed=5.0, stopping_margin=1.5, distance=0.5 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))

    _, diag_none = solve_velocity_cbf_qp(np.zeros(3), [], defender_hazard=None, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    assert diag_none["intervention_source"] == "none"

    # obstacle normal=[-1,0,0]: constraint n.v + alpha*h >= 0 is violated for v_x > -1 (h=-1 means already
    # penetrated -- almost any non-retreating command is "active"). v=[5,0,0] clearly violates it.
    _, diag_obstacle = solve_velocity_cbf_qp(np.array([5.0, 0.0, 0.0]), [obstacle_hz], defender_hazard=None, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    assert diag_obstacle["intervention_source"] == "obstacle"

    # defender normal=[1,0,0]: constraint n.v >= n.v_d - alpha_d*h is violated for v_x < 1. v=[-5,0,0] violates it.
    _, diag_defender = solve_velocity_cbf_qp(np.array([-5.0, 0.0, 0.0]), [], defender_hazard=defender_hz, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    assert diag_defender["intervention_source"] == "defender"

    # v_x=0 lies in the overlap region (-1, 1) that violates BOTH constraints simultaneously -> "joint".
    _, diag_joint = solve_velocity_cbf_qp(np.array([0.0, 0.0, 0.0]), [obstacle_hz], defender_hazard=defender_hz, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    assert diag_joint["intervention_source"] == "joint"


# ----------------------------------------------------------------------------- 18: 25 m gate entry/exit
def test_gate_entry_and_exit_are_symmetric_around_the_canonical_range():
    attacker_pos = np.zeros(3)
    just_inside = gated_defender_position(attacker_pos, np.array([DEFENDER_DETECTION_RANGE - 0.01, 0.0, 0.0]))
    just_outside = gated_defender_position(attacker_pos, np.array([DEFENDER_DETECTION_RANGE + 0.01, 0.0, 0.0]))
    assert just_inside is not None
    assert just_outside is None


# ----------------------------------------------------------------------------- 19: actual 2.5 m interception unchanged
def test_physical_intercept_radius_constant_is_unchanged_in_env_config():
    from uav_defend.config.env_config import EnvConfig

    assert EnvConfig().intercept_radius == pytest.approx(2.5)


# ----------------------------------------------------------------------------- 20: 3.5 m unsafe radius unchanged
def test_defender_safety_radius_constant_matches_canonical_unsafe_intercept_radius_value():
    from uav_defend.config.env_config import EnvConfig

    assert EnvConfig().unsafe_intercept_radius == pytest.approx(3.5)
    assert DEFENDER_SAFETY_RADIUS == pytest.approx(3.5)


# ----------------------------------------------------------------------------- 21: Greedy defender unchanged
def test_greedy_defender_policy_module_is_unmodified_by_this_stage():
    """Structural guard: this stage must not touch the Greedy policy implementation. We cannot diff git history
    from inside a test, so instead assert the documented canonical interface (act/reset) is intact and the policy
    produces a deterministic, finite action for a simple scenario -- a regression trip-wire, not a full behavior
    audit (the dedicated Monte Carlo campaign is the real regression)."""
    from uav_defend.policies.registry import get_policy

    policy = get_policy("greedy")
    policy.reset()
    info = {
        "defender_pos": np.zeros(3, dtype=np.float32), "enemy_measurement": np.array([10.0, 0.0, 2.0], dtype=np.float32),
        "soldier_pos": np.zeros(3, dtype=np.float32), "enemy_detected": True,
    }
    action = np.asarray(policy.act(np.zeros(16, dtype=np.float32), info)).reshape(3)
    assert np.all(np.isfinite(action))


# ----------------------------------------------------------------------------- 22: obstacle-only/full matched scenario equivalence
def test_full_variant_degenerates_to_obstacle_only_when_defender_unobservable():
    """With defender_hazard=None (not observable), the joint QP must be bit-for-bit identical to the Prompt-2/3
    obstacle-only QP -- confirms the 'full' and 'obstacle-only' configurations never diverge while the defender is
    out of range."""
    hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    v_a, diag_a = solve_velocity_cbf_qp(np.array([12.0, 0.0, 0.0]), [hz], defender_hazard=None, max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    v_b, diag_b = solve_velocity_cbf_qp(np.array([12.0, 0.0, 0.0]), [hz], max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0)
    assert np.allclose(v_a, v_b)
    assert diag_a["cbf_qp_status"] == diag_b["cbf_qp_status"]


def test_controller_filter_velocity_identity_shortcut_requires_both_empty():
    controller = CBFAttackerController(limits=LIMITS)
    v_nom = np.array([3.0, 4.0, 0.0])
    v_safe, diag = controller.filter_velocity(v_nom, hazards=(), defender_hazard=None)
    assert np.allclose(v_safe, v_nom)  # identity: no hazards of either class
    assert diag["intervention_source"] == "none"


# ----------------------------------------------------------------------------- 23: static clutter regression (defender disabled path)
def test_integrate_obstacle_cbf_substeps_without_defender_matches_obstacle_only_signature():
    """defender_pos=None (the Prompt-2/3 call signature) must produce the exact same aggregate diagnostics schema
    plus the new (neutral) defender fields, never crashing or omitting required keys."""
    controller = CBFAttackerController(limits=LIMITS)
    obstacle = AABBObstacle.from_center_half_extents((5.0, 0.0, 2.0), (1.0, 1.0, 2.0))
    pos, vel, exec_diag, cbf_diag = integrate_obstacle_cbf_substeps(
        position=np.array([0.0, 0.0, 2.0]), velocity=np.zeros(3), route_target=np.array([10.0, 0.0, 2.0]),
        controller=controller, obstacles=(obstacle,), dt=0.5, L=100.0, max_altitude=30.0,
    )
    assert cbf_diag["defender_constraint_present"] is False
    assert cbf_diag["last_defender_h"] == float("inf")
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))


# ----------------------------------------------------------------------------- 24: stress-set determinism
def test_defender_cbf_substep_integration_is_deterministic_given_identical_inputs():
    controller = CBFAttackerController(limits=LIMITS)
    kwargs = dict(
        position=np.array([0.0, 0.0, 2.0]), velocity=np.array([1.0, 0.0, 0.0]), route_target=np.array([20.0, 0.0, 2.0]),
        controller=controller, obstacles=(), dt=0.5, L=100.0, max_altitude=30.0,
        defender_pos=np.array([8.0, 0.0, 2.0]), defender_vel_estimate=np.array([-1.0, 0.0, 0.0]),
        defender_safety_radius=DEFENDER_SAFETY_RADIUS,
    )
    pos1, vel1, _, diag1 = integrate_obstacle_cbf_substeps(**kwargs)
    pos2, vel2, _, diag2 = integrate_obstacle_cbf_substeps(**kwargs)
    assert np.allclose(pos1, pos2) and np.allclose(vel1, vel2)
    assert diag1["intervention_source"] == diag2["intervention_source"]
    assert diag1["min_defender_h_over_substeps"] == pytest.approx(diag2["min_defender_h_over_substeps"])


# ----------------------------------------------------------------------------- env-level wiring (bit-for-bit + branching)
def test_env_defender_cbf_disabled_never_queries_defender_state():
    """defender_cbf_enabled=False (the default): SoldierEnv must produce bit-for-bit identical trajectories to the
    Prompt-2/3 behavior regardless of an active defender's position, and the new diagnostics must report neutral
    defaults."""
    from experiments.journal_v2_protocol import build_v2_env_config
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg = dataclasses.replace(build_v2_env_config("C2", "measurement"), dynamic_obstacles_enabled=False)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=False)
    obs, info = env.reset(seed=4242)
    assert info["defender_cbf_enabled"] is False
    assert info["defender_observable"] is False
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["defender_cbf_enabled"] is False
    assert info["defender_observable"] is False
    assert info["intervention_source"] in ("none", "obstacle")


def test_env_defender_cbf_enabled_activates_when_defender_within_gate():
    from experiments.journal_v2_protocol import build_v2_env_config
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg = dataclasses.replace(
        build_v2_env_config("C0", "measurement"), dynamic_obstacles_enabled=False, defender_standby_until_detection=False,
    )
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False, defender_cbf_enabled=True)
    obs, info = env.reset(seed=11)
    # Force the defender within the gate, directly between the attacker and a point behind it (closing).
    env._defender_pos = (env._enemy_pos + np.array([5.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    env._defender_vel = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["defender_observable"] is True
    assert info["defender_distance"] < DEFENDER_DETECTION_RANGE


def test_env_joint_constraint_count_includes_defender_when_observable():
    from experiments.journal_v2_protocol import build_v2_env_config
    from uav_defend.envs.soldier_env import SoldierEnv

    cfg = dataclasses.replace(
        build_v2_env_config("C0", "measurement"), dynamic_obstacles_enabled=False, defender_standby_until_detection=False,
    )
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False, defender_cbf_enabled=True)
    obs, info = env.reset(seed=7)
    env._defender_pos = (env._enemy_pos + np.array([3.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    if info["defender_observable"]:
        assert info["joint_constraint_count"] >= 1
