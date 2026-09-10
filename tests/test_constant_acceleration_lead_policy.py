"""Tests for the ANALYSIS Constant-Acceleration (CA) Lead controller and
its underlying quartic solver (Phase 6C / 19 of the learning-benefit
diagnostic study).

Covers:
 1. zero acceleration reduces to (bit-identical) CV behavior
 2. constant acceleration with a known, hand-verified quartic solution
 3. multiple positive roots -> earliest selected
 4. complex roots ignored -> falls back to Standard (CV) Lead
 5. negative roots ignored (mixed positive/negative root case)
 6. no valid root -> Standard Lead fallback
 7. near-zero coefficients / numerical degeneracy (defender at target)
 8. ConstantAccelerationLeadPolicy: finite-difference acceleration
    estimation over consecutive steps, escort/no-estimate fallbacks
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.analysis.constant_acceleration_lead_policy import ConstantAccelerationLeadPolicy
from uav_defend.policies.analysis.quartic_lead_math import solve_ca_lead, solve_quartic_intercept_time
from uav_defend.policies.baseline.lead_math import solve_cv_lead

_EPS = 1e-8


# --- 1. Zero acceleration reduces to CV behavior ---------------------------

def test_zero_acceleration_matches_cv_exactly():
    defender_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([20.0, 5.0, 3.0])
    target_vel = np.array([-2.0, 1.0, 0.5])
    v_d = 8.0

    ca_solution = solve_ca_lead(target_pos, target_vel, np.zeros(3), defender_pos, v_d, _EPS)
    cv_solution = solve_cv_lead(target_pos, target_vel, defender_pos, v_d, _EPS)

    assert np.array_equal(ca_solution.action, cv_solution.action)
    assert ca_solution.guidance_mode == cv_solution.guidance_mode
    assert ca_solution.used_acceleration is False
    assert ca_solution.fallback_used is True


# --- 2. Known, hand-verified quartic solution ------------------------------

def test_known_quartic_solution():
    # Hand-constructed 1-D scenario: r=5, v=2, a=2, v_d=8 solves
    # r + v*tau + 0.5*a*tau^2 = v_d*tau  =>  tau^2 - 6*tau + 5 = 0
    # => tau in {1, 5}; verified numerically: roots = [-9.472, 5, 1, -0.528].
    r = np.array([5.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    a = np.array([2.0, 0.0, 0.0])
    v_d = 8.0
    t_intercept, coeffs = solve_quartic_intercept_time(r, v, a, v_d, _EPS)
    assert t_intercept is not None
    assert abs(t_intercept - 1.0) < 1e-6


def test_known_quartic_solution_end_to_end():
    defender_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([5.0, 0.0, 0.0])
    target_vel = np.array([2.0, 0.0, 0.0])
    accel = np.array([2.0, 0.0, 0.0])
    solution = solve_ca_lead(target_pos, target_vel, accel, defender_pos, 8.0, _EPS)
    assert solution.used_acceleration is True
    assert solution.guidance_mode == "ca_lead"
    assert abs(solution.intercept_time - 1.0) < 1e-6


# --- 3. Multiple positive roots -> earliest selected -----------------------

def test_multiple_positive_roots_selects_earliest():
    # Same fixture as above has TWO positive roots (1.0 and 5.0); the
    # earliest (1.0) must be selected.
    r = np.array([5.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    a = np.array([2.0, 0.0, 0.0])
    t_intercept, _ = solve_quartic_intercept_time(r, v, a, 8.0, _EPS)
    assert abs(t_intercept - 1.0) < 1e-6  # not 5.0


# --- 4. Complex roots ignored -> Standard Lead fallback --------------------

def test_all_complex_roots_falls_back_to_cv():
    # r=(10,0,0), v=0, a=(2,0,0), v_d=5 => quartic roots are two complex
    # conjugate pairs (no real roots at all): verified numerically.
    defender_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    accel = np.array([2.0, 0.0, 0.0])
    t_intercept, _ = solve_quartic_intercept_time(target_pos, target_vel, accel, 5.0, _EPS)
    assert t_intercept is None

    solution = solve_ca_lead(target_pos, target_vel, accel, defender_pos, 5.0, _EPS)
    assert solution.used_acceleration is False
    assert solution.fallback_used is True
    expected_cv = solve_cv_lead(target_pos, target_vel, defender_pos, 5.0, _EPS)
    assert np.array_equal(solution.action, expected_cv.action)


# --- 5. Negative roots ignored (mixed positive/negative case) -------------

def test_negative_roots_ignored():
    # Fixture from test 2/3 also has two NEGATIVE roots (-9.472, -0.528);
    # verify the selected root is strictly positive.
    r = np.array([5.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    a = np.array([2.0, 0.0, 0.0])
    t_intercept, _ = solve_quartic_intercept_time(r, v, a, 8.0, _EPS)
    assert t_intercept > 0.0


# --- 6. No valid root -> Standard Lead fallback (duplicate-safety check) ---

def test_no_valid_root_falls_back_to_standard_lead():
    defender_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    accel = np.array([2.0, 0.0, 0.0])
    solution = solve_ca_lead(target_pos, target_vel, accel, defender_pos, 5.0, _EPS)
    assert solution.fallback_used is True
    assert solution.used_acceleration is False


# --- 7. Numerical degeneracy: defender coincides with target ---------------

def test_degenerate_zero_range_falls_back_to_pursuit():
    defender_pos = np.array([3.0, 3.0, 3.0])
    target_pos = np.array([3.0, 3.0, 3.0])
    target_vel = np.array([1.0, 0.0, 0.0])
    accel = np.array([1.0, 0.0, 0.0])
    solution = solve_ca_lead(target_pos, target_vel, accel, defender_pos, 8.0, _EPS)
    assert solution.guidance_mode == "degenerate_fallback"
    assert np.array_equal(solution.action, np.zeros(3, dtype=np.float32))


# --- 8. ConstantAccelerationLeadPolicy: acceleration-estimate bookkeeping --

def _info(**overrides):
    defaults = dict(
        defender_pos=np.array([0.0, 0.0, 0.0]),
        soldier_pos=np.array([0.0, 0.0, 0.0]),
        enemy_detected=True,
    )
    defaults.update(overrides)
    return defaults


def test_ca_policy_escort_when_not_detected():
    policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=EnvConfig())
    action = policy.act(np.zeros(16, dtype=np.float32), _info(enemy_detected=False, soldier_pos=np.array([1.0, 0.0, 0.0])))
    assert policy.last_guidance_mode == "escort"
    assert np.allclose(action, [1.0, 0.0, 0.0])


def test_ca_policy_first_step_has_no_acceleration_estimate():
    policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=EnvConfig())
    info = _info(
        enemy_measurement=np.array([20.0, 0.0, 5.0]),
        enemy_measurement_velocity_valid=True,
        enemy_measurement_velocity=np.array([-2.0, 0.0, 0.0]),
    )
    policy.act(np.zeros(16, dtype=np.float32), info)
    assert policy.last_guidance_mode == "no_acceleration_estimate_fallback"
    assert policy.last_acceleration_estimate is None


def test_ca_policy_estimates_acceleration_on_second_step():
    policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=EnvConfig(), dt=0.5)
    v1 = np.array([-2.0, 0.0, 0.0])
    v2 = np.array([-2.5, 0.0, 0.0])
    policy.act(np.zeros(16, dtype=np.float32), _info(
        enemy_measurement=np.array([20.0, 0.0, 5.0]), enemy_measurement_velocity_valid=True, enemy_measurement_velocity=v1,
    ))
    policy.act(np.zeros(16, dtype=np.float32), _info(
        enemy_measurement=np.array([19.0, 0.0, 5.0]), enemy_measurement_velocity_valid=True, enemy_measurement_velocity=v2,
    ))
    expected_accel = (v2 - v1) / 0.5
    assert np.allclose(policy.last_acceleration_estimate, expected_accel)
    assert policy.last_guidance_mode in ("ca_lead", "lead", "no_real_solution_fallback", "no_positive_root_fallback", "degenerate_fallback")


def test_ca_policy_reset_clears_velocity_history():
    policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=EnvConfig())
    policy.act(np.zeros(16, dtype=np.float32), _info(
        enemy_measurement=np.array([20.0, 0.0, 5.0]), enemy_measurement_velocity_valid=True,
        enemy_measurement_velocity=np.array([-2.0, 0.0, 0.0]),
    ))
    assert policy._prev_velocity_estimate is not None
    policy.reset()
    assert policy._prev_velocity_estimate is None


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
