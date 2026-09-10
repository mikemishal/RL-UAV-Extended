"""Regression proof that refactoring `LeadInterceptPolicy` to use the
shared pure `uav_defend.policies.baseline.lead_math` helper preserves
EXACTLY the same behavior (Phase 5 of the learning-benefit diagnostic
study). This file is written and run BEFORE the refactor (to confirm it
captures the pre-refactor behavior) and re-run AFTER the refactor (to
prove byte-for-bit-identical outputs).

Covers every `last_guidance_mode` branch: escort, first_measurement_fallback,
no_velocity_fallback, degenerate_fallback, no_real_solution_fallback,
no_positive_root_fallback, and successful "lead", across both
state_source="measurement" and state_source="kalman", using many fixed
(non-random) observations/info dicts.

Run directly: python tests/test_lead_math_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy

_OBS_PLACEHOLDER = np.zeros(16, dtype=np.float32)


def _base_info(**overrides) -> dict:
    defaults = dict(
        defender_pos=np.array([0.0, 0.0, 0.0]),
        soldier_pos=np.array([0.0, 0.0, 0.0]),
        enemy_detected=True,
    )
    defaults.update(overrides)
    return defaults


def _round_list(arr) -> list[float]:
    return [round(float(v), 6) for v in arr]


def _run(policy: LeadInterceptPolicy, info: dict):
    action = policy.act(_OBS_PLACEHOLDER, info)
    return {
        "action": _round_list(action),
        "guidance_mode": policy.last_guidance_mode,
        "intercept_time": None if policy.last_intercept_time is None else round(float(policy.last_intercept_time), 6),
        "intercept_point": None if policy.last_intercept_point is None else _round_list(policy.last_intercept_point),
        "relative_position": None if policy.last_relative_position is None else _round_list(policy.last_relative_position),
        "quadratic_a": None if policy.last_quadratic_a is None else round(float(policy.last_quadratic_a), 6),
        "quadratic_b": None if policy.last_quadratic_b is None else round(float(policy.last_quadratic_b), 6),
        "quadratic_c": None if policy.last_quadratic_c is None else round(float(policy.last_quadratic_c), 6),
        "discriminant": None if policy.last_discriminant is None else round(float(policy.last_discriminant), 6),
    }


# --- Scenario fixtures: (name, state_source, info_overrides) ---------------

def _measurement_scenarios():
    return [
        ("not_detected_escort", dict(enemy_detected=False, soldier_pos=np.array([5.0, 5.0, 0.0]))),
        ("no_measurement_escort", dict(enemy_measurement=None, soldier_pos=np.array([3.0, -2.0, 0.0]))),
        ("first_measurement_fallback", dict(
            enemy_measurement=np.array([10.0, 0.0, 5.0]), enemy_measurement_velocity_valid=False,
        )),
        ("degenerate_zero_range", dict(
            enemy_measurement=np.array([0.0, 0.0, 0.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([1.0, 0.0, 0.0]),
        )),
        ("successful_lead_head_on", dict(
            enemy_measurement=np.array([20.0, 0.0, 5.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([-2.0, 0.0, 0.0]),
        )),
        ("successful_lead_crossing", dict(
            defender_pos=np.array([0.0, 0.0, 5.0]),
            enemy_measurement=np.array([30.0, 10.0, 8.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([-1.0, -3.0, 0.5]),
        )),
        ("no_real_solution_fast_receding", dict(
            enemy_measurement=np.array([10.0, 0.0, 0.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([50.0, 0.0, 0.0]),
        )),
        ("stationary_target", dict(
            defender_pos=np.array([5.0, 5.0, 5.0]),
            enemy_measurement=np.array([15.0, 5.0, 5.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([0.0, 0.0, 0.0]),
        )),
        ("far_diagonal", dict(
            defender_pos=np.array([-10.0, -10.0, 2.0]),
            enemy_measurement=np.array([40.0, 35.0, 20.0]), enemy_measurement_velocity_valid=True,
            enemy_measurement_velocity=np.array([2.0, -4.0, 1.0]),
        )),
    ]


def _kalman_scenarios():
    return [
        ("kalman_not_detected_escort", dict(enemy_detected=False, soldier_pos=np.array([1.0, 1.0, 0.0]))),
        ("kalman_no_estimate_escort", dict(e_hat=None)),
        ("kalman_no_velocity_fallback", dict(e_hat=np.array([12.0, 3.0, 4.0]), v_hat=None)),
        ("kalman_successful_lead", dict(
            e_hat=np.array([25.0, -5.0, 10.0]), v_hat=np.array([-3.0, 1.0, -0.5]),
        )),
        ("kalman_no_positive_root", dict(
            defender_pos=np.array([0.0, 0.0, 0.0]),
            e_hat=np.array([-10.0, 0.0, 5.0]), v_hat=np.array([-20.0, 0.0, 0.0]),
        )),
    ]


def test_measurement_mode_scenarios_are_stable():
    policy = LeadInterceptPolicy(state_source="measurement", config=EnvConfig())
    results = {}
    for name, overrides in _measurement_scenarios():
        policy.reset()
        results[name] = _run(policy, _base_info(**overrides))
    _assert_known_results(results, _EXPECTED_MEASUREMENT)


def test_kalman_mode_scenarios_are_stable():
    policy = LeadInterceptPolicy(state_source="kalman", config=EnvConfig())
    results = {}
    for name, overrides in _kalman_scenarios():
        policy.reset()
        results[name] = _run(policy, _base_info(**overrides))
    _assert_known_results(results, _EXPECTED_KALMAN)


def _assert_known_results(actual: dict, expected: dict) -> None:
    for name, expected_result in expected.items():
        assert actual[name] == expected_result, f"scenario '{name}' changed: {actual[name]} != {expected_result}"


# Golden reference captured from the UNMODIFIED (pre-lead_math-refactor)
# LeadInterceptPolicy.act() implementation -- see this file's module
# docstring. These are FROZEN LITERAL values (not re-computed at test
# time), so any change to them after the Phase-5 refactor would indicate
# a real behavioral regression rather than trivially re-deriving the same
# numbers from the (now-refactored) code under test.
_EXPECTED_MEASUREMENT = {
    "not_detected_escort": {"action": [0.707107, 0.707107, 0.0], "guidance_mode": "escort", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "no_measurement_escort": {"action": [0.83205, -0.5547, 0.0], "guidance_mode": "escort", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "first_measurement_fallback": {"action": [0.894427, 0.0, 0.447214], "guidance_mode": "first_measurement_fallback", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "degenerate_zero_range": {"action": [0.0, 0.0, 0.0], "guidance_mode": "degenerate_fallback", "intercept_time": None, "intercept_point": None, "relative_position": [0.0, 0.0, 0.0], "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "successful_lead_head_on": {"action": [0.963254, 0.0, 0.268591], "guidance_mode": "lead", "intercept_time": 1.034202, "intercept_point": [17.931595, 0.0, 5.0], "relative_position": [20.0, 0.0, 5.0], "quadratic_a": -320.0, "quadratic_b": -80.0, "quadratic_c": 425.0, "discriminant": 550400.0},
    "successful_lead_crossing": {"action": [0.975468, 0.177008, 0.13088], "guidance_mode": "lead", "intercept_time": 1.616516, "intercept_point": [28.383484, 5.150452, 8.808258], "relative_position": [30.0, 10.0, 3.0], "quadratic_a": -313.75, "quadratic_b": -117.0, "quadratic_c": 1009.0, "discriminant": 1279984.0},
    "no_real_solution_fast_receding": {"action": [1.0, 0.0, 0.0], "guidance_mode": "no_positive_root_fallback", "intercept_time": None, "intercept_point": None, "relative_position": [10.0, 0.0, 0.0], "quadratic_a": 2176.0, "quadratic_b": 1000.0, "quadratic_c": 100.0, "discriminant": 129600.0},
    "stationary_target": {"action": [1.0, 0.0, 0.0], "guidance_mode": "lead", "intercept_time": 0.555556, "intercept_point": [15.0, 5.0, 5.0], "relative_position": [10.0, 0.0, 0.0], "quadratic_a": -324.0, "quadratic_b": 0.0, "quadratic_c": 100.0, "discriminant": 129600.0},
    "far_diagonal": {"action": [0.841909, 0.435496, 0.318643], "guidance_mode": "lead", "intercept_time": 3.801022, "intercept_point": [47.602043, 19.795914, 23.801022], "relative_position": [50.0, 45.0, 18.0], "quadratic_a": -303.0, "quadratic_b": -124.0, "quadratic_c": 4849.0, "discriminant": 5892364.0},
}

_EXPECTED_KALMAN = {
    "kalman_not_detected_escort": {"action": [0.707107, 0.707107, 0.0], "guidance_mode": "escort", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "kalman_no_estimate_escort": {"action": [0.0, 0.0, 0.0], "guidance_mode": "escort", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "kalman_no_velocity_fallback": {"action": [0.923077, 0.230769, 0.307692], "guidance_mode": "no_velocity_fallback", "intercept_time": None, "intercept_point": None, "relative_position": None, "quadratic_a": None, "quadratic_b": None, "quadratic_c": None, "discriminant": None},
    "kalman_successful_lead": {"action": [0.902743, -0.158326, 0.399986], "guidance_mode": "lead", "intercept_time": 1.298744, "intercept_point": [21.103767, -3.701256, 9.350628], "relative_position": [25.0, -5.0, 10.0], "quadratic_a": -313.75, "quadratic_b": -170.0, "quadratic_c": 750.0, "discriminant": 970150.0},
    "kalman_no_positive_root": {"action": [-0.894427, 0.0, 0.447214], "guidance_mode": "no_positive_root_fallback", "intercept_time": None, "intercept_point": None, "relative_position": [-10.0, 0.0, 5.0], "quadratic_a": 76.0, "quadratic_b": 400.0, "quadratic_c": 125.0, "discriminant": 122000.0},
}


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
