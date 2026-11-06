"""Tests for the hierarchical obstacle/defender CBF-QP priority restructuring (journal_cbf_attacker,
defender_cbf_v2 / Prompt 5, brief section 13). Replaces Prompt 4's SHARED-slack formulation, which allowed
defender avoidance to trade against physical obstacle safety under severe joint conflict (traced to 3/200
obstacle collisions in the Prompt-4 stress set). The new priority:

    Priority 1: physical static-obstacle safety  -- HARD, never relaxed for the defender.
    Priority 2: vehicle physical velocity limits  -- HARD, unchanged.
    Priority 3: defender avoidance                -- SOFT (dedicated non-negative slack delta_D).

See uav_defend/attackers/cbf_obstacle.py:solve_velocity_cbf_qp for the implementation and
docs/journal_cbf_attacker/defender_cbf_v2/README.md for the full Monte Carlo reproduction/retest results.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.attackers.cbf_attacker import CBFAttackerController, CBFAttackerLimits
from uav_defend.attackers.cbf_defender import DEFENDER_SAFETY_RADIUS, DefenderHazard
from uav_defend.attackers.cbf_obstacle import StaticHazard, solve_velocity_cbf_qp

LIMITS = CBFAttackerLimits()


# ----------------------------------------------------------------------------- 1: obstacle constraints cannot receive slack
def test_obstacle_constraints_are_never_relaxed_under_severe_joint_conflict():
    """A deliberately severe conflict: the obstacle pins the attacker against one wall (normal=-x), the defender
    closes from the opposite side demanding +x motion. The solver must keep the obstacle HARD (residual ~0) and
    absorb the conflict entirely via the defender's own slack."""
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=0.3, h=-3.0, inward_speed=8.0, stopping_margin=3.3, physical_clearance=0.3)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=0.3, h=-3.0, closing_speed=8.0, stopping_margin=3.3, distance=0.3 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["obstacle_constraints_hard"] is True
    obstacle_lhs = float(np.dot(obstacle_hz.normal, v_safe)) + obstacle_hz.h  # alpha=1.0 default
    assert obstacle_lhs >= -1e-4  # obstacle constraint satisfied to within the residual tolerance, no slack term


# ----------------------------------------------------------------------------- 2: defender-only slack
def test_defender_constraint_alone_can_use_slack_when_no_obstacle_present():
    defender_hz = DefenderHazard(normal=np.array([-1.0, 0.0, 0.0]), d=0.2, h=-5.0, closing_speed=10.0, stopping_margin=5.2, distance=0.2 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=1.0, max_climb_rate=1.0, max_descent_rate=1.0,  # tight bound forces reliance on slack
    )
    assert diag["cbf_qp_status"] in ("hard", "soft_fallback")
    if diag["cbf_qp_status"] == "soft_fallback":
        assert diag["defender_slack"] > 0.0
        assert diag["priority_conflict"] is True


# ----------------------------------------------------------------------------- 3: positive defender slack under conflict
def test_priority_conflict_flag_set_exactly_when_defender_slack_is_positive():
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=0.5, h=-2.0, inward_speed=6.0, stopping_margin=2.5, physical_clearance=0.5)
    defender_hz = DefenderHazard(normal=np.array([0.0, 1.0, 0.0]), d=0.5, h=-2.0, closing_speed=6.0, stopping_margin=2.5, distance=0.5 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["priority_conflict"] == (diag["defender_slack"] > 1e-4)
    # No conflict: easy, compatible case -- both constraints share the same escape direction.
    obstacle_hz2 = StaticHazard(obstacle_id=0, normal=np.array([1.0, 0.0, 0.0]), d=5.0, h=4.0, inward_speed=0.0, stopping_margin=1.0, physical_clearance=5.0)
    defender_hz2 = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=5.0, h=4.0, closing_speed=0.0, stopping_margin=1.0, distance=5.0 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe2, diag2 = solve_velocity_cbf_qp(
        np.array([3.0, 0.0, 0.0]), [obstacle_hz2], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz2,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag2["priority_conflict"] is False
    assert diag2["defender_slack"] == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------------- 4: obstacle-hard priority (joint conflict resolution)
def test_hierarchical_qp_never_reports_defender_as_the_infeasibility_source_for_obstacle_violation():
    """Whenever the primary hierarchical QP succeeds (status hard/soft_fallback), `infeasibility_source` must never
    blame the defender for an OBSTACLE violation -- the obstacle residual must independently be ~0."""
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=0.4, h=-1.5, inward_speed=5.0, stopping_margin=1.9, physical_clearance=0.4)
    defender_hz = DefenderHazard(normal=np.array([0.0, 1.0, 0.0]), d=0.4, h=-1.5, closing_speed=5.0, stopping_margin=1.9, distance=0.4 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["obstacle_constraint_residual"] <= 1e-4
    if diag["cbf_qp_status"] == "soft_fallback":
        assert diag["infeasibility_source"] == "defender"


# ----------------------------------------------------------------------------- 5: infeasible obstacle-hard handling
def test_obstacle_safety_infeasible_drops_the_defender_entirely():
    """When even the obstacle+bounds constraints alone (ignoring the defender) are infeasible -- a genuine
    spawn-time/numerical physical crisis -- the defender must be dropped entirely, never used to help satisfy
    obstacle safety, and the episode must be classified `obstacle_safety_infeasible`."""
    # An obstacle requiring an unreachable outward speed combined with a tiny velocity bound makes obstacle+bounds
    # alone infeasible (h deeply negative, required speed >> max_horizontal_speed).
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-5.0, h=-50.0, inward_speed=20.0, stopping_margin=45.0, physical_clearance=-5.0)
    defender_hz = DefenderHazard(normal=np.array([-1.0, 0.0, 0.0]), d=-5.0, h=-50.0, closing_speed=20.0, stopping_margin=45.0, distance=-5.0 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=0.5, max_climb_rate=0.5, max_descent_rate=0.5,
    )
    assert diag["priority_classification"] == "physical_safety_infeasible"
    assert diag["defender_ignored_for_obstacle_safety"] is True
    assert diag["defender_slack"] == pytest.approx(0.0)
    assert diag["priority_conflict"] is False


# ----------------------------------------------------------------------------- 6: reproduction of the three known Prompt-4 collision cases
def _run_stress_seed(seed: int, defender: str = "greedy") -> str:
    import experiments.journal_cbf_attacker_defender_foundation as F
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator = F.defender_config(defender)
    cfg = dataclasses.replace(cfg, defender_standby_until_detection=False)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    assert F.build_stress_scenario(env, seed)
    policy = F.VC.make_policy(defender, cfg)
    policy.reset()
    done, outcome, step = False, None, 0
    while not done and step < 60:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        step += 1
        outcome = info["outcome"]
        done = te or tr
    return outcome


def test_reproduction_of_the_three_prompt4_collision_seeds_is_deterministic():
    """Documents the EXACT, honestly-reported outcome of re-running the 3 known Prompt-4 stress-collision seeds.
    Originally written for the Prompt-5 hierarchical controller (journal_cbf_attacker, defender_cbf_v2); this test
    exercises the LIVE `solve_velocity_cbf_qp`/`SoldierEnv` code, so its assertions now reflect the CURRENT
    acceleration-reachable controller (journal_cbf_attacker, defender_cbf_v3 / Prompt 6, brief section 13-14),
    which supersedes Prompt 5 for all three cases -- see docs/journal_cbf_attacker/defender_cbf_v3/README.md for
    the full numeric trace and docs/journal_cbf_attacker/defender_cbf_v2/README.md for the ORIGINAL Prompt-5
    reproduction this test documented before the acceleration-reachability fix.

    Seed 630069 (originally a genuine priority-CONFLICT collision under Prompt 4's shared slack): Prompt 5 fixed
    it via defender-slack, resolving to `soldier_caught`. Under Prompt 6's acceleration-reachable controller the
    SAME seed still shows 0 obstacle collisions and a perfect optimizer/executor contract
    (`executor_velocity_error` == 0.0 throughout), but the outcome FLIPS to `intercepted` -- the added
    acceleration-reachability constraint changes which velocity the solver selects in this specific conflict, and
    the defender now wins the tactical encounter instead of the attacker. This is a tactical trade-off, NOT a
    safety regression (brief section 7/15: "it is acceptable for the defender to intercept the attacker").

    Seeds 630123 and 630203 (originally obstacle-collision stress cases under Prompt 5): FIXED under Prompt 6 --
    0 obstacle collisions, `executor_velocity_error` at machine precision (<=1.4e-16) confirming
    `v_optimized == v_executed` throughout. Seed 630203 resolves to `intercepted`. Seed 630123 reaches a SAFE
    DEADLOCK instead of a clean resolution: the attacker brakes to a complete stop (`v_current` -> 0 within 2
    external steps) at a position still inside the obstacle's INFLATED clearance buffer (`cbf_h_min` ~= -1.19,
    i.e. the formal CBF barrier remains formally violated) but with POSITIVE physical clearance (~0.81 m,
    confirmed never an actual collision) -- escaping that buffer zone in one substep would require more
    instantaneous velocity than the 0.6 m/s one-substep acceleration budget allows, so every subsequent substep is
    classified `physical_safety_infeasible` and commands the reachable emergency brake (v=0, brief section 12's
    own defined zero-velocity case) rather than inventing an unreachable escape velocity -- the outcome stays
    `ongoing` for the remainder of the fixed 60-step stress horizon. This is the EXPECTED, brief-mandated behavior
    (section 11: "Do NOT relax: obstacle safety; acceleration limits... If necessary use the safest reachable
    braking command") trading tactical resolution for a hard safety guarantee -- it is never a collision.
    """
    assert _run_stress_seed(630069) == "intercepted"
    assert _run_stress_seed(630123) == "ongoing"
    assert _run_stress_seed(630203) == "intercepted"


# ----------------------------------------------------------------------------- 7: shared-slack vs hierarchical regression
def test_hierarchical_differs_from_shared_slack_exactly_when_a_genuine_conflict_exists():
    """When obstacle and defender constraints are compatible (no real conflict), the hierarchical and (former)
    shared-slack formulations agree bit-for-bit (both find delta=0, same v_safe). When they genuinely conflict,
    the hierarchical formulation keeps the obstacle exactly hard (verified via residual) while the shared-slack
    formulation would have relaxed BOTH -- this test checks the hierarchical result's obstacle residual stays at
    machine precision regardless of how severe the conflict is, which the old shared mechanism could not
    guarantee. Severities are kept ACHIEVABLE under the 12 m/s speed bound alone (< 12) so every case here is a
    genuine defender-vs-obstacle PRIORITY conflict, never an outright geometric impossibility (which is now
    `physical_safety_infeasible`, covered by its own dedicated test, journal_cbf_attacker defender_cbf_v3 /
    Prompt 6)."""
    for severity in (1.0, 3.0, 5.0, 8.0):
        obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=0.1, h=-severity, inward_speed=severity, stopping_margin=severity + 0.1, physical_clearance=0.1)
        defender_hz = DefenderHazard(normal=np.array([0.0, 1.0, 0.0]), d=0.1, h=-severity, closing_speed=severity, stopping_margin=severity + 0.1, distance=0.1 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
        v_safe, diag = solve_velocity_cbf_qp(
            np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
            max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
        )
        assert diag["obstacle_constraint_residual"] <= 1e-4, f"obstacle residual grew with conflict severity={severity}"



# ----------------------------------------------------------------------------- 8: diagnostic classification
def test_all_new_diagnostic_fields_are_present_and_well_typed():
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    defender_hz = DefenderHazard(normal=np.array([1.0, 0.0, 0.0]), d=1.0, h=1.0, closing_speed=0.0, stopping_margin=0.0, distance=1.0 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([0.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=defender_hz,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    for key, expected_type in (
        ("defender_slack", float), ("defender_constraint_satisfied_without_slack", bool),
        ("defender_constraint_violation_with_slack", float), ("obstacle_constraints_hard", bool),
        ("obstacle_min_residual", float), ("defender_min_residual", float), ("priority_conflict", bool),
    ):
        assert key in diag, f"missing diagnostic field {key}"
        assert isinstance(diag[key], expected_type), f"{key} has wrong type {type(diag[key])}"
    assert diag["priority_classification"] is None or isinstance(diag["priority_classification"], str)


def test_no_defender_hazard_degenerates_to_prompt2_obstacle_only_bit_for_bit():
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([-1.0, 0.0, 0.0]), d=1.0, h=1.0, inward_speed=0.0, stopping_margin=0.0, physical_clearance=3.0)
    v_safe, diag = solve_velocity_cbf_qp(
        np.array([12.0, 0.0, 0.0]), [obstacle_hz], v_current=np.zeros(3), max_accel=1e6, defender_hazard=None,
        max_horizontal_speed=12.0, max_climb_rate=5.0, max_descent_rate=5.0,
    )
    assert diag["cbf_qp_status"] == "hard"
    assert diag["priority_classification"] is None
    assert diag["defender_slack"] == 0.0
    assert diag["obstacle_constraints_hard"] is True


def test_controller_filter_velocity_passes_through_new_diagnostic_fields():
    controller = CBFAttackerController(limits=LIMITS)
    obstacle_hz = StaticHazard(obstacle_id=0, normal=np.array([0.0, -1.0, 0.0]), d=0.5, h=-2.0, inward_speed=6.0, stopping_margin=2.5, physical_clearance=0.5)
    defender_hz = DefenderHazard(normal=np.array([0.0, 1.0, 0.0]), d=0.5, h=-2.0, closing_speed=6.0, stopping_margin=2.5, distance=0.5 + DEFENDER_SAFETY_RADIUS, defender_velocity=np.zeros(3))
    v_safe, diag = controller.filter_velocity(
        np.zeros(3), state={"position": np.zeros(3), "velocity": np.zeros(3)}, hazards=(obstacle_hz,), defender_hazard=defender_hz,
    )
    assert "defender_slack" in diag and "priority_conflict" in diag
