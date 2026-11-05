"""Tests for the A1-CBF attacker foundation (uav_defend/attackers/cbf_attacker.py + experiments/journal_cbf_attacker_foundation.py).

No RL/training here; everything is a deterministic controller/dynamics unit test or a small (seed-count-reduced for
speed) acceptance-style check. The full 1,000-episode foundation/no-op validations are run by
experiments/journal_cbf_attacker_run.py and recorded under docs/journal_cbf_attacker/foundation/.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.attackers.cbf_attacker import (
    CBF_DT_SUB,
    CBF_KP,
    CBFAttackerController,
    CBFAttackerLimits,
    integrate_quadrotor_substeps,
    nominal_velocity_command,
    substep_count,
)
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_cbf_attacker_foundation as F
from experiments import journal_v2_validation_core as VC

LIMITS = CBFAttackerLimits()


# ----------------------------------------------------------------------------- 1-5: nominal guidance law
def test_relative_target_nominal_command_direction_and_magnitude():
    r = np.array([10.0, 0.0, 0.0])
    v = nominal_velocity_command(r)
    assert np.allclose(v, [10.0, 0.0, 0.0], atol=1e-9)          # 10 m away: below the 12 m/s cap, v = kp*r exactly


def test_zero_distance_command_is_hover_not_random():
    v = nominal_velocity_command(np.zeros(3))
    assert np.allclose(v, [0.0, 0.0, 0.0])


def test_speed_saturation_far_away():
    r = np.array([0.0, 40.0, 0.0])
    v = nominal_velocity_command(r)
    assert np.isclose(float(np.linalg.norm(v)), 12.0) and np.allclose(v / np.linalg.norm(v), [0.0, 1.0, 0.0])


@pytest.mark.parametrize("distance,expected_speed", [(12.0, 12.0), (6.0, 6.0), (3.0, 3.0), (2.0, 2.0), (1.0, 1.0)])
def test_low_speed_command_scales_proportionally_near_asset(distance, expected_speed):
    r = np.array([distance, 0.0, 0.0])
    v = nominal_velocity_command(r)
    assert np.isclose(float(np.linalg.norm(v)), expected_speed, atol=1e-6)


def test_hover_command_via_controller():
    controller = CBFAttackerController()
    v_nom = controller.nominal_velocity(np.zeros(3))
    v_safe, diag = controller.filter_velocity(v_nom)
    assert np.allclose(v_safe, 0.0) and diag["cbf_constraint_count"] == 0


# ----------------------------------------------------------------------------- 6-10: executor dynamics limits
def test_acceleration_limiting_single_substep():
    pos = np.zeros(3)
    vel = np.zeros(3)
    v_cmd = np.array([12.0, 0.0, 0.0])
    new_pos, new_vel, diag = integrate_quadrotor_substeps(pos, vel, v_cmd, limits=LIMITS, dt=CBF_DT_SUB, L=50.0, max_altitude=30.0)
    # one substep from rest: delta_v capped at max_accel * dt_sub = 6*0.1 = 0.6 m/s
    assert np.isclose(float(np.linalg.norm(new_vel)), 0.6, atol=1e-6) and diag["accel_saturated"]


def test_horizontal_speed_limiting():
    pos = np.zeros(3)
    vel = np.array([11.9, 0.0, 0.0])
    v_cmd = np.array([30.0, 0.0, 0.0])         # commands far beyond the 12 m/s cap
    _pos, new_vel, diag = integrate_quadrotor_substeps(pos, vel, v_cmd, limits=LIMITS, dt=CBF_DT_SUB, L=50.0, max_altitude=30.0)
    assert float(np.linalg.norm(new_vel[:2])) <= 12.0 + 1e-6 and diag["horizontal_speed_saturated"]


def test_climb_limiting():
    pos = np.zeros(3)
    vel = np.array([0.0, 0.0, 4.9])
    v_cmd = np.array([0.0, 0.0, 20.0])          # commands far beyond the 5 m/s climb cap
    _pos, new_vel, diag = integrate_quadrotor_substeps(pos, vel, v_cmd, limits=LIMITS, dt=CBF_DT_SUB, L=50.0, max_altitude=30.0)
    assert new_vel[2] <= 5.0 + 1e-6 and diag["vertical_rate_saturated"] and diag["climb_saturated"]


def test_descent_limiting():
    pos = np.array([0.0, 0.0, 10.0])
    vel = np.array([0.0, 0.0, -4.9])
    v_cmd = np.array([0.0, 0.0, -20.0])         # commands far beyond the 5 m/s descent cap
    _pos, new_vel, diag = integrate_quadrotor_substeps(pos, vel, v_cmd, limits=LIMITS, dt=CBF_DT_SUB, L=50.0, max_altitude=30.0)
    assert new_vel[2] >= -5.0 - 1e-6 and diag["vertical_rate_saturated"]


def test_velocity_reversal_no_heading_rate_limit():
    """A full velocity reversal in ONE substep is impossible (accel-limited), but reversal direction is reached
    directly (monotonically) with no legacy turn-rate-style heading constraint -- i.e. the velocity passes through
    a near-zero-speed hover state rather than being forced to curve around at constant speed."""
    pos = np.zeros(3)
    vel = np.array([12.0, 0.0, 0.0])
    v_cmd = np.array([-12.0, 0.0, 0.0])
    speeds = []
    for _ in range(40):
        pos, vel, _diag = integrate_quadrotor_substeps(pos, vel, v_cmd, limits=LIMITS, dt=CBF_DT_SUB, L=50.0, max_altitude=30.0)
        speeds.append(float(np.linalg.norm(vel)))
    # velocity must pass through (near) zero on the way to full-speed reversal -- never maintains forward speed
    assert min(speeds) < 1.0 and speeds[-1] > 11.0 and vel[0] < 0


# ----------------------------------------------------------------------------- 11-12: substepping / determinism
def test_substep_count_and_0_1s_integration():
    assert substep_count(0.5, 0.1) == 5
    with pytest.raises(ValueError):
        substep_count(0.5, 0.3)   # not an integer multiple


def test_deterministic_execution():
    cfg = F.foundation_config()
    env_a = SoldierEnv(cfg, enemy_control_mode="cbf_velocity")
    env_b = SoldierEnv(cfg, enemy_control_mode="cbf_velocity")
    ea = F.run_episode(env_a, 500001)
    eb = F.run_episode(env_b, 500001)
    assert ea["outcome"] == eb["outcome"] and ea["steps"] == eb["steps"] and ea["path_length_m"] == pytest.approx(eb["path_length_m"], abs=1e-9)


# ----------------------------------------------------------------------------- 13-14: safety filter / diagnostics
def test_identity_safety_filter_behavior():
    controller = CBFAttackerController()
    v_nom = controller.nominal_velocity(np.array([5.0, 0.0, 0.0]))
    v_safe, diag = controller.filter_velocity(v_nom, hazards=())
    assert np.array_equal(v_safe, v_nom) and diag["cbf_constraint_count"] == 0 and diag["cbf_qp_status"] == "identity_no_constraints" and diag["cbf_intervention_norm"] == 0.0


def test_diagnostic_fields_present_and_correct():
    env = SoldierEnv(F.foundation_config(), enemy_control_mode="cbf_velocity")
    obs, info = env.reset(seed=500002)
    obs, _r, _te, _tr, info = env.step(np.zeros(3, dtype=np.float32))
    for key in ("cbf_enabled", "cbf_constraint_count", "cbf_qp_status", "attacker_velocity_command", "attacker_actual_velocity",
                "attacker_speed", "asset_relative_position", "asset_distance", "velocity_tracking_error", "cbf_intervention_norm"):
        assert key in info, key
    assert info["cbf_enabled"] is True and info["cbf_constraint_count"] == 0 and info["cbf_qp_status"] == "identity_no_constraints"
    assert info["attacker_speed"] == pytest.approx(float(np.linalg.norm(info["attacker_actual_velocity"])), abs=1e-6)
    assert info["asset_distance"] == pytest.approx(float(np.linalg.norm(info["asset_relative_position"])), abs=1e-5)


def test_diagnostic_fields_neutral_default_in_legacy_mode():
    env = SoldierEnv(F.foundation_config(), enemy_control_mode="legacy")
    obs, info = env.reset(seed=500003)
    obs, _r, _te, _tr, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["cbf_enabled"] is False and info["cbf_qp_status"] == "inactive"


# ----------------------------------------------------------------------------- 15-18: environment / CRN / endurance / swept capture
def test_no_defender_test_configuration_defender_cannot_terminate():
    cfg = F.foundation_config()
    assert cfg.intercept_radius == 1e-6
    episodes = F.evaluate(range(500004, 500024), enemy_control_mode="cbf_velocity")
    assert not any(e["outcome"] in ("intercepted", "unsafe_intercept") for e in episodes)


def test_common_random_number_asset_trajectory_across_control_modes():
    seeds = range(500005, 500010)
    env_legacy = SoldierEnv(F.foundation_config(), enemy_control_mode="legacy")
    env_cbf = SoldierEnv(F.foundation_config(), enemy_control_mode="cbf_velocity")
    for s in seeds:
        _obs_a, info_a = env_legacy.reset(seed=s)
        _obs_b, info_b = env_cbf.reset(seed=s)
        assert np.allclose(info_a["soldier_pos"], info_b["soldier_pos"])
        done_a = done_b = False
        while not (done_a and done_b):
            if not done_a:
                _o, _r, te, tr, info_a = env_legacy.step(np.zeros(3, dtype=np.float32))
                done_a = te or tr
            if not done_b:
                _o, _r, te, tr, info_b = env_cbf.step(np.zeros(3, dtype=np.float32))
                done_b = te or tr
            if not done_a and not done_b:
                assert np.allclose(info_a["soldier_pos"], info_b["soldier_pos"], atol=1e-5)


def test_canonical_endurance_budget_matches_initial_range():
    cfg = F.foundation_config()
    assert cfg.enemy_endurance_enabled and cfg.enemy_endurance_path_factor == 5.0
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity")
    obs, info = env.reset(seed=500006)
    expected_budget = cfg.enemy_endurance_path_factor * float(np.linalg.norm(info["enemy_pos"] - info["soldier_pos"]))
    assert info["enemy_endurance_budget_m"] == pytest.approx(expected_budget, rel=1e-6)


def test_swept_capture_detected_under_substepped_motion():
    """A scenario engineered to capture (seeds from the 200-episode smoke sample are known to capture); verify the
    terminal event is the (unchanged) swept capture contact, not a missed/endpoint-only detection."""
    episodes = F.evaluate(range(500007, 500027), enemy_control_mode="cbf_velocity")
    captures = [e for e in episodes if e["outcome"] == "soldier_caught"]
    assert len(captures) >= 15                      # overwhelmingly captures at this range distribution
    assert all(e["terminal_distance_m"] <= 2.0 + 1e-6 for e in captures)


# ----------------------------------------------------------------------------- 19: bit-for-bit legacy no-op
def test_legacy_a0_no_op_vs_parent_commit():
    result = F.legacy_noop_check(range(510_000, 510_030))
    assert result["identical"], result["mismatches"][:1]


# ----------------------------------------------------------------------------- protocol / gate / namespaces
def test_gate_boundaries():
    m = {"capture": 0.98, "exhaustion": 0.02, "timeout": 0.0, "first_pass_capture": 0.97, "repeated_flyby_fraction": 0.02, "orbit_episodes": 0, "nonfinite": 0}
    assert F.gate_status(m)["passed"]
    for key, bad in (("capture", 0.97), ("exhaustion", 0.03), ("timeout", 0.001), ("first_pass_capture", 0.96), ("repeated_flyby_fraction", 0.03), ("orbit_episodes", 1), ("nonfinite", 1)):
        assert not F.gate_status({**m, key: bad})["passed"], key


def test_namespaces_disjoint_and_protocol_stable():
    o = F.namespace_overlaps()
    assert not o["foundation_vs_legacy_noop_overlap"] and o["gap_to_curriculum_m"] > 100_000
    assert F.protocol_sha256() == F.protocol_sha256()
    spec = F.protocol_specification()
    assert spec["executor"]["substeps_per_step"] == 5 and spec["nominal_guidance"]["kp_per_s"] == 1.0


def test_constraint_verification_passes_on_a_sample():
    episodes = F.evaluate(range(500030, 500060), enemy_control_mode="cbf_velocity")
    result = F.constraint_verification(episodes, limits=LIMITS)
    assert result["passed"]
