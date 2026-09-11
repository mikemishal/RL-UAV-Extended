"""Tests for RA-LAPG (Phase 54-58): stationary/CV target, reachable vs
unreachable intercepts, turn/acceleration-limited cases, earliest-feasible
selection, no-feasible fallback, determinism, exact environment dynamics
reuse, and no environment mutation."""

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.analysis.ra_lapg_policy import CANDIDATE_TAUS, RALAPGPolicy


def _zeroed_predictor() -> HorizonConditionedPredictor:
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    model.eval()
    return model


def _make_info(defender_pos, defender_vel, target_pos, target_vel, soldier_pos=None):
    return {
        "defender_pos": np.asarray(defender_pos, dtype=np.float64),
        "defender_vel": np.asarray(defender_vel, dtype=np.float64),
        "soldier_pos": np.asarray(soldier_pos if soldier_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64),
        "enemy_detected": True,
        "enemy_measurement": np.asarray(target_pos, dtype=np.float64),
        "enemy_measurement_velocity": np.asarray(target_vel, dtype=np.float64),
        "enemy_measurement_velocity_valid": True,
    }


def test_stationary_target_selects_earliest_candidate_when_all_feasible():
    config = EnvConfig()
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    # place the (stationary) target within v_d*tau of the shortest candidate
    # tau so it is dynamically reachable at every candidate, not just the
    # longer ones
    shortest_tau = min(CANDIDATE_TAUS)
    reachable_pos = [config.v_d * shortest_tau * 0.5, 0.0, 5.0]
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], reachable_pos, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    sol = policy.last_solution
    assert sol.selected_tau == shortest_tau  # earliest feasible per Phase 57 rule A.1


def test_constant_velocity_target_zero_correction_matches_cv_prediction():
    config = EnvConfig()
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    target_pos = np.array([30.0, 0.0, 5.0])
    target_vel = np.array([-1.0, 0.0, 0.0])
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], target_pos, target_vel)
    policy.act(obs, info)
    for c in policy.last_solution.candidates:
        expected = target_pos + target_vel * c.tau  # Delta_p == 0 with zeroed decoder
        assert np.allclose(c.predicted_target_point, expected, atol=1e-5)


def test_easy_reachable_intercept_is_feasible_and_e_reach_near_zero():
    config = EnvConfig()
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    tau = 2.0
    target_pos = np.array([config.v_d * tau, 0.0, 5.0])  # exactly reachable in tau at full speed
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], target_pos, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    matching = [c for c in policy.last_solution.candidates if c.tau == tau]
    assert matching[0].feasible
    assert matching[0].E_reach < 1e-2


def test_dynamically_unreachable_kinematic_point_is_infeasible():
    """A target placed far beyond what v_d*tau could ever cover must be
    marked infeasible for that candidate."""
    config = EnvConfig()
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    tau = 1.0
    unreachable_point = np.array([config.v_d * tau * 5.0, 0.0, 5.0])  # 5x farther than max reachable distance
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], unreachable_point, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    matching = [c for c in policy.last_solution.candidates if c.tau == tau]
    assert not matching[0].feasible
    assert matching[0].E_reach > config.intercept_radius


def test_turn_limited_case_reduces_feasibility():
    config = EnvConfig(defender_max_turn_rate_deg=5.0)
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    tau = 1.0
    # target directly BEHIND the defender's current velocity -> large heading reversal required
    target_pos = np.array([-config.v_d * tau, 0.0, 5.0])
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], target_pos, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    matching = [c for c in policy.last_solution.candidates if c.tau == tau]
    assert matching[0].turn_saturated_fraction > 0.0
    assert not matching[0].feasible


def test_acceleration_limited_case_reduces_feasibility():
    config = EnvConfig(defender_max_accel=0.1)
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    tau = 1.0
    target_pos = np.array([config.v_d * tau, 0.0, 5.0])
    # defender starts at rest -> must accelerate hard to reach v_d quickly, which the tiny max_accel prevents
    info = _make_info([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], target_pos, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    matching = [c for c in policy.last_solution.candidates if c.tau == tau]
    assert matching[0].accel_saturated_fraction > 0.0
    assert not matching[0].feasible


def test_no_feasible_candidate_falls_back_to_minimum_e_reach():
    config = EnvConfig(defender_max_accel=0.05, defender_max_turn_rate_deg=2.0)
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    # a target that is essentially unreachable at every candidate tau under
    # such restrictive dynamics
    info = _make_info([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [500.0, 500.0, 5.0], [0.0, 0.0, 0.0])
    policy.act(obs, info)
    sol = policy.last_solution
    assert sol.fallback_used is True
    best_e_reach = min(c.E_reach for c in sol.candidates)
    assert sol.candidates[[c.tau for c in sol.candidates].index(sol.selected_tau)].E_reach == best_e_reach


def test_deterministic_output_same_input_same_action():
    config = EnvConfig()
    predictor = _zeroed_predictor()
    policy_a = RALAPGPolicy(predictor, config=config)
    policy_b = RALAPGPolicy(predictor, config=config)
    obs = np.zeros(16, dtype=np.float32)
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], [30.0, 0.0, 5.0], [-1.0, 0.0, 0.0])
    action_a = policy_a.act(obs, dict(info))
    action_b = policy_b.act(obs, dict(info))
    assert np.allclose(action_a, action_b)


def test_uses_exact_environment_dynamics_e_reach_matches_manual_rollout():
    from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary

    config = EnvConfig()
    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    obs = np.zeros(16, dtype=np.float32)
    tau = 2.0
    target_pos = np.array([10.0, 5.0, 5.0])
    defender_pos = np.array([0.0, 0.0, 5.0])
    defender_vel = np.array([3.0, -1.0, 0.0])
    info = _make_info(defender_pos, defender_vel, target_pos, [0.0, 0.0, 0.0])
    policy.act(obs, info)
    candidate = [c for c in policy.last_solution.candidates if c.tau == tau][0]

    from uav_defend.policies.baseline.lead_math import pure_pursuit_direction
    direction = pure_pursuit_direction(target_pos, defender_pos, config.eps)
    desired_velocity = config.v_d * direction
    pos, vel = defender_pos.copy(), defender_vel.copy()
    n_steps = int(round(tau / config.dt))
    for _ in range(n_steps):
        vel, _diag = advance_velocity(
            vel, desired_velocity, max_speed=config.v_d, max_accel=config.defender_max_accel,
            max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
            max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
            dt=config.dt, eps=config.eps,
        )
        pos = pos + vel * config.dt
        pos, vel = apply_boundary(pos, vel, L=config.L, max_altitude=config.max_altitude)
    expected_e_reach = float(np.linalg.norm(pos - target_pos))
    assert abs(candidate.E_reach - expected_e_reach) < 1e-4


def test_no_environment_mutation():
    from uav_defend.envs.soldier_env import SoldierEnv

    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=0)
    snapshot_pos = env._defender_pos.copy()
    snapshot_vel = env._defender_vel.copy()

    policy = RALAPGPolicy(_zeroed_predictor(), config=config)
    policy.act(obs, info)

    assert np.allclose(env._defender_pos, snapshot_pos)
    assert np.allclose(env._defender_vel, snapshot_vel)
