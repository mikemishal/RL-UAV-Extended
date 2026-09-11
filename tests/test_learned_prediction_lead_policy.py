"""Tests for LearnedPredictionLeadPolicy (Phase 52): causal history only,
reset behavior, zero-correction reduces to Standard Lead's corrected-point
behavior, and no ground-truth leakage."""

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.learned_prediction_lead_policy import LearnedPredictionLeadPolicy
from uav_defend.policies.baseline.lead_math import solve_cv_lead


def _zeroed_predictor() -> HorizonConditionedPredictor:
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    model.eval()
    return model


def test_reset_clears_history_and_diagnostics():
    config = EnvConfig()
    policy = LearnedPredictionLeadPolicy(_zeroed_predictor(), config=config)
    policy._history.append(np.ones(17))
    policy._steps_since_detection = 5
    policy.last_action = np.array([1.0, 0.0, 0.0])

    policy.reset()
    assert policy._history == []
    assert policy._steps_since_detection == -1
    assert policy.last_action is None
    assert policy.last_guidance_mode is None


def test_zero_correction_matches_standard_lead_intercept_point():
    """With a zeroed decoder, p_corrected must equal the plain CV Lead
    intercept point (p_hat + v_hat*tau + 0), so the commanded direction
    matches solve_cv_lead's own action exactly."""
    config = EnvConfig()
    policy = LearnedPredictionLeadPolicy(_zeroed_predictor(), config=config)

    defender_pos = np.array([0.0, 0.0, 5.0])
    target_pos = np.array([20.0, 0.0, 5.0])
    target_vel = np.array([-1.0, 0.0, 0.0])
    info = {
        "defender_pos": defender_pos, "defender_vel": np.array([0.0, 0.0, 0.0]),
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
        "enemy_detected": True,
        "enemy_measurement": target_pos, "enemy_measurement_velocity": target_vel,
        "enemy_measurement_velocity_valid": True,
    }
    obs = np.zeros(16, dtype=np.float32)
    action = policy.act(obs, info)

    cv_solution = solve_cv_lead(target_pos, target_vel, defender_pos, config.v_d, config.eps)
    assert policy.last_guidance_mode == "learned_prediction_lead"
    assert np.allclose(policy.last_corrected_point, cv_solution.intercept_point, atol=1e-5)
    assert np.allclose(action, cv_solution.action, atol=1e-5)


def test_causal_history_never_includes_ground_truth_enemy_state():
    """History features must be built ONLY from the measurement/estimate
    fields, never from info['enemy_pos']/info['enemy_vel']."""
    config = EnvConfig()
    policy = LearnedPredictionLeadPolicy(_zeroed_predictor(), config=config)

    true_enemy_pos = np.array([999.0, 999.0, 999.0])  # deliberately different from the measurement
    measured_pos = np.array([20.0, 0.0, 5.0])
    info = {
        "defender_pos": np.zeros(3), "defender_vel": np.zeros(3),
        "soldier_pos": np.zeros(3), "enemy_detected": True,
        "enemy_measurement": measured_pos, "enemy_measurement_velocity": np.array([-1.0, 0.0, 0.0]),
        "enemy_measurement_velocity_valid": True,
        "enemy_pos": true_enemy_pos, "enemy_vel": np.array([-5.0, 0.0, 0.0]),
    }
    obs = np.zeros(16, dtype=np.float32)
    policy.act(obs, info)

    assert np.allclose(policy._history[-1][0:3], measured_pos)
    assert not np.allclose(policy._history[-1][0:3], true_enemy_pos)


def test_escort_fallback_when_not_detected_resets_detection_counter():
    config = EnvConfig()
    policy = LearnedPredictionLeadPolicy(_zeroed_predictor(), config=config)
    info = {
        "defender_pos": np.array([5.0, 0.0, 5.0]), "defender_vel": np.zeros(3),
        "soldier_pos": np.zeros(3), "enemy_detected": False,
    }
    obs = np.zeros(16, dtype=np.float32)
    action = policy.act(obs, info)
    assert policy.last_guidance_mode == "escort"
    assert policy._steps_since_detection == -1
    assert policy._history == []
    assert np.isfinite(action).all()


def test_full_episode_rollout_does_not_crash_and_uses_only_causal_info():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    policy = LearnedPredictionLeadPolicy(_zeroed_predictor(), config=config)
    obs, info = env.reset(seed=100000)
    policy.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        action = policy.act(obs, info)
        assert np.isfinite(action).all()
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
    assert steps > 0
