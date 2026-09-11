"""Tests for RA-Lead (Phase 76): exact CV target prediction, reuse of the
SAME reachability code as RA-LAPG, no learned-model dependency, and
deterministic behavior."""

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.guidance.reachability_selection import ReachabilityMode
from uav_defend.policies.analysis.reachability_aware_lead_policy import (
    CANDIDATE_TAUS,
    ReachabilityAwareLeadPolicy,
)


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


def test_no_learned_model_dependency():
    """RA-Lead's constructor must not require (or accept) a predictor."""
    import inspect
    sig = inspect.signature(ReachabilityAwareLeadPolicy.__init__)
    assert "predictor" not in sig.parameters


def test_exact_cv_target_prediction():
    config = EnvConfig()
    policy = ReachabilityAwareLeadPolicy(config=config)
    obs = np.zeros(16, dtype=np.float32)
    target_pos = np.array([30.0, 0.0, 5.0])
    target_vel = np.array([-1.0, 0.0, 0.0])
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], target_pos, target_vel)
    policy.act(obs, info)
    for c in policy.last_solution.candidates:
        expected = target_pos + target_vel * c.tau
        assert np.allclose(c.predicted_target_point, expected, atol=1e-9)


def test_same_candidate_grid_as_ra_lapg():
    from uav_defend.policies.analysis.ra_lapg_policy import CANDIDATE_TAUS as RALAPG_TAUS
    assert CANDIDATE_TAUS == RALAPG_TAUS


def test_deterministic_output_same_input_same_action():
    config = EnvConfig()
    policy_a = ReachabilityAwareLeadPolicy(config=config)
    policy_b = ReachabilityAwareLeadPolicy(config=config)
    obs = np.zeros(16, dtype=np.float32)
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], [30.0, 0.0, 5.0], [-1.0, 0.0, 0.0])
    action_a = policy_a.act(obs, dict(info))
    action_b = policy_b.act(obs, dict(info))
    assert np.allclose(action_a, action_b)


def test_reachability_metric_matches_ra_lapg_with_zeroed_predictor():
    """When RA-LAPG's learned correction is exactly zero, its candidate
    E_reach/feasibility must match RA-Lead's (same CV target, same
    reachability code)."""
    import torch
    from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
    from uav_defend.policies.analysis.ra_lapg_policy import RALAPGPolicy

    config = EnvConfig()
    model = HorizonConditionedPredictor()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    model.eval()

    ra_lapg = RALAPGPolicy(model, config=config)
    ra_lead = ReachabilityAwareLeadPolicy(config=config)
    obs = np.zeros(16, dtype=np.float32)
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], [30.0, 0.0, 5.0], [-1.0, 0.0, 0.0])

    ra_lapg.act(obs, dict(info))
    ra_lead.act(obs, dict(info))

    for c1, c2 in zip(ra_lapg.last_solution.candidates, ra_lead.last_solution.candidates):
        assert c1.tau == c2.tau
        assert np.allclose(c1.predicted_target_point, c2.predicted_target_point, atol=1e-5)
        assert abs(c1.E_reach - c2.E_reach) < 1e-4
        assert c1.feasible == c2.feasible


def test_receding_target_mode_selectable():
    config = EnvConfig()
    policy = ReachabilityAwareLeadPolicy(config=config, reachability_mode=ReachabilityMode.RECEDING_TARGET)
    obs = np.zeros(16, dtype=np.float32)
    info = _make_info([0.0, 0.0, 5.0], [config.v_d, 0.0, 0.0], [30.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    policy.act(obs, info)
    assert policy.last_solution is not None
    assert np.isfinite(policy.last_action).all()
