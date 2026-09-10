"""Tests for the ANALYSIS-ONLY True-State constant-velocity Lead policy
(Phase 6B / 19 of the learning-benefit diagnostic study).

Covers:
 1. truth injection only through the explicit diagnostic API (never via
    the standard act(obs, info) two-argument signature)
 2. the normal observation_space remains 16-D, completely unaffected
 3. no truth leakage into deployable policies (structural, not just
    behavioral: the deployable LeadInterceptPolicy class has no attribute
    or code path that could accept true state)
 4. numerical agreement with solve_cv_lead given the same true inputs
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import inspect

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.true_state_lead_policy import TrueStateLeadPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.baseline.lead_math import solve_cv_lead


def test_act_signature_requires_explicit_true_state_arguments():
    """The method signature itself (not just documentation) must force
    ground truth to be passed explicitly -- never via a standard
    act(obs, info) two-positional-argument call."""
    sig = inspect.signature(TrueStateLeadPolicy.act)
    params = list(sig.parameters.keys())
    assert "true_hostile_position" in params
    assert "true_hostile_velocity" in params
    # Not the standard deployable-policy signature (obs, info).
    assert params[:2] != ["obs", "info"]


def test_deployable_lead_has_no_true_state_pathway():
    """Structural no-leakage check: the deployable LeadInterceptPolicy's
    act() signature is exactly (obs, info) and it has no method/attribute
    for accepting true hostile state."""
    sig = inspect.signature(LeadInterceptPolicy.act)
    params = list(sig.parameters.keys())
    assert params == ["self", "obs", "info"]
    assert not hasattr(LeadInterceptPolicy, "true_hostile_position")


def test_observation_space_unaffected_by_true_state_policy_existence():
    env = SoldierEnv(config=EnvConfig())
    assert env.observation_space.shape == (16,)


def test_true_state_lead_matches_solve_cv_lead_directly():
    defender_pos = np.array([0.0, 0.0, 0.0])
    true_pos = np.array([20.0, 5.0, 3.0])
    true_vel = np.array([-2.0, 1.0, 0.0])
    policy = TrueStateLeadPolicy(config=EnvConfig())
    action = policy.act(defender_pos, true_pos, true_vel)
    expected = solve_cv_lead(true_pos, true_vel, defender_pos, policy.v_d, policy.eps)
    assert np.array_equal(action, expected.action)
    assert policy.last_guidance_mode == expected.guidance_mode


def test_true_state_lead_escort_fallback_when_not_detected():
    defender_pos = np.array([0.0, 0.0, 0.0])
    policy = TrueStateLeadPolicy(config=EnvConfig())
    action = policy.act(
        defender_pos, np.array([50.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
        soldier_position=np.array([5.0, 0.0, 0.0]), enemy_detected=False,
    )
    assert policy.last_guidance_mode == "escort"
    assert np.allclose(action, [1.0, 0.0, 0.0])


def test_true_state_lead_never_registered_as_deployable():
    from uav_defend.policies import registry as policy_registry
    registry_source = inspect.getsource(policy_registry)
    assert "TrueStateLeadPolicy" not in registry_source


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
