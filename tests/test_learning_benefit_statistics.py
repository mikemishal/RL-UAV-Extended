"""Tests for Phase 14/15 paired statistical utilities and Phase 7
diagnostic logging determinism.

Covers:
 - paired outcome bootstrap difference (deterministic given fixed seed)
 - paired binary contingency counts / McNemar statistic
 - episode-level clustering (aggregation before inference)
 - diagnostic logging: deterministic output, correct episode/seed/
   controller identifiers, no policy-action changes caused by logging
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.learning_benefit.statistics_utils import (
    episode_level_aggregate,
    paired_binary_contingency,
    paired_bootstrap_difference,
)
from experiments.learning_benefit.diagnostic_logging import STEP_LOG_COLUMNS, run_episode_with_diagnostics
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy


def test_paired_bootstrap_is_deterministic_given_fixed_seed():
    a = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.float64)
    b = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=np.float64)
    result1 = paired_bootstrap_difference(a, b, n_replicates=500)
    result2 = paired_bootstrap_difference(a, b, n_replicates=500)
    assert result1 == result2


def test_paired_bootstrap_mean_diff_correct():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    result = paired_bootstrap_difference(a, b, n_replicates=200)
    assert abs(result["mean_diff"] - 1.0) < 1e-9


def test_paired_binary_contingency_counts():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)
    result = paired_binary_contingency(a, b)
    assert result["both"] == 1     # index 0: both True
    assert result["only_a"] == 1   # index 1: a True, b False
    assert result["only_b"] == 1   # index 2: a False, b True
    assert result["neither"] == 1  # index 3: both False


def test_episode_level_aggregate_collapses_per_step_to_per_episode():
    df = pd.DataFrame({
        "seed": [1, 1, 1, 2, 2],
        "value": [1.0, 2.0, 3.0, 10.0, 20.0],
    })
    result = episode_level_aggregate(df, "seed", "value")
    assert len(result) == 2  # 2 episodes, not 5 rows
    assert sorted(result.tolist()) == sorted([2.0, 15.0])


def test_diagnostic_logging_deterministic_and_identifiers_correct():
    def make_env():
        return SoldierEnv(config=EnvConfig())

    policy1 = LeadInterceptPolicy(state_source="measurement", config=EnvConfig())
    rows1 = run_episode_with_diagnostics(make_env(), policy1, seed=123, controller="lead", scenario="nominal", max_steps=20)

    policy2 = LeadInterceptPolicy(state_source="measurement", config=EnvConfig())
    rows2 = run_episode_with_diagnostics(make_env(), policy2, seed=123, controller="lead", scenario="nominal", max_steps=20)

    assert len(rows1) == len(rows2)
    for r1, r2 in zip(rows1, rows2):
        assert r1 == r2  # fully deterministic given same seed/policy/config

    assert all(r["seed"] == 123 for r in rows1)
    assert all(r["controller"] == "lead" for r in rows1)
    assert all(r["scenario"] == "nominal" for r in rows1)
    assert [r["step_index"] for r in rows1] == list(range(len(rows1)))
    assert set(rows1[0].keys()) == set(STEP_LOG_COLUMNS)


def test_diagnostic_logging_does_not_alter_policy_actions():
    """Running the SAME episode with vs. without diagnostic logging must
    produce identical environment trajectories -- logging is a pure
    observer. Each logged row records the PRE-STEP true position (the
    state used to compute that step's action), so the comparison below
    captures the plain loop's info the same way (before calling step)."""
    cfg = EnvConfig()

    env_logged = SoldierEnv(config=cfg)
    policy_logged = LeadInterceptPolicy(state_source="measurement", config=cfg)
    rows = run_episode_with_diagnostics(env_logged, policy_logged, seed=7, controller="lead", scenario="nominal", max_steps=30)

    env_plain = SoldierEnv(config=cfg)
    policy_plain = LeadInterceptPolicy(state_source="measurement", config=cfg)
    obs, info = env_plain.reset(seed=7)
    policy_plain.reset()
    from uav_defend.policies.sanitize import build_policy_info
    pre_step_true_positions = []
    for _ in range(30):
        pre_step_true_positions.append(info["enemy_pos"].copy())
        action = policy_plain.act(obs, build_policy_info(info, "measurement"))
        obs, reward, term, trunc, info = env_plain.step(action)
        if term or trunc:
            break

    assert len(rows) == len(pre_step_true_positions)
    for row, true_pos in zip(rows, pre_step_true_positions):
        assert np.allclose(row["true_enemy_pos_x"], true_pos[0])
        assert np.allclose(row["true_enemy_pos_y"], true_pos[1])
        assert np.allclose(row["true_enemy_pos_z"], true_pos[2])


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
