"""SoldierEnv obstacle integration tests, including the critical
obstacles_enabled=False conference-baseline regression test (Step 9).

Run directly: python tests/test_obstacle_env_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy


# --- basic integration -------------------------------------------------------

def test_obstacles_disabled_zero_obstacles():
    env = SoldierEnv(config=EnvConfig(obstacles_enabled=False))
    obs, info = env.reset(seed=5)
    assert info["obstacles_enabled"] is False
    assert info["obstacle_count"] == 0
    assert info["obstacle_layout"] == {"count": 0, "obstacles": []}


def test_obstacles_enabled_configured_obstacle_count():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=4,
                     obstacle_clearance_from_asset=3.0)
    env = SoldierEnv(config=cfg)
    obs, info = env.reset(seed=5)
    assert info["obstacles_enabled"] is True
    assert info["obstacle_count"] == 4
    assert len(info["obstacle_layout"]["obstacles"]) == 4


def test_same_reset_seed_same_layout():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=5,
                     obstacle_clearance_from_asset=3.0)
    env = SoldierEnv(config=cfg)
    _, info_a = env.reset(seed=77)
    _, info_b = env.reset(seed=77)
    assert info_a["obstacle_layout"] == info_b["obstacle_layout"]


def test_different_reset_seeds_different_layout():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=5,
                     obstacle_clearance_from_asset=3.0)
    env = SoldierEnv(config=cfg)
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=2)
    assert info_a["obstacle_layout"] != info_b["obstacle_layout"]


def test_observation_shape_unchanged():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=4,
                     obstacle_clearance_from_asset=3.0)
    env = SoldierEnv(config=cfg)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (16,)
    obs2, *_ = env.step(np.zeros(3, dtype=np.float32))
    assert obs2.shape == (16,)


def test_action_space_unchanged():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=4,
                     obstacle_clearance_from_asset=3.0)
    env = SoldierEnv(config=cfg)
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (16,)


# --- Step 9: critical conference-baseline regression test ------------------

def _run_episode(config: EnvConfig, seed: int, max_steps: int = 400):
    """Run one episode with GreedyInterceptPolicy actions (a realistic,
    non-trivial, non-zero action sequence) and record everything relevant
    to the conference-baseline behavior."""
    env = SoldierEnv(config=config)
    policy = GreedyInterceptPolicy()
    obs, info = env.reset(seed=seed)
    policy.reset()
    record = {
        "initial_soldier_pos": info["soldier_pos"].copy(),
        "initial_defender_pos": info["defender_pos"].copy(),
        "initial_enemy_pos": info["enemy_pos"].copy(),
        "initial_enemy_vel": info["enemy_vel"].copy(),
        "observations": [obs.copy()],
        "rewards": [],
        "enemy_measurements": [],
        "outcome": None,
    }
    for _ in range(max_steps):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        record["observations"].append(obs.copy())
        record["rewards"].append(reward)
        record["enemy_measurements"].append(
            None if info["enemy_measurement"] is None else info["enemy_measurement"].copy()
        )
        if terminated or truncated:
            record["outcome"] = info["outcome"]
            break
    return record


def _assert_records_identical(a: dict, b: dict) -> None:
    assert np.array_equal(a["initial_soldier_pos"], b["initial_soldier_pos"])
    assert np.array_equal(a["initial_defender_pos"], b["initial_defender_pos"])
    assert np.array_equal(a["initial_enemy_pos"], b["initial_enemy_pos"])
    assert np.array_equal(a["initial_enemy_vel"], b["initial_enemy_vel"])
    assert len(a["observations"]) == len(b["observations"])
    for obs_a, obs_b in zip(a["observations"], b["observations"]):
        assert np.array_equal(obs_a, obs_b)
    assert a["rewards"] == b["rewards"]
    assert len(a["enemy_measurements"]) == len(b["enemy_measurements"])
    for m_a, m_b in zip(a["enemy_measurements"], b["enemy_measurements"]):
        if m_a is None or m_b is None:
            assert m_a is None and m_b is None
        else:
            assert np.array_equal(m_a, m_b)
    assert a["outcome"] == b["outcome"]


def test_obstacles_disabled_reproduces_conference_baseline_multi_seed():
    """CRITICAL: obstacles_enabled=False must reproduce the exact
    conference-baseline environment -- initial positions, soldier random
    walk, hostile stochastic motion, sensor measurements, observations,
    rewards, and terminal outcome must all be bit-for-bit identical to a
    plain EnvConfig() run, for several seeds."""
    baseline_config = EnvConfig()
    disabled_obstacle_config = EnvConfig(
        obstacles_enabled=False,
        obstacle_layout_mode="random",  # even if mis-set, disabled must win
        obstacle_count=20,
    )
    for seed in (0, 1, 42, 12345):
        baseline = _run_episode(baseline_config, seed)
        disabled = _run_episode(disabled_obstacle_config, seed)
        _assert_records_identical(baseline, disabled)


def test_obstacles_enabled_does_not_perturb_four_exogenous_rng_streams():
    """CRITICAL: enabling obstacles (which consumes its own dedicated RNG
    stream) must not change ANY random number consumed by the four
    pre-existing streams (spawn, soldier, enemy motion, sensor) -- i.e. the
    full trajectory must remain identical whether or not obstacles are
    enabled, for several seeds and obstacle counts."""
    baseline_config = EnvConfig()
    for seed in (0, 7, 2024):
        for obstacle_count in (1, 10, 25):
            obstacle_config = EnvConfig(
                obstacles_enabled=True, obstacle_layout_mode="random",
                obstacle_count=obstacle_count, obstacle_clearance_from_asset=3.0,
            )
            baseline = _run_episode(baseline_config, seed)
            with_obstacles = _run_episode(obstacle_config, seed)
            _assert_records_identical(baseline, with_obstacles)


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
