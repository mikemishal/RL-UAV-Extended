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
    pre-existing streams (spawn, soldier, enemy motion, sensor).

    NOTE (Phase 3 update): as of the journal-extension Phase 3, obstacles
    can legitimately couple back into the SIMULATED TRAJECTORY (the
    protected asset's motion is blocked by obstacles it walks into; the
    hostile's motion includes an obstacle-avoidance term when explicitly
    enabled). So the earlier Phase-1/2-era assertion of bit-identical
    trajectories no longer holds in general and is NOT what this test
    should check. The critical invariant this test protects -- that the
    5th (obstacle) RNG stream never perturbs the 4 pre-existing streams --
    is instead verified directly against the RNG generators themselves:
    after running matched episodes, drawing one further raw sample from
    each of the four original generators must yield identical values,
    proving identical draw COUNT and CONTENT regardless of whether
    obstacles were enabled (and regardless of whether the soldier's motion
    was ever blocked in the process; hostile avoidance is left disabled
    here, matching this test's config, so its own dynamics are unaffected
    too -- see test_enemy_obstacle_avoidance.py for that regression)."""
    baseline_config = EnvConfig()
    for seed in (0, 7, 2024):
        for obstacle_count in (1, 10, 25):
            obstacle_config = EnvConfig(
                obstacles_enabled=True, obstacle_layout_mode="random",
                obstacle_count=obstacle_count, obstacle_clearance_from_asset=3.0,
            )
            env_baseline = SoldierEnv(config=baseline_config)
            env_obstacle = SoldierEnv(config=obstacle_config)

            policy_baseline = GreedyInterceptPolicy()
            policy_obstacle = GreedyInterceptPolicy()
            policy_baseline.reset()
            policy_obstacle.reset()

            obs_b, info_b = env_baseline.reset(seed=seed)
            obs_o, info_o = env_obstacle.reset(seed=seed)
            for _ in range(200):
                action_b = policy_baseline.act(obs_b, info_b)
                action_o = policy_obstacle.act(obs_o, info_o)
                obs_b, _, term_b, trunc_b, info_b = env_baseline.step(action_b)
                obs_o, _, term_o, trunc_o, info_o = env_obstacle.step(action_o)
                if term_b or trunc_b or term_o or trunc_o:
                    break

            # Draw one further raw sample from each of the four original
            # exogenous RNG streams and require exact equality: this can
            # only hold if the obstacle stream (and any obstacle-coupled
            # motion) consumed exactly zero draws from these generators.
            for attr in ("_rng_spawn", "_rng_soldier", "_rng_enemy_motion", "_rng_sensor"):
                draw_b = getattr(env_baseline, attr).normal(size=4)
                draw_o = getattr(env_obstacle, attr).normal(size=4)
                assert np.array_equal(draw_b, draw_o), f"{attr} diverged"



# --- Step 10: SoldierEnv collision detection (Phase 2) ----------------------

def _wall_config(**overrides) -> EnvConfig:
    defaults = dict(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        # A wide, thin wall (x in [-15,-5], y in [-40,40], z in [0,4]) so
        # that a crossing is detected regardless of lateral weave/evasion
        # deviation in the hostile's autonomous pursuit motion model.
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        defender_standby_until_detection=False,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def test_defender_crossing_obstacle_is_detected():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    collided = False
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        collided = collided or info["defender_obstacle_collision"]
        if term or trunc:
            break
    assert collided


def test_hostile_crossing_obstacle_is_detected():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    env._enemy_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._enemy_vel = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    collided = False
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        collided = collided or info["enemy_obstacle_collision"]
        if term or trunc:
            break
    assert collided


def test_collision_does_not_terminate_episode_or_alter_reward():
    """A collision must be a pure diagnostic in Phase 2: termination and
    reward must be identical whether or not a collision occurred, for the
    same physical trajectory."""
    baseline_env = SoldierEnv(config=EnvConfig(defender_standby_until_detection=False))
    obstacle_env = SoldierEnv(config=_wall_config())
    baseline_env.reset(seed=1)
    obstacle_env.reset(seed=1)
    for env in (baseline_env, obstacle_env):
        env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, r_base, term_base, trunc_base, _ = baseline_env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        _, r_obs, term_obs, trunc_obs, info_obs = obstacle_env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert r_base == r_obs
        assert term_base == term_obs
        assert trunc_base == trunc_obs
        if term_base or trunc_base:
            break


def test_collision_does_not_modify_trajectory():
    """Positions/velocities must be identical to the no-obstacle baseline
    even while a collision is being flagged (Phase 2 never blocks motion)."""
    baseline_env = SoldierEnv(config=EnvConfig(defender_standby_until_detection=False))
    obstacle_env = SoldierEnv(config=_wall_config())
    baseline_env.reset(seed=1)
    obstacle_env.reset(seed=1)
    for env in (baseline_env, obstacle_env):
        env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term_base, trunc_base, info_base = baseline_env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        _, _, term_obs, trunc_obs, info_obs = obstacle_env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert np.array_equal(info_base["defender_pos"], info_obs["defender_pos"])
        assert np.array_equal(info_base["defender_vel"], info_obs["defender_vel"])
        if term_base or trunc_base:
            break


def test_collision_count_and_event_semantics():
    """Remaining inside/overlapping an obstacle across multiple steps must
    count as ONE event, not once per step."""
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    counts = []
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        counts.append(info["defender_collision_count"])
        if term or trunc:
            break
    # The count must be non-decreasing and, since there is exactly one
    # obstacle astride the straight-line path, must saturate at 1 -- not
    # increment again on every subsequent overlapping step.
    assert counts == sorted(counts)
    assert max(counts) == 1
    first_hit = counts.index(1)
    assert all(c == 1 for c in counts[first_hit:])


def test_no_collision_when_flying_above_obstacle():
    cfg = _wall_config()
    env = SoldierEnv(config=cfg)
    env.reset(seed=1)
    # Fly well above the obstacle's top (z=4) in a straight line across its
    # x/y footprint.
    env._defender_pos = np.array([-30.0, 0.0, 20.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    collided = False
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        collided = collided or info["defender_obstacle_collision"]
        if term or trunc:
            break
    assert not collided


def test_minimum_clearance_diagnostic_correct():
    cfg = _wall_config()
    env = SoldierEnv(config=cfg)
    _, info = env.reset(seed=1)
    # Origin (0,0,0) to nearest face of obstacle spanning x in [-15,-5]: 5.0
    assert np.isclose(info["defender_min_obstacle_clearance"], 5.0)


# --- Part A: pre-detection standby collision-semantics fix ------------------

def test_standby_synchronization_not_reported_as_swept_defender_collision():
    """Part A fix: while the defender is held in pre-detection standby, its
    position is synchronized onto the just-moved soldier as bookkeeping,
    NOT physical UAV flight. A thin obstacle sits exactly on the straight
    line between the defender's previous position and the newly synced
    soldier position, but neither endpoint is inside the obstacle --
    before the fix this would have been misreported as a swept collision
    ("flying through a wall"); after the fix it must not be."""
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p1 = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        # Thin wall crossing the p0->p1 line at its midpoint (x=2), while
        # containing neither p0 (x=0) nor p1 (x=4).
        obstacle_fixed_spec=((2.0, 0.0, 5.0, 0.3, 5.0, 5.0),),
        # defender_standby_until_detection defaults to True.
    )
    env = SoldierEnv(config=cfg)
    env.reset(seed=1)
    env._soldier_pos = p0.copy()
    env._defender_pos = p0.copy()
    env._defender_vel = np.zeros(3, dtype=np.float32)
    # Force a controlled, deterministic soldier displacement (bypassing the
    # stochastic random walk) so the scenario is exact and reproducible.
    env._move_soldier = lambda: setattr(env, "_soldier_pos", p1.copy())

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert info["controller_action_executed"] is False  # still in standby
    assert np.array_equal(info["defender_pos"], p1)     # synced onto the soldier
    assert info["defender_obstacle_collision"] is False  # NOT a fictitious swept strike
    assert info["defender_collision_count"] == 0


def test_controller_executed_motion_through_same_obstacle_is_still_detected():
    """Retained: once the controller is actually in control (post-
    detection, or legacy mode), REAL swept defender motion through the same
    obstacle placement must still be detected -- the Part A fix only
    suppresses the pre-detection bookkeeping synchronization, not genuine
    flight."""
    cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((2.0, 0.0, 5.0, 0.3, 5.0, 5.0),),
        defender_standby_until_detection=False,  # legacy mode: action always executed
    )
    env = SoldierEnv(config=cfg)
    env.reset(seed=1)
    env._defender_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert info["controller_action_executed"] is True
    assert info["defender_obstacle_collision"] is True


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

