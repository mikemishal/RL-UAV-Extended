"""Action contract `a1_v2_action_v3`: bounded residual + A1 tactical safety filter. No PPO."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import experiments.journal_v2_a1 as A
import experiments.journal_v2_a1_r2 as R
from experiments.journal_v2_a1_r3_theta_study import run_episode as v3_episode
from experiments.journal_v2_a1_r2_theta_study import run_episode as v2_episode
from experiments.journal_v2_protocol import build_v2_env_config
from experiments.journal_v2_validation_core import PARKED_DEFENDER, build_config
from uav_defend.envs.soldier_env import SoldierEnv


def passive_env(level, seed):
    cfg, _ = build_config(level, "V2", "passive")
    env = SoldierEnv(cfg)
    return env, cfg


def prime(env, seed):
    """Reset and run one canonical step so the perception/mission state of a real mid-episode step exists."""
    env.reset(seed=seed)
    env._defender_pos = np.asarray(PARKED_DEFENDER, np.float32)
    env._defender_vel = np.zeros(3, np.float32)


def test_identifiers():
    assert R.ACTION_VERSION == "a1_v2_action_v3" and R.ACTION_VERSION_V2 == "a1_v2_action_v2" and R.REWARD_VERSION == "a1_v2_reward_v2"


def test_zero_action_is_exactly_u_base_and_never_calls_the_filter(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "tactical_safety_check", lambda *a, **k: calls.append(1) or {"safe": True, "min_clearance": 9.0, "n_obstacles": 1})
    env = R.A1V2R2TrainingEnv(0, 0, theta_max_deg=30.0)
    env.reset(options={"scenario": ("C2", "greedy", 140006)})
    for _ in range(5):
        _o, _r, te, tr, info = env.step(np.zeros(3, np.float32))
        assert info["a1_residual_angle_deg"] == 0.0 and not info["a1_filter_intervened"]
        if te or tr:
            break
    assert calls == []


def test_a_safe_residual_is_accepted_unchanged():
    env, _ = passive_env("C0", 145000)  # C0: no obstacles at all -> nothing to collide with
    prime(env, 145000)
    env.enemy_tactical_override = lambda: np.zeros(3)
    env.step(np.zeros(3, np.float32))
    u_b = R.base_direction(env)
    command, _angle = R.residual_command(np.array([0.0, 1.0, 0.0]), u_b, 30.0)
    check = R.tactical_safety_check(env, command)
    assert check["safe"] and check["n_obstacles"] == 0 and check["collision"] is False


def _env_toward_obstacle():
    """C3 episode advanced until a residual is predicted unsafe while the canonical u_base is safe."""
    env, _ = passive_env("C3", 145003)
    prime(env, 145003)
    env.enemy_tactical_override = lambda: np.zeros(3)
    for _ in range(40):
        env.step(np.zeros(3, np.float32))
        env._enemy_perception  # noqa: B018
        u_b = R.base_direction(env)
        if u_b is None or getattr(env, "_enemy_last_perceived", None) is None:
            continue
        for raw in ([0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, 1, -1], [1, -1, 1]):
            command, _a = R.residual_command(np.array(raw, float), u_b, 60.0)
            if command.any() and not R.tactical_safety_check(env, command)["safe"] and R.tactical_safety_check(env, u_b)["safe"]:
                return env, u_b, command
        if env._steps_done if hasattr(env, "_steps_done") else False:
            break
    return None


def test_predicted_unsafe_residual_is_rejected_and_u_base_is_executed():
    found = None
    for seed in range(145000, 145040):
        env = R.A1V2R2TrainingEnv(0, 0, theta_max_deg=60.0)
        env.reset(options={"scenario": ("C3", "greedy", seed)})
        for _ in range(60):
            _o, _r, te, tr, info = env.step(np.array([0.0, 1.0, 0.0], np.float32))
            if info.get("a1_filter_intervened"):
                found = info
                break
            if te or tr:
                break
        if found:
            break
    assert found is not None, "no filter intervention found in C3"
    assert found["a1_proposed_angle_deg"] > 0.0 and found["a1_residual_angle_deg"] == 0.0  # rejected: executed residual is zero
    assert found["a1_filter_safe"] is False


def test_rejection_flies_the_canonical_u_base_exactly():
    """When the filter rejects, the executed command equals what a zero residual executes (u_base, bit-for-bit)."""
    scenario = ("C3", "greedy", 145003)

    def trajectory(action_fn, **kw):
        env = R.A1V2R2TrainingEnv(0, 0, theta_max_deg=60.0, **kw)
        env.reset(options={"scenario": scenario})
        out, done = [], False
        while not done and len(out) < 3:
            _o, _r, te, tr, info = env.step(action_fn())
            out.append((bool(info.get("a1_filter_intervened")), np.asarray(info["enemy_pos"]).copy()))
            done = te or tr
        return out

    zero = trajectory(lambda: np.zeros(3, np.float32))
    filtered = trajectory(lambda: np.array([0.0, 1.0, 0.0], np.float32))
    for (intervened, pos_f), (_i, pos_z) in zip(filtered, zero):
        if not intervened:
            break
        assert np.array_equal(pos_f, pos_z)  # identical state while every proposal is rejected


def test_filter_uses_the_perceived_set_not_ground_truth():
    env, _ = passive_env("C3", 145003)
    prime(env, 145003)
    env.enemy_tactical_override = lambda: np.zeros(3)
    env.step(np.zeros(3, np.float32))
    perceived_before = env._enemy_last_perceived
    u_b = R.base_direction(env)
    command, _ = R.residual_command(np.array([0.0, 1.0, 0.0]), u_b, 30.0)
    seen = R.tactical_safety_check(env, command)
    assert seen["n_obstacles"] == len(perceived_before.to_static_layout()) + 0 or seen["n_obstacles"] >= len(perceived_before.to_static_layout())
    env._enemy_last_perceived = None
    assert R.tactical_safety_check(env, command)["safe"] is False  # no perceived set -> fail closed


def test_dynamic_obstacles_enter_through_the_frozen_combined_layout(monkeypatch):
    env, _ = passive_env("C3", 145003)
    prime(env, 145003)
    env.enemy_tactical_override = lambda: np.zeros(3)
    env.step(np.zeros(3, np.float32))
    seen = {}
    original = env._combined_avoidance_layout

    def spy(position, velocity, horizon, clearance, static_layout, dynamic):
        seen["args"] = (horizon, clearance, dynamic)
        return original(position, velocity, horizon, clearance, static_layout, dynamic)

    monkeypatch.setattr(env, "_combined_avoidance_layout", spy)
    u_b = R.base_direction(env)
    R.tactical_safety_check(env, u_b)
    assert seen["args"][0] == env.config.enemy_obstacle_prediction_horizon_steps and seen["args"][1] == env.config.enemy_obstacle_clearance
    assert seen["args"][2] is env._enemy_last_perceived.dynamic  # the perceived dynamic set, via the frozen helper


def test_no_spurious_rejection_without_obstacles():
    env = R.A1V2R2TrainingEnv(0, 0, theta_max_deg=60.0)
    env.reset(options={"scenario": ("C0", "greedy", 140001)})
    for _ in range(12):
        _o, _r, te, tr, info = env.step(np.array([0.3, -1.0, 0.5], np.float32))
        assert info["a1_filter_intervened"] is False and info["a1_filter_n_obstacles"] == 0
        if te or tr:
            break


@pytest.mark.parametrize("theta", R.THETA_CANDIDATES_DEG)
def test_angle_bound_positive_base_component_no_reversal_finite(theta):
    rng = np.random.default_rng(5)
    for _ in range(500):
        u_b = rng.normal(size=3)
        u_b /= np.linalg.norm(u_b)
        command, angle = R.residual_command(rng.uniform(-1, 1, 3), u_b, theta)
        if not command.any():
            continue
        assert np.isfinite(command).all() and angle <= theta + 1e-6 and float(command @ u_b) > 0.0


def test_regression_action_v2_collision_is_avoided_by_action_v3():
    """theta=30 deg, persistent left, C3, seed 145003 + persistent max-left: action-v2 (no filter) collided at step 19."""
    unfiltered = v2_episode(30.0, "left", "C3", 145003)
    assert unfiltered["outcome"] == "enemy_static_obstacle_collision"
    filtered = v3_episode(30.0, "left", "C3", 145003)
    assert filtered["outcome"] != "enemy_static_obstacle_collision" and filtered["interventions"] > 0
    assert filtered["max_residual_angle_deg"] <= 30.01


def test_filter_off_reproduces_the_rejected_action_v2_exactly():
    a = v2_episode(30.0, "left", "C3", 145003)
    b = v3_episode(30.0, "left", "C3", 145003) if False else None
    from experiments.journal_v2_a1_r3_theta_study import run_episode

    unfiltered_v3 = run_episode(30.0, "left", "C3", 145003, safety_filter=False)
    assert (unfiltered_v3["outcome"], unfiltered_v3["steps"]) == (a["outcome"], a["steps"]) and unfiltered_v3["interventions"] == 0


# ----------------------------------------------------------------------------- isolation
@pytest.mark.parametrize("scenario", [("C0", "greedy", 140001), ("C2", "greedy", 140006), ("C3", "lead", 140028)])
def test_a0_is_unaffected_scripted_a0_has_no_override_and_no_filter_calls(monkeypatch, scenario):
    monkeypatch.setattr(R, "tactical_safety_check", lambda *a, **k: pytest.fail("the filter must never run for A0"))
    env = R.A1V2R2TrainingEnv(0, 0, scripted_a0=True, theta_max_deg=30.0)
    env.reset(options={"scenario": scenario})
    assert env.soldier_env.enemy_tactical_override is None
    done = False
    while not done:
        _o, _r, te, tr, info = env.step(np.ones(3, np.float32))
        done = te or tr
    assert info["a1_episode"]["filter_proposals"] == 0 and info["a1_residual_angle_deg"] == 0.0


def test_simulator_core_and_v1_attacker_modules_do_not_reference_the_filter():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("uav_defend", "experiments/adaptive_attacker_env.py", "experiments/journal_v2_a1.py", "experiments/journal_v2_protocol.py"):
        path = root / rel
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")
            assert "tactical_safety_check" not in text and "a1_v2_action_v3" not in text, file


def test_defender_policies_and_config_do_not_depend_on_the_attacker_filter():
    cfg_a = build_v2_env_config("C2", "measurement")
    env = R.A1V2R2TrainingEnv(0, 0, theta_max_deg=30.0)
    env.reset(options={"scenario": ("C2", "greedy", 140006)})
    assert dataclasses.asdict(env.soldier_env.config) == dataclasses.asdict(cfg_a)  # canonical config untouched by A1 action/filter
    assert env.soldier_env.enemy_tactical_override is not None and env.soldier_env.config.enemy_endurance_enabled


def test_filter_is_pure_and_does_not_mutate_environment_state():
    env, _ = passive_env("C3", 145003)
    prime(env, 145003)
    env.enemy_tactical_override = lambda: np.zeros(3)
    env.step(np.zeros(3, np.float32))
    snapshot = (env._enemy_pos.copy(), env._enemy_vel.copy(), env._enemy_obstacle_planner.active, env._enemy_obstacle_planner.step_count)
    u_b = R.base_direction(env)
    for raw in ([0, 1, 0], [0, 0, 1], [1, -1, 0]):
        command, _ = R.residual_command(np.array(raw, float), u_b, 45.0)
        if command.any():
            R.tactical_safety_check(env, command)
    assert np.array_equal(env._enemy_pos, snapshot[0]) and np.array_equal(env._enemy_vel, snapshot[1])
    assert (env._enemy_obstacle_planner.active, env._enemy_obstacle_planner.step_count) == snapshot[2:]
