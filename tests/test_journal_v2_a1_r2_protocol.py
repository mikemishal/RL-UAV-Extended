"""Reward `a1_v2_reward_v2`, action `a1_v2_action_v2`, hover diagnostic, frozen gates. No PPO."""

from __future__ import annotations

import math

import numpy as np
import pytest

import experiments.journal_v2_a1 as A
import experiments.journal_v2_a1_r2 as R

GAMMA = R.GAMMA


def trajectory(ranges, path_steps, d0, outcome):
    budget = 5.0 * d0
    prev, comps = d0, []
    for i, (r, dl) in enumerate(zip(ranges, path_steps)):
        comps.append(R.step_reward(range_prev_m=prev, range_next_m=r, initial_range_m=d0, path_step_m=dl, path_budget_m=budget,
                                   outcome=outcome if i == len(ranges) - 1 else "ongoing"))
        prev = r
    return comps


def ret(comps, discounted=True):
    return sum((GAMMA ** i if discounted else 1.0) * c.total for i, c in enumerate(comps))


def test_identifiers_and_failed_v1_is_untouched():
    assert (R.REWARD_VERSION, R.ACTION_VERSION, R.TRAINING_PROTOCOL) == ("a1_v2_reward_v2", "a1_v2_action_v2", "a1_v2_training_v2")
    assert A.A1_V2_REWARD_VERSION == "a1_v2_reward_v1" and A.A1_V2_TRAINING_PROTOCOL == "a1_v2_training_v1"
    assert A.W_PROGRESS == 2.0 and A.W_TIME == 0.25 and A.potential(50.0, 100.0) == -0.5  # v1 reward unchanged
    assert R.RESULTS_ROOT.name == "a1_v2_training_v2" and R.FAILED_V1_RESULTS_ROOT.name == "a1_v2_training_v1"
    assert R.FAILED_V1["label"] == "NON-CANONICAL — FAILED REWARD/ACTION FORMULATION"


def test_step_cost_derivation():
    assert GAMMA == 0.99 and R.R_F == 10.0 and R.STEP_COST == -0.1
    assert sum(GAMMA ** t * R.STEP_COST for t in range(20000)) == pytest.approx(-R.R_F, rel=1e-6)


def test_terminal_mapping():
    assert R.terminal_reward("soldier_caught") == 10.0 and R.terminal_reward("ongoing") == 0.0
    for outcome in ("intercepted", "unsafe_intercept", "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
                    "enemy_endurance_exhausted", "timeout", "defender_collision", "x"):
        assert R.terminal_reward(outcome) == -10.0


def test_potential_is_bounded_and_formula_matches_spec():
    assert R.phi(100.0, 100.0) == 0.0 and R.phi(0.0, 100.0) == 1.0 and R.phi(250.0, 100.0) == 0.0 and R.phi(-5.0, 100.0) == 1.0
    r = R.step_reward(range_prev_m=80.0, range_next_m=70.0, initial_range_m=100.0, path_step_m=10.0, path_budget_m=500.0, outcome="ongoing")
    assert r.progress == pytest.approx(2.0 * (0.99 * 0.3 - 0.2)) and r.path == pytest.approx(-10.0 / 500.0)
    assert r.step == -0.1 and r.terminal == 0.0 and r.total == pytest.approx(r.progress + r.path + r.step)


@pytest.mark.parametrize("d", np.linspace(0.0, 150.0, 31))
def test_constant_range_never_earns_positive_progress(d):
    r = R.step_reward(range_prev_m=d, range_next_m=d, initial_range_m=100.0, path_step_m=0.0, path_budget_m=500.0, outcome="ongoing")
    assert r.progress <= 0.0 and r.total < 0.0


def test_capture_beats_every_failure_and_efficiency_ordering():
    direct = trajectory([100 - 8.0 * i for i in range(1, 13)], [8.0] * 12, 100.0, "soldier_caught")
    long = trajectory([100 - 3.0 * i for i in range(1, 34)], [9.0] * 33, 100.0, "soldier_caught")
    early = trajectory([100 - 8.0 * i for i in range(1, 11)], [8.0] * 10, 100.0, "intercepted")
    hover = trajectory([60.0] * 2000, [0.35] * 2000, 100.0, "timeout")
    exhaust = trajectory([60.0] * 400, [1.25] * 400, 100.0, "enemy_endurance_exhausted")
    for disc in (True, False):
        assert ret(direct, disc) > ret(long, disc) > 0.0
        assert min(ret(direct, disc), ret(long, disc)) > max(ret(early, disc), ret(hover, disc), ret(exhaust, disc))
        assert ret(hover, disc) <= ret(early, disc)  # a 2000-step hover never beats an early interception


def test_failed_v1_exploit_is_closed_for_hover_at_any_range():
    for d in (20.0, 60.0, 100.0, 150.0):
        hover = trajectory([d] * 2000, [0.35] * 2000, 100.0, "timeout")
        assert ret(hover) < -9.0  # ~ -R_F: surviving forever is worth no more than failing immediately


def test_approach_retreat_oscillation_is_bounded_and_negative():
    ranges, paths = [], []
    for i in range(80):
        ranges.append(60.0 if i % 2 else 48.0)
        paths.append(6.0)
    comps = trajectory(ranges, paths, 60.0, "enemy_endurance_exhausted")
    nonterminal = sum(c.progress + c.path + c.step for c in comps)
    assert nonterminal < 0.0 and ret(comps) < 0.0
    assert sum(GAMMA ** i * c.progress for i, c in enumerate(comps)) <= 2.0  # shaping gain is bounded by 2 * Phi_max


# ----------------------------------------------------------------------------- action contract
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("theta", R.THETA_CANDIDATES_DEG)
def test_residual_angle_never_exceeds_theta_and_never_reverses(theta):
    rng = np.random.default_rng(1)
    for _ in range(2000):
        u_b = unit(rng.normal(size=3))
        command, angle = R.residual_command(rng.uniform(-1, 1, 3), u_b, theta)
        if np.linalg.norm(command) == 0:
            continue
        measured = math.degrees(math.acos(np.clip(np.dot(unit(command), u_b), -1, 1)))
        assert measured <= theta + 1e-6 and angle <= theta + 1e-6 and abs(measured - angle) < 1e-6
        assert np.dot(command, u_b) > 0.0 and abs(np.linalg.norm(command) - 1.0) < 1e-9


def test_zero_and_parallel_actions_return_the_base_guidance_exactly():
    u_b = unit([0.3, -0.5, 0.2])
    for raw in (np.zeros(3), 0.7 * u_b, -0.9 * u_b):
        command, angle = R.residual_command(raw, u_b, 45.0)
        assert angle == 0.0 and not command.any()  # zero vector => SoldierEnv keeps its own u_base bit-for-bit


def test_maximum_residual_reaches_theta_exactly_and_speed_cannot_be_commanded():
    u_b = unit([1.0, 0.0, 0.0])
    command, angle = R.residual_command([0.0, 1.0, 0.0], u_b, 30.0)
    assert angle == pytest.approx(30.0)
    small, small_angle = R.residual_command([0.0, 0.2, 0.0], u_b, 30.0)
    assert 0.0 < small_angle < angle and abs(np.linalg.norm(small) - 1.0) < 1e-9
    # the parallel component of the raw action has no effect
    assert np.allclose(R.residual_command([0.9, 0.5, 0.0], u_b, 30.0)[0], R.residual_command([-0.9, 0.5, 0.0], u_b, 30.0)[0])


def test_oversized_perpendicular_residual_is_clamped():
    u_b = unit([1.0, 0.0, 0.0])
    _command, angle = R.residual_command([0.0, 1.0, 1.0], u_b, 45.0)  # |a_perp| = sqrt(2) -> clamped to 1
    assert angle == pytest.approx(45.0)


@pytest.mark.parametrize("scenario", [("C0", "greedy", 140001), ("C2", "greedy", 140006), ("C3", "lead", 140028), ("C1", "pn", 140010)])
def test_zero_action_reproduces_canonical_a0_exactly(scenario):
    bank = A.A1V2DefenderBank()

    def run(**kw):
        env = R.A1V2R2TrainingEnv(0, 0, defender_bank=bank, theta_max_deg=30.0, **kw)
        env.reset(options={"scenario": scenario})
        done = False
        while not done:
            _o, _r, te, tr, info = env.step(np.zeros(3, np.float32))
            done = te or tr
        return info["a1_episode"]

    a0, zero = run(scripted_a0=True), run()
    assert (a0["outcome"], a0["steps"]) == (zero["outcome"], zero["steps"]) and a0["path_length_m"] == pytest.approx(zero["path_length_m"], abs=1e-9)


def test_environment_interface_unchanged_and_reward_decomposes():
    env = R.A1V2R2TrainingEnv(201, 0, theta_max_deg=15.0)
    obs, _ = env.reset(options={"scenario": ("C1", "greedy", 140000)})
    assert env.observation_space.shape == (37,) and env.action_space.shape == (3,) and obs[-1] == 1.0
    obs, reward, _t, _u, info = env.step(np.array([0.0, 1.0, 0.0], np.float32))
    assert reward == pytest.approx(sum(info["a1_reward_components"].values())) and info["a1_reward_components"]["step"] == -0.1
    assert 0.0 < info["a1_residual_angle_deg"] <= 15.0 and info["a1_action_version"] == "a1_v2_action_v2"
    assert env.soldier_env.config.enemy_endurance_enabled and env.soldier_env.config.asset_motion_model == "legacy_gaussian_random_walk"


# ----------------------------------------------------------------------------- hover diagnostic
def test_hover_diagnostic_detects_the_failed_v1_ceiling_hover_and_not_pursuit():
    n = 600
    ceiling = R.hover_metrics(np.full(n, 0.7), np.full(n, 29.6), 30.0)  # failed-v1 signature: ~0.7 m/s at the ceiling
    assert ceiling["sustained_hover"] and ceiling["ceiling_hover"] and ceiling["near_hover_fraction"] == 1.0
    pursuit = R.hover_metrics(np.full(n, 11.0), np.full(n, 12.0), 30.0)
    assert not pursuit["sustained_hover"] and not pursuit["ceiling_hover"] and pursuit["near_hover_fraction"] == 0.0
    low = R.hover_metrics(np.full(n, 0.7), np.full(n, 12.0), 30.0)
    assert low["sustained_hover"] and not low["ceiling_hover"]
    brief = np.full(n, 11.0)
    brief[100:110] = 0.5  # a 5 s slow patch is not a hover
    assert not R.hover_metrics(brief, np.full(n, 12.0), 30.0)["sustained_hover"]
    assert R.hover_metrics(np.array([]), np.array([]), 30.0)["near_hover_fraction"] == 0.0


def test_frozen_gate_constants():
    g = R.GATES
    assert g["c0_endurance_max_rate"] == 0.05 and g["collision_max_rate_per_seed_and_level"] == 0.02
    assert "zero capture" in g["capture_every_seed_positive"] and "at least 2 of 3" in g["multi_seed_viability"]
    assert R.HOVER_SPEED_MPS == 2.0 and R.HOVER_MIN_RUN_STEPS == 20


def test_ppo_hyperparameters_and_observation_unchanged():
    desc = {"gamma": 0.99, "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
            "vf_coef": 0.5, "max_grad_norm": 0.5, "gae_lambda": 0.95}
    assert A.A1_V2_PPO_HYPERPARAMS == desc and A.A1_V2_NUM_ENVS == 16 and A.A1_V2_OBSERVATION_DIM == 37
    assert R.OBSERVATION_VERSION == "a1_v2_obs_v1" and A.A1_V2_OBSERVATION_SCHEMA[-1][0] == "remaining_endurance_fraction"
