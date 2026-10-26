"""V2 A1 protocol (`a1_v2_training_v1`, reward `a1_v2_reward_v1`): reward accounting, observation schema,
sampling distributions, seed namespaces, acceptance gates. No PPO training is executed here."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from experiments import ppo_hyperparams as H
from experiments.journal_v2_a1 import (
    A1_V2_ACTION_DIM,
    A1_V2_CLUTTER_LEVELS,
    A1_V2_CLUTTER_PROBABILITIES,
    A1_V2_DEFENDER_FAMILIES,
    A1_V2_DEFENDER_PROBABILITIES,
    A1_V2_EVAL_EPISODES,
    A1_V2_EVAL_SEED_START,
    A1_V2_OBSERVATION_DIM,
    A1_V2_OBSERVATION_SCHEMA,
    A1_V2_PPO_HYPERPARAMS,
    A1_V2_PPO_SEEDS,
    A1_V2_REWARD_VERSION,
    A1_V2_TRAIN_SCENARIO_SEED_HIGH,
    A1_V2_TRAIN_SCENARIO_SEED_LOW,
    A1_V2_TRAINING_PROTOCOL,
    A1_V2_VALIDATION_EPISODES,
    A1_V2_VALIDATION_SEED_START,
    RESERVED_EXTERNAL_SEED_RANGES,
    A1V2TrainingEnv,
    a1_v2_step_reward,
    build_episode_config,
    evaluation_seeds,
    potential,
    sample_training_scenario,
    terminal_reward,
    training_scenario_stream,
    validation_scenarios,
)
from experiments.journal_v2_a1_evaluation import (
    A1_V2_ACCEPTANCE_THRESHOLDS,
    evaluate_acceptance,
    evaluation_plan,
    run_episode,
    select_representative_episodes,
    summarize,
)
from experiments.journal_v2_protocol import (
    CanonicalV2ConfigError,
    assert_canonical_v2_training_config,
    build_v2_env_config,
)

GAMMA = A1_V2_PPO_HYPERPARAMS["gamma"]
CONFIG = build_v2_env_config("C0", "measurement")
V_E, DT = float(CONFIG.v_e), float(CONFIG.dt)
D0 = 100.0
B = 5.0 * D0


# ----------------------------------------------------------------------------- synthetic reward episodes
def episode_rewards(ranges, path_steps, outcome="soldier_caught"):
    """Per-step A1Reward list for a range trace (ranges[0] is the initial range) and per-step path (m)."""
    assert len(ranges) == len(path_steps) + 1
    steps = []
    for i, path in enumerate(path_steps):
        last = i == len(path_steps) - 1
        steps.append(a1_v2_step_reward(
            range_prev_m=ranges[i], range_next_m=ranges[i + 1], initial_range_m=ranges[0], path_step_m=path,
            path_budget_m=B, dt_s=DT, v_e_mps=V_E, outcome=outcome if last else "ongoing", gamma=GAMMA))
    return steps


def total(steps, discounted=False):
    return sum((GAMMA ** i if discounted else 1.0) * s.total for i, s in enumerate(steps))


def components(steps):
    return {k: sum(getattr(s, k) for s in steps) for k in ("terminal", "progress", "path", "time")}


def direct(distance=D0, step_m=5.0, outcome="soldier_caught", stop_at=0.0):
    n = int(math.ceil((distance - stop_at) / step_m))
    ranges = [max(distance - step_m * i, stop_at) for i in range(n + 1)]
    return ranges, [step_m] * n


def test_terminal_mapping_symmetric_and_complete():
    assert terminal_reward("soldier_caught") == 10.0
    for outcome in ("intercepted", "unsafe_intercept", "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
                    "enemy_endurance_exhausted", "timeout", "defender_collision", "anything_else"):
        assert terminal_reward(outcome) == -10.0
    assert terminal_reward("ongoing") == 0.0


def test_potential_and_progress_formula():
    assert potential(100.0, 100.0) == -1.0 and potential(0.0, 100.0) == 0.0
    r = a1_v2_step_reward(range_prev_m=80.0, range_next_m=70.0, initial_range_m=100.0, path_step_m=10.0,
                          path_budget_m=500.0, dt_s=0.05, v_e_mps=10.0, outcome="ongoing", gamma=0.99)
    assert r.progress == pytest.approx(2.0 * (0.99 * -0.7 - -0.8))
    assert r.path == pytest.approx(-1.0 * 10.0 / 500.0)
    assert r.time == pytest.approx(-0.25 * 0.05 / (500.0 / 10.0))
    assert r.terminal == 0.0 and r.total == pytest.approx(r.progress + r.path + r.time)


def test_reward_uses_ppo_gamma_by_default():
    assert GAMMA == H.GAMMA == 0.99
    a = a1_v2_step_reward(range_prev_m=50, range_next_m=50, initial_range_m=100, path_step_m=0, path_budget_m=500,
                          dt_s=0, v_e_mps=10, outcome="ongoing")
    assert a.progress == pytest.approx(2.0 * (GAMMA - 1.0) * -0.5)


def test_step_costs_scale_with_actual_path_and_time_not_commands():
    base = dict(range_prev_m=50, range_next_m=50, initial_range_m=100, path_budget_m=500, dt_s=DT, v_e_mps=V_E, outcome="ongoing")
    assert a1_v2_step_reward(path_step_m=0.0, **base).path == 0.0
    assert a1_v2_step_reward(path_step_m=20.0, **base).path == pytest.approx(2 * a1_v2_step_reward(path_step_m=10.0, **base).path)
    # a full budget of path costs exactly -1 (w_s); a full T_ref of time costs exactly -0.25 (w_t)
    full = a1_v2_step_reward(path_step_m=500.0, **{**base, "dt_s": 500.0 / V_E})
    assert full.path == pytest.approx(-1.0) and full.time == pytest.approx(-0.25)


def test_direct_capture_is_strongly_positive_and_decomposes():
    steps = episode_rewards(*direct())
    c = components(steps)
    assert c["terminal"] == 10.0 and c["progress"] > 1.9
    assert c["path"] == pytest.approx(-1.0 * 100.0 / B)
    assert total(steps) > 11.0


def test_longer_success_scores_lower_than_shorter():
    short = total(episode_rewards(*direct(step_m=5.0)))
    ranges, paths = direct(step_m=5.0)
    longer = total(episode_rewards(ranges, [p * 2.0 for p in paths]))
    assert short > longer > 0.0
    assert short - longer < 1.0  # efficiency matters but never rivals mission completion (terminal 10)


def test_circling_then_capture_scores_below_direct():
    ranges, paths = direct(step_m=5.0)
    circ_ranges = [ranges[0]] * 61 + ranges[1:]
    circ_paths = [6.0] * 60 + paths
    # PPO optimizes the DISCOUNTED return (gamma = 0.99): circling is strictly worse, mostly through the
    # discounted terminal reward and the path/time cost.
    assert total(episode_rewards(circ_ranges, circ_paths), True) < total(episode_rewards(ranges, paths), True) - 3.0
    # The UNDISCOUNTED sum carries the known bounded discount leak (gamma < 1 at constant long range); it must
    # stay far below the terminal scale.
    assert total(episode_rewards(circ_ranges, circ_paths)) - total(episode_rewards(ranges, paths)) < 1.0


def test_failures_remain_strongly_negative_even_after_progress():
    approach, paths = direct(distance=D0, stop_at=40.0)  # make 60 m of progress first
    for outcome in ("intercepted", "unsafe_intercept", "enemy_static_obstacle_collision", "timeout"):
        t = total(episode_rewards(approach, paths, outcome))
        assert t < -7.0, outcome
    # endurance exhaustion after progress and then long circling
    ranges = approach + [approach[-1]] * 60
    path_steps = paths + [(B - sum(paths)) / 60.0] * 60
    steps = episode_rewards(ranges, path_steps, "enemy_endurance_exhausted")
    assert components(steps)["progress"] > 0.0 and total(steps) < -8.0


def test_success_beats_every_failure_for_representative_traces():
    success = total(episode_rewards(*direct()))
    worst_success = total(episode_rewards(*direct(step_m=5.0)[:1], [9.0] * len(direct(step_m=5.0)[1])))  # very wasteful, still captured
    approach, paths = direct(distance=D0, stop_at=5.0)  # best possible failure: nearly reached the asset
    failures = [total(episode_rewards(approach, paths, o)) for o in ("intercepted", "enemy_static_obstacle_collision", "timeout")]
    assert min(success, worst_success) > max(failures) + 15.0


def test_temporary_retreat_does_not_prevent_positive_capture_reward():
    ranges = [100.0, 95.0, 90.0, 95.0, 100.0, 105.0, 100.0, 95.0]
    path_steps = [5.0] * 7
    r, p = direct(distance=95.0, step_m=5.0)
    ranges = ranges + r[1:]
    path_steps = path_steps + p
    steps = episode_rewards(ranges, path_steps, "soldier_caught")
    assert total(steps) > 8.0 and components(steps)["terminal"] == 10.0


@pytest.mark.parametrize("amplitude,step_m", [(5.0, 5.0), (10.0, 10.0), (2.0, 4.0)])
def test_approach_retreat_oscillation_is_not_exploitable(amplitude, step_m):
    cycles = int((B / step_m) * 0.9 / 2)  # stay below the endurance budget
    ranges, paths = [60.0], []
    for _ in range(cycles):
        ranges += [60.0 - amplitude, 60.0]
        paths += [step_m, step_m]
    steps = episode_rewards(ranges, paths, "enemy_endurance_exhausted")
    nonterminal = components(steps)["progress"] + components(steps)["path"] + components(steps)["time"]
    # Undiscounted, the leak (gamma < 1) makes a long-range cycle at most marginally positive and BOUNDED
    # (< 1.5 for the whole endurance budget, <15% of the terminal scale); it never grows with more cycles
    # because the integrated-path budget ends the episode.
    assert nonterminal < 1.5
    # Discounted (what PPO optimizes): a failed oscillation is negative and far below even a direct capture.
    assert total(steps, True) < 0.0
    assert total(steps, True) < total(episode_rewards(*direct()), True) - 10.0
    assert abs(components(steps)["progress"]) <= 2.0 * 2.0


def test_discounted_shaping_telescopes_and_terminal_dominates():
    # sum_t gamma^t (gamma Phi(s_{t+1}) - Phi(s_t)) = gamma^T Phi(s_T) - Phi(s_0)  (bounded, path-independent)
    ranges = [100.0, 130.0, 150.0, 120.0, 90.0, 60.0, 30.0, 10.0]
    paths = [10.0] * (len(ranges) - 1)
    steps = episode_rewards(ranges, paths, "intercepted")
    shaping = sum(GAMMA ** i * s.progress for i, s in enumerate(steps))
    expected = 2.0 * (GAMMA ** len(steps) * potential(ranges[-1], ranges[0]) - potential(ranges[0], ranges[0]))
    assert shaping == pytest.approx(expected)
    c = components(steps)
    assert abs(c["progress"]) + abs(c["path"]) + abs(c["time"]) < 4.0 < abs(c["terminal"])


# ----------------------------------------------------------------------------- observation / spaces
def test_observation_schema_layout_and_dimension():
    assert A1_V2_OBSERVATION_DIM == 37 and A1_V2_ACTION_DIM == 3
    names = [n for n, _ in A1_V2_OBSERVATION_SCHEMA]
    assert len(set(names)) == 37
    assert names[-1] == "remaining_endurance_fraction"
    categories = Counter(c for _, c in A1_V2_OBSERVATION_SCHEMA)
    assert categories == {"attacker_own_state": 6, "asset_state": 3, "defender_state": 7,
                          "obstacle_perception_state": 20, "mission_endurance_state": 1}


def test_environment_spaces_and_endurance_feature():
    env = A1V2TrainingEnv(201, 0)
    obs, info = env.reset(options={"scenario": ("C1", "greedy", 140000)})
    assert env.observation_space.shape == (37,) and env.action_space.shape == (3,)
    assert obs.shape == (37,) and obs.dtype == np.float32 and obs[-1] == pytest.approx(1.0)
    assert info["a1_reward_version"] == A1_V2_REWARD_VERSION
    e_prev = 1.0
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert 0.0 <= obs[-1] <= e_prev + 1e-9
        e_prev = obs[-1]
        assert reward == pytest.approx(sum(info["a1_reward_components"].values()))
        if terminated or truncated:
            break
    assert e_prev < 1.0
    soldier = env.soldier_env
    assert soldier.config.asset_motion_model == "legacy_gaussian_random_walk"
    assert soldier.config.enemy_endurance_enabled and soldier.config.enemy_endurance_path_factor == 5


def test_training_env_episode_reports_components_and_outcome():
    def asset_ward(env, obs):
        direction = obs[3:6] - obs[0:3]
        return (direction / (np.linalg.norm(direction) + 1e-9)).astype(np.float32)

    env = A1V2TrainingEnv(201, 0)
    obs, _ = env.reset(options={"scenario": ("C0", "greedy", 140000)})
    done, info = False, {}
    while not done:
        obs, _r, terminated, truncated, info = env.step(asset_ward(env, obs))
        done = terminated or truncated
    episode = info["a1_episode"]
    assert episode["outcome"] in {"soldier_caught", "intercepted", "enemy_endurance_exhausted", "timeout"}
    assert episode["return_undiscounted"] == pytest.approx(sum(episode["components"].values()))
    assert episode["components"]["terminal"] == terminal_reward(episode["outcome"])


def test_run_episode_returns_complete_record_for_a0_and_policy():
    class Straight:
        def predict(self, obs, deterministic=True):
            d = obs[3:6] - obs[0:3]
            return (d / (np.linalg.norm(d) + 1e-9)).astype(np.float32), None

    a0 = run_episode(None, "A0", "C0", "greedy", 140000)
    a1 = run_episode(Straight(), "A1-test", "C0", "greedy", 140000)
    for record in (a0, a1):
        for key in ("outcome", "path_length_m", "path_excess", "min_defender_separation_m", "captured", "endurance_exhausted",
                    "obstacle_collision", "overshoots", "orbit_revolutions_max", "avoidance_steps", "reward_components"):
            assert key in record
    assert a0["seed"] == a1["seed"] == 140000


# ----------------------------------------------------------------------------- canonical requirement
def test_episode_configs_are_canonical_for_every_cell():
    for level in A1_V2_CLUTTER_LEVELS:
        for family in A1_V2_DEFENDER_FAMILIES:
            assert_canonical_v2_training_config(build_episode_config(level, family))


@pytest.mark.parametrize("field,value", [("asset_motion_model", "gauss_markov_velocity"), ("enemy_endurance_enabled", False),
                                         ("enemy_endurance_path_factor", 3.0)])
def test_drift_from_canonical_asset_or_endurance_fails_fast(field, value):
    import dataclasses

    with pytest.raises(CanonicalV2ConfigError):
        assert_canonical_v2_training_config(dataclasses.replace(build_episode_config("C1", "greedy"), **{field: value}))


# ----------------------------------------------------------------------------- distributions and seeds
def test_probability_tables_are_exactly_uniform():
    assert A1_V2_DEFENDER_FAMILIES == ("greedy", "kalman_greedy", "pn", "lead", "rmpc")
    assert "passive" not in A1_V2_DEFENDER_FAMILIES
    assert A1_V2_DEFENDER_PROBABILITIES == (0.2,) * 5 and sum(A1_V2_DEFENDER_PROBABILITIES) == pytest.approx(1.0)
    assert A1_V2_CLUTTER_LEVELS == ("C0", "C1", "C2", "C3")
    assert A1_V2_CLUTTER_PROBABILITIES == (0.25,) * 4


def test_sampling_is_uniform_and_independent():
    rng = training_scenario_stream(201, 0)
    draws = [sample_training_scenario(rng) for _ in range(8000)]
    levels = Counter(d[0] for d in draws)
    families = Counter(d[1] for d in draws)
    assert all(abs(levels[l] / 8000 - 0.25) < 0.025 for l in A1_V2_CLUTTER_LEVELS)
    assert all(abs(families[f] / 8000 - 0.20) < 0.025 for f in A1_V2_DEFENDER_FAMILIES)
    joint = Counter((d[0], d[1]) for d in draws)
    assert len(joint) == 20 and all(abs(n / 8000 - 0.05) < 0.02 for n in joint.values())
    assert all(A1_V2_TRAIN_SCENARIO_SEED_LOW <= d[2] < A1_V2_TRAIN_SCENARIO_SEED_HIGH for d in draws)


def test_scenario_stream_is_deterministic_and_separated():
    a = [sample_training_scenario(training_scenario_stream(201, 0)) for _ in range(1)]
    b = [sample_training_scenario(training_scenario_stream(201, 0)) for _ in range(1)]
    assert a == b

    def first(seed, worker, resume=0):
        rng = training_scenario_stream(seed, worker, resume)
        return [sample_training_scenario(rng) for _ in range(20)]

    streams = {(s, w, r): first(s, w, r) for s in A1_V2_PPO_SEEDS for w in range(3) for r in range(2)}
    values = list(streams.values())
    assert all(values[i] != values[j] for i in range(len(values)) for j in range(i + 1, len(values)))


def test_training_stream_does_not_touch_global_numpy_state():
    np.random.seed(5)
    before = np.random.get_state()[1].copy()
    rng = training_scenario_stream(201, 0)
    for _ in range(10):
        sample_training_scenario(rng)
    assert (np.random.get_state()[1] == before).all()


def test_seed_namespaces_are_disjoint():
    assert list(evaluation_seeds()) == list(range(140_000, 140_100)) and A1_V2_EVAL_SEED_START == 140_000
    assert A1_V2_EVAL_EPISODES == 100
    validation = {s for _, _, s in validation_scenarios()}
    assert validation == set(range(150_000, 150_000 + A1_V2_VALIDATION_EPISODES)) and A1_V2_VALIDATION_SEED_START == 150_000
    assert validation.isdisjoint(set(evaluation_seeds()))
    assert A1_V2_TRAIN_SCENARIO_SEED_LOW > 150_000 + A1_V2_VALIDATION_EPISODES
    for lo, hi in RESERVED_EXTERNAL_SEED_RANGES.values():
        assert hi < A1_V2_TRAIN_SCENARIO_SEED_LOW
        assert hi < 140_000 or lo > 140_099
        assert hi < 150_000 or lo > 150_039


def test_validation_set_is_stratified():
    cells = Counter((lvl, fam) for lvl, fam, _ in validation_scenarios())
    assert len(cells) == 20 and set(cells.values()) == {2}


def test_ppo_hyperparameters_are_inherited_from_v1_unchanged():
    expected = dict(learning_rate=H.LEARNING_RATE, gamma=H.GAMMA, batch_size=H.BATCH_SIZE, n_steps=H.N_STEPS,
                    n_epochs=H.N_EPOCHS, clip_range=H.CLIP_RANGE, ent_coef=H.ENT_COEF, vf_coef=H.VF_COEF,
                    max_grad_norm=H.MAX_GRAD_NORM, gae_lambda=H.GAE_LAMBDA)
    assert A1_V2_PPO_HYPERPARAMS == expected
    assert (A1_V2_PPO_HYPERPARAMS["learning_rate"], A1_V2_PPO_HYPERPARAMS["batch_size"], A1_V2_PPO_HYPERPARAMS["n_steps"]) == (3e-4, 64, 2048)
    assert A1_V2_PPO_SEEDS == (201, 202, 203)
    assert A1_V2_TRAINING_PROTOCOL == "a1_v2_training_v1" and A1_V2_REWARD_VERSION == "a1_v2_reward_v1"


# ----------------------------------------------------------------------------- evaluation protocol / gates
def test_full_evaluation_plan_size_and_matched_a0():
    plan = evaluation_plan()
    assert len(plan) == (3 + 1) * 5 * 4 * 100
    assert Counter(p[0] for p in plan) == {"A1-201": 2000, "A1-202": 2000, "A1-203": 2000, "A0": 2000}
    a0 = {(l, d, s) for a, l, d, s in plan if a == "A0"}
    assert all({(l, d, s) for a, l, d, s in plan if a == att} == a0 for att in ("A1-201", "A1-202", "A1-203"))


def _record(attacker, level, defender, seed, outcome="soldier_caught", excess=1.1, sep=30.0, avoid=0):
    return dict(attacker=attacker, level=level, defender=defender, seed=seed, outcome=outcome, steps=100, flight_time_s=10.0,
                path_length_m=100.0 * excess, chord_m=100.0, path_excess=excess, initial_range_m=100.0,
                min_defender_separation_m=sep, obstacle_collision=outcome.endswith("obstacle_collision"),
                endurance_exhausted=outcome == "enemy_endurance_exhausted", captured=outcome == "soldier_caught",
                overshoots=0, orbit_revolutions_max=0.0, avoidance_steps=avoid,
                return_undiscounted=11.0 if outcome == "soldier_caught" else -9.0, reward_components={})


def _synthetic(a1_capture=0.8, a0_capture=0.5, c0_exhaust=0.0, collisions=0.0, excess=1.1):
    records = []
    for attacker, rate in (("A0", a0_capture), ("A1-201", a1_capture), ("A1-202", a1_capture), ("A1-203", a1_capture)):
        for level in A1_V2_CLUTTER_LEVELS:
            for family in A1_V2_DEFENDER_FAMILIES:
                for seed in range(20):
                    frac = seed / 20.0
                    outcome = "soldier_caught" if frac < rate else "intercepted"
                    if attacker != "A0" and level == "C0" and frac >= 1.0 - c0_exhaust:
                        outcome = "enemy_endurance_exhausted"
                    elif attacker != "A0" and frac >= 1.0 - collisions:
                        outcome = "enemy_static_obstacle_collision"
                    records.append(_record(attacker, level, family, seed, outcome, excess if outcome == "soldier_caught" else float("nan")))
    return records


def test_acceptance_gates_pass_on_a_good_policy():
    gates = evaluate_acceptance(_synthetic())
    assert gates["all_passed"], gates


@pytest.mark.parametrize("kwargs,failed_gate", [
    (dict(c0_exhaust=0.25), "c0_directness_and_endurance"),
    (dict(a1_capture=0.45, a0_capture=0.5), "mission_advantage_over_a0"),
    (dict(a1_capture=0.3, a0_capture=0.5), "seed_generalization"),
    (dict(collisions=0.10), "obstacle_safety"),
    (dict(excess=3.0), "path_efficiency"),
])
def test_acceptance_gates_reject_pathological_policies(kwargs, failed_gate):
    gates = evaluate_acceptance(_synthetic(**kwargs))
    assert not gates[failed_gate]["passed"] and not gates["all_passed"]


def test_acceptance_thresholds_are_the_pre_registered_values():
    assert A1_V2_ACCEPTANCE_THRESHOLDS["c0_max_endurance_exhaustion_rate"] == 0.05
    assert A1_V2_ACCEPTANCE_THRESHOLDS["obstacle_max_collision_rate"] == 0.02


def test_summarize_flags_path_excess_buckets():
    records = [_record("A1", "C1", "greedy", i, excess=e) for i, e in enumerate((1.0, 1.4, 2.5, 6.0))]
    summary = summarize(records)
    assert summary["path_excess_flags_given_success"] == {">1.3x": 3, ">2.0x": 2, ">5.0x": 1}


def test_representative_selection_is_deterministic_and_complete():
    records = _synthetic()
    records.append(_record("A1-201", "C3", "rmpc", 999, outcome="enemy_endurance_exhausted"))
    records.append(_record("A1-202", "C3", "lead", 998, excess=4.0, sep=8.0, avoid=50))
    picked = select_representative_episodes(records)
    assert set(picked) == {"best_mission_success", "successful_evasive", "endurance_exhaustion", "worst_path_excess",
                           "obstacle_heavy_success"}
    assert picked["endurance_exhaustion"]["seed"] == 999
    assert picked["worst_path_excess"]["seed"] == 998
    assert picked == select_representative_episodes(list(reversed(records)))
