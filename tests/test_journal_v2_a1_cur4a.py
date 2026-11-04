"""Tests for curriculum Stage 4A (moving canonical asset, no active defender). Real environments / real SB3 (small), no long training."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4 as P
from experiments import journal_v2_a1_cur4a as Y
from experiments import journal_v2_a1_cur4a_state as TS
from experiments import journal_v2_a1_cur4a_training as T4
from uav_defend.envs.soldier_env import SoldierEnv

SEEDS = (171_000, 171_001, 171_002, 171_003, 171_004, 171_005)


def rand_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda _obs: rng.uniform(-1, 1, 3).astype(np.float32)


# ----------------------------------------------------------------------------- environment definition
def test_asset_motion_is_the_exact_canonical_v2_model_and_only_the_two_curriculum_settings_differ():
    check = Y.canonical_asset_motion_check()
    assert check["identical"] and check["model_is_legacy_gaussian_random_walk"]
    p = Y.asset_motion_parameters()
    assert p["model"] == "legacy_gaussian_random_walk" and p["v_s_mps"] == 1.5 and p["dt_s"] == 0.5 and p["sigma_m_per_axis_per_step"] == 0.75 and "z = 0" in p["axes"]
    assert set(Y.config_differences_from_canonical()) == {"intercept_radius", "defender_standby_until_detection"}
    cfg = Y.stage4a_config()
    assert cfg.v_s == 1.5 and cfg.intercept_radius == 1e-6 and cfg.defender_standby_until_detection is False and not cfg.enemy_obstacle_avoidance_enabled


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_realized_asset_motion_is_the_documented_gaussian_random_walk(seed):
    env = Y.Stage4AEnv(0, 0, 0, scripted_a0=True)
    C.run_episode(env, seed, None)
    trace = env.asset_trace()
    stream = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[1])
    pos = np.zeros(3)
    for k in range(1, len(trace) - 1):
        pos = pos + np.array([*stream.normal(0.0, 0.75, size=2), 0.0])
        assert np.allclose(trace[k], pos, atol=1e-4) and trace[k][2] == 0.0
    assert len(trace) > 3


def test_defender_is_masked_parked_and_cannot_terminate_the_episode_under_the_moving_asset():
    m = Y.masking_check(SEEDS)
    assert m["max_abs_defender_observation"] == 0.0 and m["interception_outcomes"] == 0 and m["defender_slots"] == [9, 10, 11, 12, 13, 14, 15]
    env = Y.Stage4AEnv(0, 0, 0)
    obs, _ = env.reset(options={"scenario": SEEDS[0]})
    e = env.soldier_env
    e._enemy_pos = (e._defender_pos + np.array([0.5, 0.0, 0.5], dtype=np.float32)).astype(np.float32)      # 0.7 m from the structurally present defender
    obs2, _r, te, tr, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert not np.any(obs2[Y.DEFENDER_SLOTS]) and not te and info["outcome"] == "ongoing"
    for s in SEEDS[:3]:
        env.reset(options={"scenario": s})
        done = False
        while not done:
            _o, _r, te, tr, info = env.step(rand_policy(s)(None))
            done = te or tr
            assert np.allclose(env.soldier_env._defender_pos, C.PARKED_DEFENDER) and not np.any(env.soldier_env._defender_vel)
        assert info["outcome"] not in Y.INTERCEPT_OUTCOMES


def test_observation_action_reward_interfaces_are_unchanged():
    assert (Y.OBSERVATION_VERSION, Y.ACTION_VERSION, Y.REWARD_VERSION) == ("a1_v2_obs_v1", "a1_cur1_action_v1", "a1_v2_reward_v2")
    env = Y.Stage4AEnv(0, 0, 0)
    assert env.observation_space.shape == (37,) and env.action_space.shape == (3,)


# ----------------------------------------------------------------------------- common random numbers / no leakage
def test_common_random_numbers_same_asset_trajectory_for_a0_every_policy_and_the_stage4_environment():
    seeds = SEEDS[:4]
    ref = Y.asset_traces(seeds, {"a0": None, "random": rand_policy(1), "other_random": rand_policy(2)})
    report = Y.crn_report(ref)
    assert report["identical"] and report["compared_asset_positions"] > 10
    stage4 = Y.asset_traces(seeds, {"a0": None}, env_class=P.Stage4Env)["a0"]
    cross = Y.crn_report({"stage4a": ref["a0"], "stage4": stage4})
    assert cross["identical"] and cross["compared_asset_positions"] > 5            # Stage 4A realises the very same asset walk as the failed Stage 4 on the same seed


def test_same_seed_means_same_initial_state_and_no_future_information_in_the_observation():
    for s in SEEDS[:4]:
        e1, e2 = Y.Stage4AEnv(0, 0, 0, scripted_a0=True), Y.Stage4AEnv(7, 3, 5)
        o1, _ = e1.reset(options={"scenario": s})
        o2, _ = e2.reset(options={"scenario": s})
        assert np.array_equal(o1, o2) and np.array_equal(e1.soldier_env._enemy_pos, e2.soldier_env._enemy_pos) and np.array_equal(e1.soldier_env._enemy_vel, e2.soldier_env._enemy_vel)
        env = e1.soldier_env
        for _ in range(4):
            obs, _r, te, tr, _i = e1.step(np.zeros(3, dtype=np.float32))
            if te or tr:
                break
            assert np.allclose(obs[3:6], np.asarray(env._soldier_pos) / float(env.config.L), atol=1e-6)      # the CURRENT asset position only
    names_ = [n for n, _g in A.A1_V2_OBSERVATION_SCHEMA]
    assert not any("future" in n or "predicted" in n or "asset_vel" in n for n in names_) and len(names_) == 37


def test_evaluation_is_deterministic_and_replays_exactly():
    a = Y.evaluate(None, SEEDS, scripted_a0=True)
    b = Y.evaluate(None, SEEDS, scripted_a0=True)
    keys = ("outcome", "steps", "asset_final_position", "path_length_m", "closest_approach_count")
    assert [{k: e[k] for k in keys} for e in a] == [{k: e[k] for k in keys} for e in b]
    assert all(e["asset_moving"] and e["asset_displacement_max_m"] > 0.0 for e in a)


def test_asset_increments_have_the_canonical_scale():
    env = SoldierEnv(Y.stage4a_config())
    env.reset(seed=7)
    pos, inc = [env._soldier_pos.copy()], []
    for _ in range(600):
        env._move_soldier()
        inc.append(env._soldier_pos[:2] - pos[-1][:2])
        pos.append(env._soldier_pos.copy())
    inside = np.abs(np.array(pos)[1:, :2]).max(axis=1) < 45.0
    assert abs(np.array(inc)[inside].std() - 0.75) < 0.06


# ----------------------------------------------------------------------------- metrics / gate / retention
def test_closest_approach_and_miss_distance_extraction():
    count, misses = Y.closest_approaches([30, 20, 10, 4, 5, 12, 6, 3.4, 5, 9])
    assert count == 2 and misses == [4.0, 3.4]
    count, misses = Y.closest_approaches([30, 20, 10, 4, 1])            # closing to the end: one (terminal) approach, no miss
    assert count == 1 and misses == []


def test_metrics_report_gt3_first_pass_and_repeated_flybys():
    def ep(outcome, pe, overshoots, misses):
        return {"outcome": outcome, "path_excess": pe, "overshoots": overshoots, "orbit_revolutions_max": 0.0, "nonclosing_run": 0, "pathology": False, "finite": True, "invalid_actions": 0,
                "held_steps": 0, "duration_s": 5.0, "path_length_m": 50.0, "terminal_distance_m": 1.0, "return_undiscounted": 0.0, "return_discounted": 0.0,
                "components": {"terminal": 0.0, "progress": 0.0, "path": 0.0, "step": 0.0}, "asset_displacement_max_m": 2.0, "asset_final_displacement_m": 1.0,
                "min_defender_separation_m": 30.0, "closest_approach_count": overshoots + 1, "miss_distances_m": misses}
    eps = [ep("soldier_caught", 1.0, 0, []), ep("soldier_caught", 1.1, 1, [3.4]), ep("soldier_caught", 3.5, 2, [3.0, 3.8]), ep("enemy_endurance_exhausted", 6.0, 4, [3.0] * 4)]
    m = Y.metrics(eps)
    assert m["capture"] == 0.75 and m["gt3"] == 0.5 and m["gt5"] == 0.25 and m["repeated_flyby_fraction"] == 0.5 and m["first_pass_capture"] == 0.25
    assert m["first_pass_fraction_of_captures"] == pytest.approx(1 / 3) and m["miss_distance_count"] == 7 and m["interception"] == 0.0


def test_namespaces_are_fresh_and_disjoint_and_the_gate_is_frozen():
    o = Y.namespace_overlaps()
    for key in ("validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "baseline", "mutual", "training_overlap"):
        assert o[key] == [], key
    assert o["suggested_block_166000_166199"], "the suggested block is occupied (Stage-4 anchors) and must be documented as the reason for the new block"
    assert Y.VALIDATION_A == (169_000, 169_099) and Y.VALIDATION_B == (169_100, 169_199) and len(Y.final_evaluation_seeds()) == 200 and len(Y.seeds_of(Y.BASELINE)) == 1000
    assert Y.GATE == {"capture_min": 0.95, "exhaustion_max": 0.02, "timeout_max": 0.01, "median_path_excess_max": 1.50, "gt2_max": 0.05, "gt3_max": 0.02, "gt5_max": 0.0, "orbit_episodes_max": 0,
                      "repeated_flyby_max": 0.05, "anchor1_capture_min": 0.98, "anchor2_capture_min": 0.80, "anchor3_capture_min": 0.75, "nonfinite_max": 0}
    assert Y.MAX_ROLLOUTS == 12 and T4.SPEC.max_timesteps == 393_216 and T4.TORCH_THREADS == 2


def _metrics(**kw):
    base = {"capture": 0.95, "exhaustion": 0.02, "timeout": 0.01, "median_path_excess": 1.5, "gt2": 0.05, "gt3": 0.02, "gt5": 0.0, "orbit_episodes": 0, "repeated_flyby_fraction": 0.05, "nonfinite": 0}
    return base | kw


def test_gate_boundaries():
    kw = dict(a0_capture=1.0, anchor1=0.98, anchor2=0.80, anchor3=0.75)
    assert Y.gate_status(_metrics(), **kw)["passed"]
    for key, bad in (("capture", 0.94), ("exhaustion", 0.03), ("timeout", 0.02), ("median_path_excess", 1.51), ("gt2", 0.06), ("gt3", 0.03), ("gt5", 0.01), ("orbit_episodes", 1),
                     ("repeated_flyby_fraction", 0.06), ("nonfinite", 1)):
        assert not Y.gate_status(_metrics(**{key: bad}), **kw)["passed"], key
    for key, bad in (("anchor1", 0.97), ("anchor2", 0.79), ("anchor3", 0.74)):
        assert not Y.gate_status(_metrics(), **{**kw, key: bad})["passed"], key
    assert Y.gate_status(_metrics(), **{**kw, "a0_capture": 0.0})["passed"]                      # no improvement-over-A0 requirement


def _val(passed, a1=1.0, a2=1.0, a3=1.0):
    return {"passed": passed, "anchor1_capture": a1, "anchor2_capture": a2, "anchor3_capture": a3}


def test_seed_decision_mastery_and_any_anchor_retention_failure():
    assert T4.seed_decision([_val(True), _val(False), _val(True)]) == (None, None)
    assert T4.seed_decision([_val(False), _val(True), _val(True)]) == ("MASTERED", 3)
    assert T4.seed_decision([_val(False, a1=0.97), _val(False, a1=0.9)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a2=0.79), _val(False, a2=0.5)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a3=0.74), _val(False, a3=0.7)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a3=0.74), _val(False, a3=0.9), _val(False, a3=0.7)]) == (None, None)
    assert T4.seed_decision([_val(True, a3=0.7), _val(True, a3=0.7)]) == ("MASTERED", 2)
    assert T4.failed_anchors([_val(False, a3=0.74), _val(False, a3=0.7)], 2) == ["anchor3_moving_defender"]


def test_three_anchors_are_the_unchanged_earlier_stage_tasks():
    from stable_baselines3 import PPO

    act = C.model_actor(PPO.load(Y.parent_checkpoint(203)["path"].removesuffix(".zip"), device="cpu"))
    a1 = Y.evaluate_anchor1(act, range(169_200, 169_210))
    a2 = Y.evaluate_anchor2(act, range(169_300, 169_310))
    a3 = Y.evaluate_anchor3(act, range(169_400, 169_410))
    assert all(e["asset_displacement_max_m"] == 0.0 for e in a1 + a2 + a3)                    # stationary asset in all three anchors
    assert all("release_step" in e for e in a3) and all("release_step" not in e for e in a2)
    assert not any("defender_position" in e for e in a1)


def test_baseline_verdict_is_configuration_validity_only():
    def summ(parent_capture):
        s = {"asset_displacement_max_m": {"median": 2.0}, "interception": 0.0, "nonfinite_episodes": 0, "capture": 1.0}
        return {"a0": dict(s), "parents": {"201": dict(s, capture=parent_capture), "202": dict(s, capture=parent_capture)}, "masking": {"max_abs_defender_observation": 0.0, "interception_outcomes": 0},
                "crn": {"identical": True}, "crn_vs_stage4_environment": {"identical": True}, "deterministic_replay_first_50": True}
    assert Y.baseline_verdict(summ(0.1))["passed"] and Y.baseline_verdict(summ(0.99))["passed"]      # difficulty never blocks training
    bad = copy.deepcopy(summ(0.9))
    bad["masking"]["max_abs_defender_observation"] = 0.1
    assert not Y.baseline_verdict(bad)["passed"]
    bad = copy.deepcopy(summ(0.9))
    bad["crn"]["identical"] = False
    assert not Y.baseline_verdict(bad)["passed"]


# ----------------------------------------------------------------------------- parents / protocol / restart
@pytest.mark.parametrize("seed", [201, 202, 203])
def test_stage3_parents_are_the_preregistered_hashes_and_not_the_failed_stage4_models(seed):
    p = Y.parent_checkpoint(seed)
    assert p["model_sha256"] == Y.PARENT_SHA256[seed] and p["stage3_v2_training_status"] == "MASTERED" and "stage3_v2" in p["path"] and "stage4" not in p["path"].replace("stage4a", "")
    assert p["model_sha256"] != Y.STAGE4_ACCEPTED_FINAL_SHA256[seed]


def test_parent_initialisation_copies_actor_critic_log_std_and_resets_the_optimizer():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from experiments import journal_v2_a1_cur1v2_bc as BC

    kwargs = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=32, batch_size=32)
    env = DummyVecEnv([lambda: Y.make_env(201, 0, 0)])
    model, record = T4.initialize_model(201, env, kwargs)
    parent = PPO.load(Y.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    assert BC.tensor_sha256(BC.actor_state(model)) == BC.tensor_sha256(BC.actor_state(parent)) and BC.value_state_sha256(model) == BC.value_state_sha256(parent)
    assert np.array_equal(model.policy.log_std.detach().numpy(), parent.policy.log_std.detach().numpy())
    assert model.num_timesteps == 0 and len(model.policy.optimizer.state) == 0 and len(parent.policy.optimizer.state) > 0
    assert record["parent"]["model_sha256"] == Y.PARENT_SHA256[201] and record["protocol_sha256"] == Y.protocol_sha256()
    assert "failed Stage-4 models" in record["not_loaded"] or "Stage-2 models" in record["not_loaded"]
    env.close()


def test_protocol_decomposes_stage4_changes_only_the_defender_side_and_is_stable():
    spec = Y.protocol_specification()
    assert spec["reward"].startswith("a1_v2_reward_v2 unchanged") and spec["environment"]["asset_motion"]["model"] == "legacy_gaussian_random_walk" and spec["ppo"]["torch_threads"] == 2
    assert spec["decomposition"]["failed_stage4"]["status"].startswith("non-canonical") and "failed Stage-4 models" in spec["initialization"]["not_loaded"]
    assert spec["parents"]["status"] == "quantitatively_accepted_curriculum_parent" and "166000-166199" in spec["validation"]["suggested_block_conflict"]
    assert Y.protocol_sha256() == Y.protocol_sha256() != P.protocol_sha256() and spec["training_budget"]["max_timesteps"] == 393_216


def test_rollouts_of_another_protocol_are_never_accepted(tmp_path, monkeypatch):
    spec = T4.Spec(n_envs=2, n_steps=32, max_rollouts=2)
    layout = T4.make_layout(201, tmp_path, tmp_path / "res")
    layout.scratch_rollouts.mkdir(parents=True)

    class _M:
        num_timesteps, _n_updates = 64, 1

        def save(self, buf):
            buf.write(b"model")

    monkeypatch.setattr(T4, "state_sha256", lambda m: "0" * 64)
    monkeypatch.setattr(T4, "summarize_training", lambda e: {})
    monkeypatch.setattr(T4.T, "capture_rng", lambda: b"rng")
    manifest = T4.commit_rollout(layout, 1, _M(), [], spec=spec, ppo_kwargs={}, parent={"model_sha256": "a" * 64})
    assert T4.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=spec) == manifest and manifest["parent_sha256"] == "a" * 64
    monkeypatch.setattr(T4, "_PROTOCOL_SHA", "f" * 64)
    assert T4.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=spec) is None


def test_state_machine_has_the_fifteen_required_stages_and_isolated_output(tmp_path):
    assert len(TS.STAGES) == 15 and TS.STAGES[0] == "01_repository_verification" and TS.STAGES[-1] == "15_commit_push" and TS.STAGES[12] == "13_stage4b_handoff"
    assert str(Y.RESULTS_ROOT).endswith("stage4a") and TS.STATE_DIR.name == "stage4a" and str(T4.DEFAULT_SCRATCH).endswith("stage4a")
    with pytest.raises(T4.RunConflictError):
        T4.make_layout(201, tmp_path, T4.PROJECT_ROOT / "results" / "elsewhere")
    assert Y.RESULTS_ROOT != P.RESULTS_ROOT and Y.RESULTS_ROOT != X.RESULTS_ROOT


def test_real_sb3_resume_with_parent_initialisation_matches_uninterrupted_run(tmp_path):
    result = T4.resume_test(tmp_path / "rt", n_envs=2, n_steps=32, rollouts=3, interrupt_after=1)
    assert result["all_states_equal"] and result["all_episode_logs_equal"] and result["init_record_equal"] and result["crash_injected_between_model_write_and_manifest"]
    assert result["environment"].startswith("Stage4AEnv")


# ----------------------------------------------------------------------------- final-evaluation rules
def _seed_result(capture=0.97, ci_lo=5.0, fly=0.02, pfly=0.30, anchors=(1.0, 0.9, 0.8), orbit=0, rep=1, episodes=200):
    return {"metrics": {"capture": capture, "repeated_flyby_fraction": fly, "orbit_episodes": orbit, "repeated_flyby_episodes": rep, "nonclosing_episodes": 0, "nonfinite": 0, "episodes": episodes},
            "parent_metrics": {"repeated_flyby_fraction": pfly}, "paired_capture_vs_parent": {"ci95_pp": [ci_lo, ci_lo + 10]},
            "anchor_metrics": {k: {"capture": v} for k, v in zip(("anchor1", "anchor2", "anchor3"), anchors)}}


def test_decision_rules_pass_B_and_C():
    from experiments import journal_v2_a1_cur4a_final as F

    ok = {s: {"status": "MASTERED"} for s in (201, 202, 203)}
    per = {s: _seed_result() for s in (201, 202, 203)}
    assert F.decide(per, ok)["verdict"] == "A"
    two = {201: {"status": "MASTERED"}, 202: {"status": "MASTERED"}, 203: {"status": "FAIL"}}
    assert F.decide(per, two)["stage4a_pass"]
    assert not F.decide(per, {201: {"status": "MASTERED"}, 202: {"status": "FAIL"}, 203: {"status": "FAIL"}})["stage4a_pass"]
    assert F.decide(per, {201: {"status": "MASTERED"}, 202: {"status": "FAIL"}, 203: {"status": "FAIL"}})["verdict"] == "B"               # >= 1 mastered but < 2 => needs manual review
    lost = per | {201: _seed_result(anchors=(1.0, 0.9, 0.7))}                                                                                  # a mastered seed that lost an anchor does not count
    assert not F.decide(lost, {201: {"status": "MASTERED"}, 202: {"status": "MASTERED"}, 203: {"status": "FAIL"}})["stage4a_pass"]
    flat = per | {202: _seed_result(ci_lo=-1.0)}                                                                                              # mastered but no tracking improvement over the parent
    assert not F.decide(flat, {201: {"status": "MASTERED"}, 202: {"status": "MASTERED"}, 203: {"status": "FAIL"}})["stage4a_pass"]
    shared = {s: _seed_result(rep=10) for s in (201, 202, 203)}                                                                               # 5 % repeated fly-bys in >= 2 mastered seeds
    assert F.decide(shared, ok)["common_structural_pathology"] and not F.decide(shared, ok)["stage4a_pass"]
    fail = {s: {"status": "FAIL"} for s in (201, 202, 203)}
    weak = {s: _seed_result(capture=0.8, ci_lo=2.0) for s in (201, 202, 203)}
    assert F.decide(weak, fail)["verdict"] == "C"
    strong = {s: _seed_result(capture=0.92, ci_lo=2.0) for s in (201, 202, 203)}
    assert F.decide(strong, fail)["verdict"] == "B"


def test_tracking_row_distinguishes_current_from_obsolete_target():
    from experiments import journal_v2_a1_cur4a_final as F

    asset = np.array([[0.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]])            # the asset has moved 10 m along +x
    pos = np.array([[0.0, 30.0, 0], [0.0, 20.0, 0], [0.0, 10.0, 0], [0.0, 3.0, 0]])      # attacker approaching along -y, offset from the asset
    unit = lambda v: v / np.linalg.norm(v)                                                  # noqa: E731
    ep = {"scenario_seed": 1, "outcome": "soldier_caught", "steps": 3, "overshoots": 0, "closest_approach_count": 1, "miss_distances_m": [], "asset_final_displacement_m": 10.0,
          "asset_displacement_max_m": 10.0, "path_excess": 1.0}

    def trace(direction_to):
        u = np.array([unit(direction_to(a, p)) for a, p in zip(asset[:3], pos[:3])])
        los = np.array([unit(a - p) for a, p in zip(asset[:3], pos[:3])])
        return {"pos": pos, "u": u, "los": los, "asset_at_decision": asset[:3], "asset": asset[3], "asset_trace": asset}

    current = F.tracking_row(trace(lambda a, p: a - p), ep)
    obsolete = F.tracking_row(trace(lambda a, p: asset[0] - p), ep)
    assert current["cos_current_mean"] == pytest.approx(1.0) and current["cos_initial_mean"] < 1.0 and current["informative_current_better"] == current["informative_decisions"] > 0
    assert obsolete["cos_initial_mean"] == pytest.approx(1.0) and obsolete["cos_current_mean"] < 1.0 and obsolete["informative_current_better"] == 0
    summary = F.summarize_tracking([current, obsolete])
    assert summary["informative_fraction_current_better"] == pytest.approx(0.5) and summary["first_pass_fraction"] == 1.0
