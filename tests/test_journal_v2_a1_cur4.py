"""Tests for curriculum Stage 4 (moving canonical asset + hold-until-detection Greedy defender)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4 as Y
from experiments import journal_v2_a1_cur4_state as TS
from experiments import journal_v2_a1_cur4_training as T4
from uav_defend.envs.soldier_env import SoldierEnv

SEEDS = (168_000, 168_001, 168_002, 168_003, 168_004, 168_005)


def rand_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda _obs: rng.uniform(-1, 1, 3).astype(np.float32)


def test_asset_motion_is_the_exact_canonical_v2_model_and_only_standby_differs():
    check = Y.canonical_asset_motion_check()
    assert check["identical"] and check["model_is_legacy_gaussian_random_walk"]
    p = Y.asset_motion_parameters()
    assert p["model"] == "legacy_gaussian_random_walk" and p["v_s_mps"] == 1.5 and p["dt_s"] == 0.5 and p["sigma_m_per_axis_per_step"] == 0.75
    assert set(Y.config_differences_from_canonical()) == {"defender_standby_until_detection"}
    cfg = Y.stage4_config()
    assert cfg.detection_radius == 15.0 and cfg.v_s == 1.5 and cfg.intercept_radius == 2.5 and cfg.defender_standby_until_detection is False


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_realized_asset_motion_is_the_documented_gaussian_random_walk(seed):
    env = Y.Stage4Env(0, 0, 0, scripted_a0=True)
    env.reset(options={"scenario": seed})
    C.run_episode(env, seed, None)
    trace = env.asset_trace()
    stream = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[1])
    pos = np.zeros(3)
    for k in range(1, len(trace) - 1):                       # completed steps (the terminal element is the swept-contact event state)
        pos = pos + np.array([*stream.normal(0.0, 0.75, size=2), 0.0])
        assert np.allclose(trace[k], pos, atol=1e-4) and trace[k][2] == 0.0
    assert len(trace) > 3 and np.abs(trace).max() < 50.0


def test_asset_increments_have_the_canonical_scale_and_no_velocity_persistence():
    env = SoldierEnv(Y.stage4_config())
    env.reset(seed=7)
    pos, inc = [env._soldier_pos.copy()], []
    for _ in range(600):
        env._move_soldier()
        inc.append(env._soldier_pos[:2] - pos[-1][:2])
        pos.append(env._soldier_pos.copy())
    inc = np.array(inc)
    inside = np.abs(np.array(pos)[1:, :2]).max(axis=1) < 45.0              # ignore steps near the reflecting boundary
    assert abs(inc[inside].std() - 0.75) < 0.06
    lag1 = float(np.mean(inc[1:][inside[1:] & inside[:-1]] * inc[:-1][inside[1:] & inside[:-1]]))
    assert abs(lag1) < 0.08                                                  # i.i.d. displacements (lag-1 autocovariance ~ 0 vs variance 0.5625)


@pytest.mark.parametrize("controller", ["greedy", "kalman_greedy", "pn", "lead"])
def test_common_random_numbers_same_asset_trajectory_for_every_attacker_policy_and_defender(controller):
    seeds = SEEDS[:4]
    policies = {"a0": None, "random": rand_policy(1), "other_random": rand_policy(2)}
    ref = Y.asset_traces(seeds, policies)
    report = Y.crn_report(ref)
    assert report["identical"] and report["compared_asset_positions"] > 10
    env = Y.Stage4Env(0, 0, 0, scripted_a0=True, controller=controller)
    for s in seeds:
        C.run_episode(env, s, None)
        a, b = env.asset_trace(), ref["a0"][s]
        k = min(len(a), len(b)) - 1
        assert np.array_equal(a[:k], b[:k])                                   # independent of the defender controller as well


def test_asset_motion_is_independent_of_the_hold_wrapper_and_defender_placement_stream():
    for s in SEEDS[:3]:
        e1, e2 = Y.Stage4Env(0, 0, 0, scripted_a0=True), Y.Stage4Env(0, 0, 0, scripted_a0=True, hold_until_detection=False)
        C.run_episode(e1, s, None)
        C.run_episode(e2, s, None)
        a, b = e1.asset_trace(), e2.asset_trace()
        k = min(len(a), len(b)) - 1
        assert np.array_equal(a[:k], b[:k])


def test_no_future_information_leakage_in_placement_and_observation():
    for s in SEEDS[:4]:
        e3 = X.Stage3V2Env(0, 0, 0, scripted_a0=True)
        e3.reset(options={"scenario": s})
        e4 = Y.Stage4Env(0, 0, 0, scripted_a0=True)
        obs0, _ = e4.reset(options={"scenario": s})
        assert np.array_equal(e3._env._defender_pos, e4._env._defender_pos) and e3.placement["lambda"] == e4.placement["lambda"] and e3.placement["offset_m"] == e4.placement["offset_m"]
        assert np.array_equal(e3._env._enemy_pos, e4._env._enemy_pos) and np.array_equal(e3._env._enemy_vel, e4._env._enemy_vel)       # same initial attacker state
        path3 = S.nominal_a0_path(s)
        assert np.allclose(path3[0], e4._env._enemy_pos, atol=1e-4)           # nominal path comes from the INITIAL state with a stationary asset
        env = e4._env
        for _ in range(4):
            obs, _r, te, tr, _i = e4.step(np.zeros(3, dtype=np.float32))
            if te or tr:
                break
            assert np.allclose(obs[3:6], np.asarray(env._soldier_pos) / float(env.config.L), atol=1e-6)    # the CURRENT asset position, nothing from the future
    names_ = [n for n, _g in A.A1_V2_OBSERVATION_SCHEMA]
    assert not any("future" in n or "predicted" in n or "asset_vel" in n for n in names_) and len(names_) == 37


def test_hold_until_detection_semantics_are_unchanged_under_the_moving_asset():
    for s in SEEDS[:4]:
        env = Y.Stage4Env(0, 0, 0, scripted_a0=True)
        env.reset(options={"scenario": s})
        e = env._env
        p0, held, ep = e._defender_pos.copy(), True, None
        done = False
        while not done:
            _o, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            done = te or tr
            ep = info.get("a1_episode") or ep
            if not env.released:
                assert np.array_equal(e._defender_pos, p0) and not e._defender_vel.any()
        assert ep["defender_speed_at_release_mps"] == 0.0 and ep["defender_moved_before_release_m"] == 0.0 and ep["first_controller_step"] == ep["detection_step"] + 1
    assert X.detection_radius() == 15.0 and type(env._policy).__name__ == "GreedyInterceptPolicy"


def test_validation_is_deterministic_and_replays_exactly():
    a = Y.evaluate(None, SEEDS, scripted_a0=True)
    b = Y.evaluate(None, SEEDS, scripted_a0=True)
    keys = ("outcome", "steps", "detection_step", "release_step", "defender_final_position", "asset_final_position", "path_length_m")
    assert [{k: e[k] for k in keys} for e in a] == [{k: e[k] for k in keys} for e in b]
    assert all(e["asset_moving"] and e["asset_displacement_max_m"] > 0.0 for e in a)


def test_namespaces_are_fresh_and_disjoint_and_the_gate_is_frozen():
    o = Y.namespace_overlaps()
    for key in ("validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "diagnostic", "scenario_validation", "mutual", "training_overlap"):
        assert o[key] == [], key
    assert Y.VALIDATION_A == (165_000, 165_099) and Y.VALIDATION_B == (165_100, 165_199) and len(Y.final_evaluation_seeds()) == 200
    assert Y.GATE == {"capture_min": 0.70, "improvement_over_a0_min_pp": 20.0, "interception_max": 0.30, "timeout_max": 0.03, "exhaustion_max": 0.03, "median_path_excess_max": 2.00, "gt3_max": 0.05,
                      "gt5_max": 0.0, "orbit_episodes_max": 0, "anchor1_capture_min": 0.98, "anchor2_capture_min": 0.80, "anchor3_capture_min": 0.75, "nonfinite_max": 0}


def test_gate_boundaries():
    m = {"capture": 0.70, "interception": 0.30, "timeout": 0.03, "exhaustion": 0.03, "median_path_excess": 2.0, "gt3": 0.05, "gt5": 0.0, "orbit_episodes": 0, "nonfinite": 0}
    kw = dict(a0_capture=0.0, anchor1=0.98, anchor2=0.80, anchor3=0.75)
    assert Y.gate_status(m, **kw)["passed"]
    for key, bad in (("capture", 0.69), ("interception", 0.31), ("timeout", 0.04), ("exhaustion", 0.04), ("median_path_excess", 2.01), ("gt3", 0.06), ("gt5", 0.01), ("orbit_episodes", 1), ("nonfinite", 1)):
        assert not Y.gate_status({**m, key: bad}, **kw)["passed"], key
    assert not Y.gate_status(m, **{**kw, "a0_capture": 0.51})["passed"]
    for key, bad in (("anchor1", 0.97), ("anchor2", 0.79), ("anchor3", 0.74)):
        assert not Y.gate_status(m, **{**kw, key: bad})["passed"], key


def _val(passed, a1=1.0, a2=1.0, a3=1.0):
    return {"passed": passed, "anchor1_capture": a1, "anchor2_capture": a2, "anchor3_capture": a3}


def test_seed_decision_mastery_and_any_anchor_retention_failure():
    assert T4.seed_decision([_val(True), _val(False), _val(True)]) == (None, None)
    assert T4.seed_decision([_val(False), _val(True), _val(True)]) == ("MASTERED", 3)
    assert T4.seed_decision([_val(False, a1=0.97), _val(False, a1=0.9)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a2=0.79), _val(False, a2=0.5)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a3=0.74), _val(False, a3=0.7)]) == ("CURRICULUM_RETENTION_FAILURE", 2)
    assert T4.seed_decision([_val(False, a3=0.74), _val(False, a3=0.9), _val(False, a3=0.7)]) == (None, None)
    assert T4.seed_decision([_val(True, a3=0.7), _val(True, a3=0.7)]) == ("MASTERED", 2)                  # mastery wins a tie
    assert T4.failed_anchors([_val(False, a3=0.74), _val(False, a3=0.7)], 2) == ["anchor3_moving_defender"]


def test_scenario_validation_verdict_rules():
    base = {"a0": {"capture": 0.0, "interception": 1.0, "nonfinite_episodes": 0, "asset_displacement_max_m": {"median": 2.0}},
            "a0_hold_release_validity": {"episodes_steps_le_1": 0, "intercepted_with_fewer_than_2_visible_decisions": 0, "max_defender_speed_at_release_mps": 0.0, "max_defender_displacement_before_release_m": 0.0,
                                         "released_episodes_with_time_after_release": 5, "released_episodes_where_greedy_moved_defender": 5},
            "parents": {"201": {"capture": 0.5, "nonfinite_episodes": 0}, "202": {"capture": 0.1, "nonfinite_episodes": 0}}, "deterministic_replay_first_50": True, "crn": {"identical": True}}
    assert Y.validation_verdict(base)["passed"]
    dominated = copy.deepcopy(base)
    dominated["a0"].update(capture=0.9, interception=0.05)
    assert Y.validation_verdict(dominated)["a0_dominates"] and not Y.validation_verdict(dominated)["passed"]
    collapsed = copy.deepcopy(base)
    collapsed["parents"] = {k: {"capture": 0.2, "nonfinite_episodes": 0} for k in ("201", "202")}
    assert Y.validation_verdict(collapsed)["parents_collapse"] and not Y.validation_verdict(collapsed)["passed"]
    no_crn = copy.deepcopy(base)
    no_crn["crn"] = {"identical": False}
    assert not Y.validation_verdict(no_crn)["passed"]


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_stage3_parents_are_the_preregistered_hashes(seed):
    p = Y.parent_checkpoint(seed)
    assert p["model_sha256"] == Y.PARENT_SHA256[seed] and p["parent_status"] == "quantitatively_accepted_curriculum_parent" and p["stage3_v2_training_status"] == "MASTERED"
    assert p["rollout"] == {201: 3, 202: 5, 203: 7}[seed]


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
    env.close()


def test_three_anchors_are_the_unchanged_earlier_stage_tasks():
    from stable_baselines3 import PPO

    act = C.model_actor(PPO.load(Y.parent_checkpoint(203)["path"].removesuffix(".zip"), device="cpu"))
    a1 = Y.evaluate_anchor1(act, range(166_000, 166_010))
    a2 = Y.evaluate_anchor2(act, range(166_100, 166_110))
    a3 = Y.evaluate_anchor3(act, range(166_200, 166_210))
    assert all(e["asset_displacement_max_m"] == 0.0 for e in a1 + a2 + a3)                    # stationary asset in all three anchors
    assert all("release_step" in e for e in a3) and all("release_step" not in e for e in a2)
    assert not any("defender_position" in e for e in a1)                                        # anchor 1 has no defender


def test_protocol_documents_exactly_one_conceptual_change_and_is_stable():
    spec = Y.protocol_specification()
    v3 = X.protocol_specification()
    assert "moves with the canonical V2 asset-motion model" in spec["single_conceptual_change_from_stage3_v2"]
    assert spec["reward"].startswith("a1_v2_reward_v2 unchanged") and v3["reward"].startswith("a1_v2_reward_v2 unchanged")
    assert spec["environment"]["defender"] == v3["environment"]["defender"] and spec["parents"]["status"] == "quantitatively_accepted_curriculum_parent"
    assert spec["environment"]["asset_motion"]["model"] == "legacy_gaussian_random_walk" and spec["environment"]["detection_radius_m"] == 15.0 and spec["ppo"]["torch_threads"] == 2
    assert Y.protocol_sha256() == Y.protocol_sha256() != X.protocol_sha256() and T4.SPEC.max_timesteps == 524_288


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
    assert len(TS.STAGES) == 15 and TS.STAGES[0] == "01_verify_stage3_parent" and TS.STAGES[-1] == "15_commit_push"
    assert str(Y.RESULTS_ROOT).endswith("stage4") and "stage4" in str(TS.STATE_DIR)
    with pytest.raises(T4.RunConflictError):
        T4.make_layout(201, tmp_path, T4.PROJECT_ROOT / "results" / "elsewhere")


def test_real_sb3_resume_with_parent_initialisation_matches_uninterrupted_run(tmp_path):
    result = T4.resume_test(tmp_path / "rt", n_envs=2, n_steps=32, rollouts=3, interrupt_after=1)
    assert result["all_states_equal"] and result["all_episode_logs_equal"] and result["init_record_equal"] and result["crash_injected_between_model_write_and_manifest"]
