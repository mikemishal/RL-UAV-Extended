"""Tests for curriculum Stage3-v2 (moving Greedy defender, hold-until-detection)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3 as V1
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur3v2_state as TS
from experiments import journal_v2_a1_cur3v2_training as T3
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info

SEEDS = (157_000, 157_001, 157_002, 157_003, 157_004, 157_005)


def trace(seed: int, hold: bool = True):
    env = X.Stage3V2Env(0, 0, 0, scripted_a0=True, hold_until_detection=hold)
    env.reset(options={"scenario": int(seed)})
    e = env._env
    rows = [{"pos": e._defender_pos.copy(), "vel": e._defender_vel.copy(), "detected": bool(e._enemy_detected), "att": e._enemy_pos.copy(), "action": None, "released": env.released}]
    done, info = False, {}
    while not done:
        _o, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        done = te or tr
        rows.append({"pos": e._defender_pos.copy(), "vel": e._defender_vel.copy(), "detected": bool(e._enemy_detected), "att": e._enemy_pos.copy(), "action": env._def_action.copy(),
                     "released": env.released, "info": dict(e._defender_dynamics_info)})
    return env, rows, info["a1_episode"]


def test_detection_radius_and_config_are_the_canonical_ones():
    cfg = X.stage3v2_config()
    assert cfg.detection_radius == 15.0 and cfg.v_s == 0.0 and cfg.defender_standby_until_detection is False and cfg.intercept_radius == 2.5
    assert cfg.v_d == 18.0 and cfg.defender_max_accel == 8.0 and cfg.defender_max_turn_rate_deg == 90.0
    assert X.config_sha256() == S.config_sha256()                         # identical to the Stage-2 / Stage3-v1 configuration (the hold lives in the wrapper)
    assert X.namespace_overlaps()["training_overlap"] == []


@pytest.mark.parametrize("seed", SEEDS)
def test_defender_is_exactly_fixed_with_zero_velocity_and_acceleration_before_release(seed):
    env, rows, ep = trace(seed)
    release = ep["release_step"]
    assert release is not None and ep["detection_step"] == release
    for r in rows[: release + 1]:                                          # includes the transition step: detection became true but the action is not yet executed
        assert np.array_equal(r["pos"], rows[0]["pos"]) and not r["vel"].any()
    for r in rows[1: release + 1]:
        assert r["info"]["accel_used"] == 0.0 and r["info"]["turn_rate_used_deg"] == 0.0 and not r["info"]["accel_saturated"]
    assert ep["defender_moved_before_release_m"] == 0.0 and ep["defender_speed_at_release_mps"] == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_release_happens_at_the_first_canonical_detection_and_never_reverts(seed):
    env, rows, ep = trace(seed)
    d = ep["detection_step"]
    first_detected = next(i for i, r in enumerate(rows) if r["detected"])
    assert d == first_detected
    # detection is the canonical true-3-D <= detection_radius criterion evaluated with the post-move attacker position and the (held) defender position
    assert np.linalg.norm(rows[d]["att"] - rows[d]["pos"]) <= X.detection_radius() + 1e-3
    assert all(np.linalg.norm(rows[i]["att"] - rows[i]["pos"]) > X.detection_radius() for i in range(0, d))
    assert all(r["detected"] for r in rows[d:])                              # sticky
    assert rows[d]["released"] is False and all(r["released"] for r in rows[d + 1:])       # the release flag is set when the NEXT step starts (the transition step is still held)
    assert ep["first_controller_step"] == d + 1
    if len(rows) > d + 1:
        assert np.linalg.norm(rows[d + 1]["vel"]) > 0.0                       # the defender starts moving at the first controller step


def test_after_release_the_unmodified_greedy_action_drives_the_unmodified_dynamics():
    raw = SoldierEnv(X.stage3v2_config())
    raw.reset(seed=1)
    checked = 0
    for seed in SEEDS:
        env, rows, ep = trace(seed)
        d = ep["release_step"]
        for i in range(d + 1, len(rows) - 1):                                 # the terminal step is excluded: swept-contact arbitration may clamp positions
            action = rows[i]["action"]
            assert action is not None and np.isclose(np.linalg.norm(action), 1.0, atol=1e-5)
            raw._defender_pos, raw._defender_vel = rows[i - 1]["pos"].astype(np.float32).copy(), rows[i - 1]["vel"].astype(np.float32).copy()
            raw._move_defender(action)                                        # canonical dynamics with the SAME action
            assert np.allclose(raw._defender_pos, rows[i]["pos"], atol=1e-5) and np.allclose(raw._defender_vel, rows[i]["vel"], atol=1e-5)
            assert np.linalg.norm(rows[i]["vel"]) <= 18.0 + 1e-4 and np.linalg.norm(rows[i]["vel"] - rows[i - 1]["vel"]) / 0.5 <= 8.0 + 1e-3
            checked += 1
    assert checked >= 3


def test_greedy_action_is_pursuit_of_the_measurement_after_detection_and_the_policy_is_unmodified():
    env = X.Stage3V2Env(0, 0, 0, scripted_a0=True)
    env.reset(options={"scenario": SEEDS[0]})
    e = env._env
    done = False
    greedy = get_policy("greedy")
    greedy.reset()
    while not done:
        _o, _r, te, tr, _i = env.step(np.zeros(3, dtype=np.float32))
        done = te or tr
        if env.released and not done:
            info = build_policy_info(e._get_info(), "measurement")
            expected = (np.asarray(info["enemy_measurement"]) - np.asarray(info["defender_pos"]))
            expected = expected / np.linalg.norm(expected)
            assert np.allclose(greedy.act(e._get_obs(), info), expected, atol=1e-5)
            break
    assert type(env._policy).__name__ == "GreedyInterceptPolicy"


def test_without_hold_the_wrapper_reproduces_stage3_v1_escort_exactly():
    for s in SEEDS[:4]:
        v1 = V1.Stage3Env(0, 0, 0, scripted_a0=True)
        v2 = X.Stage3V2Env(0, 0, 0, scripted_a0=True, hold_until_detection=False)
        a, b = C.run_episode(v1, s, None), C.run_episode(v2, s, None)
        assert a["outcome"] == b["outcome"] and a["steps"] == b["steps"] and a["defender_final_position"] == b["defender_final_position"]


def test_placement_is_the_unchanged_stage2_nominal_path_placement():
    for s in SEEDS[:3]:
        env = X.Stage3V2Env(0, 0, 0, scripted_a0=True)
        env.reset(options={"scenario": int(s)})
        ref = S.Stage2Env(0, 0, 0, scripted_a0=True)
        ref.reset(options={"scenario": int(s)})
        assert np.array_equal(env._env._defender_pos, ref._env._defender_pos) and env.placement["lambda"] == ref.placement["lambda"]
        assert 0.35 <= env.placement["lambda"] <= 0.65 and env.placement["offset_m"] <= 1.5


def test_observation_is_the_unchanged_37d_gate_without_release_information():
    env = X.Stage3V2Env(0, 0, 0)
    obs, _ = env.reset(options={"scenario": SEEDS[0]})
    assert obs.shape == (A.A1_V2_OBSERVATION_DIM,) == (37,) and env.observation_space.shape == (37,) and env.action_space.shape == (3,)
    e = env._env
    dist = float(np.linalg.norm(e._enemy_pos - e._defender_pos))
    assert dist > 25.0 and not obs[9:16].any()                               # defender slots are zero outside the 25 m true-range gate
    names_ = json.dumps(A.A1_V2_OBSERVATION_SCHEMA).lower()
    assert "released" not in names_ and "release" not in names_


def test_validation_is_deterministic_and_replays_exactly():
    a = X.evaluate(None, SEEDS, scripted_a0=True)
    b = X.evaluate(None, SEEDS, scripted_a0=True)
    keys = ("outcome", "steps", "detection_step", "release_step", "defender_final_position", "min_defender_separation_m", "path_length_m")
    assert [{k: e[k] for k in keys} for e in a] == [{k: e[k] for k in keys} for e in b]


def test_namespaces_are_fresh_disjoint_and_the_rejected_range_is_documented():
    o = X.namespace_overlaps()
    for key in ("validation_A", "validation_B", "stage1_anchor", "static_anchor", "diagnostic", "scenario_validation", "mutual", "training_overlap"):
        assert o[key] == [], key
    assert X.VALIDATION_A == (154_000, 154_099) and X.VALIDATION_B == (154_100, 154_199) and len(X.final_evaluation_seeds()) == 200
    assert [X.validation_set_for(k) for k in (1, 2, 15, 16)] == ["A", "B", "A", "B"]
    assert not any(not (hi < 149_000 or lo > 149_999) for lo, hi in (X.VALIDATION_A, X.VALIDATION_B, X.STAGE1_ANCHOR, X.STATIC_ANCHOR, X.DIAGNOSTIC, X.SCENARIO_VALIDATION))


def test_gate_is_identical_to_stage3_v1_and_the_frozen_requirements():
    assert X.GATE == V1.GATE == {"capture_min": 0.80, "improvement_over_a0_min_pp": 20.0, "interception_max": 0.20, "timeout_max": 0.02, "exhaustion_max": 0.02, "median_path_excess_max": 1.80,
                                 "gt3_max": 0.05, "gt5_max": 0.0, "orbit_episodes_max": 0, "stage1_anchor_capture_min": 0.98, "static_anchor_capture_min": 0.80, "nonfinite_max": 0}
    m = {"capture": 0.80, "interception": 0.20, "timeout": 0.02, "exhaustion": 0.02, "median_path_excess": 1.80, "gt3": 0.05, "gt5": 0.0, "orbit_episodes": 0, "nonfinite": 0}
    assert X.gate_status(m, a0_capture=0.60, stage1_anchor=0.98, static_anchor=0.80)["passed"]
    for key, bad in (("capture", 0.79), ("interception", 0.21), ("timeout", 0.03), ("median_path_excess", 1.81), ("gt3", 0.06), ("gt5", 0.01), ("orbit_episodes", 1), ("nonfinite", 1)):
        assert not X.gate_status({**m, key: bad}, a0_capture=0.0, stage1_anchor=1.0, static_anchor=1.0)["passed"], key
    assert not X.gate_status(m, a0_capture=0.61, stage1_anchor=0.98, static_anchor=0.80)["passed"]        # +19 pp only
    assert not X.gate_status(m, a0_capture=0.0, stage1_anchor=0.97, static_anchor=0.80)["passed"]
    assert not X.gate_status(m, a0_capture=0.0, stage1_anchor=0.98, static_anchor=0.79)["passed"]


def _val(passed, a1=1.0, static=1.0):
    return {"passed": passed, "stage1_anchor_capture": a1, "static_anchor_capture": static}


def test_seed_decision_mastery_and_both_retention_stops():
    assert T3.seed_decision([_val(True), _val(False), _val(True)]) == (None, None)
    assert T3.seed_decision([_val(False), _val(True), _val(True)]) == ("MASTERED", 3)
    assert T3.seed_decision([_val(False, 0.97), _val(False, 0.97)]) == ("PURSUIT_RETENTION_FAILURE", 2)
    assert T3.seed_decision([_val(False, 0.97), _val(False, 0.99), _val(False, 0.97)]) == (None, None)
    assert T3.seed_decision([_val(False, 1.0, 0.79), _val(False, 1.0, 0.7)]) == ("STATIC_EVASION_RETENTION_FAILURE", 2)
    assert T3.seed_decision([_val(False, 0.97), _val(True, 0.97)]) == ("PURSUIT_RETENTION_FAILURE", 2)
    assert T3.seed_decision([_val(True, 0.97), _val(True, 0.97)]) == ("MASTERED", 2)               # mastery wins a tie


def test_scenario_validation_verdict_rules():
    ok = {"a0_interception_fraction": 0.9, "validity": {"nonfinite_episodes": 0, "episodes_steps_le_1": 0, "intercepted_with_fewer_than_2_visible_decisions": 0, "defender_initial_distance_min_m": 17.0,
                                                         "max_defender_speed_at_release_mps": 0.0, "max_defender_displacement_before_release_m": 0.0,
                                                         "released_episodes_with_time_after_release": 10, "released_episodes_where_greedy_moved_defender": 10}}
    assert X.validation_verdict(ok)["passed"]
    assert not X.validation_verdict({**ok, "a0_interception_fraction": 0.69})["passed"] and X.validation_verdict({**ok, "a0_interception_fraction": 0.69})["too_easy"]
    assert X.validation_verdict({**ok, "a0_interception_fraction": 1.0})["passed"]
    bad = copy.deepcopy(ok)
    bad["validity"]["max_defender_speed_at_release_mps"] = 0.1
    assert not X.validation_verdict(bad)["passed"]
    bad = copy.deepcopy(ok)
    bad["validity"]["intercepted_with_fewer_than_2_visible_decisions"] = 1
    assert not X.validation_verdict(bad)["passed"]


def test_actual_scenario_validation_sample_passes_the_validity_rules():
    summary = X.scenario_validation(range(X.SCENARIO_VALIDATION[0], X.SCENARIO_VALIDATION[0] + 60))
    verdict = X.validation_verdict(summary)
    assert verdict["passed"], verdict
    assert summary["validity"]["max_defender_speed_at_release_mps"] == 0.0 and summary["a0_interception_fraction"] >= 0.70


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_stage2_v2_parents_are_the_preregistered_rollout_16_checkpoints_and_provisional(seed):
    p = X.parent_checkpoint(seed)
    assert p["model_sha256"].startswith(X.STAGE2V2_PREFIX[seed]) and p["rollout"] == 16 and p["parent_status"] == "provisional_curriculum_transfer" and p["stage2_v2_training_status"] != "MASTERED"
    assert p["num_timesteps"] == 524_288


def test_parent_initialisation_copies_actor_critic_log_std_and_resets_the_optimizer():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from experiments import journal_v2_a1_cur1v2_bc as BC

    kwargs = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=32, batch_size=32)
    env = DummyVecEnv([lambda: X.make_env(201, 0, 0)])
    model, record = T3.initialize_model(201, env, kwargs)
    parent = PPO.load(X.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    assert BC.tensor_sha256(BC.actor_state(model)) == BC.tensor_sha256(BC.actor_state(parent)) and BC.value_state_sha256(model) == BC.value_state_sha256(parent)
    assert np.array_equal(model.policy.log_std.detach().numpy(), parent.policy.log_std.detach().numpy())
    assert model.num_timesteps == 0 and len(model.policy.optimizer.state) == 0 and len(parent.policy.optimizer.state) > 0
    assert record["parent_status"] == "provisional_curriculum_transfer" and record["parent"]["model_sha256"].startswith("d70b85a6") and record["protocol_sha256"] == X.protocol_sha256()
    env.close()


def test_protocol_documents_exactly_one_conceptual_change_and_is_stable():
    spec = X.protocol_specification()
    v1 = V1.protocol_specification()
    assert "Defender motion is suppressed before canonical detection" in spec["single_conceptual_change_from_stage3_v1"]
    for key in ("reward", "gate", "retention", "success_rule", "paired_statistics"):
        assert spec[key] == v1[key], key
    assert spec["ppo"] == {**v1["ppo"], "hyperparameters": A.A1_V2_PPO_HYPERPARAMS}
    assert spec["parents"]["status"] == "provisional_curriculum_transfer" and spec["environment"]["detection_radius_m"] == 15.0
    assert X.protocol_sha256() == X.protocol_sha256() != V1.protocol_sha256()
    assert T3.SPEC.max_timesteps == 524_288 and T3.SPEC.steps_per_rollout == 32_768


def test_rollouts_of_another_protocol_are_never_accepted(tmp_path, monkeypatch):
    spec = T3.Spec(n_envs=2, n_steps=32, max_rollouts=2)
    layout = T3.make_layout(201, tmp_path, tmp_path / "res")
    layout.scratch_rollouts.mkdir(parents=True)

    class _M:
        num_timesteps, _n_updates = 64, 1

        def save(self, buf):
            buf.write(b"model")

    monkeypatch.setattr(T3, "state_sha256", lambda m: "0" * 64)
    monkeypatch.setattr(T3, "summarize_training", lambda e: {})
    monkeypatch.setattr(T3.T, "capture_rng", lambda: b"rng")
    manifest = T3.commit_rollout(layout, 1, _M(), [], spec=spec, ppo_kwargs={}, parent={"model_sha256": "a" * 64})
    assert T3.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=spec) == manifest
    monkeypatch.setattr(T3, "_PROTOCOL_SHA", "f" * 64)
    assert T3.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=spec) is None


def test_state_machine_has_the_fourteen_required_stages_and_isolated_output(tmp_path):
    assert len(TS.STAGES) == 14 and TS.STAGES[0].startswith("01_repository") and TS.STAGES[-1].endswith("commit_push")
    assert str(X.RESULTS_ROOT).endswith("stage3_v2") and "stage3_v2" in str(TS.STATE_DIR)
    with pytest.raises(T3.RunConflictError):
        T3.make_layout(201, tmp_path, T3.PROJECT_ROOT / "results" / "elsewhere")
    state = TS.load_state()
    assert all(s in state["stages"] for s in TS.STAGES)


def test_real_sb3_resume_with_parent_initialisation_matches_uninterrupted_run(tmp_path):
    result = T3.resume_test(tmp_path / "rt", n_envs=2, n_steps=32, rollouts=3, interrupt_after=1)
    assert result["all_states_equal"] and result["all_episode_logs_equal"] and result["init_record_equal"] and result["crash_injected_between_model_write_and_manifest"]
