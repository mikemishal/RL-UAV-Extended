"""Tests for curriculum Stage 2 (static-defender evasion). PPO `learn()` is only exercised by the tiny real-SB3 resume check."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_state as TS
from experiments import journal_v2_a1_cur2_training as T2
from experiments import journal_v2_a1_r2 as R

DEFENDER_SLOTS = [i for i, (name, _g) in enumerate(A.A1_V2_OBSERVATION_SCHEMA) if "defender" in name]


# ----------------------------------------------------------------------------- environment / defender placement
def test_config_is_the_stage1_config_with_canonical_interception_radius():
    cfg = S.stage2_config()
    assert set(S.config_differences_from_canonical()) == {"v_s", "defender_standby_until_detection"}
    assert cfg.v_s == 0.0 and cfg.intercept_radius == 2.5 == A.build_episode_config("C0", "greedy").intercept_radius and cfg.enemy_endurance_path_factor == 5.0
    assert S.REWARD_VERSION == R.REWARD_VERSION == "a1_v2_reward_v2" and S.ACTION_VERSION == "a1_cur1_action_v1" and S.OBSERVATION_VERSION == "a1_v2_obs_v1"


def test_placement_is_deterministic_inside_the_pre_registered_distribution_and_near_the_nominal_path():
    env = S.Stage2Env(0, 0, 0)
    for seed in (149_000, 149_001, 149_017, 149_503):
        _o, _ = env.reset(options={"scenario": seed})
        first = dict(env.placement)
        env.reset(options={"scenario": seed})
        assert np.array_equal(first["position"], env.placement["position"])
        path = S.nominal_a0_path(seed)
        q = first["position"].astype(np.float64)
        assert S.LAMBDA_RANGE[0] <= first["lambda"] <= S.LAMBDA_RANGE[1] and 0.0 <= first["offset_m"] <= S.OFFSET_MAX_M and first["attempts"] == 1
        point, tangent = S.polyline_point(path, first["lambda"])
        assert np.linalg.norm(q - point) == pytest.approx(first["offset_m"], abs=1e-4) and abs(float(np.dot(q - point, tangent))) < 1e-4       # perpendicular to the local path
        seg = np.diff(path, axis=0)
        d = [np.linalg.norm(q - (path[i] + np.clip(np.dot(q - path[i], seg[i]) / np.dot(seg[i], seg[i]), 0, 1) * seg[i])) for i in range(len(seg))]
        assert min(d) <= S.OFFSET_MAX_M + 1e-4                                                                                                     # within r of the nominal path
        assert abs(q[0]) <= 50 and abs(q[1]) <= 50 and 0 <= q[2] <= 30


def test_invalid_placements_are_resampled_never_clipped():
    path = S.nominal_a0_path(149_005)
    q0, _ = S.polyline_point(path, 0.5)
    small = float(max(abs(q0[0]), abs(q0[1]))) + 0.2                          # arena just big enough that some perpendicular draws fall outside
    outs = [S.place_defender(s, path, arena_half=small, max_altitude=30.0) for s in range(40)]
    assert any(o["attempts"] > 1 for o in outs)
    for o in outs:
        q = o["position"]
        assert abs(q[0]) <= small + 1e-5 and abs(q[1]) <= small + 1e-5 and 0 <= q[2] <= 30
    with pytest.raises(RuntimeError):
        S.place_defender(1, path, arena_half=0.01, max_altitude=30.0)


def test_chord_placement_reproduces_the_superseded_design_and_the_nominal_path_fixes_it():
    seeds = list(range(149_000, 149_060))
    env = S.Stage2Env(0, 0, 0, scripted_a0=True)
    hits = 0
    for s in seeds:
        env.reset(options={"scenario": s})
        e = env._env
        chord = S.place_defender_on_chord(s, e._enemy_pos, e._soldier_pos, arena_half=50.0, max_altitude=30.0)
        a0 = S.nominal_a0_path(s)
        hits += int(np.min(np.linalg.norm(a0 - chord["position"], axis=1)) <= 2.5)
    assert hits < 0.8 * len(seeds)                                                  # the chord design could not meet the 70% rule reliably
    assert np.mean([e["intercepted"] for e in S.evaluate(None, seeds, scripted_a0=True)]) >= 0.97


def test_defender_is_stationary_active_and_canonically_terminates_the_episode():
    env = S.Stage2Env(0, 0, 0, scripted_a0=True)
    obs, _ = env.reset(options={"scenario": 149_010})
    e = env.soldier_env
    fixed = e._defender_pos.copy()
    done, info = False, {}
    while not done:
        _o, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        done = te or tr
        assert np.array_equal(e._defender_pos, fixed) and not np.any(e._defender_vel) and np.allclose(e._soldier_pos, 0) and float(np.linalg.norm(e._soldier_vel if hasattr(e, "_soldier_vel") else 0)) == 0.0
    assert info["outcome"] == "intercepted" and info["a1_episode"]["min_defender_separation_m"] == pytest.approx(2.5, abs=1e-4) and info["a1_episode"]["intercepted"]


def test_defender_observation_follows_the_canonical_gate():
    env = S.Stage2Env(0, 0, 0)
    obs, _ = env.reset(options={"scenario": 149_020})
    e = env.soldier_env
    cfg = e.config
    assert cfg.enemy_defender_detection_range == 25.0 and env._detection_range == 25.0
    saw_visible = saw_hidden = False
    for _ in range(12):
        dist = float(np.linalg.norm(e._defender_pos - e._enemy_pos))
        if dist <= 25.0:
            saw_visible = True
            assert obs[DEFENDER_SLOTS[0]] == 1.0 and np.allclose(obs[DEFENDER_SLOTS[1:4]], np.clip(e._normalize_pos(e._defender_pos), -1, 1), atol=1e-6) and not np.any(obs[DEFENDER_SLOTS[4:]])
        else:
            saw_hidden = True
            assert not np.any(obs[DEFENDER_SLOTS])
        obs, _r, te, tr, _i = env.step(env.soldier_env._soldier_pos.astype(np.float32) * 0 + (np.asarray(e._soldier_pos) - np.asarray(e._enemy_pos)).astype(np.float32))
        if te or tr:
            break
    assert saw_hidden and saw_visible
    assert obs.shape == (37,) == env.observation_space.shape and env.action_space.shape == (3,)


def test_paired_scenarios_share_the_initial_state_between_a0_and_a1():
    bc = BC.load_model(Path(T2.bc_parent(201)["path"]))
    seeds = [147_000, 147_003, 147_150]
    a0 = S.evaluate(None, seeds, scripted_a0=True)
    a1 = S.evaluate(C.model_actor(bc), seeds)
    for x, y in zip(a0, a1):
        assert x["scenario_seed"] == y["scenario_seed"] and x["defender_position"] == y["defender_position"] and x["initial_range_m"] == y["initial_range_m"]
        assert x["initial_heading_offset_deg"] == y["initial_heading_offset_deg"] and x["placement_lambda"] == y["placement_lambda"]


def test_namespaces_are_disjoint_from_every_reserved_range_and_from_each_other():
    o = S.namespace_overlaps()
    assert o["validation_A"] == o["validation_B"] == o["anchor"] == o["placement_validation"] == [] and o["mutual"] == []
    assert len(S.final_evaluation_seeds()) == 200 and S.final_evaluation_seeds()[0] == 147_000 and S.final_evaluation_seeds()[-1] == 147_199
    assert max(S.anchor_seeds()) < C.TRAIN_SEED_LOW


# ----------------------------------------------------------------------------- BC -> PPO initialisation
def test_bc_to_ppo_initialisation_copies_actor_and_log_std_and_keeps_a_fresh_critic_and_optimizer():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([functools.partial(S.make_env, 202, 0, 0)])
    model, record = T2.initialize_model(202, env, dict(A.A1_V2_PPO_HYPERPARAMS))
    bc = BC.load_model(Path(T2.bc_parent(202)["path"]))
    reference = PPO(policy=A.A1_V2_POLICY, env=env, seed=202, verbose=0, device="cpu", **A.A1_V2_PPO_HYPERPARAMS)
    other = PPO(policy=A.A1_V2_POLICY, env=env, seed=203, verbose=0, device="cpu", **A.A1_V2_PPO_HYPERPARAMS)
    assert BC.tensor_sha256(BC.actor_state(model)) == BC.tensor_sha256(BC.actor_state(bc)) and np.allclose(model.policy.log_std.detach().numpy(), -2.0)
    assert np.allclose(record["initial_action_std"], np.exp(-2.0))
    assert BC.value_state_sha256(model) == BC.value_state_sha256(reference) == record["fresh_critic_sha256"] != BC.value_state_sha256(other)       # critic = PPO-seed initialisation
    assert BC.tensor_sha256(BC.actor_state(model)) != BC.tensor_sha256(BC.actor_state(reference))                                                  # actor is NOT the PPO-seed one
    assert model.num_timesteps == 0 and len(model.policy.optimizer.state_dict()["state"]) == 0                                                      # fresh optimizer, no BC optimizer state
    assert record["parent"]["attacker_id"] == "A1-BC1-202" and "failed A1-v1 PPO weights" in record["not_loaded"]
    obs = env.reset()
    assert np.array_equal(model.predict(obs, deterministic=True)[0], bc.predict(obs, deterministic=True)[0])
    env.close()


def test_bc_parent_hash_is_verified_against_the_handoff(tmp_path):
    root = tmp_path
    (root / "docs/journal_v2_a1_curriculum/stage1_v2").mkdir(parents=True)
    (root / "m.zip").write_bytes(b"x")
    (root / S.STAGE1_V2_HANDOFF).write_text(json.dumps({"log_std": -2.0, "checkpoints": {"201": {"path": "m.zip", "model_sha256": "0" * 64}}}))
    with pytest.raises(T2.RunConflictError):
        T2.bc_parent(201, project_root=root)


# ----------------------------------------------------------------------------- anchor / evaluation / mastery / retention
def test_anchor_evaluation_is_the_masked_stage1_no_defender_task_and_the_bc_parent_passes_it():
    bc = BC.load_model(Path(T2.bc_parent(203)["path"]))
    eps = S.evaluate_anchor(C.model_actor(bc), range(S.ANCHOR[0], S.ANCHOR[0] + 12))
    m = V1V2.metrics(eps)
    assert m["capture"] == 1.0 and m["intercepted"] == 0.0 and m["expert_cosine_mean"] > 0.999


def _m(**kw):
    base = {"capture": 1.0, "timeout": 0.0, "exhaustion": 0.0, "median_path_excess": 1.1, "gt3": 0.0, "gt5": 0.0, "orbit_episodes": 0, "nonfinite": 0}
    base.update(kw)
    return base


def test_stage2_gate_boundaries_and_improvement_requirement():
    g = lambda m, a0=0.0, anchor=1.0: S.gate_status(m, a0_capture=a0, anchor_capture=anchor)  # noqa: E731
    assert g(_m())["passed"] and g(_m(capture=0.90, timeout=0.02, exhaustion=0.02, median_path_excess=1.60, gt3=0.02), a0=0.70, anchor=0.98)["passed"]
    for bad in ({"capture": 0.8999}, {"timeout": 0.0201}, {"exhaustion": 0.0201}, {"median_path_excess": 1.6001}, {"gt3": 0.0201}, {"gt5": 0.005}, {"orbit_episodes": 1}, {"nonfinite": 1}):
        assert not g(_m(**bad))["passed"], bad
    assert not g(_m(capture=0.95), a0=0.76)["passed"] and g(_m(capture=0.95), a0=0.75)["passed"] and not g(_m(), anchor=0.9799)["passed"]


def _v(passed, anchor=1.0):
    return {"passed": passed, "anchor_capture": anchor}


def test_seed_decision_needs_two_consecutive_passes_and_stops_on_two_consecutive_anchor_failures():
    assert T2.seed_decision([_v(True), _v(False), _v(True)]) == (None, None)
    assert T2.seed_decision([_v(False), _v(True), _v(True)]) == ("MASTERED", 3)
    assert T2.seed_decision([_v(True), _v(True)]) == ("MASTERED", 2)
    assert T2.seed_decision([_v(False, 0.97), _v(False, 0.99), _v(False, 0.97)]) == (None, None)
    assert T2.seed_decision([_v(False, 0.97), _v(False, 0.97)]) == ("PURSUIT_RETENTION_FAILURE", 2)
    assert S.retention_failure_rollout([1.0, 0.5, 0.99, 0.5, 0.5]) == 5
    assert T2.seed_decision([_v(False, 1.0), _v(True, 0.99), _v(True, 0.99)]) == ("MASTERED", 3)


def test_paired_bootstrap_difference_and_interval():
    a, b = np.array([1] * 90 + [0] * 10), np.array([0] * 100)
    r = S.paired_bootstrap(a, b, resamples=2000)
    assert r["difference_pp"] == pytest.approx(90.0) and r["ci95_pp"][0] > 80 and r["ci95_pp"][1] <= 96 and r["a1_only"] == 90 and r["a0_only"] == 0
    same = S.paired_bootstrap(a, a, resamples=500)
    assert same["difference_pp"] == 0.0 and same["ci95_pp"] == [0.0, 0.0]
    assert S.paired_bootstrap(a, b, resamples=500) == S.paired_bootstrap(a, b, resamples=500)
    with pytest.raises(ValueError):
        S.paired_bootstrap(a, b[:-1])


def test_deterministic_evaluation_and_validation_payload_integrity(tmp_path):
    bc = BC.load_model(Path(T2.bc_parent(201)["path"]))
    seeds = [147_000, 147_001, 147_002, 147_003]
    x, y = S.evaluate(C.model_actor(bc), seeds), S.evaluate(C.model_actor(bc), seeds)
    assert [e["outcome"] for e in x] == [e["outcome"] for e in y] and [e["path_length_m"] for e in x] == [e["path_length_m"] for e in y]
    t = S.evaluate(C.model_actor(bc), seeds, record_trajectory=True)
    assert [e["outcome"] for e in t] == [e["outcome"] for e in x] and all(len(e["trajectory"]) == e["steps"] + 1 for e in t)
    manifest = {"model_sha256": "a" * 64, "ppo_seed": 201, "num_timesteps": 32768}
    payload = T2.run_validation(bc, 1, manifest)
    assert payload["validation_set"] == "A" and payload["a0_metrics"]["interception"] == 1.0 and payload["anchor_capture"] == 1.0 and not payload["passed"]
    layout = T2.make_layout(201, tmp_path / "s", tmp_path / "r")
    layout.scratch_rollouts.mkdir(parents=True)
    path = layout.scratch_rollouts / T2.names(1)["validation"]
    path.write_text(json.dumps(payload))
    assert T2.verify_validation(layout.scratch_rollouts, 1, manifest)["rollout"] == 1
    tampered = json.loads(path.read_text())
    tampered["metrics"]["capture"] = 1.0
    path.write_text(json.dumps(tampered))
    assert T2.verify_validation(layout.scratch_rollouts, 1, manifest) is None
    assert T2.verify_validation(layout.scratch_rollouts, 2, manifest) is None


def test_isolated_output_namespace(tmp_path):
    assert "a1_curriculum" in str(S.RESULTS_ROOT) and S.RESULTS_ROOT.name == "stage2" and S.RESULTS_ROOT != V1V2.RESULTS_ROOT and S.RESULTS_ROOT != C.RESULTS_ROOT
    with pytest.raises(T2.RunConflictError):
        T2.make_layout(201, tmp_path, T2.PROJECT_ROOT / "results" / "elsewhere")


# ----------------------------------------------------------------------------- checkpoint / resume (real SB3, tiny)
def test_real_sb3_resume_with_bc_initialisation_matches_uninterrupted_run(tmp_path):
    result = T2.resume_test(tmp_path / "rt", n_envs=2, n_steps=32, rollouts=3, interrupt_after=1)
    assert result["crash_injected_between_model_write_and_manifest"] and result["partial_files_before_resume"]
    assert result["all_states_equal"] and result["all_episode_logs_equal"] and result["init_record_equal"]


def test_rollout_recovery_discards_partial_and_tampered_rollouts(tmp_path):
    layout = T2.make_layout(201, tmp_path / "s", tmp_path / "r")
    spec = T2.Spec(n_envs=2, n_steps=8, max_rollouts=3)
    layout.scratch_rollouts.mkdir(parents=True)
    n1 = T2.names(1)
    for part in ("model", "rng", "episodes"):
        (layout.scratch_rollouts / n1[part]).write_bytes(part.encode())
    T2.T.atomic_write_json(layout.scratch_rollouts / n1["manifest"], {"schema": T2.SCHEMA, "rollout": 1, "ppo_seed": 201, "num_timesteps": spec.steps_per_rollout, "training_protocol": S.TRAINING_PROTOCOL,
                                                                    "model_sha256": T2.T.sha256_bytes(b"model"), "rng_sha256": T2.T.sha256_bytes(b"rng"), "episodes_sha256": T2.T.sha256_bytes(b"episodes")})
    (layout.scratch_rollouts / T2.names(2)["model"]).write_bytes(b"partial")                # rollout 2 never got a manifest
    best, _m = T2.recover(layout, spec=spec)
    assert best == 1 and not (layout.scratch_rollouts / T2.names(2)["model"]).exists()
    (layout.scratch_rollouts / n1["rng"]).write_bytes(b"tampered")
    best, _m = T2.recover(layout, spec=spec)
    assert best == 0 and not list(layout.scratch_rollouts.glob("rollout_*"))


def test_protocol_state_and_runner_conventions(tmp_path, monkeypatch):
    parent = {"201": {"attacker_id": "A1-BC1-201", "model_sha256": "1" * 64}}
    spec = S.protocol_specification(parent)
    assert spec["gate"] == S.GATE and spec["ppo"]["rollout_steps"] == 32768 and spec["ppo"]["max_timesteps"] == 262144 and spec["reward"].startswith("a1_v2_reward_v2")
    assert spec["ppo"]["hyperparameters"] == A.A1_V2_PPO_HYPERPARAMS and spec["placement"]["lambda"] == [0.35, 0.65] and spec["initialization"]["critic"].startswith("FRESH")
    assert S.protocol_sha256(parent) == S.protocol_sha256(parent) != S.protocol_sha256({})
    assert len(TS.STAGES) == 13 and TS.STAGES[6] == "07_seed_201_ppo" and TS.STAGES[3] == "04_deterministic_scenario_validation"
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    state = TS.load_state()
    TS.skip(state, TS.STAGES[10], "visualizer only if >= 2/3 seeds master")
    assert TS.stage_is_verified(TS.load_state(), TS.STAGES[10]) and TS.next_stage(TS.load_state()) == TS.STAGES[0]


def _final():
    from experiments import journal_v2_a1_cur2_final as F

    return F


def test_stage2_decision_requires_two_mastered_seeds_that_retain_pursuit_without_shared_pathology():
    F = _final()
    ok_metrics = {"episodes": 200, "orbit_episodes": 0, "repeated_flyby_episodes": 0, "nonclosing_episodes": 0, "nonfinite": 0, "capture": 0.95}
    per = {s: {"anchor_metrics": {"capture": 1.0}, "metrics": dict(ok_metrics)} for s in (201, 202, 203)}
    train = lambda *st: {s: {"status": x} for s, x in zip((201, 202, 203), st)}  # noqa: E731
    assert F.decide(per, train("MASTERED", "MASTERED", "FAIL"))["stage2_pass"]
    assert F.decide(per, train("MASTERED", "MASTERED", "MASTERED"))["all_three_mastered"]
    assert not F.decide(per, train("MASTERED", "FAIL", "FAIL"))["stage2_pass"]
    low = {**per, 202: {"anchor_metrics": {"capture": 0.9}, "metrics": dict(ok_metrics)}}
    assert not F.decide(low, train("MASTERED", "MASTERED", "FAIL"))["stage2_pass"]
    shared = {s: {"anchor_metrics": {"capture": 1.0}, "metrics": {**ok_metrics, "repeated_flyby_episodes": 5}} for s in (201, 202, 203)}
    d = F.decide(shared, train("MASTERED", "MASTERED", "MASTERED"))
    assert not d["stage2_pass"] and "repeated_flyby_episodes" in d["common_structural_pathology"]
    assert not F.decide(per, train("FAIL", "FAIL", "FAIL"))["stage2_pass"] and F.decide(per, train("FAIL", "FAIL", "FAIL"))["learning_without_mastery"]


def test_evasion_features_detect_a_detour_and_the_return_to_pursuit():
    F = _final()
    asset = np.zeros(3)
    xs = np.linspace(40.0, 0.0, 11)
    bump = np.array([0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0], dtype=float)                       # lateral (y) detour around a defender placed on the straight line
    pos = np.stack([xs, bump, np.linspace(10.0, 0.0, 11)], axis=1)
    los = np.array([(asset - p) / np.linalg.norm(asset - p) for p in pos[:-1]])
    u = np.array([(pos[i + 1] - pos[i]) / np.linalg.norm(pos[i + 1] - pos[i]) for i in range(10)])
    trace = {"pos": pos, "u": u, "los": los, "u_bc": los.copy(), "defender_dist": np.linalg.norm(pos[:-1] - np.array([20.0, 0.0, 5.0]), axis=1), "defender": np.array([20.0, 0.0, 5.0]), "asset": asset}
    f = F.evasion_features(trace, {"scenario_seed": 1, "outcome": "soldier_caught", "path_excess": 1.1, "defender_first_visible_step": 3})
    assert f["max_lateral_dev_m"] > 2.5 and f["side"] in ("left", "right") and f["min_defender_distance_m"] > 2.5 and f["first_divergence_step"] is not None
    assert f["post_encounter_mean_cos_to_asset"] > 0.9 and f["pursuit_evasion_pursuit"] and f["resumes_pursuit"]
    summary = F.summarize_evasion([f])
    assert summary["n_successful"] == 1 and summary["side_counts"][f["side"]] == 1 and F.summarize_evasion([]) == {"n_successful": 0}

