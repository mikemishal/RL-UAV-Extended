"""Tests for curriculum Stage 1 (stationary asset, C0, parked defender; continuous direction action; mastery-based restart-safe PPO runner)."""

from __future__ import annotations

import functools
import hashlib
import json
import math

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1_state as TS
from experiments import journal_v2_a1_cur1_training as TR
from experiments import journal_v2_a1_r2 as R


# ----------------------------------------------------------------------------- environment definition
def test_stage1_config_differs_from_canonical_c0_only_in_asset_speed_and_defender_standby():
    assert set(C.config_differences_from_canonical()) == {"v_s", "defender_standby_until_detection"}
    cfg = C.stage1_config()
    assert cfg.v_s == 0.0 and cfg.defender_standby_until_detection is False and cfg.enemy_obstacle_avoidance_enabled is False
    assert cfg.enemy_nominal_guidance_mode == "direct_obstacle_only" and cfg.enemy_endurance_enabled and cfg.enemy_endurance_path_factor == 5.0


def test_asset_is_stationary_at_the_canonical_origin_and_defender_stays_parked():
    env = C.Stage1Env(0, 0, 0)
    for seed in (146000, 146050, 146120):
        obs, _ = env.reset(options={"scenario": seed})
        s = env.soldier_env
        assert np.allclose(s._soldier_pos, [0, 0, 0]) and np.allclose(s._defender_pos, C.PARKED_DEFENDER)
        done, steps = False, 0
        while not done and steps < 40:
            _o, _r, te, tr, info = env.step(np.array([0.3, -0.2, 0.1], dtype=np.float32))
            done, steps = te or tr, steps + 1
            assert np.allclose(s._soldier_pos, [0, 0, 0]) and np.allclose(s._defender_pos, C.PARKED_DEFENDER) and not np.any(s._defender_vel)
        assert not any(o in info["outcome"] for o in ("intercept",)) or steps > 0
    ep = C.run_episode(C.Stage1Env(0, 0, 0, scripted_a0=True), 146001, None)
    assert ep["asset_displacement_max_m"] == 0.0 and ep["min_defender_separation_m"] > 5.0


def test_scripted_a0_captures_every_stage1_scenario_without_pathology():
    env = C.Stage1Env(0, 0, 0, scripted_a0=True)
    eps = [C.run_episode(env, s, None) for s in range(146000, 146040)]
    m = C.metrics(eps)
    assert m["capture"] == 1.0 and m["median_path_excess"] < 1.05 and m["pathology_rate"] == 0.0 and m["gt5"] == 0.0 and m["asset_displacement_max_m"] == 0.0


def test_observation_is_the_canonical_37d_and_defender_features_are_zero_outside_the_gate():
    env = C.Stage1Env(0, 0, 0)
    obs, _ = env.reset(options={"scenario": 146000})
    assert obs.shape == (37,) == env.observation_space.shape and env.observation_space.contains(obs)
    assert np.array_equal(obs, A.build_a1_v2_observation(env.soldier_env, env._detection_range).astype(np.float32))
    idx = [i for i, (name, _g) in enumerate(A.A1_V2_OBSERVATION_SCHEMA) if "defender" in name]
    assert idx
    far = 0
    for seed in range(146000, 146030):
        o, _ = env.reset(options={"scenario": seed})
        s = env.soldier_env
        if float(np.linalg.norm(s._defender_pos - s._enemy_pos)) > C.stage1_config().enemy_defender_detection_range:
            far += 1
            assert not np.any(o[idx])
    assert far > 20 and env.observation_space.shape == (37,) and A.A1_V2_OBSERVATION_VERSION == C.OBSERVATION_VERSION == "a1_v2_obs_v1"


# ----------------------------------------------------------------------------- geometry distribution
def test_training_geometry_covers_bearings_ranges_altitudes_and_headings():
    env = C.Stage1Env(0, 0, 0)
    bearings, ranges, alts, offsets, speeds = [], [], [], [], []
    for seed in range(1_000_000, 1_000_300):
        env.reset(options={"scenario": seed})
        s = env.soldier_env
        rel = np.asarray(s._enemy_pos, dtype=float)
        bearings.append(math.degrees(math.atan2(rel[1], rel[0])))
        ranges.append(float(np.linalg.norm(rel)))
        alts.append(float(rel[2]))
        offsets.append(env._initial_offset_deg)
        speeds.append(float(np.linalg.norm(s._enemy_vel)))
    assert min(bearings) < -150 and max(bearings) > 150 and any(-30 < b < 30 for b in bearings) and any(60 < b < 120 for b in bearings)
    assert min(ranges) >= 50.0 and max(ranges) <= 90.0 and 10.0 <= min(alts) and max(alts) <= 30.0 and max(alts) - min(alts) > 15.0
    assert 0.0 <= min(offsets) and max(offsets) <= C.MAX_HEADING_OFFSET_DEG + 1e-6 and 35.0 < float(np.mean(offsets)) < 55.0 and max(offsets) > 80.0
    assert max(speeds) <= C.stage1_config().v_e + 1e-4


def test_initial_heading_is_deterministic_per_scenario_and_respects_the_vertical_rate_limit():
    los = np.array([1.0, 0.0, -0.4])
    assert np.allclose(C.initial_heading_direction(1234, los), C.initial_heading_direction(1234, los))
    assert not np.allclose(C.initial_heading_direction(1234, los), C.initial_heading_direction(1235, los))
    env = C.Stage1Env(0, 0, 0)
    cfg = C.stage1_config()
    for seed in range(146000, 146060):
        env.reset(options={"scenario": seed})
        v = env.soldier_env._enemy_vel
        assert -cfg.enemy_max_descent_rate - 1e-4 <= v[2] <= cfg.enemy_max_climb_rate + 1e-4


def test_scenario_streams_are_pure_functions_of_seed_worker_and_rollout():
    a = [C.sample_scenario_seed(C.scenario_stream(201, 3, 5)) for _ in range(1)]
    b = [C.sample_scenario_seed(C.scenario_stream(201, 3, 5)) for _ in range(1)]
    assert a == b and C.sample_scenario_seed(C.scenario_stream(201, 3, 6)) != a[0] and C.sample_scenario_seed(C.scenario_stream(202, 3, 5)) != a[0]
    assert C.TRAIN_SEED_LOW <= a[0] < C.TRAIN_SEED_HIGH


def test_validation_and_training_seed_ranges_are_disjoint_from_every_reserved_range():
    reserved = [(130_000, 130_099), (140_000, 140_099), (145_000, 145_099), (150_000, 150_039), (180_000, 180_999), (190_000, 190_999), (95_000, 95_999)]
    a, b = set(C.validation_seeds("A")), set(C.validation_seeds("B"))
    assert len(a) == len(b) == 100 and not (a & b) and min(a) == 146_000 and max(b) == 146_199 and len(C.final_evaluation_seeds()) == 200
    for lo, hi in reserved:
        assert not any(lo <= s <= hi for s in a | b)
    assert C.TRAIN_SEED_LOW > max(b)


# ----------------------------------------------------------------------------- action semantics
def test_action_is_normalized_to_a_unit_direction_independent_of_magnitude():
    v = np.array([5.0, 5.0, 0.0])
    for a in ([1.0, 0.0, 0.0], [0.2, -0.4, 0.1], [-1.0, 1.0, -1.0]):
        d, held = C.desired_direction(a, v)
        assert not held and np.linalg.norm(d) == pytest.approx(1.0) and np.allclose(d, np.array(a) / np.linalg.norm(a))
    assert np.allclose(C.desired_direction([0.1, 0.0, 0.0], v)[0], C.desired_direction([1.0, 0.0, 0.0], v)[0])
    assert np.allclose(C.desired_direction([5.0, 0.0, 0.0], v)[0], [1, 0, 0])      # out-of-range actions are clipped first


@pytest.mark.parametrize("action", ([0.0, 0.0, 0.0], [1e-9, -1e-9, 0.0], [np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]))
def test_near_zero_or_invalid_actions_hold_the_current_heading_not_a0_guidance(action):
    v = np.array([0.0, 6.0, -3.0])
    d, held = C.desired_direction(action, v)
    assert held and np.allclose(d, v / np.linalg.norm(v))
    d0, held0 = C.desired_direction(action, np.zeros(3))
    assert held0 and np.allclose(d0, [1, 0, 0])


def test_zero_actions_do_not_fall_back_to_mission_pursuit_in_the_environment():
    env = C.Stage1Env(0, 0, 0, record_trajectory=True)
    a0 = C.Stage1Env(0, 0, 0, scripted_a0=True, record_trajectory=True)
    for seed in (146002, 146003, 146004, 146005):
        env.reset(options={"scenario": seed})
        if env._initial_offset_deg < 40.0:
            continue
        done, info = False, {}
        while not done:
            _o, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            done = te or tr
        ep = info["a1_episode"]
        ref = C.run_episode(a0, seed, None)
        assert ep["held_steps"] == ep["steps"] and ep["outcome"] != "soldier_caught"            # flew its own heading, never captured
        assert ref["outcome"] == "soldier_caught"
        return
    pytest.fail("no scenario with a large initial heading offset found")


def test_policy_actions_steer_the_vehicle_under_canonical_dynamics_and_speed_is_not_commanded():
    env = C.Stage1Env(0, 0, 0)
    env.reset(options={"scenario": 146010})
    cfg = C.stage1_config()
    speeds = []
    for _ in range(6):
        env.step(np.array([0.05, 0.0, 0.0], dtype=np.float32))     # tiny magnitude must not slow the vehicle
        speeds.append(float(np.linalg.norm(env.soldier_env._enemy_vel)))
    assert max(speeds) <= cfg.v_e + 1e-4 and speeds[-1] > 0.5 * cfg.v_e


# ----------------------------------------------------------------------------- reward
def test_reward_v2_is_unchanged_and_decomposes_exactly():
    env = C.Stage1Env(0, 0, 0, scripted_a0=True)
    obs, _ = env.reset(options={"scenario": 146020})
    total, done, info = 0.0, False, {}
    while not done:
        _o, r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        total += r
        done = te or tr
    ep = info["a1_episode"]
    assert C.REWARD_VERSION == R.REWARD_VERSION == "a1_v2_reward_v2" and total == pytest.approx(ep["return_undiscounted"])
    assert ep["components"]["terminal"] == 10.0 and ep["components"]["step"] == pytest.approx(-0.1 * ep["steps"])
    assert R.step_reward(range_prev_m=10.0, range_next_m=9.0, initial_range_m=20.0, path_step_m=1.0, path_budget_m=100.0, outcome="timeout").total < 0
    assert not hasattr(C, "STAGE1_REWARD")


def test_determinism_of_scripted_episode():
    a = C.run_episode(C.Stage1Env(0, 0, 0, scripted_a0=True, record_trajectory=True), 146030, None)
    b = C.run_episode(C.Stage1Env(0, 0, 0, scripted_a0=True, record_trajectory=True), 146030, None)
    assert a["trajectory"] == b["trajectory"] and a["return_undiscounted"] == b["return_undiscounted"]


# ----------------------------------------------------------------------------- gate + mastery
def _m(**kw):
    base = {"capture": 1.0, "collision": 0.0, "exhaustion": 0.0, "timeout": 0.0, "median_path_excess": 1.0, "gt2": 0.0, "gt5": 0.0, "pathology_rate": 0.0, "orbit_episodes": 0, "nonfinite": 0}
    base.update(kw)
    return base


def test_mastery_gate_boundaries_are_the_pre_registered_ones():
    assert C.gate_status(_m())["passed"] and C.gate_status(_m(capture=0.98, exhaustion=0.01, timeout=0.01, median_path_excess=1.20, gt2=0.02, pathology_rate=0.02))["passed"]
    for bad in ({"capture": 0.9799}, {"exhaustion": 0.0101}, {"timeout": 0.0101}, {"median_path_excess": 1.2001}, {"gt2": 0.0201}, {"gt5": 0.005}, {"collision": 0.005},
                {"pathology_rate": 0.0201}, {"orbit_episodes": 1}, {"nonfinite": 1}):
        assert not C.gate_status(_m(**bad))["passed"], bad
    assert C.GATE["capture_min"] == 0.98 and C.GATE["gt5_max"] == 0.0 and C.GATE["median_path_excess_max"] == 1.20


def test_metrics_flag_orbit_repeated_flyby_nonclosing_and_nonfinite_episodes():
    def ep(**kw):
        e = {"outcome": "soldier_caught", "path_excess": 1.0, "orbit_revolutions_max": 0.0, "overshoots": 0, "nonclosing_run": 0, "pathology": False, "finite": True, "invalid_actions": 0,
             "held_steps": 0, "duration_s": 5.0, "path_length_m": 60.0, "terminal_distance_m": 2.0, "return_undiscounted": 10.0, "return_discounted": 9.0,
             "components": {"terminal": 10.0, "progress": 2.0, "path": -0.2, "step": -1.0}, "asset_displacement_max_m": 0.0, "min_defender_separation_m": 40.0}
        e.update(kw)
        return e

    eps = [ep() for _ in range(96)] + [ep(orbit_revolutions_max=1.2, pathology=True), ep(overshoots=2, pathology=True), ep(nonclosing_run=10, pathology=True), ep(finite=False)]
    m = C.metrics(eps)
    assert m["orbit_episodes"] == 1 and m["repeated_flyby_episodes"] == 1 and m["nonclosing_episodes"] == 1 and m["nonfinite"] == 1 and m["pathology_rate"] == pytest.approx(0.03)
    assert not C.gate_status(m)["passed"]


def test_mastery_requires_two_consecutive_passes_and_stops_at_the_second():
    f = TR.mastery_from_passes
    assert f([]) is None and f([True]) is None and f([True, True]) == 2 and f([False, True, True]) == 3 and f([True, False, True]) is None
    assert f([True, False, True, True]) == 4 and f([False] * 8) is None
    assert [TR.validation_set_for(k) for k in range(1, 7)] == ["A", "B", "A", "B", "A", "B"]


def test_protocol_specification_is_stable_complete_and_matches_the_frozen_artifact_if_present():
    spec = C.protocol_specification()
    assert spec["mastery_gate"] == C.GATE and spec["training"]["steps_per_rollout"] == 32_768 and spec["training"]["max_timesteps"] == 262_144
    assert spec["ppo"]["hyperparameters"]["n_steps"] == 2048 and spec["ppo"]["hyperparameters"]["batch_size"] == 64 and spec["reward"]["special_stage_reward"] is False
    assert C.protocol_sha256() == hashlib.sha256(json.dumps(C.protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()
    path = TS.STATE_DIR / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text())["protocol_sha256"] == C.protocol_sha256()


def test_ppo_settings_are_the_unchanged_a1_settings():
    h = A.A1_V2_PPO_HYPERPARAMS
    assert (h["learning_rate"], h["gamma"], h["gae_lambda"], h["n_steps"], h["batch_size"], h["n_epochs"], h["clip_range"], h["ent_coef"], h["vf_coef"], h["max_grad_norm"]) == (
        3e-4, 0.99, 0.95, 2048, 64, 10, 0.2, 0.01, 0.5, 0.5)
    assert C.N_ENVS == 16 and C.PPO_SEEDS == (201, 202, 203) and C.MAX_ROLLOUTS == 8 and TR.CANONICAL_SPEC.steps_per_rollout == 32_768


# ----------------------------------------------------------------------------- runner (tiny real-SB3 on DummyVecEnv)
TINY = TR.Spec(n_envs=2, n_steps=64, max_rollouts=3)
TINY_KW = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=64, batch_size=64)


def _dummy_factory(seed):
    def factory():
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv([functools.partial(C.make_env, seed, i, 0) for i in range(TINY.n_envs)])

    return factory


def _passing(monkeypatch, pattern):
    """Replace the 100-episode validation by a scripted pass/fail pattern (the production metric code is covered above)."""
    calls = {"n": 0}

    def fake(model, k, manifest):
        ok = pattern[k - 1]
        calls["n"] += 1
        m = _m(capture=1.0 if ok else 0.5)
        gate = C.gate_status(m)
        body = json.dumps(m, sort_keys=True).encode()
        return {"schema": TR.VALIDATION_SCHEMA, "rollout": k, "ppo_seed": manifest["ppo_seed"], "model_sha256": manifest["model_sha256"], "num_timesteps": manifest["num_timesteps"],
                "validation_set": TR.validation_set_for(k), "seeds": [0, 0], "metrics": m, "gate": gate, "passed": gate["passed"], "gate_spec": C.GATE,
                "content_sha256": TR.T.sha256_bytes(body), "created_utc": "t"}

    monkeypatch.setattr(TR, "run_validation", fake)
    return calls


def test_layout_isolation_and_namespaces(tmp_path):
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    assert layout.canonical == tmp_path / "r" / "seed_201" and layout.accepted_model.name == "accepted_model.zip"
    with pytest.raises(TR.RunConflictError):
        TR.assert_isolated(TR.PROJECT_ROOT / "results" / "journal_v2_corrected_semantics" / "a1_training" / "a1_v2_training_v1" / "x")
    assert TR.assert_isolated(C.RESULTS_ROOT / "seed_201")
    assert "a1_curriculum" in str(C.RESULTS_ROOT) and "a1_v2_training_v1" not in str(C.RESULTS_ROOT)
    assert TR.select_scratch_root().as_posix().lower().find("onedrive") < 0


def test_early_stop_at_the_second_consecutive_pass_and_accepted_checkpoint(tmp_path, monkeypatch):
    calls = _passing(monkeypatch, [False, True, True])
    result = TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    assert result["status"] == "MASTERED" and result["mastery_rollout"] == 3 and result["timesteps_to_mastery"] == 3 * TINY.steps_per_rollout
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    assert layout.accepted_model.exists() and TR.T.sha256_file(layout.accepted_model) == result["accepted_model_sha256"]
    assert [r["validation_set"] for r in result["rollouts"]] == ["A", "B", "A"] and result["model_role"].startswith("curriculum initialization")
    assert calls["n"] == 3
    again = TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    assert again == result and calls["n"] == 3                                           # a finished seed is never retrained or revalidated


def test_seed_stops_early_after_mastery_even_with_rollouts_left(tmp_path, monkeypatch):
    _passing(monkeypatch, [True, True, False])
    spec = TR.Spec(n_envs=2, n_steps=64, max_rollouts=3)
    result = TR.train_seed(202, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=spec, env_factory=_dummy_factory(202), ppo_kwargs=TINY_KW)
    layout = TR.make_layout(202, tmp_path / "s", tmp_path / "r")
    assert result["mastery_rollout"] == 2 and result["rollouts_trained"] == 2 and not (layout.scratch_rollouts / TR.names(3)["manifest"]).exists()


def test_failure_after_the_maximum_rollouts_is_recorded_and_not_extended(tmp_path, monkeypatch):
    _passing(monkeypatch, [True, False, True])
    result = TR.train_seed(203, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(203), ppo_kwargs=TINY_KW)
    assert result["status"] == "FAIL" and result["mastery_rollout"] is None and result["timesteps_to_mastery"] is None and result["rollouts_trained"] == 3
    assert result["model_role"].startswith("failed Stage-1 seed")


def test_commit_marker_recover_discards_partial_rollout_and_resumes_from_the_highest_verified(tmp_path, monkeypatch):
    _passing(monkeypatch, [False, False, False])

    def hook(stage, k):
        if stage == "before_manifest" and k == 2:
            raise TR.SimulatedPowerLoss()

    with pytest.raises(TR.SimulatedPowerLoss):
        TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, fault_hook=hook)
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    assert (layout.scratch_rollouts / TR.names(2)["model"]).exists() and not (layout.scratch_rollouts / TR.names(2)["manifest"]).exists()
    assert not list(layout.scratch.glob("*.lock"))                                       # the lock was released by the finally block
    completed, manifest = TR.recover(layout, spec=TINY)
    assert completed == 1 and manifest["rollout"] == 1 and not (layout.scratch_rollouts / TR.names(2)["model"]).exists()
    result = TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    assert result["status"] == "FAIL" and result["rollouts_trained"] == 3


def test_tampered_or_missing_artifacts_invalidate_a_rollout_and_the_mirror_restores_a_lost_scratch(tmp_path, monkeypatch):
    _passing(monkeypatch, [False, False, False])
    TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, stop_after=2)
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    assert TR.verify_rollout(layout.scratch_rollouts, 2, seed=201, spec=TINY) and TR.verify_rollout(layout.canonical_rollouts, 2, seed=201, spec=TINY)
    (layout.scratch_rollouts / TR.names(2)["model"]).write_bytes(b"corrupt")
    assert TR.verify_rollout(layout.scratch_rollouts, 2, seed=201, spec=TINY) is None
    completed, _m2 = TR.recover(layout, spec=TINY)
    assert completed == 2                                                                # restored from the verified OneDrive mirror
    import shutil

    shutil.rmtree(layout.scratch_rollouts)
    completed, manifest = TR.recover(layout, spec=TINY)
    assert completed == 2 and TR.verify_rollout(layout.scratch_rollouts, 2, seed=201, spec=TINY)


def test_validation_files_are_hash_bound_to_their_model_and_recomputed_when_invalid(tmp_path, monkeypatch):
    calls = _passing(monkeypatch, [True, True, True])
    TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, stop_after=1)
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    manifest = TR.verify_rollout(layout.scratch_rollouts, 1, seed=201, spec=TINY)
    assert TR.verify_validation(layout.scratch_rollouts, 1, manifest)
    path = layout.scratch_rollouts / TR.names(1)["validation"]
    payload = json.loads(path.read_text())
    payload["metrics"]["capture"] = 0.0
    path.write_text(json.dumps(payload))
    assert TR.verify_validation(layout.scratch_rollouts, 1, manifest) is None
    before = calls["n"]
    TR.ensure_validation(layout, 1, manifest)
    assert calls["n"] == before + 1 and TR.verify_validation(layout.scratch_rollouts, 1, manifest)


def test_duplicate_launch_for_the_same_seed_is_refused_and_a_dead_lock_is_taken_over(tmp_path, monkeypatch):
    _passing(monkeypatch, [False, False, False])
    layout = TR.make_layout(201, tmp_path / "s", tmp_path / "r")
    layout.scratch.mkdir(parents=True)
    holder = TR.CellLock(layout.lock_base)
    assert holder.acquire()
    with pytest.raises(TR.RunConflictError):
        TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, stop_after=1)
    holder.release()
    (layout.scratch / "train.lock").write_text("999999")
    assert TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, stop_after=1)


def test_rollout_is_a_pure_function_of_state_and_rollout_index_dummy_vec_env(tmp_path, monkeypatch):
    """Interrupt-and-resume vs uninterrupted gives identical policy+optimizer state (fast Dummy-VecEnv version of the real-SubprocVecEnv validation)."""
    _passing(monkeypatch, [False, False, False])
    TR.train_seed(201, scratch_root=tmp_path / "a_s", results_root=tmp_path / "a_r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    TR.train_seed(201, scratch_root=tmp_path / "b_s", results_root=tmp_path / "b_r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW, stop_after=1)
    TR.train_seed(201, scratch_root=tmp_path / "b_s", results_root=tmp_path / "b_r", spec=TINY, env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    la, lb = TR.make_layout(201, tmp_path / "a_s", tmp_path / "a_r"), TR.make_layout(201, tmp_path / "b_s", tmp_path / "b_r")
    for k in (1, 2, 3):
        ma, mb = TR.verify_rollout(la.scratch_rollouts, k, seed=201, spec=TINY), TR.verify_rollout(lb.scratch_rollouts, k, seed=201, spec=TINY)
        assert ma["state_sha256"] == mb["state_sha256"] and ma["episodes_sha256"] == mb["episodes_sha256"]


def test_fresh_initialization_never_loads_a_prior_model(tmp_path, monkeypatch):
    _passing(monkeypatch, [False, False, False])
    import stable_baselines3

    loaded = []
    original = stable_baselines3.PPO.load
    monkeypatch.setattr(stable_baselines3.PPO, "load", classmethod(lambda cls, *a, **k: loaded.append(a) or original.__func__(cls, *a, **k)))
    TR.train_seed(201, scratch_root=tmp_path / "s", results_root=tmp_path / "r", spec=TR.Spec(n_envs=2, n_steps=64, max_rollouts=1), env_factory=_dummy_factory(201), ppo_kwargs=TINY_KW)
    assert loaded == []                                                                  # a fresh seed constructs PPO(...) and loads nothing (in particular no failed A1 model)


def test_learning_curve_data_and_state_stages():
    assert len(TS.STAGES) == 12 and TS.STAGES[0] == "01_repository_verification" and TS.STAGES[-1] == "12_commit_push"
    assert [s for s in TS.STAGES if "training" in s] == ["06_seed_201_training", "07_seed_202_training", "08_seed_203_training"]


def _seed_manifest(status):
    return {"status": status, "mastery_rollout": 4 if status == "MASTERED" else None, "timesteps_to_mastery": 4 * 32768 if status == "MASTERED" else None}


def _eval(passed=True, **over):
    m = _m(**over)
    m.update({"orbit_episodes": 0, "repeated_flyby_episodes": 0, "nonclosing_episodes": 0})
    m.update({k: v for k, v in over.items() if k in m})
    return {"metrics": m, "gate": {"passed": passed}}


def test_stage_decision_rule_requires_two_mastered_seeds_final_gates_and_no_common_pathology():
    from experiments import journal_v2_a1_cur1_eval as EV

    ok = {"per_seed": {"201": _eval(), "202": _eval(), "203": _eval(False)}, "episodes": 200}
    d = EV.decide(ok, {201: _seed_manifest("MASTERED"), 202: _seed_manifest("MASTERED"), 203: _seed_manifest("FAIL")})
    assert d["stage1_pass"] and d["mastered_seeds"] == [201, 202] and d["seeds_failed"] == [203] and not d["all_three_mastered"]
    one = EV.decide(ok, {201: _seed_manifest("MASTERED"), 202: _seed_manifest("FAIL"), 203: _seed_manifest("FAIL")})
    assert not one["stage1_pass"]
    bad_final = {"per_seed": {"201": _eval(), "202": _eval(False)}, "episodes": 200}
    assert not EV.decide(bad_final, {201: _seed_manifest("MASTERED"), 202: _seed_manifest("MASTERED")})["stage1_pass"]
    common = {"per_seed": {"201": _eval(), "202": _eval()}, "episodes": 200}
    for s in ("201", "202"):
        common["per_seed"][s]["metrics"]["repeated_flyby_episodes"] = 5
    d2 = EV.decide(common, {201: _seed_manifest("MASTERED"), 202: _seed_manifest("MASTERED")})
    assert not d2["stage1_pass"] and "repeated_flyby" in d2["common_structural_pathology"]
    assert not EV.decide({"per_seed": {}, "episodes": 200}, {201: _seed_manifest("FAIL"), 202: _seed_manifest("FAIL"), 203: _seed_manifest("FAIL")})["stage1_pass"]


def test_learning_curves_render_png_and_pdf(tmp_path):
    from experiments import journal_v2_a1_cur1_eval as EV

    def roll(k):
        return {"num_timesteps": k * 32768, "validation": {"capture": 0.1 * k, "median_path_excess": 1.5, "exhaustion": 0.1, "timeout": 0.5}, "training_summary": {"mean_return": -18.0 + k}}

    manifests = {s: {"status": "FAIL", "timesteps_to_mastery": None, "rollouts": [roll(k) for k in range(1, 9)]} for s in (201, 202, 203)}
    paths = EV.plot_curves(manifests, tmp_path)
    assert [p.rsplit(".", 1)[1] for p in paths] == ["png", "pdf"] and all((tmp_path / f"stage1_learning_curves.{e}").stat().st_size > 1000 for e in ("png", "pdf"))


def test_not_applicable_stage_needs_a_recorded_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    state = TS.load_state()
    state["stages"]["10_visualizer_recordings"] = {"status": "not_applicable"}
    assert not TS.stage_is_verified(state, "10_visualizer_recordings")
    TS.skip(state, "10_visualizer_recordings", "fewer than 2/3 seeds mastered")
    assert TS.stage_is_verified(TS.load_state(), "10_visualizer_recordings")
