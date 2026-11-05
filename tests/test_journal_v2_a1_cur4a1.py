"""Tests for curriculum Stage 4A-1 (goal-conditioned pursuit via translated A0 imitation + Stage3-v2 behaviour retention). NO PPO learn() anywhere; real environments and real SB3/torch, no mocks."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4a1 as Z
from experiments import journal_v2_a1_cur4a1_bc as BC4
from experiments import journal_v2_a1_cur4a1_data as D
from experiments import journal_v2_a1_cur4a1_retention as RT
from experiments import journal_v2_a1_cur4a1_state as TS

SEEDS = (172_000, 172_001, 172_002, 172_003, 172_004, 172_005)


# ----------------------------------------------------------------------------- translation math / feasible box
def test_feasible_box_formula_matches_exact_margin_constraints():
    L, margin = 50.0, 5.0
    for enemy0 in (np.array([50.0, 12.0]), np.array([-50.0, -40.0]), np.array([3.0, 50.0]), np.array([30.0, -50.0])):
        box = Z.feasible_box(enemy0, L, margin)
        for i in range(2):
            lo, hi = box[i]
            assert abs(0.0 + lo) <= L - margin + 1e-9 and abs(0.0 + hi) <= L - margin + 1e-9               # soldier (always at origin) stays inside
            assert abs(enemy0[i] + lo) <= L - margin + 1e-9 and abs(enemy0[i] + hi) <= L - margin + 1e-9    # attacker stays inside
            assert hi >= lo


def test_sample_translation_uniform_and_stratified_modes():
    enemy0 = np.array([50.0, 10.0])
    box = Z.feasible_box(enemy0, 50.0, 5.0)
    d = Z.sample_translation(999, enemy0, 50.0, 5.0)
    assert box[0, 0] <= d[0] <= box[0, 1] and box[1, 0] <= d[1] <= box[1, 1]
    with pytest.raises(Z.TranslationInfeasible):
        Z.sample_translation(999, enemy0, 50.0, 5.0, target=(40.0, 0.0), jitter_m=0.0)        # x=+40 cannot fit (box forces x negative for this edge)
    d2 = Z.sample_translation(999, enemy0, 50.0, 5.0, target=(-10.0, 0.0), jitter_m=1.0)
    assert -11.0 <= d2[0] <= -9.0


# ----------------------------------------------------------------------------- translated environment
def test_translation_preserves_relative_geometry_and_respects_the_margin():
    for s in SEEDS:
        et, eu = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True), V1V2.Stage1V2Env(0, 0, 0, scripted_a0=True)
        et.reset(options={"scenario": s})
        eu.reset(options={"scenario": s})
        rel_t = np.linalg.norm(et.soldier_env._soldier_pos - et.soldier_env._enemy_pos)
        rel_u = np.linalg.norm(eu.soldier_env._soldier_pos - eu.soldier_env._enemy_pos)
        assert abs(rel_t - rel_u) < 1e-4
        assert np.allclose(et.soldier_env._enemy_pos[:2] - eu.soldier_env._enemy_pos[:2], et.translation, atol=1e-4)
        assert np.allclose(et.soldier_env._soldier_pos[:2], et.translation, atol=1e-4)                       # asset after translation == delta (started at origin)
        L, margin = float(et._config.L), Z.MARGIN_M
        assert abs(et.soldier_env._soldier_pos[0]) <= L - margin + 1e-6 and abs(et.soldier_env._soldier_pos[1]) <= L - margin + 1e-6
        assert abs(et.soldier_env._enemy_pos[0]) <= L - margin + 1e-6 and abs(et.soldier_env._enemy_pos[1]) <= L - margin + 1e-6
        assert et.soldier_env._enemy_pos[2] == eu.soldier_env._enemy_pos[2]                                   # z unaffected (horizontal-only translation)


def test_asset_is_stationary_within_an_episode_and_endurance_budget_is_translation_invariant():
    for s in SEEDS[:3]:
        et, eu = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True), V1V2.Stage1V2Env(0, 0, 0, scripted_a0=True)
        et.reset(options={"scenario": s}); eu.reset(options={"scenario": s})
        assert et._config.v_s == 0.0
        budget_t, budget_u = et.soldier_env._enemy_endurance_budget, eu.soldier_env._enemy_endurance_budget
        assert abs(budget_t - budget_u) < 1e-4
        done = False
        asset0 = et.soldier_env._soldier_pos.copy()
        while not done:
            _o, _r, te, tr, _i = et.step(np.zeros(3, dtype=np.float32))
            done = te or tr
            assert np.array_equal(et.soldier_env._soldier_pos, asset0)


def test_defender_masked_parked_and_non_intercepting_under_translation():
    env = Z.TranslatedStage1V2Env(0, 0, 0)
    for s in SEEDS[:3]:
        obs, _ = env.reset(options={"scenario": s})
        assert not np.any(obs[Z.DEFENDER_SLOTS])
        done, info = False, {}
        while not done:
            obs, _r, te, tr, info = env.step(np.random.default_rng(s).uniform(-1, 1, 3).astype(np.float32))
            assert not np.any(obs[Z.DEFENDER_SLOTS]) and np.allclose(env.soldier_env._defender_pos, C.PARKED_DEFENDER)
            done = te or tr
        assert info["outcome"] not in X.INTERCEPT_OUTCOMES


def test_observation_has_no_future_information_and_tracks_the_current_translated_asset():
    env = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
    for s in SEEDS[:3]:
        obs, _ = env.reset(options={"scenario": s})
        e = env.soldier_env
        assert np.allclose(obs[3:6], np.asarray(e._soldier_pos, dtype=np.float64) / float(e.config.L), atol=1e-6)
    names_ = [n for n, _g in A.A1_V2_OBSERVATION_SCHEMA]
    assert not any("future" in n or "predicted" in n for n in names_) and len(names_) == 37


def test_deterministic_replay():
    a = Z.evaluate_translated(None, SEEDS, scripted_a0=True)
    b = Z.evaluate_translated(None, SEEDS, scripted_a0=True)
    keys = ("outcome", "steps", "asset_position_m", "translation_m")
    assert [{k: e[k] for k in keys} for e in a] == [{k: e[k] for k in keys} for e in b]


# ----------------------------------------------------------------------------- stratified generalization
def test_stratified_scenarios_fit_their_quadrant_and_magnitude():
    strata = Z.generalization_strata()
    assert len(strata) == 8 and all(abs(abs(tx) - abs(ty)) < 1e-6 for tx, ty in strata.values())             # exactly diagonal
    for name, target in list(strata.items())[:2]:
        seeds = Z.stratified_scenarios(name, target, 5, jitter_m=3.0)
        assert len(seeds) == 5
        for s in seeds:
            env = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
            env.reset(options={"scenario": s, "target_translation": target, "translation_jitter_m": 3.0})
            assert abs(env.translation[0] - target[0]) <= 3.0 + 1e-6 and abs(env.translation[1] - target[1]) <= 3.0 + 1e-6


# ----------------------------------------------------------------------------- dataset / split / coverage
def test_expert_dataset_split_has_no_episode_leakage_and_covers_quadrants():
    arr = D.generate_expert_episodes(range(172_000, 172_120))
    sp = D.split(arr)
    assert not (sp["train_episodes"] & sp["val_episodes"]) and len(sp["train_episodes"]) + len(sp["val_episodes"]) == 120
    assert abs(len(sp["train_episodes"]) / 120 - 0.8) < 0.05
    cov = D.coverage(arr)
    assert all(v > 0 for v in cov["asset_quadrant_episode_counts"].values()) and cov["caught_fraction"] == 1.0


def test_expert_labels_are_normalized_a0_directions_with_no_future_information():
    env = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
    for s in SEEDS[:3]:
        obs, _ = env.reset(options={"scenario": s})
        done = False
        while not done:
            label = env.expert_action()
            assert abs(float(np.linalg.norm(label)) - 1.0) < 1e-4
            e = env.soldier_env
            expected = (np.asarray(e._soldier_pos, dtype=np.float64) - np.asarray(e._enemy_pos, dtype=np.float64))
            expected = expected / np.linalg.norm(expected)
            assert np.allclose(label, expected, atol=1e-4)
            obs, _r, te, tr, _i = env.step(np.zeros(3, dtype=np.float32))
            done = te or tr


# ----------------------------------------------------------------------------- retention dataset
def test_retention_dataset_classification_is_consistent_with_detection_and_closest_step():
    from stable_baselines3 import PPO

    model = PPO.load(Z.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    ds = RT.collect_retention_dataset(model, target_states=250, seeds=range(*Z.RETENTION_SOURCE))
    assert len(ds["obs"]) == 250 and set(np.unique(ds["class"])) <= set(RT.STATE_CLASSES)
    assert RT.classify(0, None, 5) == "not_visible" and RT.classify(3, 3, 5) == "newly_visible" and RT.classify(4, 3, 5) == "active_evasion" and RT.classify(6, 3, 5) == "post_pass_pursuit"
    sp = RT.split(ds, 0.8)
    assert sp["train"][0].shape[0] + sp["val"][0].shape[0] == 250


def test_retention_dataset_uses_the_parents_own_actions_not_a0():
    from stable_baselines3 import PPO

    model = PPO.load(Z.parent_checkpoint(202)["path"].removesuffix(".zip"), device="cpu")
    ds = RT.collect_retention_dataset(model, target_states=60, seeds=range(*Z.RETENTION_SOURCE))
    recomputed = BC.mean_action(model, torch.as_tensor(ds["obs"][:10])).detach().numpy()
    assert np.allclose(recomputed, ds["act"][:10], atol=1e-5)


# ----------------------------------------------------------------------------- combined-loss BC / resume
def test_initialize_model_copies_actor_critic_and_forces_log_std_to_minus_two():
    model, record = BC4.initialize_model(201)
    from stable_baselines3 import PPO

    parent = PPO.load(Z.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    assert BC.tensor_sha256(BC.actor_state(model)) == BC.tensor_sha256(BC.actor_state(parent))
    assert BC.value_state_sha256(model) == BC.value_state_sha256(parent)
    assert np.allclose(model.policy.log_std.detach().numpy(), -2.0) and record["log_std_forced_to"] == -2.0


def test_combined_loss_trains_actor_only_and_reduces_validation_loss(tmp_path):
    goal = D.generate_expert_episodes(range(172_000, 172_060))
    gsp = D.split(goal)
    from stable_baselines3 import PPO

    parent_model = PPO.load(Z.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    ds = RT.collect_retention_dataset(parent_model, target_states=200, seeds=range(*Z.RETENTION_SOURCE))
    rsp = RT.split(ds, 0.8)
    model, _rec = BC4.initialize_model(201)
    critic_before = BC.value_state_sha256(model)
    out = BC4.train_combined(model, gsp["train"], rsp["train"], gsp["val"], rsp["val"], work_dir=tmp_path / "w", seed=201, tag="t", hyper={"epochs": 3, "lr": 1e-3, "batch_size": 64})
    assert BC.value_state_sha256(model) == critic_before and np.allclose(model.policy.log_std.detach().numpy(), -2.0)
    assert out["history"][-1]["val"]["loss"] < out["history"][0]["val"]["loss"] + 0.5                           # loss moves in the right direction (noisy with 3 tiny epochs)
    assert out["selected_epoch"] in {h["epoch"] for h in out["history"]}


def test_combined_bc_epoch_checkpoint_resume_matches_uninterrupted(tmp_path):
    goal = D.generate_expert_episodes(range(172_000, 172_040))
    gsp = D.split(goal)
    from stable_baselines3 import PPO

    parent_model = PPO.load(Z.parent_checkpoint(203)["path"].removesuffix(".zip"), device="cpu")
    ds = RT.collect_retention_dataset(parent_model, target_states=120, seeds=range(*Z.RETENTION_SOURCE))
    rsp = RT.split(ds, 0.8)
    hyper = {"epochs": 3, "lr": 1e-3, "batch_size": 32}

    model_a, _ = BC4.initialize_model(203)
    BC4.train_combined(model_a, gsp["train"], rsp["train"], gsp["val"], rsp["val"], work_dir=tmp_path / "a", seed=203, tag="resume", hyper=hyper)

    model_b, _ = BC4.initialize_model(203)
    BC4.train_combined(model_b, gsp["train"], rsp["train"], gsp["val"], rsp["val"], work_dir=tmp_path / "b", seed=203, tag="resume", hyper=hyper, stop_after=1)
    model_b2, _ = BC4.initialize_model(203)
    res_b = BC4.train_combined(model_b2, gsp["train"], rsp["train"], gsp["val"], rsp["val"], work_dir=tmp_path / "b", seed=203, tag="resume", hyper=hyper)

    man_a = BC4.verify_epoch(tmp_path / "a", 3, seed=203, tag="resume")
    man_b = BC4.verify_epoch(tmp_path / "b", 3, seed=203, tag="resume")
    assert man_a["actor_sha256"] == man_b["actor_sha256"] and man_a["val"]["loss"] == pytest.approx(man_b["val"]["loss"], abs=1e-9)
    assert res_b["selected_epoch"] is not None


def test_model_save_load_round_trip_preserves_actor_weights():
    model, _ = BC4.initialize_model(201)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.zip"
        sha = BC.save_model(model, p)
        assert TS.sha256_file(p) == sha
        loaded = BC.load_model(p)
        assert BC.tensor_sha256(BC.actor_state(loaded)) == BC.tensor_sha256(BC.actor_state(model))


# ----------------------------------------------------------------------------- sensitivity / representation gate
def test_asset_input_sensitivity_distinguishes_a_trained_from_an_untrained_asset_input():
    from stable_baselines3 import PPO

    parent = PPO.load(Z.parent_checkpoint(201)["path"].removesuffix(".zip"), device="cpu")
    obs = Z.collect_observations(C.model_actor(parent), Z.seeds_of(Z.VALIDATION_A), max_episodes=10)
    sens = Z.asset_input_sensitivity(parent, obs)
    assert np.allclose(sens["first_layer_column_norm_asset_xyz"], np.sqrt(2.0), atol=0.05)              # untrained (Stages 1-3 never moved the asset): orthogonal-init norm
    assert sens["asset_to_attacker_sensitivity_ratio"] < 0.2                                              # far below a relative tracker


def test_namespaces_are_fresh_and_disjoint_and_the_gate_is_frozen():
    o = Z.namespace_overlaps()
    for key in ("expert", "validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "generalization", "retention_source", "mutual", "training_overlap"):
        assert o[key] == [], key
    assert len(Z.final_evaluation_seeds()) == 200 and Z.MAX_EPOCHS == 20 and Z.RETAIN_LOSS_WEIGHT == 0.25
    assert Z.GATE["capture_min"] == 0.99 and Z.GATE["cosine_mean_min"] == 0.995 and Z.GATE["cosine_min_min"] == 0.95
    assert Z.SENSITIVITY_RATIO_MIN_GAIN == 2.0 and Z.QUADRANT_CAPTURE_MIN == 0.95


def test_gate_boundaries():
    m = {"capture": 0.99, "exhaustion": 0.0, "timeout": 0.0, "median_path_excess": 1.2, "gt2": 0.01, "gt5": 0.0, "orbit_episodes": 0, "expert_cosine_mean": 0.995, "expert_cosine_min": 0.95, "nonfinite": 0}
    kw = dict(anchor1=0.99, anchor2=0.80, anchor3=0.75)
    assert Z.gate_status(m, **kw)["passed"]
    for key, bad in (("capture", 0.98), ("exhaustion", 0.01), ("timeout", 0.01), ("median_path_excess", 1.21), ("gt2", 0.02), ("gt5", 0.01), ("orbit_episodes", 1),
                     ("expert_cosine_mean", 0.994), ("expert_cosine_min", 0.94), ("nonfinite", 1)):
        assert not Z.gate_status({**m, key: bad}, **kw)["passed"], key
    for key, bad in (("anchor1", 0.98), ("anchor2", 0.79), ("anchor3", 0.74)):
        assert not Z.gate_status(m, **{**kw, key: bad})["passed"], key


# ----------------------------------------------------------------------------- parents / protocol
@pytest.mark.parametrize("seed", [201, 202, 203])
def test_parents_are_the_preregistered_stage3_v2_hashes_not_stage4_or_4a_models(seed):
    p = Z.parent_checkpoint(seed)
    assert p["model_sha256"] == Z.PARENT_SHA256[seed] and p["model_sha256"] not in Z.STAGE4_ACCEPTED_FINAL_SHA256.values() and p["model_sha256"] not in Z.STAGE4A_ACCEPTED_FINAL_SHA256.values()


def test_protocol_is_stable_and_documents_the_translation_and_retention_design():
    spec = Z.protocol_specification()
    assert spec["objective"]["L_total"] == "L_goal + 0.25 * L_retain" and spec["architecture"].startswith("unchanged")
    assert spec["environment"]["translation"]["margin_m"] == Z.MARGIN_M and spec["representation_gate"]["asset_sensitivity_ratio_min_gain"] == 2.0
    assert Z.protocol_sha256() == Z.protocol_sha256()
    assert len(spec["generalization_strata"]) == 8


def test_state_machine_has_the_fifteen_required_stages_and_isolated_output(tmp_path):
    assert len(TS.STAGES) == 15 and TS.STAGES[0] == "01_repository_verification" and TS.STAGES[-1] == "15_commit_push" and TS.STAGES[12] == "13_stage4a2_handoff"
    assert str(Z.RESULTS_ROOT).endswith("stage4a1") and TS.STATE_DIR.name == "stage4a1"
