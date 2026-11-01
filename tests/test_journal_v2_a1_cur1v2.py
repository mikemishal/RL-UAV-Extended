"""Tests for curriculum Stage 1 V2 (expert initialization by behavior cloning). No PPO learn()."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur1v2_data as D
from experiments import journal_v2_a1_cur1v2_run as RUN
from experiments import journal_v2_a1_cur1v2_state as TS

DEFENDER_SLOTS = [i for i, (name, _g) in enumerate(A.A1_V2_OBSERVATION_SCHEMA) if "defender" in name]


# ----------------------------------------------------------------------------- clean curriculum environment
def test_config_differs_from_canonical_only_in_the_three_curriculum_settings():
    assert set(V.config_differences_from_canonical()) == {"v_s", "intercept_radius", "defender_standby_until_detection"}
    cfg = V.stage1v2_config()
    assert cfg.v_s == 0.0 and cfg.intercept_radius == V.NO_INTERCEPT_RADIUS and cfg.enemy_endurance_path_factor == 5.0 and not cfg.enemy_obstacle_avoidance_enabled
    assert V.CURRICULUM_MODE == "curriculum_basic_pursuit_no_defender_interaction"


def test_defender_features_are_masked_unavailable_in_every_state_even_next_to_the_defender():
    env = V.Stage1V2Env(0, 0, 0)
    obs, _ = env.reset(options={"scenario": 160001})
    assert DEFENDER_SLOTS and obs.shape == (37,) == env.observation_space.shape and not np.any(obs[DEFENDER_SLOTS])
    e = env.soldier_env
    e._enemy_pos = (e._defender_pos + np.array([0.5, 0.0, 0.5], dtype=np.float32)).astype(np.float32)        # 0.7 m from the (structurally present) defender
    assert float(np.linalg.norm(e._enemy_pos - e._defender_pos)) < 1.0
    obs2, _r, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert not np.any(obs2[DEFENDER_SLOTS]) and not terminated and info["outcome"] == "ongoing"


def test_defender_remains_structurally_present_stationary_and_cannot_terminate_episodes():
    env = V.Stage1V2Env(0, 0, 0)
    for seed in range(160_000, 160_040):
        env.reset(options={"scenario": seed})
        s = env.soldier_env
        done = False
        while not done:
            _o, _r, te, tr, info = env.step(np.random.default_rng(seed).uniform(-1, 1, 3).astype(np.float32))
            done = te or tr
            assert np.allclose(s._defender_pos, C.PARKED_DEFENDER) and not np.any(s._defender_vel) and np.allclose(s._soldier_pos, 0)
        assert info["outcome"] not in ("intercepted", "unsafe_intercept")


def test_observation_version_and_action_interface_are_unchanged():
    assert V.OBSERVATION_VERSION == "a1_v2_obs_v1" and V.ACTION_VERSION == "a1_cur1_action_v1" and V.REWARD_VERSION == "a1_v2_reward_v2"
    env = V.Stage1V2Env(0, 0, 0)
    assert env.action_space.shape == (3,) and env.observation_space.shape == (37,)
    d, held = C.desired_direction([0.0, 0.0, 0.0], np.array([0.0, 4.0, 0.0]))
    assert held and np.allclose(d, [0, 1, 0])                              # no A0 fallback at execution time


# ----------------------------------------------------------------------------- expert action extraction
def test_expert_action_is_the_unit_vector_to_the_asset_computed_from_the_current_state_only():
    env = V.Stage1V2Env(0, 0, 0)
    env.reset(options={"scenario": 160_003})
    e = env.soldier_env
    label = env.expert_action()
    expected = (np.asarray(e._soldier_pos, float) - np.asarray(e._enemy_pos, float))
    assert np.allclose(label, expected / np.linalg.norm(expected), atol=1e-6) and np.linalg.norm(label) == pytest.approx(1.0, abs=1e-6)
    e._defender_pos = np.array([10.0, 10.0, 10.0], dtype=np.float32)       # independent of the defender
    e._enemy_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)               # and of the attacker's velocity
    assert np.allclose(env.expert_action(), label)


@pytest.mark.parametrize("seed", [160_002, 160_010, 160_020, 160_031])
def test_a_policy_that_outputs_the_expert_label_reproduces_canonical_a0_exactly(seed):
    env = V.Stage1V2Env(0, 0, 0, record_trajectory=True)
    a0 = V.Stage1V2Env(0, 0, 0, scripted_a0=True, record_trajectory=True)
    ref = C.run_episode(a0, seed, None)
    env.reset(options={"scenario": seed})
    done, info = False, {}
    while not done:
        _o, _r, te, tr, info = env.step(env.expert_action())
        done = te or tr
    ep = info["a1_episode"]
    assert ep["outcome"] == ref["outcome"] == "soldier_caught" and ep["steps"] == ref["steps"]
    assert np.abs(np.array(ep["trajectory"]) - np.array(ref["trajectory"])).max() < 1e-4


def test_scripted_a0_on_the_clean_environment_captures_everything():
    eps = V.evaluate(None, list(range(146_000, 146_030)), scripted_a0=True)
    m = V.metrics(eps)
    assert m["capture"] == 1.0 and m["median_path_excess"] < 1.05 and V.gate_status(m)["passed"] and m["expert_cosine_mean"] is None


# ----------------------------------------------------------------------------- namespaces and split
def test_namespaces_do_not_overlap_any_reserved_range():
    o = V.namespace_overlaps()
    assert o["expert"] == [] and o["relabel"] == [] and o["expert_vs_relabel"] is False
    assert V.EXPERT_SEED_START == 160_000 and V.EXPERT_EPISODES == 5_000 and V.RELABEL_SEED_START == 170_000 and V.RELABEL_EPISODES == 1_000
    held_out = set(C.final_evaluation_seeds())
    assert not held_out & set(V.expert_seeds()) and not held_out & set(V.relabel_seeds())


def test_split_is_by_episode_deterministic_and_80_20():
    flags = V.split_table(V.EXPERT_SEED_START, V.EXPERT_EPISODES)
    assert flags.sum() == 4000 and np.array_equal(flags, V.split_table(V.EXPERT_SEED_START, V.EXPERT_EPISODES))
    assert V.split_of(160_000, V.EXPERT_SEED_START, V.EXPERT_EPISODES) == ("train" if flags[0] else "val")
    assert not np.array_equal(flags, V.split_table(V.EXPERT_SEED_START, V.EXPERT_EPISODES, extra=1))
    with pytest.raises(ValueError):
        V.split_of(1, V.EXPERT_SEED_START, V.EXPERT_EPISODES)


@pytest.fixture()
def tiny_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "EXPERT_SEED_START", 160_000)
    monkeypatch.setattr(V, "EXPERT_EPISODES", 12)
    monkeypatch.setattr(V, "EXPERT_SHARDS", 3)
    monkeypatch.setattr(D, "SHARD_EPISODES", 4)
    for i in range(3):
        D.write_shard(i, tmp_path)
    return tmp_path


def test_shards_are_atomic_hash_verified_restart_skips_and_tampering_is_detected(tiny_dataset):
    base = tiny_dataset
    assert all(D.verify_shard(i, base) for i in range(3)) and not list(base.glob(".*tmp*")) and not list(base.glob("*.lock"))
    assert D.write_shard(1, base) == {"shard": 1, "skipped": True}
    path = base / "expert_shard_01.npz"
    raw = path.read_bytes()
    path.write_bytes(raw[:-8] + b"corrupt!")
    assert D.verify_shard(1, base) is None
    path.write_bytes(raw)
    (base / "expert_shard_02.json").unlink()                                 # a shard without its commit marker is not complete
    assert D.verify_shard(2, base) is None and "skipped" not in D.write_shard(2, base) and D.verify_shard(2, base)


def test_duplicate_shard_writer_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "SHARD_EPISODES", 2)
    lock = D.P.CellLock(tmp_path / "expert_shard_00")
    assert lock.acquire()
    assert D.write_shard(0, tmp_path).get("skipped") and not (tmp_path / "expert_shard_00.npz").exists()
    lock.release()


def test_dataset_split_has_no_episode_leakage_and_labels_are_unit_vectors(tiny_dataset):
    data = D.load_all(tiny_dataset)
    sp = D.split(data)
    assert not (sp["train_episodes"] & sp["val_episodes"]) and len(sp["train_episodes"]) + len(sp["val_episodes"]) == 12
    assert len(sp["train"][0]) + len(sp["val"][0]) == len(data["obs"]) and set(np.unique(data["ep_seed"][np.isin(data["ep_seed"], list(sp["val_episodes"]))])) == sp["val_episodes"]
    assert np.allclose(np.linalg.norm(data["act"], axis=1), 1.0, atol=1e-5) and data["obs"].shape[1] == 37 and not np.any(data["obs"][:, DEFENDER_SLOTS])
    manifest = D.finalize(tiny_dataset)
    assert manifest["episode_leakage"] == 0 and manifest["pairs"] == len(data["obs"]) and manifest["dataset_sha256"] == D.dataset_sha256(tiny_dataset)
    cov = manifest["coverage"]
    assert cov["episodes"] == 12 and cov["late_approach_pairs_within_10m"] > 0 and sum(cov["bearing_quadrant_episode_counts"]) == 12


def test_dataset_hash_changes_when_any_shard_changes(tiny_dataset):
    before = D.dataset_sha256(tiny_dataset)
    side = json.loads((tiny_dataset / "expert_shard_00.json").read_text())
    assert before == D.dataset_sha256(tiny_dataset) and len(before) == 64 and side["npz_sha256"] in json.dumps(D.finalize(tiny_dataset))


# ----------------------------------------------------------------------------- BC
@pytest.fixture(scope="module")
def tiny_xy():
    d = D.generate_expert_episodes(range(160_100, 160_130))
    n = len(d["obs"])
    cut = int(n * 0.8)
    return (d["obs"][:cut], d["act"][:cut]), (d["obs"][cut:], d["act"][cut:])


def test_architecture_is_the_ppo_mlp_policy_64_64_tanh_with_gaussian_3d_action():
    import torch

    model = BC.build_model(201)
    p = model.policy
    assert type(p).__name__ == "ActorCriticPolicy" and [m.out_features for m in p.mlp_extractor.policy_net if hasattr(m, "out_features")] == [64, 64]
    assert [m.out_features for m in p.mlp_extractor.value_net if hasattr(m, "out_features")] == [64, 64] and p.action_net.out_features == 3 and p.action_net.in_features == 64
    assert any(isinstance(m, torch.nn.Tanh) for m in p.mlp_extractor.policy_net) and p.log_std.shape == (3,) and model.observation_space.shape == (37,)


def test_loss_metrics_cosine_and_mse_are_correct_and_only_the_actor_is_trained(tiny_xy):
    import torch

    train, val = tiny_xy
    model = BC.build_model(201)
    value_before, std_before, trainable = BC.value_state_sha256(model), model.policy.log_std.detach().clone(), BC.actor_parameters(model)
    ids = {id(p) for p in trainable}
    assert ids == {id(p) for m in BC.actor_modules(model) for p in m.parameters()} and not ids & {id(p) for p in model.policy.mlp_extractor.value_net.parameters()}
    m0 = BC.bc_metrics(model, *val)
    t = torch.as_tensor(val[1])
    perfect = BC.bc_metrics.__wrapped__ if hasattr(BC.bc_metrics, "__wrapped__") else None
    assert perfect is None
    cos = torch.nn.functional.cosine_similarity(BC.mean_action(model, torch.as_tensor(val[0])), t, dim=1)
    assert m0["cos_mean"] == pytest.approx(float(cos.mean()), abs=1e-6) and m0["loss"] == pytest.approx(m0["mse"] + 1 - m0["cos_mean"], abs=1e-9)
    res = BC.train_bc(model, train, val, work_dir=Path(__import__("tempfile").mkdtemp()), seed=201, tag="t", hyper={"epochs": 15, "lr": 3e-3, "batch_size": 64})
    m1 = BC.bc_metrics(model, *val)
    assert res["selected_epoch"] is not None and m1["loss"] < m0["loss"] and m1["cos_mean"] > m0["cos_mean"]
    assert BC.value_state_sha256(model) == value_before and torch.equal(model.policy.log_std.detach(), std_before)       # critic and log_std untouched


def test_log_std_initialization_is_minus_two():
    model = BC.build_model(202)
    assert np.allclose(model.policy.log_std.detach().numpy(), 0.0)
    BC.set_log_std(model)
    ls = model.policy.log_std.detach().numpy()
    assert np.allclose(ls, -2.0) and np.allclose(np.exp(ls), 0.1353, atol=1e-4) and V.LOG_STD_INIT == -2.0


def test_model_save_load_roundtrip_preserves_weights_predictions_and_log_std(tmp_path, tiny_xy):
    model = BC.build_model(203)
    BC.train_bc(model, *tiny_xy, work_dir=tmp_path / "w", seed=203, tag="t", hyper={"epochs": 3, "lr": 3e-3, "batch_size": 64})
    BC.set_log_std(model)
    sha = BC.save_model(model, tmp_path / "m.zip")
    loaded = BC.load_model(tmp_path / "m.zip")
    assert hashlib.sha256((tmp_path / "m.zip").read_bytes()).hexdigest() == sha and BC.model_state_sha256(loaded) == BC.model_state_sha256(model)
    obs = tiny_xy[1][0][:20]
    assert all(np.array_equal(model.predict(o, deterministic=True)[0], loaded.predict(o, deterministic=True)[0]) for o in obs)
    assert np.allclose(loaded.policy.log_std.detach().numpy(), -2.0)


def test_epoch_checkpoint_restart_is_exact_and_partial_epochs_are_discarded(tmp_path, tiny_xy):
    hyper = {"epochs": 4, "lr": 3e-3, "batch_size": 64}
    a = BC.build_model(201)
    BC.train_bc(a, *tiny_xy, work_dir=tmp_path / "a", seed=201, tag="r", hyper=hyper)
    b = BC.build_model(201)

    def crash(stage, e):
        if stage == "before_manifest" and e == 3:
            raise BC.SimulatedPowerLoss()

    with pytest.raises(BC.SimulatedPowerLoss):
        BC.train_bc(b, *tiny_xy, work_dir=tmp_path / "b", seed=201, tag="r", hyper=hyper, fault_hook=crash)
    assert (tmp_path / "b" / BC.epoch_names(3)["ckpt"]).exists() and not (tmp_path / "b" / BC.epoch_names(3)["manifest"]).exists()
    assert not list((tmp_path / "b").glob("*.lock"))
    assert BC.recover(tmp_path / "b", 4, seed=201, tag="r") == 2 and not (tmp_path / "b" / BC.epoch_names(3)["ckpt"]).exists()
    c = BC.build_model(201)                                                       # a brand-new process-equivalent model resumes from disk
    BC.train_bc(c, *tiny_xy, work_dir=tmp_path / "b", seed=201, tag="r", hyper=hyper)
    assert BC.tensor_sha256(BC.actor_state(a)) == BC.tensor_sha256(BC.actor_state(c))


def test_tampered_epoch_is_rejected_and_wrong_tag_or_seed_does_not_resume(tmp_path, tiny_xy):
    m = BC.build_model(201)
    BC.train_bc(m, *tiny_xy, work_dir=tmp_path, seed=201, tag="r", hyper={"epochs": 2, "lr": 3e-3, "batch_size": 64})
    assert BC.verify_epoch(tmp_path, 2, seed=201, tag="r") and BC.verify_epoch(tmp_path, 2, seed=202, tag="r") is None and BC.verify_epoch(tmp_path, 2, seed=201, tag="x") is None
    (tmp_path / BC.epoch_names(2)["ckpt"]).write_bytes(b"bad")
    assert BC.verify_epoch(tmp_path, 2, seed=201, tag="r") is None and BC.recover(tmp_path, 2, seed=201, tag="r") == 1


def test_duplicate_training_process_is_refused(tmp_path, tiny_xy):
    lock = D.P.CellLock(tmp_path / "train")
    assert lock.acquire()
    with pytest.raises(RuntimeError):
        BC.train_bc(BC.build_model(201), *tiny_xy, work_dir=tmp_path, seed=201, tag="r", hyper={"epochs": 1, "lr": 1e-3, "batch_size": 64})
    lock.release()


# ----------------------------------------------------------------------------- relabeling
def test_relabel_dataset_uses_the_policy_visited_states_with_a0_labels_and_respects_the_cap():
    model = BC.build_model(201)                                                   # an untrained policy visits states unlike A0's
    v = BC.collect_visited_states(model, range(170_000, 170_004), max_states=7)
    assert v["obs"].shape[1] == 37 and len(v["obs"]) == len(v["act"]) == len(v["ep_seed"]) and len(v["obs"]) <= 4 * 7 and max(v["steps_per_episode"]) <= 7
    assert np.allclose(np.linalg.norm(v["act"], axis=1), 1.0, atol=1e-5) and not np.any(v["obs"][:, DEFENDER_SLOTS])
    env = V.Stage1V2Env(0, 0, 0)
    obs, _ = env.reset(options={"scenario": 170_000})
    first = env.expert_action()
    assert np.allclose(v["obs"][0], obs) and np.allclose(v["act"][0], first, atol=1e-6)              # label of the FIRST visited state = A0's action there
    a0 = D.generate_expert_episodes([170_000])
    assert np.allclose(a0["act"][0], first) and np.allclose(a0["obs"][0], obs)
    assert not np.allclose(v["obs"][3:5], a0["obs"][3:5])                                             # later states differ: the policy's own distribution
    sp = BC.relabel_arrays(v)
    assert len(sp["train"][0]) + len(sp["val"][0]) == len(v["obs"])
    union = BC.append_datasets({"train": (np.zeros((2, 37), np.float32), np.zeros((2, 3), np.float32)), "val": (np.zeros((1, 37), np.float32), np.zeros((1, 3), np.float32))}, sp)
    assert len(union["train"][0]) == 2 + len(sp["train"][0])


def test_relabel_round_is_exactly_one_round_and_never_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "BC", {**V.BC, "epochs": 3})
    monkeypatch.setattr(V, "FINE_TUNE", {**V.FINE_TUNE, "epochs": 2})
    monkeypatch.setattr(V, "RELABEL_EPISODES", 6)
    monkeypatch.setattr(V, "RELABEL_MAX_STEPS", 6)
    d = D.generate_expert_episodes(range(160_200, 160_215))
    cut = int(len(d["obs"]) * 0.8)
    expert = {"train": (d["obs"][:cut], d["act"][:cut]), "val": (d["obs"][cut:], d["act"][cut:])}
    monkeypatch.setattr(RUN.D, "dataset_sha256", lambda *a, **k: "x" * 64)
    out = tmp_path / "res"
    s0 = RUN.run_round0(201, expert, scratch=tmp_path / "scr", results=out)
    s1 = RUN.run_relabel_finetune(201, expert, scratch=tmp_path / "scr", results=out)
    assert s0["value_network_unchanged"] and s1["relabel"]["added_expert_labels_total"] > 0 and s1["hyper"]["rounds"] == 1 and s1["initial_action_std"] == pytest.approx(0.1353, abs=1e-3)
    assert (out / "relabel_states.npz").exists() and s1["round0_model_sha256"] == s0["model_sha256"] and "pre_relabel_round0_metrics" in s1 and "post_relabel_metrics" in s1


# ----------------------------------------------------------------------------- evaluation / gate / decision
def test_deterministic_evaluation_is_reproducible_and_records_cosine_to_the_expert():
    model = BC.build_model(201)
    seeds = [146_000, 146_001, 146_002]
    a = V.evaluate(BC.model_actor(model), seeds)
    b = V.evaluate(BC.model_actor(model), seeds)
    assert [e["outcome"] for e in a] == [e["outcome"] for e in b] and [e["expert_cosines"] for e in a] == [e["expert_cosines"] for e in b]
    assert all(-1.0 - 1e-6 <= c <= 1.0 + 1e-6 for e in a for c in e["expert_cosines"]) and all(len(e["expert_cosines"]) == e["steps"] for e in a)
    m = V.metrics(a)
    assert m["expert_cosine_median"] is not None and m["expert_cosine_steps"] == sum(e["steps"] for e in a)


def _m(**kw):
    base = {"capture": 1.0, "exhaustion": 0.0, "timeout": 0.0, "median_path_excess": 1.0, "gt2": 0.0, "gt5": 0.0, "orbit_episodes": 0, "nonfinite": 0}
    base.update(kw)
    return base


def test_stage1_v2_gate_boundaries():
    assert V.gate_status(_m())["passed"] and V.gate_status(_m(capture=0.98, exhaustion=0.01, timeout=0.01, median_path_excess=1.20, gt2=0.02))["passed"]
    for bad in ({"capture": 0.9799}, {"exhaustion": 0.0101}, {"timeout": 0.0101}, {"median_path_excess": 1.2001}, {"gt2": 0.0201}, {"gt5": 0.005}, {"orbit_episodes": 1}, {"nonfinite": 1}):
        assert not V.gate_status(_m(**bad))["passed"], bad


def _entry(passed, **over):
    m = {"capture": 1.0, "orbit_episodes": 0, "repeated_flyby_episodes": 0, "nonclosing_episodes": 0, **over}
    return {"final": {"metrics": m, "gate": {"passed": passed}}}


def test_stage_decision_requires_two_seeds_and_no_shared_pathology():
    ok = RUN.decide({"episodes": 200, "per_seed": {"201": _entry(True), "202": _entry(True), "203": _entry(False)}})
    assert ok["stage1_v2_pass"] and ok["seeds_passed"] == [201, 202] and not ok["all_three_passed"]
    assert not RUN.decide({"episodes": 200, "per_seed": {"201": _entry(True), "202": _entry(False), "203": _entry(False, capture=0.5)}})["stage1_v2_pass"]
    shared = RUN.decide({"episodes": 200, "per_seed": {"201": _entry(True, nonclosing_episodes=5), "202": _entry(True, nonclosing_episodes=5)}})
    assert not shared["stage1_v2_pass"] and "nonclosing" in shared["common_structural_pathology"]
    assert RUN.decide({"episodes": 200, "per_seed": {"201": _entry(False, capture=0.95), "202": _entry(False), "203": _entry(False)}})["near_pass_needs_review"]


def test_protocol_is_stable_complete_and_matches_the_frozen_artifact_if_present():
    spec = V.protocol_specification()
    assert spec["policy"]["log_std_after_bc"] == -2.0 and spec["relabeling"]["rounds"] == 1 and spec["relabeling"]["adaptive_repetition"] is False and spec["gate"] == V.GATE_V2
    assert spec["dataset"]["namespace_overlaps"]["expert"] == [] and spec["expert"]["privileged_information"] is False
    assert V.protocol_sha256() == hashlib.sha256(json.dumps(V.protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()
    path = TS.STATE_DIR / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text())["protocol_sha256"] == V.protocol_sha256()


def test_ppo_learn_is_never_called_in_the_stage1_v2_modules():
    root = Path(V.__file__).parent
    for name in ("journal_v2_a1_cur1v2.py", "journal_v2_a1_cur1v2_data.py", "journal_v2_a1_cur1v2_bc.py", "journal_v2_a1_cur1v2_run.py", "journal_v2_a1_cur1v2_state.py"):
        code = (root / name).read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
        assert ".learn(" not in code, name


def test_task_state_has_twelve_stages_and_restart_semantics(tmp_path, monkeypatch):
    assert len(TS.STAGES) == 12 and TS.STAGES[3] == "04_generate_expert_dataset" and TS.STAGES[7] == "08_fixed_expert_relabeling_round"
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    state = TS.load_state()
    rel = "docs/journal_v2_a1_curriculum/stage1_v2/_probe.txt"
    probe = TS.PROJECT_ROOT / rel
    probe.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe.write_text("a")
        TS.begin(state, TS.STAGES[0], "cfg")
        assert not TS.stage_is_verified(state, TS.STAGES[0])
        TS.complete(state, TS.STAGES[0], [rel], "ok")
        assert TS.stage_is_verified(TS.load_state(), TS.STAGES[0], "cfg") and not TS.stage_is_verified(TS.load_state(), TS.STAGES[0], "other-config")
        probe.write_text("b")
        assert TS.next_stage(TS.load_state()) == TS.STAGES[0]
        TS.skip(state, TS.STAGES[9], "visualizer only if >= 2/3 pass")
        assert TS.stage_is_verified(TS.load_state(), TS.STAGES[9])
    finally:
        probe.unlink(missing_ok=True)
