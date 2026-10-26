"""V2 A1 training runner: isolation, manifests, resume state machine and the dry-run (seeds 201/202/203).
PPO is monkeypatched to raise: nothing here constructs or trains a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.journal_v2_a1_training as T
from experiments.journal_v2_a1 import A1_V2_OBSERVATION_DIM, A1_V2_PPO_SEEDS
from experiments.journal_v2_protocol import V2_RESULTS_DIR


@pytest.fixture(autouse=True)
def ppo_forbidden(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("PPO must not be constructed or trained in these tests")

    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm

    monkeypatch.setattr(BaseAlgorithm, "__init__", forbid)
    monkeypatch.setattr(PPO, "learn", forbid)


class FakeModel:
    num_timesteps = 123_456

    def save(self, path):
        Path(str(path) + ".zip").write_bytes(b"fake-model-bytes")


def test_default_output_namespace_is_v2_and_per_seed():
    for seed in A1_V2_PPO_SEEDS:
        paths = T.run_paths(seed)
        assert V2_RESULTS_DIR.resolve() in paths.root.parents
        assert "a1_training" in paths.root.parts and "a1_v2_training_v1" in paths.root.parts
        assert paths.root.name == f"seed_{seed}"
    roots = {T.run_paths(s).root for s in A1_V2_PPO_SEEDS}
    assert len(roots) == 3
    with pytest.raises(ValueError):
        T.run_paths(999)


@pytest.mark.parametrize("name", T.V1_PROTECTED_ROOTS)
def test_v1_namespaces_are_never_writable(name):
    with pytest.raises(T.RunConflictError):
        T.assert_output_isolated(name)
    with pytest.raises(T.RunConflictError):
        T.assert_output_isolated(name / "seed_201" / "best_model.zip")
    with pytest.raises(T.RunConflictError):
        T.run_paths(201, root=name)


def test_results_tree_outside_v2_is_refused_but_external_tmp_is_allowed(tmp_path):
    with pytest.raises(T.RunConflictError):
        T.assert_output_isolated(T.PROJECT_ROOT / "results" / "some_other_namespace")
    assert T.assert_output_isolated(tmp_path) == tmp_path.resolve()


def test_registered_v1_a1_checkpoints_are_outside_the_writable_namespace():
    registry = json.loads((T.PROJECT_ROOT / "results" / "defender_adaptation_model_registry.json").read_text())
    attackers = registry["attackers"]
    for entry in (attackers.values() if isinstance(attackers, dict) else attackers):
        path = (T.PROJECT_ROOT / entry["path"]).resolve()
        with pytest.raises(T.RunConflictError):
            T.assert_output_isolated(path)
        assert V2_RESULTS_DIR.resolve() not in path.parents


def test_run_state_machine(tmp_path):
    paths = T.run_paths(201, root=tmp_path)
    assert T.decide_run_state(paths) == "fresh"
    paths.make_dirs()
    assert T.decide_run_state(paths) == "fresh"            # empty directories are fine
    paths.resume_model.write_bytes(b"x")
    with pytest.raises(T.RunConflictError):                # resume model without state: refuse to guess
        T.decide_run_state(paths)
    paths.resume_state.write_text("{}")
    assert T.decide_run_state(paths) == "resume"
    paths.final_model.write_bytes(b"x")
    with pytest.raises(T.RunAlreadyCompleteError):         # completed runs are never overwritten or resumed
        T.decide_run_state(paths)


def test_stray_files_without_resume_state_are_a_conflict(tmp_path):
    paths = T.run_paths(202, root=tmp_path)
    paths.make_dirs()
    (paths.logs / "episodes.jsonl").write_text("{}")
    with pytest.raises(T.RunConflictError):
        T.decide_run_state(paths)


def test_atomic_json_write_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "a" / "m.json"
    T.write_json_atomic(target, {"b": 1, "a": [1, 2]})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 1}
    assert [p.name for p in target.parent.iterdir()] == ["m.json"]


def test_model_manifest_structure_and_hash(tmp_path):
    paths = T.run_paths(203, root=tmp_path)
    paths.make_dirs()
    manifest = T.save_model_with_manifest(FakeModel(), paths.final_model, ppo_seed=203, role="final", canonical=True)
    sidecar = json.loads((paths.checkpoints / "final_model.manifest.json").read_text())
    assert sidecar == json.loads(json.dumps(manifest, default=str))
    required = {"ppo_seed", "canonical_a0_commit", "canonical_a0_tag", "config_sha256", "observation_dim", "action_dim",
                "reward_version", "requested_total_timesteps", "num_timesteps", "ppo_hyperparameters", "defender_sampling",
                "clutter_sampling", "endurance_path_factor", "asset_motion_model", "created_utc", "model_sha256",
                "training_protocol", "observation_schema_sha256", "scenario_seed_namespaces", "research_commit", "canonical"}
    assert required <= set(sidecar)
    assert sidecar["ppo_seed"] == 203 and sidecar["observation_dim"] == A1_V2_OBSERVATION_DIM == 37 and sidecar["action_dim"] == 3
    assert sidecar["reward_version"] == "a1_v2_reward_v1" and sidecar["training_protocol"] == "a1_v2_training_v1"
    assert sidecar["canonical_a0_commit"] == "bdbee45dc08a583a0da5ca882dc6b1269207a8cd"
    assert sidecar["canonical_a0_tag"] == "journal-v2-a0-canonical-v1"
    assert sidecar["endurance_path_factor"] == 5 and sidecar["asset_motion_model"] == "legacy_gaussian_random_walk"
    assert sidecar["defender_sampling"] == {"greedy": 0.2, "kalman_greedy": 0.2, "pn": 0.2, "lead": 0.2, "rmpc": 0.2}
    assert sidecar["clutter_sampling"] == {"C0": 0.25, "C1": 0.25, "C2": 0.25, "C3": 0.25}
    assert sidecar["model_sha256"] == T.sha256_file(paths.final_model)
    assert set(sidecar["config_sha256"]) == {f"{l}/{e}" for l in ("C0", "C1", "C2", "C3") for e in ("measurement", "kalman")}
    assert all(len(v) == 64 for v in sidecar["config_sha256"].values())


def test_saved_models_are_never_overwritten_except_the_diagnostic_best(tmp_path):
    paths = T.run_paths(201, root=tmp_path)
    paths.make_dirs()
    T.save_model_with_manifest(FakeModel(), paths.final_model, ppo_seed=201, role="final", canonical=True)
    with pytest.raises(T.RunConflictError):
        T.save_model_with_manifest(FakeModel(), paths.final_model, ppo_seed=201, role="final", canonical=True)
    T.save_model_with_manifest(FakeModel(), paths.best_validation_model, ppo_seed=201, role="best_validation_diagnostic",
                               canonical=False, allow_replace=True)
    T.save_model_with_manifest(FakeModel(), paths.best_validation_model, ppo_seed=201, role="best_validation_diagnostic",
                               canonical=False, allow_replace=True)


def test_saving_into_v1_namespace_is_refused():
    v1 = T.V1_PROTECTED_ROOTS[0] / "seed_201" / "best_model.zip"
    with pytest.raises(T.RunConflictError):
        T.save_model_with_manifest(FakeModel(), v1, ppo_seed=201, role="final", canonical=True)
    assert not v1.exists() or v1.stat().st_size > 20  # the real V1 model, if present, is untouched (not the fake bytes)


def test_worker_seeds_and_env_functions():
    assert T.worker_env_seeds(201, 4) == [201, 10_201, 20_201, 30_201]
    fns = T.worker_env_fns(202, 3)
    assert len(fns) == 3
    envs = [fn() for fn in fns]
    assert [e.worker_index for e in envs] == [0, 1, 2] and all(e.ppo_seed == 202 for e in envs)


@pytest.mark.parametrize("seed", A1_V2_PPO_SEEDS)
def test_dry_run_stops_before_ppo_and_writes_nothing(seed, tmp_path):
    report = T.dry_run(seed, num_envs=4, root=tmp_path, probe_resets=40)
    assert report["ppo_constructed"] is False and report["canonical_guard"] == "passed"
    assert report["observation_dim"] == 37 and report["action_dim"] == 3
    assert report["run_state"] == "fresh"
    assert set(report["sampled_levels"]) == {"C0", "C1", "C2", "C3"}
    assert set(report["sampled_defenders"]) == {"greedy", "kalman_greedy", "pn", "lead", "rmpc"}
    assert all(step["terminal"] == 0.0 for step in report["reward_probe"])
    assert list(tmp_path.rglob("*")) == []


def test_dry_run_refuses_completed_run(tmp_path):
    paths = T.run_paths(201, root=tmp_path)
    paths.make_dirs()
    paths.final_model.write_bytes(b"x")
    with pytest.raises(T.RunAlreadyCompleteError):
        T.dry_run(201, num_envs=1, root=tmp_path)


def test_training_entry_point_fails_fast_on_noncanonical_environment(monkeypatch, tmp_path):
    from experiments.journal_v2_protocol import CanonicalV2ConfigError

    def drift(*args, **kwargs):
        raise CanonicalV2ConfigError("drift")

    monkeypatch.setattr(T, "assert_canonical_v2_env_fn", drift)
    with pytest.raises(CanonicalV2ConfigError):
        T.run_a1_v2_training(201, num_envs=1, root=tmp_path, vec_env_cls="dummy")
    assert list(tmp_path.rglob("*")) == []  # nothing is created before the guard passes


def test_handoff_a1_section_matches_live_protocol_constants():
    from experiments import journal_v2_a1 as A

    handoff = json.loads((T.PROJECT_ROOT / "docs" / "journal_v2_training_handoff.json").read_text())["a1_v2_training_protocol"]
    assert handoff["protocol"] == A.A1_V2_TRAINING_PROTOCOL and handoff["reward_version"] == A.A1_V2_REWARD_VERSION
    assert handoff["observation_dim"] == A.A1_V2_OBSERVATION_DIM and handoff["action_dim"] == A.A1_V2_ACTION_DIM
    assert handoff["ppo_seeds"] == list(A.A1_V2_PPO_SEEDS)
    assert handoff["ppo"]["hyperparameters_inherited_from_v1"] == A.A1_V2_PPO_HYPERPARAMS
    assert handoff["ppo"]["requested_total_timesteps"] == A.A1_V2_TOTAL_TIMESTEPS
    assert handoff["clutter_training_distribution"] == dict(zip(A.A1_V2_CLUTTER_LEVELS, A.A1_V2_CLUTTER_PROBABILITIES))
    reward = handoff["reward"]
    assert (reward["terminal_success"], reward["terminal_failure_all_other_outcomes"]) == (A.REWARD_TERMINAL_SUCCESS, A.REWARD_TERMINAL_FAILURE)
    assert (reward["w_progress"], reward["w_path"], reward["w_time"]) == (A.W_PROGRESS, A.W_PATH, A.W_TIME)
