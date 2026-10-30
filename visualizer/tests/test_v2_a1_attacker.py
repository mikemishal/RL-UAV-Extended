"""Canonical-V2 A1 attacker support: identifiers, registry/manifest/SHA verification, observation/action validation.

No simulation logic is exercised here; the visualizer only resolves, verifies and labels the research checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from visualizer.backend import replay_engine as E
from visualizer.backend.application_error import ApplicationError
from visualizer.backend.config import Settings
from visualizer.backend.data_access import ResearchDataStore
from visualizer.backend.replay_contract import Checkpoint, PolicySelection
from visualizer.backend.replay_requests import (
    ATTACKER_CONDITIONS,
    ReplayBatchRequest,
    ReplayRequest,
    V2_ATTACKER_SEEDS,
)

MODEL = b"synthetic-a1-v2-model-bytes"
SHA = hashlib.sha256(MODEL).hexdigest()


def research_root(tmp_path, *, registry_sha=SHA, manifest_sha=SHA, obs_dim=37, with_registry=True, with_manifest=True):
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("")
    base = root / E.V2_A1_RESULTS_RELATIVE
    model_dir = base / "seed_201" / "checkpoints"
    model_dir.mkdir(parents=True)
    (model_dir / "final_model.zip").write_bytes(MODEL)
    if with_manifest:
        (model_dir / "final_model.manifest.json").write_text(json.dumps({
            "model_sha256": manifest_sha, "ppo_seed": 201, "training_protocol": E.V2_A1_PROTOCOL,
            "reward_version": E.V2_A1_REWARD_VERSION, "observation_dim": 37}))
    if with_registry:
        (base / "a1_v2_model_registry.json").write_text(json.dumps({
            "protocol": E.V2_A1_PROTOCOL, "reward_version": E.V2_A1_REWARD_VERSION,
            "attackers": {"A1-201-V2": {
                "ppo_seed": 201, "path": f"{E.V2_A1_RESULTS_RELATIVE}/seed_201/checkpoints/final_model.zip",
                "sha256": registry_sha, "observation_dim": obs_dim, "action_dim": 3}}}))
    return root


def store_for(root):
    return ResearchDataStore(Settings(research_data_root=root))


def request(attacker, semantics="V2", seed_model=None, **overrides):
    return ReplayRequest(
        defender_policy_id="greedy", attacker_condition=attacker, attacker_model_seed=seed_model,
        clutter_level="C0", semantics=semantics, evaluation_seed=140000, **overrides)


def test_attacker_identifiers_are_explicit_and_separate_from_v1():
    assert ATTACKER_CONDITIONS == ("A0", "A1-201", "A1-202", "A1-203", "A1-201-V2", "A1-202-V2", "A1-203-V2")
    assert V2_ATTACKER_SEEDS == {"A1-201-V2": 201, "A1-202-V2": 202, "A1-203-V2": 203}


@pytest.mark.parametrize("attacker,seed", V2_ATTACKER_SEEDS.items())
def test_v2_attackers_require_v2_semantics_and_their_seed(attacker, seed):
    assert request(attacker, "V2", seed).attacker_condition == attacker
    with pytest.raises(ValidationError):
        request(attacker, "V1", seed)
    with pytest.raises(ValidationError):
        request(attacker, "V2", None)
    with pytest.raises(ValidationError):
        request(attacker, "V2", seed + 1)


def test_batch_request_enforces_the_same_rules():
    with pytest.raises(ValidationError):
        ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A1-201-V2", clutter_level="C0", semantics="V1")
    ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A1-201-V2", clutter_level="C0", semantics="V2", seed_start=140000)


def test_v1_attackers_keep_working_with_either_semantics():
    for semantics in ("V1", "V2"):
        assert request("A1-203", semantics, 203).attacker_condition == "A1-203"


def test_resolution_requires_registry_manifest_and_file_to_agree(tmp_path):
    store = store_for(research_root(tmp_path))
    checkpoint = E._resolve_attacker_checkpoint("A1-201-V2", {"attackers": {}}, store)
    assert checkpoint.role == "attacker_v2_final" and checkpoint.sha256 == SHA and checkpoint.training_seed == 201
    assert checkpoint.relative_path.endswith("seed_201/checkpoints/final_model.zip")
    assert E._verify_artifact(store, checkpoint).read_bytes() == MODEL


def test_registry_manifest_sha_disagreement_is_rejected(tmp_path):
    store = store_for(research_root(tmp_path, manifest_sha="0" * 64))
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-201-V2", {"attackers": {}}, store)
    assert error.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_tampered_model_file_is_rejected(tmp_path):
    root = research_root(tmp_path)
    store = store_for(root)
    checkpoint = E._resolve_attacker_checkpoint("A1-201-V2", {"attackers": {}}, store)
    (root / checkpoint.relative_path).write_bytes(MODEL + b"x")
    with pytest.raises(ApplicationError) as error:
        E._verify_artifact(store, checkpoint)
    assert error.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_missing_registry_or_manifest_is_unavailable_not_aliased_to_v1(tmp_path):
    v1_registry = {"attackers": {"201": {"path": "x", "sha256": "a" * 64}}}
    for name, kwargs in (("no_registry", {"with_registry": False}), ("no_manifest", {"with_manifest": False})):
        store = store_for(research_root(tmp_path / name, **kwargs))
        with pytest.raises(ApplicationError) as error:
            E._resolve_attacker_checkpoint("A1-201-V2", v1_registry, store)
        assert error.value.code == "CHECKPOINT_NOT_REGISTERED"


def test_unregistered_v2_seed_is_unavailable(tmp_path):
    store = store_for(research_root(tmp_path))
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-202-V2", {"attackers": {}}, store)
    assert error.value.code == "CHECKPOINT_NOT_REGISTERED"


def test_registry_declaring_a_36d_model_is_rejected(tmp_path):
    store = store_for(research_root(tmp_path, obs_dim=36))
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-201-V2", {"attackers": {}}, store)
    assert error.value.code == "INCOMPATIBLE_CHECKPOINT"


def test_model_space_validation_rejects_wrong_observation_or_action_shape():
    good = SimpleNamespace(observation_space=SimpleNamespace(shape=(37,)), action_space=SimpleNamespace(shape=(3,)))
    E._checkpoint_model_input(good, (37,), "a1_v2:201:final")
    legacy = SimpleNamespace(observation_space=SimpleNamespace(shape=(36,)), action_space=SimpleNamespace(shape=(3,)))
    with pytest.raises(ApplicationError) as error:
        E._checkpoint_model_input(legacy, (37,), "a1_v2:201:final")
    assert error.value.code == "INCOMPATIBLE_CHECKPOINT"
    bad_action = SimpleNamespace(observation_space=SimpleNamespace(shape=(37,)), action_space=SimpleNamespace(shape=(2,)))
    with pytest.raises(ApplicationError):
        E._checkpoint_model_input(bad_action, (37,), "a1_v2:201:final")


def test_observation_builder_selection_and_shape_guard():
    env = SimpleNamespace(config=SimpleNamespace(enemy_defender_detection_range=25.0))
    modules = {
        "experiments.journal_v2_a1": SimpleNamespace(build_a1_v2_observation=lambda e, r: np.zeros(37)),
        "experiments.adaptive_attacker_env": SimpleNamespace(build_attacker_observation=lambda e, r: np.zeros(36)),
    }
    assert E._build_attacker_observation("A1-201-V2", modules, env).shape == (37,)
    assert E._build_attacker_observation("A1-201", modules, env).shape == (36,)
    modules["experiments.journal_v2_a1"] = SimpleNamespace(build_a1_v2_observation=lambda e, r: np.zeros(36))
    with pytest.raises(ApplicationError):
        E._build_attacker_observation("A1-201-V2", modules, env)


def test_contract_accepts_v2_checkpoint_role_and_label_parameters():
    selection = PolicySelection(
        policy_kind="learned", policy_id="A1-201-V2", architecture="a1_v2_ppo", training_seed=201,
        checkpoint=Checkpoint(artifact_id="a1_v2:201:final", research_relative_path="results/x/final_model.zip",
                              sha256=SHA, role="attacker_v2_final"),
        observation_shape=(37,), estimator="attacker_local",
        parameters={"display_label": "Attacker: A1-201 V2", "canonical_v2_attacker": True})
    assert selection.observation_shape == (37,)
