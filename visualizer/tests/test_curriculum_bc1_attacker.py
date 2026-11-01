"""Curriculum Stage-1-v2 BC initialization attackers (A1-BC1-201/202/203): identifiers, handoff-record/SHA verification,
masked observation, and real replay fidelity. The visualizer only resolves, verifies and labels research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from visualizer.backend import replay_engine as E
from visualizer.backend.application_error import ApplicationError
from visualizer.backend.config import Settings
from visualizer.backend.data_access import ResearchDataStore
from visualizer.backend.replay_contract import EnvironmentSpec
from visualizer.backend.replay_requests import (
    ALL_ATTACKER_SEEDS,
    ATTACKER_CONDITIONS,
    CURRICULUM_ATTACKER_SEEDS,
    ReplayBatchRequest,
    ReplayRequest,
    V2_ATTACKER_SEEDS,
)

MODEL = b"synthetic-bc-model-bytes"
SHA = hashlib.sha256(MODEL).hexdigest()


def research_root(tmp_path, *, sha=SHA, passed=True, path=None, mode=E.CURRICULUM_MODE, with_handoff=True):
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("")
    model_dir = root / E.CURRICULUM_RESULTS_RELATIVE / "seed_203"
    model_dir.mkdir(parents=True)
    (model_dir / "bc_final_model.zip").write_bytes(MODEL)
    if with_handoff:
        handoff = root / E.CURRICULUM_REGISTRY_RELATIVE_PATH
        handoff.parent.mkdir(parents=True)
        handoff.write_text(json.dumps({
            "stage": E.CURRICULUM_STAGE, "curriculum_mode": mode, "decision": {"stage1_v2_pass": passed},
            "versions": {"observation": "a1_v2_obs_v1", "action": "a1_cur1_action_v1", "reward": "a1_v2_reward_v2"},
            "checkpoints": {"203": {"model_sha256": sha,
                                    "path": (path or f"{E.CURRICULUM_RESULTS_RELATIVE}/seed_203/bc_final_model.zip").replace("/", "\\")}}}))
    return root


def store_for(root):
    return ResearchDataStore(Settings(research_data_root=root))


def request(attacker="A1-BC1-203", seed_model=203, **overrides):
    fields = dict(defender_policy_id="passive", attacker_condition=attacker, attacker_model_seed=seed_model,
                  clutter_level="C0", semantics="V2", evaluation_seed=146000)
    fields.update(overrides)
    return ReplayRequest(**fields)


def test_identifiers_are_explicit_and_not_aliases_of_other_attackers():
    assert CURRICULUM_ATTACKER_SEEDS == {"A1-BC1-201": 201, "A1-BC1-202": 202, "A1-BC1-203": 203}
    assert not set(CURRICULUM_ATTACKER_SEEDS) & set(V2_ATTACKER_SEEDS)
    assert set(CURRICULUM_ATTACKER_SEEDS) <= set(ATTACKER_CONDITIONS) and set(CURRICULUM_ATTACKER_SEEDS.items()) <= set(ALL_ATTACKER_SEEDS.items())


@pytest.mark.parametrize("attacker,seed", CURRICULUM_ATTACKER_SEEDS.items())
def test_requests_require_the_curriculum_environment_and_their_seed(attacker, seed):
    assert request(attacker, seed).attacker_condition == attacker
    for bad in ({"semantics": "V1"}, {"clutter_level": "C1"}, {"defender_policy_id": "greedy"}, {"attacker_model_seed": seed + 1}, {"attacker_model_seed": None}):
        with pytest.raises(ValidationError):
            request(attacker, seed, **bad)


def test_batch_request_enforces_the_same_rules():
    ReplayBatchRequest(defender_policy_id="passive", attacker_condition="A1-BC1-203", clutter_level="C0", semantics="V2", seed_start=146000)
    with pytest.raises(ValidationError):
        ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A1-BC1-203", clutter_level="C0", semantics="V2")


def test_resolution_verifies_the_passed_handoff_record_and_the_file(tmp_path):
    root = research_root(tmp_path)
    store = store_for(root)
    checkpoint = E._resolve_attacker_checkpoint("A1-BC1-203", {"attackers": {}}, store)
    assert checkpoint.role == "attacker_curriculum_bc" and checkpoint.sha256 == SHA and checkpoint.training_seed == 203
    assert checkpoint.relative_path.endswith("seed_203/bc_final_model.zip")
    assert E._verify_artifact(store, checkpoint).read_bytes() == MODEL
    (root / checkpoint.relative_path).write_bytes(MODEL + b"x")
    with pytest.raises(ApplicationError) as error:
        E._verify_artifact(store, checkpoint)
    assert error.value.code == "CHECKPOINT_HASH_MISMATCH"


@pytest.mark.parametrize("kwargs,code", [
    ({"passed": False}, "INVALID_MODEL_REGISTRY"),
    ({"mode": "other"}, "INVALID_MODEL_REGISTRY"),
    ({"path": "results/elsewhere/bc_final_model.zip"}, "INVALID_MODEL_REGISTRY"),
    ({"sha": "short"}, "INVALID_MODEL_REGISTRY"),
    ({"with_handoff": False}, "CHECKPOINT_NOT_REGISTERED"),
])
def test_unpassed_or_inconsistent_handoff_is_rejected(tmp_path, kwargs, code):
    store = store_for(research_root(tmp_path, **kwargs))
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-BC1-203", {"attackers": {}}, store)
    assert error.value.code == code


def test_unregistered_seed_is_unavailable(tmp_path):
    store = store_for(research_root(tmp_path))
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-BC1-201", {"attackers": {}}, store)
    assert error.value.code == "INVALID_MODEL_REGISTRY"


def test_observation_uses_the_masked_detection_range_and_37d_shape():
    seen = {}

    def build(env, detection_range):
        seen["range"] = detection_range
        return np.zeros(37)

    env = SimpleNamespace(config=SimpleNamespace(enemy_defender_detection_range=25.0))
    modules = {"experiments.journal_v2_a1": SimpleNamespace(build_a1_v2_observation=build),
               "experiments.journal_v2_a1_cur1v2": SimpleNamespace(MASKED_DETECTION_RANGE=-1.0)}
    assert E._build_attacker_observation("A1-BC1-203", modules, env).shape == (37,)
    assert seen["range"] == -1.0
    assert E._build_attacker_observation("A1-201-V2", modules, env).shape == (37,) and seen["range"] == 25.0


def test_contract_accepts_the_curriculum_builder_only_with_v2_semantics():
    config = {"mission_contact_mode": "swept", "enemy_nominal_guidance_mode": "direct_obstacle_only"}
    spec = dict(clutter="C0", awareness="map_local", builder="experiments.journal_v2_a1_cur1v2.stage1v2_config",
                semantics="V2", config=config, config_sha256="a" * 64, dt_s=0.5)
    assert EnvironmentSpec(**spec).builder.endswith("stage1v2_config")
    with pytest.raises(ValidationError):
        EnvironmentSpec(**{**spec, "semantics": "V1", "config": {}})


def _simulation_python_and_root():
    root = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
    python = os.environ.get("VISUALIZER_SIMULATION_PYTHON")
    if not root or not python:
        pytest.skip("Set VISUALIZER_RESEARCH_DATA_ROOT and VISUALIZER_SIMULATION_PYTHON to run the real simulator tests.")
    if not (Path(root) / E.CURRICULUM_RESULTS_RELATIVE / "seed_203" / "bc_final_model.zip").exists():
        pytest.skip("research Stage-1-v2 models are not available")
    return python, root


def _run_worker(python, root, code):
    # A subprocess mirrors the backend (simulation interpreter, research root first on sys.path).
    env = {**os.environ, "VISUALIZER_RESEARCH_DATA_ROOT": root, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[2], timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_real_replay_is_a_clean_basic_pursuit_capture_without_defender_interaction():
    python, root = _simulation_python_and_root()
    code = (
        "import json\n"
        "from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode\n"
        "r = ReplayRequest(defender_policy_id='passive', attacker_condition='A1-BC1-203', attacker_model_seed=203,"
        " clutter_level='C0', semantics='V2', evaluation_seed=146009)\n"
        "e = run_replay_episode(r).episodes[0]\n"
        "print(json.dumps({'outcome': e.summary.outcome, 'steps': e.summary.steps, 'builder': e.environment.builder,"
        " 'v_s': e.environment.config['v_s'], 'flag': e.attacker.parameters['curriculum_initialization_checkpoint'],"
        " 'asset_positions': len({tuple(f.asset.position_m) for f in e.frames}),"
        " 'min_defender_distance': e.summary.safety_metrics['minimum_defender_attacker_distance_m']}))\n"
    )
    result = _run_worker(python, root, code)
    assert result["outcome"] == "soldier_caught" and result["steps"] == 11 and result["builder"].endswith("stage1v2_config")
    assert result["v_s"] == 0.0 and result["flag"] is True and result["asset_positions"] == 1 and result["min_defender_distance"] > 1.0


def test_real_catalog_lists_the_curriculum_attackers_with_the_pinned_source_commit():
    python, root = _simulation_python_and_root()
    code = (
        "import json\n"
        "from visualizer.backend.config import load_settings\n"
        "from visualizer.backend.replay_catalog import build_replay_catalog\n"
        "c = build_replay_catalog(load_settings())\n"
        "print(json.dumps({'source_commit': c['source_commit'], 'registry_sha256': c['registry_sha256'],"
        " 'attackers': {a['attacker_id']: a['available'] for a in c['attackers']}}))\n"
    )
    result = _run_worker(python, root, code)
    assert result["source_commit"] == E.EXPECTED_RESEARCH_COMMIT and len(result["registry_sha256"]) == 64
    assert all(result["attackers"].get(name) for name in CURRICULUM_ATTACKER_SEEDS)
