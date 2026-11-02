"""Stage2-v2 REVIEW policies (A1-CUR2V2-20x-REVIEW) and the A0 reference: identifiers, manifest/hash gating, exact research-environment replay, and the committed shortlist.

The visualizer only resolves, verifies and labels research artifacts; the scientific status (NOT MASTERED / NON-CANONICAL) is never changed here."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from visualizer.backend import replay_engine as E
from visualizer.backend.application_error import ApplicationError
from visualizer.backend.config import Settings
from visualizer.backend.data_access import ResearchDataStore
from visualizer.backend.replay_requests import (
    ALL_ATTACKER_SEEDS,
    ATTACKER_CONDITIONS,
    REVIEW_A0_REFERENCE,
    REVIEW_ATTACKER_SEEDS,
    REVIEW_LABEL,
    V2_ATTACKER_SEEDS,
    ReplayBatchRequest,
    ReplayRequest,
)

REPO = Path(__file__).resolve().parents[2]
SHORTLIST = REPO / "visualizer" / "review" / "stage2v2_review_shortlist.json"
EXPECTED_SHA_PREFIX = {201: "d70b85a6", 202: "97a934bc", 203: "06993dca"}


def synthetic_root(tmp_path, *, seed=201, sha=None, **overrides):
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("")
    docs = root / E.REVIEW_DOCS_RELATIVE
    docs.mkdir(parents=True)
    manifest = {"seed": seed, "status": "FAIL", "mastery_rollout": None, "accepted_rollout": 16, "rollouts_trained": 16, "training_protocol": "a1_cur2_training_v1",
                "placement_protocol": E.REVIEW_PLACEMENT_PROTOCOL, "observation_version": "a1_v2_obs_v1", "action_version": "a1_cur1_action_v1", "reward_version": "a1_v2_reward_v2",
                "accepted_model": "checkpoints/accepted_model.zip", "accepted_model_sha256": sha or (EXPECTED_SHA_PREFIX[seed] + "0" * 56)}
    manifest.update(overrides)
    (docs / f"seed_{seed}_manifest.json").write_text(json.dumps(manifest))
    return root


def store_for(root):
    return ResearchDataStore(Settings(research_data_root=root))


def request(attacker="A1-CUR2V2-201-REVIEW", seed_model=201, **overrides):
    fields = dict(defender_policy_id="passive", attacker_condition=attacker, attacker_model_seed=seed_model, clutter_level="C0", semantics="V2", evaluation_seed=147149)
    fields.update(overrides)
    return ReplayRequest(**fields)


def test_identifiers_are_explicit_non_canonical_and_never_alias_existing_attackers():
    assert REVIEW_ATTACKER_SEEDS == {"A1-CUR2V2-201-REVIEW": 201, "A1-CUR2V2-202-REVIEW": 202, "A1-CUR2V2-203-REVIEW": 203}
    assert all(name.endswith("-REVIEW") for name in REVIEW_ATTACKER_SEEDS) and "NOT MASTERED" in REVIEW_LABEL and "NON-CANONICAL" in REVIEW_LABEL
    existing = set(ATTACKER_CONDITIONS) - set(REVIEW_ATTACKER_SEEDS) - {REVIEW_A0_REFERENCE}
    assert not existing & (set(REVIEW_ATTACKER_SEEDS) | {REVIEW_A0_REFERENCE}) and not set(REVIEW_ATTACKER_SEEDS) & set(V2_ATTACKER_SEEDS)
    assert {"A0", "A1-201", "A1-203-V2", "A1-BC1-203"} <= existing                       # earlier ids untouched
    assert set(REVIEW_ATTACKER_SEEDS.items()) <= set(ALL_ATTACKER_SEEDS.items())


@pytest.mark.parametrize("attacker,seed", REVIEW_ATTACKER_SEEDS.items())
def test_requests_require_the_exact_stage2_environment_slot_and_their_seed(attacker, seed):
    assert request(attacker, seed).attacker_condition == attacker
    for bad in ({"semantics": "V1"}, {"clutter_level": "C1"}, {"defender_policy_id": "greedy"}, {"attacker_model_seed": seed + 1}, {"attacker_model_seed": None}):
        with pytest.raises(ValidationError):
            request(attacker, seed, **bad)


def test_reference_attacker_is_scripted_and_batches_follow_the_same_rules():
    assert request(REVIEW_A0_REFERENCE, None).attacker_condition == REVIEW_A0_REFERENCE
    with pytest.raises(ValidationError):
        request(REVIEW_A0_REFERENCE, 201)
    with pytest.raises(ValidationError):
        request(REVIEW_A0_REFERENCE, None, clutter_level="C2")
    ReplayBatchRequest(defender_policy_id="passive", attacker_condition="A1-CUR2V2-203-REVIEW", clutter_level="C0", semantics="V2", seed_start=147000)
    with pytest.raises(ValidationError):
        ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A1-CUR2V2-203-REVIEW", clutter_level="C0", semantics="V2")


def test_resolution_accepts_only_the_unmastered_rollout16_manifest_with_the_preregistered_hash(tmp_path):
    checkpoint = E._resolve_attacker_checkpoint("A1-CUR2V2-201-REVIEW", {"attackers": {}}, store_for(synthetic_root(tmp_path / "ok")))
    assert checkpoint.role == "attacker_review_stage2v2" and checkpoint.training_seed == 201 and checkpoint.sha256.startswith("d70b85a6")
    assert checkpoint.relative_path.endswith("stage2_v2/seed_201/checkpoints/accepted_model.zip") and E._resolve_attacker_checkpoint(REVIEW_A0_REFERENCE, {"attackers": {}}, store_for(synthetic_root(tmp_path / "ref"))) is None


@pytest.mark.parametrize("overrides,code", [
    ({"status": "MASTERED"}, "INVALID_MODEL_REGISTRY"),
    ({"mastery_rollout": 12}, "INVALID_MODEL_REGISTRY"),
    ({"accepted_rollout": 8, "rollouts_trained": 8}, "INVALID_MODEL_REGISTRY"),
    ({"training_protocol": "other"}, "INVALID_MODEL_REGISTRY"),
    ({"reward_version": "a1_v2_reward_v1"}, "INVALID_MODEL_REGISTRY"),
    ({"accepted_model_sha256": "ab" * 32}, "CHECKPOINT_HASH_MISMATCH"),
    ({"accepted_model_sha256": "short"}, "CHECKPOINT_HASH_MISMATCH"),
])
def test_resolution_rejects_inconsistent_manifests(tmp_path, overrides, code):
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-CUR2V2-201-REVIEW", {"attackers": {}}, store_for(synthetic_root(tmp_path, **overrides)))
    assert error.value.code == code


def test_missing_manifest_is_unavailable(tmp_path):
    root = synthetic_root(tmp_path)
    (root / E.REVIEW_DOCS_RELATIVE / "seed_201_manifest.json").unlink()
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-CUR2V2-201-REVIEW", {"attackers": {}}, store_for(root))
    assert error.value.code == "CHECKPOINT_NOT_REGISTERED"


def test_tampered_model_file_is_rejected(tmp_path):
    payload = b"synthetic-model"
    digest = hashlib.sha256(payload).hexdigest()
    root = synthetic_root(tmp_path, sha=EXPECTED_SHA_PREFIX[201] + digest[8:])
    model = root / E.REVIEW_RESULTS_RELATIVE / "seed_201" / "checkpoints" / "accepted_model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    store = store_for(root)
    checkpoint = E._resolve_attacker_checkpoint("A1-CUR2V2-201-REVIEW", {"attackers": {}}, store)
    with pytest.raises(ApplicationError) as error:                                       # the file hash cannot equal the pre-registered one
        E._verify_artifact(store, checkpoint)
    assert error.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_review_observation_uses_the_canonical_25m_gate_not_the_masked_curriculum_gate():
    import numpy as np
    from types import SimpleNamespace

    seen = {}

    def build(env, detection_range):
        seen["range"] = detection_range
        return np.zeros(37)

    env = SimpleNamespace(config=SimpleNamespace(enemy_defender_detection_range=25.0))
    modules = {"experiments.journal_v2_a1": SimpleNamespace(build_a1_v2_observation=build), "experiments.journal_v2_a1_cur1v2": SimpleNamespace(MASKED_DETECTION_RANGE=-1.0)}
    assert E._build_attacker_observation("A1-CUR2V2-203-REVIEW", modules, env).shape == (37,) and seen["range"] == 25.0
    E._build_attacker_observation("A1-BC1-203", modules, env)
    assert seen["range"] == -1.0


def test_committed_shortlist_is_complete_verified_and_prioritised():
    data = json.loads(SHORTLIST.read_text(encoding="utf-8"))
    rows = data["recordings"]
    assert data["label"] == REVIEW_LABEL and "no scientific status changed" in data["scientific_status"] and data["all_verified"] and 10 <= len(rows) <= 13
    assert [(r["attacker"], r["kind"]) for r in rows[:4]] == [("A1-CUR2V2-201-REVIEW", "representative_high_quality_success"), ("A1-CUR2V2-203-REVIEW", "representative_high_quality_success"),
                                                              ("A1-CUR2V2-203-REVIEW", "interception_failure"), ("A0-CUR2V2-REFERENCE", "matched_a0_reference")]
    for seed in (201, 202, 203):
        kinds = {r["kind"] for r in rows if r["model_seed"] == seed}
        assert {"closest_successful_pass", "largest_successful_detour", "interception_failure"} <= kinds and kinds & {"representative_success", "representative_high_quality_success"}
    assert next(r for r in rows if r["kind"] == "matched_a0_reference")["stats"]["outcome"] == "intercepted"
    assert rows[0]["evaluation_seed"] == next(r for r in rows if r["kind"] == "matched_a0_reference")["evaluation_seed"]
    for r in rows:
        v = r["verification"]
        assert v["passed"] and all(v["checks"].values()) and v["max_trajectory_difference_m"] <= data["tolerance_m"]
        assert r["what_to_look_for"].count("\n") > 8 and all(q in r["what_to_look_for"] for q in data["checklist"]) and LABEL_IN(r["what_to_look_for"])
        assert r["url"].endswith(r["replay_id"]) and len(r["replay_id"]) == 32
        if r["model_seed"]:
            assert r["verification"]["replay_sha256"].startswith(EXPECTED_SHA_PREFIX[r["model_seed"]]) and r["verification"]["replay_sha256"] == r["verification"]["manifest_sha256"]


def LABEL_IN(text: str) -> bool:
    return REVIEW_LABEL in text


def _sim_env():
    root, python = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT"), os.environ.get("VISUALIZER_SIMULATION_PYTHON")
    if not root or not python:
        pytest.skip("Set VISUALIZER_RESEARCH_DATA_ROOT and VISUALIZER_SIMULATION_PYTHON to run the real simulator tests.")
    if not (Path(root) / E.REVIEW_RESULTS_RELATIVE / "seed_201" / "checkpoints" / "accepted_model.zip").exists():
        pytest.skip("research Stage2-v2 models are not available")
    return python, root


def _worker(python, root, code):
    env = {**os.environ, "VISUALIZER_RESEARCH_DATA_ROOT": root, "PYTHONPATH": str(REPO)}
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_real_replays_reproduce_the_research_outcome_and_steps_and_use_the_exact_stage2_environment():
    python, root = _sim_env()
    final = json.loads((Path(root) / E.REVIEW_DOCS_RELATIVE / "final_evaluation.json").read_text())
    ref201 = next(e for e in final["per_seed"]["201"]["episodes"] if e["scenario_seed"] == 147149)
    ref_a0 = next(e for e in final["a0"]["episodes"] if e["scenario_seed"] == 147149)
    code = (
        "import json\n"
        "from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode\n"
        "out = {}\n"
        "for name, seed in (('A1-CUR2V2-201-REVIEW', 201), ('A0-CUR2V2-REFERENCE', None)):\n"
        "    r = ReplayRequest(defender_policy_id='passive', attacker_condition=name, attacker_model_seed=seed, clutter_level='C0', semantics='V2', evaluation_seed=147149)\n"
        "    e = run_replay_episode(r).episodes[0]\n"
        "    out[name] = {'outcome': e.summary.outcome, 'steps': e.summary.steps, 'builder': e.environment.builder, 'v_s': e.environment.config['v_s'], 'intercept_radius': e.environment.config['intercept_radius'],\n"
        "                 'asset': len({tuple(f.asset.position_m) for f in e.frames}), 'defender': len({tuple(f.defender.position_m) for f in e.frames}), 'label': e.attacker.parameters['display_label'],\n"
        "                 'sha': e.attacker.checkpoint.sha256 if e.attacker.checkpoint else None, 'not_mastered': e.attacker.parameters.get('stage2_v2_not_mastered'), 'canonical': e.attacker.parameters.get('canonical'),\n"
        "                 'min_sep': e.summary.safety_metrics['minimum_defender_attacker_distance_m']}\n"
        "print(json.dumps(out))\n"
    )
    out = _worker(python, root, code)
    a1, a0 = out["A1-CUR2V2-201-REVIEW"], out["A0-CUR2V2-REFERENCE"]
    assert (a1["outcome"], a1["steps"]) == (ref201["outcome"], ref201["steps"]) == ("soldier_caught", 10)
    assert (a0["outcome"], a0["steps"]) == (ref_a0["outcome"], ref_a0["steps"]) == ("intercepted", 5) and a0["min_sep"] == pytest.approx(2.5, abs=1e-3)
    for item in (a1, a0):
        assert item["builder"].endswith("journal_v2_a1_cur2.stage2_config") and item["v_s"] == 0.0 and item["intercept_radius"] == 2.5 and item["asset"] == 1 and item["defender"] == 1
    assert a1["sha"].startswith("d70b85a6") and a1["not_mastered"] is True and a1["canonical"] is False and "NOT MASTERED" in a1["label"] and a0["sha"] is None
