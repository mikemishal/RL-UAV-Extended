"""Stage3-v2 REVIEW policies (A1-CUR3V2-20x) and the A0 reference: identifiers, manifest/hash gating, hold-until-detection replay semantics, exact research-environment replay, and the committed shortlist.

The visualizer only resolves, verifies and labels research artifacts; the scientific status (PROVISIONAL curriculum checkpoint, NON-FINAL, NON-CANONICAL) is never changed here."""

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
    REVIEW3_A0_REFERENCE,
    REVIEW3_ATTACKER_SEEDS,
    REVIEW3_LABEL,
    REVIEW_A0_REFERENCE,
    REVIEW_ATTACKER_SEEDS,
    V2_ATTACKER_SEEDS,
    ReplayBatchRequest,
    ReplayRequest,
)

REPO = Path(__file__).resolve().parents[2]
SHORTLIST = REPO / "visualizer" / "review" / "stage3v2_review_shortlist.json"
EXPECTED_SHA_PREFIX = {201: "c60ae85d", 202: "3dfabf5c", 203: "818338a5"}
MASTERY_ROLLOUT = {201: 3, 202: 5, 203: 7}


def synthetic_root(tmp_path, *, seed=201, sha=None, **overrides):
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("")
    docs = root / E.REVIEW3_DOCS_RELATIVE
    docs.mkdir(parents=True)
    manifest = {"seed": seed, "status": "MASTERED", "mastery_rollout": MASTERY_ROLLOUT[seed], "accepted_rollout": MASTERY_ROLLOUT[seed], "training_protocol": "a1_cur3v2_training_v1",
                "scenario_protocol": E.REVIEW3_SCENARIO_PROTOCOL, "parent_status": "provisional_curriculum_transfer", "observation_version": "a1_v2_obs_v1", "action_version": "a1_cur1_action_v1",
                "reward_version": "a1_v2_reward_v2", "accepted_model": "checkpoints/accepted_model.zip", "accepted_model_sha256": sha or (EXPECTED_SHA_PREFIX[seed] + "0" * 56)}
    manifest.update(overrides)
    (docs / f"seed_{seed}_manifest.json").write_text(json.dumps(manifest))
    return root


def store_for(root):
    return ResearchDataStore(Settings(research_data_root=root))


def request(attacker="A1-CUR3V2-201", seed_model=201, **overrides):
    fields = dict(defender_policy_id="greedy", attacker_condition=attacker, attacker_model_seed=seed_model, clutter_level="C0", semantics="V2", evaluation_seed=154001)
    fields.update(overrides)
    return ReplayRequest(**fields)


def test_identifiers_are_explicit_non_final_and_never_alias_existing_attackers():
    assert REVIEW3_ATTACKER_SEEDS == {"A1-CUR3V2-201": 201, "A1-CUR3V2-202": 202, "A1-CUR3V2-203": 203} and REVIEW3_A0_REFERENCE == "A0-CUR3V2-REFERENCE"
    assert "PROVISIONAL" in REVIEW3_LABEL and "NON-FINAL" in REVIEW3_LABEL and "NON-CANONICAL" in REVIEW3_LABEL
    new = set(REVIEW3_ATTACKER_SEEDS) | {REVIEW3_A0_REFERENCE}
    existing = set(ATTACKER_CONDITIONS) - new
    assert not existing & new and not set(REVIEW3_ATTACKER_SEEDS) & (set(V2_ATTACKER_SEEDS) | set(REVIEW_ATTACKER_SEEDS))
    assert {"A0", "A1-201", "A1-203-V2", "A1-BC1-203", "A1-CUR2V2-203-REVIEW", REVIEW_A0_REFERENCE} <= existing         # earlier ids untouched
    assert set(REVIEW3_ATTACKER_SEEDS.items()) <= set(ALL_ATTACKER_SEEDS.items())


@pytest.mark.parametrize("attacker,seed", REVIEW3_ATTACKER_SEEDS.items())
def test_requests_require_the_exact_stage3_environment_a_scripted_defender_and_their_seed(attacker, seed):
    for defender in ("greedy", "kalman_greedy", "pn", "lead"):
        assert request(attacker, seed, defender_policy_id=defender).attacker_condition == attacker
    for bad in ({"semantics": "V1"}, {"clutter_level": "C1"}, {"defender_policy_id": "passive"}, {"defender_policy_id": "rmpc"}, {"attacker_model_seed": seed + 1}, {"attacker_model_seed": None}):
        with pytest.raises(ValidationError):
            request(attacker, seed, **bad)


def test_reference_attacker_is_scripted_and_batches_follow_the_same_rules():
    assert request(REVIEW3_A0_REFERENCE, None).attacker_condition == REVIEW3_A0_REFERENCE
    with pytest.raises(ValidationError):
        request(REVIEW3_A0_REFERENCE, 201)
    with pytest.raises(ValidationError):
        request(REVIEW3_A0_REFERENCE, None, clutter_level="C2")
    ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A1-CUR3V2-203", clutter_level="C0", semantics="V2", seed_start=154000)
    with pytest.raises(ValidationError):
        ReplayBatchRequest(defender_policy_id="passive", attacker_condition="A1-CUR3V2-203", clutter_level="C0", semantics="V2")


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_resolution_accepts_only_the_mastered_manifest_with_the_registered_hash(tmp_path, seed):
    checkpoint = E._resolve_attacker_checkpoint(f"A1-CUR3V2-{seed}", {"attackers": {}}, store_for(synthetic_root(tmp_path / "ok", seed=seed)))
    assert checkpoint.role == "attacker_review_stage3v2" and checkpoint.training_seed == seed and checkpoint.sha256.startswith(EXPECTED_SHA_PREFIX[seed])
    assert checkpoint.relative_path.endswith(f"stage3_v2/seed_{seed}/checkpoints/accepted_model.zip") and checkpoint.artifact_id.endswith(f"rollout{MASTERY_ROLLOUT[seed]}")
    assert E._resolve_attacker_checkpoint(REVIEW3_A0_REFERENCE, {"attackers": {}}, store_for(synthetic_root(tmp_path / "ref"))) is None


@pytest.mark.parametrize("overrides,code", [
    ({"status": "FAIL"}, "INVALID_MODEL_REGISTRY"),
    ({"mastery_rollout": 12}, "INVALID_MODEL_REGISTRY"),
    ({"accepted_rollout": 16}, "INVALID_MODEL_REGISTRY"),
    ({"training_protocol": "a1_cur2_training_v1"}, "INVALID_MODEL_REGISTRY"),
    ({"scenario_protocol": "a1_cur2_static_defender_nominal_path_v1"}, "INVALID_MODEL_REGISTRY"),
    ({"parent_status": "canonical"}, "INVALID_MODEL_REGISTRY"),
    ({"reward_version": "a1_v2_reward_v1"}, "INVALID_MODEL_REGISTRY"),
    ({"accepted_model_sha256": "ab" * 32}, "CHECKPOINT_HASH_MISMATCH"),
    ({"accepted_model_sha256": "short"}, "CHECKPOINT_HASH_MISMATCH"),
])
def test_resolution_rejects_inconsistent_manifests(tmp_path, overrides, code):
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-CUR3V2-201", {"attackers": {}}, store_for(synthetic_root(tmp_path, **overrides)))
    assert error.value.code == code


def test_missing_manifest_is_unavailable_and_tampered_model_rejected(tmp_path):
    root = synthetic_root(tmp_path)
    payload = b"synthetic-model"
    digest = hashlib.sha256(payload).hexdigest()
    (root / E.REVIEW3_DOCS_RELATIVE / "seed_201_manifest.json").unlink()
    with pytest.raises(ApplicationError) as error:
        E._resolve_attacker_checkpoint("A1-CUR3V2-201", {"attackers": {}}, store_for(root))
    assert error.value.code == "CHECKPOINT_NOT_REGISTERED"
    root2 = synthetic_root(tmp_path / "t", sha=EXPECTED_SHA_PREFIX[201] + digest[8:])
    model = root2 / E.REVIEW3_RESULTS_RELATIVE / "seed_201" / "checkpoints" / "accepted_model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    store = store_for(root2)
    checkpoint = E._resolve_attacker_checkpoint("A1-CUR3V2-201", {"attackers": {}}, store)
    with pytest.raises(ApplicationError) as hash_error:
        E._verify_artifact(store, checkpoint)
    assert hash_error.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_review3_observation_uses_the_canonical_25m_gate_and_checkpoint_shape():
    import numpy as np
    from types import SimpleNamespace

    seen = {}

    def build(env, detection_range):
        seen["range"] = detection_range
        return np.zeros(37)

    env = SimpleNamespace(config=SimpleNamespace(enemy_defender_detection_range=25.0))
    modules = {"experiments.journal_v2_a1": SimpleNamespace(build_a1_v2_observation=build), "experiments.journal_v2_a1_cur1v2": SimpleNamespace(MASKED_DETECTION_RANGE=-1.0)}
    assert E._build_attacker_observation("A1-CUR3V2-203", modules, env).shape == (37,) and seen["range"] == 25.0
    assert set(REVIEW3_ATTACKER_SEEDS) <= set(E.REVIEW_MODEL_SEEDS)


def test_committed_shortlist_is_complete_verified_and_prioritised():
    data = json.loads(SHORTLIST.read_text(encoding="utf-8"))
    rows = data["recordings"]
    assert data["label"] == REVIEW3_LABEL and "no scientific status changed" in data["scientific_status"] and data["all_verified"] and 10 <= len(rows) <= 15
    assert [(r["attacker"], r["kind"]) for r in rows[:5]] == [("A1-CUR3V2-203", "representative_high_quality_success"), ("A1-CUR3V2-203", "closest_successful_pass"),
                                                              ("A1-CUR3V2-203", "interception_failure"), ("A0-CUR3V2-REFERENCE", "matched_a0_reference"), ("A1-CUR3V2-201", "representative_high_quality_success")]
    kinds = {r["kind"] for r in rows}
    assert {"lateral_evasion", "vertical_evasion", "mixed_evasion", "interception_failure", "matched_a0_reference", "generalization_capture", "closest_successful_pass"} <= kinds
    for seed in (201, 202, 203):
        assert {"closest_successful_pass", "interception_failure"} <= {r["kind"] for r in rows if r["model_seed"] == seed}
    assert next(r for r in rows if r["kind"] == "matched_a0_reference")["stats"]["outcome"] == "intercepted"
    assert rows[0]["evaluation_seed"] == next(r for r in rows if r["kind"] == "matched_a0_reference")["evaluation_seed"] and rows[0]["stats"]["outcome"] == "soldier_caught"
    for r in rows:
        v = r["verification"]
        assert v["passed"] and all(v["checks"].values()) and v["max_attacker_trajectory_difference_m"] <= data["tolerance_m"] and v["max_defender_trajectory_difference_m"] <= data["tolerance_m"]
        assert r["what_to_look_for"].count("\n") > 8 and all(q in r["what_to_look_for"] for q in data["checklist"]) and REVIEW3_LABEL in r["what_to_look_for"]
        assert r["url"].endswith(r["replay_id"]) and len(r["replay_id"]) == 32 and ("generalization" in r) == (r["kind"] == "generalization_capture")
        if r["kind"] == "generalization_capture":
            assert r["defender"] in ("lead", "pn") and r["model_seed"] == 203 and r["stats"]["outcome"] == "soldier_caught"
        if r["model_seed"]:
            assert v["replay_sha256"].startswith(EXPECTED_SHA_PREFIX[r["model_seed"]]) and v["replay_sha256"] == v["manifest_sha256"]


def test_research_module_list_keeps_every_module_the_defender_catalog_needs_and_adds_stage3v2():
    import inspect

    source = inspect.getsource(E._research_modules)
    for name in ("experiments.post_detection_diagnostic_protocol", "experiments.lead_residual_diagnostic_protocol", "experiments.clutter_benchmark_protocol",
                 "experiments.journal_v2_a1_cur2", "experiments.journal_v2_a1_cur3v2"):
        assert f'"{name}"' in source


def _sim_env():
    root, python = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT"), os.environ.get("VISUALIZER_SIMULATION_PYTHON")
    if not root or not python:
        pytest.skip("Set VISUALIZER_RESEARCH_DATA_ROOT and VISUALIZER_SIMULATION_PYTHON to run the real simulator tests.")
    if not (Path(root) / E.REVIEW3_RESULTS_RELATIVE / "seed_203" / "checkpoints" / "accepted_model.zip").exists():
        pytest.skip("research Stage3-v2 models are not available")
    return python, root


def _worker(python, root, code):
    env = {**os.environ, "VISUALIZER_RESEARCH_DATA_ROOT": root, "PYTHONPATH": str(REPO)}
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_real_catalog_lists_the_stage3v2_review_attackers_as_available_with_the_pinned_commit():
    python, root = _sim_env()
    out = subprocess.run([python, "-m", "visualizer.backend.catalog_cli"], capture_output=True, text=True, cwd=REPO, timeout=900,
                         env={**os.environ, "VISUALIZER_RESEARCH_DATA_ROOT": root, "VISUALIZER_SIMULATION_PYTHON": python, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr[-2000:]
    catalog = json.loads(out.stdout.strip().splitlines()[-1])
    assert catalog["source_commit"] == E.EXPECTED_RESEARCH_COMMIT
    by_id = {a["attacker_id"]: a for a in catalog["attackers"]}
    for attacker, seed in REVIEW3_ATTACKER_SEEDS.items():
        entry = by_id[attacker]
        assert entry["available"] and entry["checkpoint_sha256"].startswith(EXPECTED_SHA_PREFIX[seed]) and entry["architecture"] == "a1_cur3v2_review" and "PROVISIONAL" in entry["display_name"]
    assert by_id[REVIEW3_A0_REFERENCE]["available"] and by_id[REVIEW3_A0_REFERENCE]["classification"] == "scripted"
    assert {"greedy", "kalman_greedy", "pn", "lead"} <= {d["policy_id"] for d in catalog["defenders"]}


def test_real_replays_hold_then_release_the_defender_and_match_the_research_run():
    python, root = _sim_env()
    rows = json.loads(SHORTLIST.read_text(encoding="utf-8"))["recordings"]
    a1 = rows[0]
    final = json.loads((Path(root) / E.REVIEW3_DOCS_RELATIVE / "final_evaluation.json").read_text())
    ref = next(e for e in final["per_seed"]["203"]["episodes"] if e["scenario_seed"] == a1["evaluation_seed"])
    code = (
        "import json\n"
        "from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode\n"
        f"r = ReplayRequest(defender_policy_id='greedy', attacker_condition='A1-CUR3V2-203', attacker_model_seed=203, clutter_level='C0', semantics='V2', evaluation_seed={a1['evaluation_seed']})\n"
        "e = run_replay_episode(r).episodes[0]\n"
        "rel = e.attacker.parameters['defender_release_step']\n"
        "pos = [tuple(f.defender.position_m) for f in e.frames]\n"
        "vel = [sum(x * x for x in f.defender.velocity_mps) ** 0.5 for f in e.frames]\n"
        "print(json.dumps({'outcome': e.summary.outcome, 'steps': e.summary.steps, 'release': rel, 'held_positions': len(set(pos[:rel + 1])), 'held_speed': max(vel[:rel + 1]), 'speed_after': max(vel[rel + 1:]) if len(vel) > rel + 1 else 0.0,\n"
        "                  'builder': e.environment.builder, 'v_s': e.environment.config['v_s'], 'standby': e.environment.config['defender_standby_until_detection'], 'sha': e.attacker.checkpoint.sha256,\n"
        "                  'label': e.attacker.parameters['display_label'], 'canonical': e.attacker.parameters.get('canonical'), 'hold': e.attacker.parameters.get('hold_until_detection'),\n"
        "                  'frame_note': any('curriculum.defender_hold' in f.unavailable_fields for f in e.frames)}))\n"
    )
    out = _worker(python, root, code)
    assert (out["outcome"], out["steps"]) == (ref["outcome"], ref["steps"]) == (a1["stats"]["outcome"], a1["stats"]["steps"]) and out["release"] == ref["release_step"]
    assert out["held_positions"] == 1 and out["held_speed"] == 0.0 and (out["speed_after"] > 0.0)          # fixed and at rest until release, then moving
    assert out["builder"].endswith("journal_v2_a1_cur3v2.stage3v2_config") and out["v_s"] == 0.0 and out["standby"] is False and out["sha"].startswith("818338a5")
    assert out["canonical"] is False and out["hold"] is True and "PROVISIONAL" in out["label"] and out["frame_note"] is True
