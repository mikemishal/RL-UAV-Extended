"""Tests for curriculum Stage2-v2 (budget-only continuation of Stage2-v1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_training as T2
from experiments import journal_v2_a1_cur2v2 as V2
from experiments import journal_v2_a1_cur2v2_run as RUN
from experiments import journal_v2_a1_cur2v2_state as TS


def test_protocol_differs_from_stage2_v1_only_in_the_budget():
    diff = V2.differences_from_v1()
    assert diff == {"ppo.max_rollouts": [8, 16], "ppo.max_timesteps": [262_144, 524_288]}
    spec = V2.protocol_specification()
    v1 = S.protocol_specification(V2.parent_hashes())
    for key in ("environment", "placement", "initialization", "reward", "validation", "gate", "mastery", "retention", "success_rule"):
        assert spec[key] == v1[key]
    assert spec["ppo"]["hyperparameters"] == A.A1_V2_PPO_HYPERPARAMS and spec["gate"] == S.GATE and spec["version"]["id"] == "a1_curriculum_stage2_v2"
    assert V2.protocol_sha256() == V2.protocol_sha256() != V2.v1_protocol_sha256()
    assert S.protocol_sha256(V2.parent_hashes()) == V2.v1_protocol_sha256()                    # Stage-2 v1 definitions have not drifted
    assert V2.SPEC.max_timesteps == 524_288 and V2.SPEC.steps_per_rollout == 32_768 and V2.SPEC.max_rollouts - V2.V1_SPEC.max_rollouts == 8


def test_validation_alternation_continues_from_stage2_v1():
    assert [T2.validation_set_for(k) for k in (8, 9, 10, 15, 16)] == ["B", "A", "B", "A", "B"]
    assert S.validation_seeds("A")[0] == 147_000 and S.validation_seeds("B")[-1] == 147_199


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_stage2_v1_parents_are_the_preregistered_verified_rollout_8_commits(seed):
    p = V2.verify_v1_parent(seed)
    assert p["model_sha256"].startswith(V2.V1_PARENT_PREFIX[seed]) and p["completed_rollout"] == 8 and p["num_timesteps"] == 262_144 and len(p["state_sha256_policy_and_optimizer"]) == 64


def test_seeding_is_read_only_on_v1_and_leaves_a_recoverable_rollout_8_state(tmp_path):
    before = {s: V2.verify_v1_parent(s)["rollout8_manifest_sha256"] for s in (201,)}
    v1_files = sorted(V2.V1_RESULTS_ROOT.glob("seed_201/**/*"))
    v1_hash = {str(p): T2.T.sha256_file(p) for p in v1_files if p.is_file() and p.suffix in (".json", ".zip")}
    parent = V2.seed_from_v1(201, tmp_path)
    again = V2.seed_from_v1(201, tmp_path)                                                    # idempotent
    assert parent["model_sha256"] == again["model_sha256"]
    with V2.activated():
        default_layout = T2.make_layout(201, tmp_path)
        assert V2.STAGE in str(default_layout.canonical) and V1_STAGE_NOT_IN(default_layout.canonical)
        layout = T2.make_layout(201, tmp_path, tmp_path / "res")                           # isolated results root (the real Stage2-v2 results may already exist)
        best, manifest = T2.recover(layout, spec=V2.SPEC)
        validations = T2.collect_validations(layout, best, V2.SPEC)
    assert best == 8 and manifest["model_sha256"] == parent["model_sha256"] and len(validations) == 8 and T2.seed_decision(validations) == (None, None)
    assert {s: V2.verify_v1_parent(s)["rollout8_manifest_sha256"] for s in (201,)} == before
    assert {str(p): T2.T.sha256_file(p) for p in v1_files if p.is_file() and p.suffix in (".json", ".zip")} == v1_hash


def V1_STAGE_NOT_IN(path: Path) -> bool:
    return "stage2_v2" in path.parts and "stage2" not in path.parts


def test_activation_scopes_the_output_namespace_and_protects_stage2_v1_results():
    original = S.RESULTS_ROOT
    with V2.activated():
        assert S.RESULTS_ROOT == V2.RESULTS_ROOT != V2.V1_RESULTS_ROOT
        with pytest.raises(T2.RunConflictError):
            T2.make_layout(201, Path("C:/tmp_scratch"), V2.V1_RESULTS_ROOT)
    assert S.RESULTS_ROOT == original == V2.V1_RESULTS_ROOT


def test_extension_of_a_finished_run_equals_the_uninterrupted_run_real_sb3(tmp_path):
    """Run to rollout 2, stop, then call again with a larger budget (the Stage2-v2 mechanism) == an uninterrupted run to rollout 4 (real SB3, SubprocVecEnv, BC initialisation)."""
    kw = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=32, batch_size=64)
    small, big = T2.Spec(n_envs=2, n_steps=32, max_rollouts=2), T2.Spec(n_envs=2, n_steps=32, max_rollouts=4)
    a, b = {"s": tmp_path / "a_s", "r": tmp_path / "a_r"}, {"s": tmp_path / "b_s", "r": tmp_path / "b_r"}
    T2.train_seed(201, scratch_root=a["s"], results_root=a["r"], spec=big, ppo_kwargs=kw, validate=False)
    T2.train_seed(201, scratch_root=b["s"], results_root=b["r"], spec=small, ppo_kwargs=kw, validate=False)
    T2.train_seed(201, scratch_root=b["s"], results_root=b["r"], spec=big, ppo_kwargs=kw, validate=False)
    la, lb = T2.make_layout(201, a["s"], a["r"]), T2.make_layout(201, b["s"], b["r"])
    for k in range(1, 5):
        ma, mb = T2.verify_rollout(la.scratch_rollouts, k, seed=201, spec=big), T2.verify_rollout(lb.scratch_rollouts, k, seed=201, spec=big)
        assert ma["state_sha256"] == mb["state_sha256"] and ma["episodes_sha256"] == mb["episodes_sha256"], k


def test_trend_description_is_descriptive_and_threshold_based():
    assert V2.describe_trend([0.3, 0.4, 0.6, 0.7])["label"] == "rising"
    assert V2.describe_trend([0.60, 0.62, 0.61, 0.63, 0.60, 0.62])["label"] == "plateau"
    assert V2.describe_trend([0.8, 0.7, 0.5, 0.4])["label"] == "regressing"
    assert V2.describe_trend([0.1, 0.2])["label"] == "insufficient"


def test_budget_sensitivity_pairs_the_same_scenarios():
    ep = lambda outs: [{"scenario_seed": 147_000 + i, "outcome": "soldier_caught" if o else "intercepted"} for i, o in enumerate(outs)]  # noqa: E731
    v1 = {"per_seed": {str(s): {"episodes": ep([1] * 60 + [0] * 40), "metrics": {"interception": 0.4}, "training_status": "FAIL"} for s in S.PPO_SEEDS}}
    v2 = {"per_seed": {str(s): {"episodes": ep([1] * 92 + [0] * 8), "metrics": {"interception": 0.08}, "training_status": "MASTERED"} for s in S.PPO_SEEDS}}
    r = RUN.budget_sensitivity(v1, v2)["201"]
    assert r["paired_v2_minus_v1_capture"]["difference_pp"] == pytest.approx(32.0) and r["stage2_v1_capture"] == 0.6 and r["stage2_v2_capture"] == 0.92
    v2["per_seed"]["201"]["episodes"][0]["scenario_seed"] = 1
    with pytest.raises(AssertionError):
        RUN.budget_sensitivity(v1, v2)


def test_task_state_stages_and_namespace():
    assert len(TS.STAGES) == 11 and TS.STAGES[4] == "05_seed_201_ppo_rollouts_9_16" and TS.STATE_DIR.name == "stage2_v2"
    assert V2.RESULTS_ROOT.name == "stage2_v2" and V2.V1_RESULTS_ROOT.name == "stage2"
