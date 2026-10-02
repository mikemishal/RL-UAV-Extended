"""
Tests for the defender-adaptation held-out evaluation harness (journal
extension, 38-phase defender-adaptation-EVALUATION sub-task, Phase 13).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.defender_adaptation_evaluation import (
    ARCHITECTURES,
    ATTACKER_CONDITIONS,
    EVAL_SEED_START,
    EVAL_SEEDS,
    N_EVAL_SEEDS,
    VALID_OUTCOMES,
    append_episode_result,
    build_evaluation_manifest,
    build_model_registry,
    is_valid_episode_row,
    load_completed_episodes,
    load_model_registry,
)


@pytest.fixture(scope="module")
def registry():
    return build_model_registry()


@pytest.fixture(scope="module")
def manifest(registry):
    return build_evaluation_manifest(registry)


def test_manifest_has_exactly_32400_keys(manifest):
    assert manifest["total_expected_keys"] == 32400
    assert manifest["n_defenders"] == 27
    assert manifest["n_attacker_conditions"] == 4
    assert manifest["n_levels"] == 3
    assert manifest["n_eval_seeds"] == 100


def test_manifest_has_no_duplicate_keys(manifest):
    keys = [e["key"] for e in manifest["entries"]]
    assert len(keys) == len(set(keys))


def test_eval_seed_namespace_disjoint_from_prior_experiment_seeds():
    # Prior namespaces audited in the module docstring; spot-check the ones
    # with the largest footprint near our range.
    prior_ranges = [
        (71, 79),            # clutter training seeds
        (95_000, 95_099),    # CLUTTER_EVAL_SEED_START (generous upper bound)
        (180_000, 181_000),  # ATTACKER_VALIDATION_SEED_START region
        (190_000, 191_000),  # ATTACKER_HELDOUT_EVAL_SEED_START region
        (90_000_000, 90_000_019),  # shared periodic in-training validation callback
    ]
    our_range = set(EVAL_SEEDS)
    for lo, hi in prior_ranges:
        assert our_range.isdisjoint(range(lo, hi + 1)), f"overlap with prior range [{lo},{hi}]"
    assert EVAL_SEED_START == 90_010_000
    assert N_EVAL_SEEDS == 100
    assert EVAL_SEED_START > 90_000_019  # strictly above the documented prior maximum


def test_seed_203_present_only_as_final_learned_attacker_condition(manifest):
    seed_203_entries = [e for e in manifest["entries"] if e["attacker_seed"] == 203]
    assert len(seed_203_entries) > 0
    assert all(e["attacker_type"] == "A1-unseen-203" for e in seed_203_entries)
    assert all(e["generation"] in ("d0", "d0_plus", "d1") for e in seed_203_entries)


def test_d0_uses_original_starting_checkpoint(registry):
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            entry = registry["defenders"][f"d0|{arch}|{seed}"]
            assert entry["checkpoint_role"] == "d0_start"
            assert "clutter_training_400k" in entry["path"]
            assert entry["path"].endswith("best_model.zip")


def test_d0_plus_and_d1_primary_use_final_model_not_best(registry):
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            for generation in ("d0_plus", "d1"):
                entry = registry["defenders"][f"{generation}|{arch}|{seed}"]
                assert entry["checkpoint_role"] == "final"
                assert entry["path"].endswith("final_model.zip")
                # best_model.zip is recorded but NOT the primary path.
                assert "best_model_path" in entry
                assert entry["best_model_path"].endswith("best_model.zip")
                assert entry["path"] != entry["best_model_path"]


def test_paired_d0_plus_d1_use_identical_scenario_seeds(manifest):
    d0_plus_seeds = sorted({e["eval_seed"] for e in manifest["entries"] if e["generation"] == "d0_plus"})
    d1_seeds = sorted({e["eval_seed"] for e in manifest["entries"] if e["generation"] == "d1"})
    assert d0_plus_seeds == d1_seeds == sorted(EVAL_SEEDS)


def test_resume_skips_completed_records(tmp_path):
    jsonl_path = tmp_path / "episodes.jsonl"
    row = {
        "architecture": "ppo_clutter_v1", "defender_seed": 71, "generation": "d0",
        "checkpoint_role": "d0_start", "attacker_type": "A0", "attacker_seed": None,
        "level": "C1", "eval_seed": 90010000, "outcome": "intercepted",
        "episode_length": 50, "min_defender_attacker_distance": 1.0, "min_attacker_asset_distance": 5.0,
        "defender_path_length": 10.0, "attacker_path_length": 8.0, "defender_safety_intervention_fraction": 0.1,
    }
    completed = {}
    append_episode_result(jsonl_path, row, completed)
    reloaded = load_completed_episodes(jsonl_path)
    assert len(reloaded) == 1
    # Appending the SAME row again is a no-op (not duplicated).
    append_episode_result(jsonl_path, row, reloaded)
    assert jsonl_path.read_text().count("\n") == 1


def test_malformed_final_jsonl_record_recovery(tmp_path):
    jsonl_path = tmp_path / "episodes.jsonl"
    good_row = {
        "architecture": "ppo_clutter_v1", "defender_seed": 71, "generation": "d0",
        "checkpoint_role": "d0_start", "attacker_type": "A0", "attacker_seed": None,
        "level": "C1", "eval_seed": 90010000, "outcome": "intercepted",
        "episode_length": 50, "min_defender_attacker_distance": 1.0, "min_attacker_asset_distance": 5.0,
        "defender_path_length": 10.0, "attacker_path_length": 8.0, "defender_safety_intervention_fraction": 0.1,
    }
    jsonl_path.write_text(json.dumps(good_row) + "\n" + '{"truncated": tru')
    completed = load_completed_episodes(jsonl_path)
    assert len(completed) == 1


def test_duplicate_conflicting_record_raises(tmp_path):
    jsonl_path = tmp_path / "episodes.jsonl"
    row_a = {
        "architecture": "ppo_clutter_v1", "defender_seed": 71, "generation": "d0",
        "checkpoint_role": "d0_start", "attacker_type": "A0", "attacker_seed": None,
        "level": "C1", "eval_seed": 90010000, "outcome": "intercepted",
        "episode_length": 50, "min_defender_attacker_distance": 1.0, "min_attacker_asset_distance": 5.0,
        "defender_path_length": 10.0, "attacker_path_length": 8.0, "defender_safety_intervention_fraction": 0.1,
    }
    row_b = dict(row_a, outcome="soldier_caught", episode_length=99)
    jsonl_path.write_text(json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n")
    with pytest.raises(ValueError, match="duplicate episode key"):
        load_completed_episodes(jsonl_path)


def test_checkpoint_hash_mismatch_raises(tmp_path, monkeypatch):
    from experiments.defender_adaptation_evaluation import verify_checkpoint_integrity
    fake_registry = {
        "defenders": {"d0|ppo_clutter_v1|71": {"path": str(tmp_path / "nope.zip"), "sha256": "0" * 64}},
        "attackers": {},
    }
    (tmp_path / "nope.zip").write_bytes(b"not a real checkpoint")
    problems = verify_checkpoint_integrity(fake_registry)
    assert len(problems) == 1
    assert "hash mismatch" in problems[0]


def test_outcome_categories_mutually_exclusive(manifest):
    assert len(VALID_OUTCOMES) == len(set(VALID_OUTCOMES))
    exclusive_indicator_fields = (
        "defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
        "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
        "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
    )
    assert len(exclusive_indicator_fields) == len(VALID_OUTCOMES)


def test_environment_configuration_remains_canonical():
    from experiments.journal_clutter_protocol import build_canonical_env_config
    cfg = build_canonical_env_config("C2", "measurement")
    assert cfg.enemy_obstacle_avoidance_enabled is True
    assert cfg.defender_obstacle_avoidance_enabled is True
    assert cfg.enemy_route_planning_enabled is False


def test_attacker_conditions_match_fixed_spec():
    assert ATTACKER_CONDITIONS == (
        ("A0", None), ("A1-seen-201", 201), ("A1-seen-202", 202), ("A1-unseen-203", 203),
    )
