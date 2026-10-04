"""
Canonical no-obstacle (C0) LR-PPO-semantics evaluation for the D0/D0+/D1
defender-adaptation checkpoints (journal "first publication evaluation"
sub-task). Supersedes the framing of the earlier
`defender_adaptation_c0_evaluation.py` run (tag `journal-paper-reproduction-v1`)
as the canonical LR-PPO baseline comparison -- see
`docs/lrppo_vs_journal_environment_audit.md` Phase 13 for the
reclassification. This module is a NEW, independently-seeded dataset in its
own result namespace; the prior C0 dataset is left untouched for provenance.

PRIMARY CONFIGURATION (Phase 1 of this sub-task): the "C0 + A0 + canonical
LR-PPO semantics" environment identified as authoritative by the audit doc
is simply `journal_clutter_protocol.build_canonical_env_config("C0", estimator)`
with NO extra_overrides -- this already resolves to
`defender_standby_until_detection=True` (the bare EnvConfig default, shared
identically by the frozen LR-PPO reference branch and every
publication-relevant journal campaign; see the audit doc Section 0/Phase 9).
This module asserts that resolution explicitly and writes the full resolved
config to the evaluation manifest -- "fail loudly if any canonical setting
is wrong" (Phase 1).

Seed namespace: audited every SEED_START/SEED_OFFSET constant across
experiments/*.py (see docs/canonical_c0_evaluation_seed_audit.md). The
highest reserved block below the attacker's 180,000-199,999 range is the
(written-but-never-run) C0 robustness grid at 120,000-120,999. This module
reserves the next free 10,000-aligned block: CANONICAL_C0_EVAL_SEED_START
=130_000, N=100 (130,000-130,099) -- disjoint from the legacy C0 evaluation's
100,000-100,099.

Reuses, UNCHANGED, every piece of validated infrastructure from
`experiments/defender_adaptation_evaluation.py` (frozen model registry,
`DefenderInstance`/`AttackerCondition` construction, the common-random-
numbers matched-episode runner, outcome taxonomy, JSONL persistence). No
training or model modification happens anywhere in this module.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.defender_adaptation_evaluation import (
    PROJECT_ROOT,
    MODEL_REGISTRY_PATH,
    ATTACKER_CONDITIONS,
    load_model_registry,
    build_model_registry,
    verify_checkpoint_integrity,
    build_defender_instances,
    build_attacker_conditions,
    _run_matched_episode,
    _episode_key,
    _config_hash,
    _eval_git_commit,
    load_completed_episodes,
    append_episode_result,
)

OUT_DIR = PROJECT_ROOT / "results" / "defender_adaptation_canonical_c0_evaluation"

CANONICAL_C0_LEVEL = "C0"
CANONICAL_C0_EVAL_SEED_START = 130_000
N_EVAL_SEEDS = 100
EVAL_SEEDS: tuple[int, ...] = tuple(range(CANONICAL_C0_EVAL_SEED_START, CANONICAL_C0_EVAL_SEED_START + N_EVAL_SEEDS))

# `build_canonical_env_config("C0", estimator)` alone leaves
# enemy_obstacle_avoidance_enabled/defender_obstacle_avoidance_enabled=True
# and both awareness modes="map_local" (dormant no-ops with zero obstacles,
# but NOT bit-for-bit the bare-EnvConfig state validated by the audit's
# Phase 10 runtime-equivalence check). Phase 1 of this sub-task requires
# the LITERAL "no defender/attacker obstacle avoidance" state, so these are
# explicitly forced off here -- matching the exact configuration
# independently verified against the frozen LR-PPO reference branch.
CANONICAL_EXTRA_OVERRIDES: dict = {
    "enemy_obstacle_avoidance_enabled": False,
    "defender_obstacle_avoidance_enabled": False,
    "enemy_obstacle_awareness_mode": "disabled",
    "defender_obstacle_awareness_mode": "disabled",
}

# Previously-used ranges this evaluation must never collide with (Phase 3
# audit trail, recorded here for anyone re-deriving the next free block).
EXCLUDED_RANGES_AUDITED: tuple[tuple[str, int, int], ...] = (
    ("training_seeds", 42, 44), ("clutter_training_seeds", 71, 79),
    ("paper_validation", 10_000, 10_099), ("paper_diagnostic", 12_000, 12_199),
    ("mobility_audit_diagnostic", 18_000, 18_199),
    ("paper_final_nominal", 20_000, 24_999), ("acquisition_ablation", 25_000, 25_999),
    ("robustness_detection", 30_000, 30_999), ("robustness_measurement_noise", 31_000, 31_999),
    ("robustness_mobility", 32_000, 32_999), ("robustness_evasion", 33_000, 33_999),
    ("robustness_maneuverability_and_dynamics", 34_000, 34_999), ("robustness_mobility_grid", 35_000, 35_499),
    ("post_detection_validation", 40_000, 40_099), ("post_detection_diagnostic", 40_100, 40_299),
    ("post_detection_final_test", 41_000, 45_999), ("post_detection_future_robustness", 46_000, 46_999),
    ("lrppo_residual_validation", 60_000, 60_099), ("lrppo_residual_diagnostic", 60_100, 60_299),
    ("lrppo_expanded_final", 61_000, 65_999), ("lrppo_robustness", 66_000, 66_999),
    ("expanded_robustness_evasion", 66_000, 66_999), ("expanded_robustness_maneuverability", 67_000, 67_999),
    ("expanded_robustness_measurement_noise", 68_000, 68_999), ("expanded_robustness_mobility", 69_000, 69_999),
    ("expanded_robustness_detection", 70_000, 70_999), ("expanded_robustness_future", 71_000, 71_999),
    ("rmpc_final_nominal", 72_000, 76_999), ("rmpc_robustness", 77_000, 77_999),
    ("clutter_benchmark_legacy", 80_000, 89_999), ("clutter_eval", 95_000, 99_999),
    ("legacy_c0_evaluation", 100_000, 100_099),
    ("c0_robustness_sweep_reserved_unused", 110_000, 110_999),
    ("c0_robustness_grid_reserved_unused", 120_000, 120_999),
    ("attacker_validation", 180_000, 189_999), ("attacker_heldout_eval", 190_000, 199_999),
    ("d0_training_validation_callback", 90_000_000, 90_000_099),
    ("defender_adaptation_eval", 90_010_000, 90_010_099),
)


def assert_canonical_lrppo_config(cfg: EnvConfig) -> dict:
    """Phase 1: programmatically assert the resolved config matches
    'C0 + A0 + canonical LR-PPO semantics'. Fails loudly (AssertionError) if
    any canonical setting is wrong -- never silently proceeds."""
    checks = {
        "defender_standby_until_detection": (cfg.defender_standby_until_detection, True),
        "obstacles_enabled": (cfg.obstacles_enabled, False),
        "dynamic_obstacles_enabled": (cfg.dynamic_obstacles_enabled, False),
        "defender_obstacle_avoidance_enabled": (cfg.defender_obstacle_avoidance_enabled, False),
        "enemy_obstacle_avoidance_enabled": (cfg.enemy_obstacle_avoidance_enabled, False),
        "enemy_route_planning_enabled": (cfg.enemy_route_planning_enabled, False),
        "enemy_obstacle_awareness_mode": (cfg.enemy_obstacle_awareness_mode, "disabled"),
        "defender_obstacle_awareness_mode": (cfg.defender_obstacle_awareness_mode, "disabled"),
    }
    failures = [f"{k}: expected {expected!r}, got {actual!r}" for k, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise AssertionError("Canonical C0 + A0 + LR-PPO configuration check FAILED:\n" + "\n".join(failures))
    return {k: v[0] for k, v in checks.items()}


def build_evaluation_manifest(registry: dict | None = None) -> dict:
    registry = registry if registry is not None else load_model_registry()
    defenders = build_defender_instances(registry)
    attacker_conditions_meta = [
        {"name": name, "seed": seed, **(
            {"checkpoint_path": registry["attackers"][str(seed)]["path"],
             "checkpoint_sha256": registry["attackers"][str(seed)]["sha256"]}
            if seed is not None else {"checkpoint_path": None, "checkpoint_sha256": None}
        )}
        for name, seed in ATTACKER_CONDITIONS
    ]
    config_hash_cache: dict[str, str] = {}
    resolved_config_cache: dict[str, dict] = {}

    def _get_config(estimator: str) -> EnvConfig:
        if estimator not in resolved_config_cache:
            cfg = build_canonical_env_config(CANONICAL_C0_LEVEL, estimator, extra_overrides=CANONICAL_EXTRA_OVERRIDES)
            assert_canonical_lrppo_config(cfg)
            resolved_config_cache[estimator] = dataclasses.asdict(cfg)
            config_hash_cache[estimator] = _config_hash(cfg)
        return resolved_config_cache[estimator]

    keys = []
    for d in defenders:
        _get_config(d.estimator)
        for attacker_meta in attacker_conditions_meta:
            for eval_seed in EVAL_SEEDS:
                keys.append({
                    "key": "|".join(str(x) for x in (
                        d.architecture, d.defender_seed, d.generation, d.checkpoint_role,
                        attacker_meta["name"], attacker_meta["seed"], CANONICAL_C0_LEVEL, eval_seed,
                    )),
                    "architecture": d.architecture, "defender_seed": d.defender_seed,
                    "generation": d.generation, "checkpoint_role": d.checkpoint_role,
                    "defender_checkpoint_path": d.checkpoint_path, "defender_checkpoint_sha256": d.checkpoint_sha256,
                    "attacker_type": attacker_meta["name"], "attacker_seed": attacker_meta["seed"],
                    "attacker_checkpoint_path": attacker_meta["checkpoint_path"],
                    "attacker_checkpoint_sha256": attacker_meta["checkpoint_sha256"],
                    "level": CANONICAL_C0_LEVEL, "eval_seed": eval_seed,
                    "clutter_config_hash": config_hash_cache[d.estimator],
                    "observation_version": "36-D (PPO/PPO-Kalman)" if not d.is_lr_ppo else "39-D (LR-PPO)",
                })
    manifest = {
        "total_expected_keys": len(keys),
        "n_defenders": len(defenders),
        "n_attacker_conditions": len(attacker_conditions_meta),
        "n_levels": 1,
        "level": CANONICAL_C0_LEVEL,
        "n_eval_seeds": len(EVAL_SEEDS),
        "eval_seed_range": [CANONICAL_C0_EVAL_SEED_START, CANONICAL_C0_EVAL_SEED_START + N_EVAL_SEEDS - 1],
        "resolved_env_config": resolved_config_cache,
        "canonical_assertions_passed": True,
        "code_commit": _eval_git_commit(),
        "entries": keys,
    }
    return manifest


def write_evaluation_manifest(path: Path | None = None) -> dict:
    path = path if path is not None else (OUT_DIR / "evaluation_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_evaluation_manifest()
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def build_recovery_manifest(jsonl_path: Path | None = None, evaluation_manifest: dict | None = None) -> dict:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    evaluation_manifest = evaluation_manifest if evaluation_manifest is not None else build_evaluation_manifest()
    completed = load_completed_episodes(jsonl_path)
    expected_keys = [e["key"] for e in evaluation_manifest["entries"]]
    total_expected = len(expected_keys)
    n_valid = sum(1 for k in expected_keys if k in completed)
    return {
        "total_expected": total_expected,
        "complete_valid": n_valid,
        "missing": total_expected - n_valid,
        "percent_complete": (100.0 * n_valid / total_expected) if total_expected else 100.0,
    }


def evaluate_all(jsonl_path: Path | None = None, registry: dict | None = None) -> pd.DataFrame:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else load_model_registry()
    completed = load_completed_episodes(jsonl_path)

    defenders = build_defender_instances(registry)
    attacker_conditions = build_attacker_conditions(registry)
    lead_policy_cache: dict[str, object] = {}
    eval_code_commit = _eval_git_commit()

    rows = list(completed.values())
    n_skipped = 0
    n_executed = 0
    for d in defenders:
        base_config = build_canonical_env_config(CANONICAL_C0_LEVEL, d.estimator, extra_overrides=CANONICAL_EXTRA_OVERRIDES)
        assert_canonical_lrppo_config(base_config)
        defender_policy = d.build(base_config)
        lead_policy_for_residual = None
        if d.is_lr_ppo:
            cache_key = d.estimator
            if cache_key not in lead_policy_cache:
                lead_policy_cache[cache_key] = get_policy("lead", config=base_config)
            lead_policy_for_residual = lead_policy_cache[cache_key]
        env = SoldierEnv(base_config)
        for attacker in attacker_conditions:
            t0 = time.time()
            n_new = 0
            for eval_seed in EVAL_SEEDS:
                key = _episode_key(d, attacker, CANONICAL_C0_LEVEL, eval_seed)
                if key in completed:
                    n_skipped += 1
                    continue
                metrics = _run_matched_episode(env, d, defender_policy, lead_policy_for_residual, attacker.model, eval_seed)
                metrics.update(
                    architecture=d.architecture, defender_seed=d.defender_seed, generation=d.generation,
                    checkpoint_role=d.checkpoint_role, attacker_type=attacker.name, attacker_seed=attacker.seed,
                    level=CANONICAL_C0_LEVEL, eval_seed=eval_seed, eval_code_commit=eval_code_commit,
                )
                append_episode_result(jsonl_path, metrics, completed)
                rows.append(metrics)
                n_executed += 1
                n_new += 1
            if n_new:
                print(f"{d.architecture:>22s} seed={d.defender_seed} {d.generation:<8s} level=C0(canonical) "
                      f"attacker={attacker.name:<16s} (+{n_new} new, {time.time()-t0:.1f}s)", flush=True)
    print(f"=== canonical C0 evaluation loop done: {n_executed} executed, {n_skipped} skipped ===", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-registry", action="store_true")
    args = parser.parse_args()

    if args.rebuild_registry or not MODEL_REGISTRY_PATH.exists():
        print("Building model registry (shared with C1-C3 / legacy-C0 evaluations)...", flush=True)
        build_model_registry()

    registry = load_model_registry()
    print("Checkpoint integrity check (before)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED before evaluation:\n" + "\n".join(problems))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "evaluation_manifest.json"
    if not manifest_path.exists():
        write_evaluation_manifest(manifest_path)
    evaluation_manifest = json.loads(manifest_path.read_text())
    print("=== resolved canonical config (Phase 1 assertion passed) ===", flush=True)
    print(json.dumps(evaluation_manifest["resolved_env_config"], indent=2), flush=True)

    recovery = build_recovery_manifest(OUT_DIR / "episodes.jsonl", evaluation_manifest)
    print("=== recovery manifest (before) ===", flush=True)
    print(json.dumps(recovery, indent=2), flush=True)

    df = evaluate_all(OUT_DIR / "episodes.jsonl", registry)

    final_recovery = build_recovery_manifest(OUT_DIR / "episodes.jsonl", evaluation_manifest)
    print("=== recovery manifest (after) ===", flush=True)
    print(json.dumps(final_recovery, indent=2), flush=True)
    if final_recovery["missing"] != 0:
        raise RuntimeError(f"evaluation incomplete: {final_recovery['missing']} missing -- re-run to resume")

    print("Checkpoint integrity check (after)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED after evaluation:\n" + "\n".join(problems))

    df.to_csv(OUT_DIR / "episodes.csv", index=False)
    print(f"wrote {len(df)} episode rows to {OUT_DIR / 'episodes.csv'}", flush=True)

    n_collisions = int(
        df.get("defender_static_obstacle_collision", pd.Series(dtype=int)).sum()
        + df.get("defender_dynamic_obstacle_collision", pd.Series(dtype=int)).sum()
        + df.get("enemy_static_obstacle_collision", pd.Series(dtype=int)).sum()
        + df.get("enemy_dynamic_obstacle_collision", pd.Series(dtype=int)).sum()
    )
    if n_collisions != 0:
        raise RuntimeError(
            f"CONFIGURATION ERROR: {n_collisions} obstacle-collision outcomes occurred in canonical C0 "
            "(expected exactly zero -- obstacles must be fully disabled). STOP and diagnose."
        )
    print("Obstacle-collision sanity check passed: 0 collisions (as required for C0).", flush=True)
