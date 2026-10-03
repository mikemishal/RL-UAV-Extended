"""
No-obstacle (C0) retention evaluation for the D0/D0+/D1 defender-adaptation
checkpoints (journal "accepted-paper reproduction" sub-task, Phases 2/4/7/8/9).

PURPOSE: the accepted paper characterized defender performance in an
open-air (no-obstacle) environment. Clutter training (D0) and adversarial
continuation (D0+/D1) happened entirely under C1-C3 obstacle density. This
module answers: does the ORIGINAL no-obstacle operating envelope still
hold for these journal-era checkpoints, against BOTH the original scripted
hostile (A0) and the 3 Adaptive Attacker v1 conditions (A1-seen-201,
A1-seen-202, A1-unseen-203)?

Reuses, UNCHANGED, every piece of validated infrastructure from
`experiments/defender_adaptation_evaluation.py`: the frozen model registry,
`DefenderInstance`/`AttackerCondition` construction, the common-random-
numbers matched-episode runner (`_run_matched_episode`), and the
resumable, integrity-checked JSONL persistence layer. ONLY the swept
clutter level (C0 instead of C1-C3) and the held-out eval-seed namespace
differ -- both are new, disjoint constants, never substituted into the
original module's own namespace.

Seed namespace: `docs/original_paper_protocol_audit.md` Section 4 audits
every existing seed range in this repository and finds the next free
10,000-aligned block begins at 100,000 (clear of the defender-adaptation
evaluation's 90,010,000-90,010,099 and the attacker protocol's
180,000-199,999). This module reserves C0_EVAL_SEED_START=100_000,
N_EVAL_SEEDS=100 (100,000-100,099) -- the SAME N as the C1-C3 evaluation,
for direct comparability, just a disjoint seed range (never reused).

No training or model modification happens anywhere in this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
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
    is_valid_episode_row,
    load_completed_episodes,
    append_episode_result,
    EPISODE_KEY_FIELDS,
)

OUT_DIR = PROJECT_ROOT / "results" / "defender_adaptation_c0_evaluation"

C0_LEVEL = "C0"
C0_EVAL_SEED_START = 100_000
N_EVAL_SEEDS = 100
EVAL_SEEDS: tuple[int, ...] = tuple(range(C0_EVAL_SEED_START, C0_EVAL_SEED_START + N_EVAL_SEEDS))


def build_evaluation_manifest(registry: dict | None = None) -> dict:
    """Mirrors `defender_adaptation_evaluation.build_evaluation_manifest`
    exactly, with EVAL_LEVELS fixed to ("C0",) and the seed namespace above.
    Expected total: 27 defenders x 4 attacker conditions x 100 eval seeds
    = 10,800 keys."""
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

    def _get_config_hash(estimator: str) -> str:
        if estimator not in config_hash_cache:
            config_hash_cache[estimator] = _config_hash(build_canonical_env_config(C0_LEVEL, estimator))
        return config_hash_cache[estimator]

    keys = []
    for d in defenders:
        for attacker_meta in attacker_conditions_meta:
            for eval_seed in EVAL_SEEDS:
                keys.append({
                    "key": "|".join(str(x) for x in (
                        d.architecture, d.defender_seed, d.generation, d.checkpoint_role,
                        attacker_meta["name"], attacker_meta["seed"], C0_LEVEL, eval_seed,
                    )),
                    "architecture": d.architecture, "defender_seed": d.defender_seed,
                    "generation": d.generation, "checkpoint_role": d.checkpoint_role,
                    "defender_checkpoint_path": d.checkpoint_path, "defender_checkpoint_sha256": d.checkpoint_sha256,
                    "attacker_type": attacker_meta["name"], "attacker_seed": attacker_meta["seed"],
                    "attacker_checkpoint_path": attacker_meta["checkpoint_path"],
                    "attacker_checkpoint_sha256": attacker_meta["checkpoint_sha256"],
                    "level": C0_LEVEL, "eval_seed": eval_seed,
                    "clutter_config_hash": _get_config_hash(d.estimator),
                    "awareness_mode": "map_local (dormant: zero obstacles at C0)",
                    "observation_version": "36-D (PPO/PPO-Kalman)" if not d.is_lr_ppo else "39-D (LR-PPO)",
                })
    manifest = {
        "total_expected_keys": len(keys),
        "n_defenders": len(defenders),
        "n_attacker_conditions": len(attacker_conditions_meta),
        "n_levels": 1,
        "level": C0_LEVEL,
        "n_eval_seeds": len(EVAL_SEEDS),
        "eval_seed_range": [C0_EVAL_SEED_START, C0_EVAL_SEED_START + N_EVAL_SEEDS - 1],
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
        base_config = build_canonical_env_config(C0_LEVEL, d.estimator)
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
                key = _episode_key(d, attacker, C0_LEVEL, eval_seed)
                if key in completed:
                    n_skipped += 1
                    continue
                metrics = _run_matched_episode(env, d, defender_policy, lead_policy_for_residual, attacker.model, eval_seed)
                metrics.update(
                    architecture=d.architecture, defender_seed=d.defender_seed, generation=d.generation,
                    checkpoint_role=d.checkpoint_role, attacker_type=attacker.name, attacker_seed=attacker.seed,
                    level=C0_LEVEL, eval_seed=eval_seed, eval_code_commit=eval_code_commit,
                )
                append_episode_result(jsonl_path, metrics, completed)
                rows.append(metrics)
                n_executed += 1
                n_new += 1
            if n_new:
                print(f"{d.architecture:>22s} seed={d.defender_seed} {d.generation:<8s} level=C0 "
                      f"attacker={attacker.name:<16s} (+{n_new} new, {time.time()-t0:.1f}s)", flush=True)
    print(f"=== C0 evaluation loop done: {n_executed} executed, {n_skipped} skipped (already complete) ===", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-registry", action="store_true")
    args = parser.parse_args()

    if args.rebuild_registry or not MODEL_REGISTRY_PATH.exists():
        print("Building model registry (shared with C1-C3 evaluation)...", flush=True)
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
