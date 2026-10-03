"""
Phase 3: historical reproduction check for the accepted-paper nominal
evaluation (journal "accepted-paper reproduction" sub-task).

Does NOT modify or re-run `experiments/run_final_nominal_evaluation.py`
(the frozen, locked script) -- that script's own `EnvConfig()` bare-default
call has silently drifted (see `docs/original_paper_protocol_audit.md`
Section 0: `defender_standby_until_detection` now defaults to True, but did
not exist when `results/final_nominal/` was generated). This script:

  1. Re-hashes the 6 frozen `training_400k` PPO checkpoints referenced by
     `results/final_nominal/model_hashes.json` / `experiment_manifest.json`
     -- confirms the exact model weights are unchanged since the accepted
     paper's results were generated.
  2. Re-runs a SMALL sample (`N_SAMPLE=50` episodes, seeds 20000-20049 --
     already-opened final-test seeds, a strict subset of the original
     20000-24999 range actually evaluated) for all 12 policy instances,
     TWICE: once under the CORRECTED config
     (`defender_standby_until_detection=False`, the true original
     semantics) and once under TODAY'S drifted bare `EnvConfig()`
     default (`defender_standby_until_detection=True`), to make the drift's
     behavioral effect concrete rather than asserted.
  3. Compares both fresh samples, per episode AND in aggregate, against the
     EXISTING `results/final_nominal/raw_episode_results.csv` rows for the
     SAME (policy_instance, eval_seed) pairs (exact seeds, common random
     numbers -- an exact-match check is possible, not just a statistical
     one).

No training or model modification happens anywhere in this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import POLICY_INSTANCES, FINAL_NOMINAL_OUTPUT_ROOT
from experiments.final_eval_utils import run_final_episode, sha256_of_file

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "paper_reproduction_check"

N_SAMPLE = 50
SAMPLE_SEEDS = tuple(range(20000, 20000 + N_SAMPLE))
DT = 0.5  # matches EnvConfig default dt, recorded in the original manifest


def check_checkpoint_hashes() -> list[dict]:
    model_hashes = json.loads((FINAL_NOMINAL_OUTPUT_ROOT / "model_hashes.json").read_text())
    rows = []
    for entry in model_hashes:
        path = PROJECT_ROOT / entry["model_path"]
        actual = sha256_of_file(path)
        rows.append({
            "policy_instance": entry["policy_instance"],
            "model_path": entry["model_path"],
            "recorded_sha256": entry["sha256"],
            "actual_sha256": actual,
            "match": actual == entry["sha256"],
        })
    return rows


def _build_policy(instance):
    if instance.model_path is not None:
        return get_policy(instance.policy_name, model_path=str(instance.model_path), deterministic=True)
    return get_policy(instance.policy_name)


def run_sample(standby: bool) -> pd.DataFrame:
    rows = []
    for instance in POLICY_INSTANCES:
        cfg = EnvConfig(use_kalman_tracking=(instance.estimator == "kalman"), defender_standby_until_detection=standby)
        env = SoldierEnv(config=cfg)
        policy = _build_policy(instance)
        for seed in SAMPLE_SEEDS:
            row = run_final_episode(env, policy, seed, DT)
            row.update(paper_method=instance.paper_method, policy_instance=instance.policy_instance,
                       estimator=instance.estimator, training_seed=instance.training_seed, standby=standby)
            rows.append(row)
    return pd.DataFrame(rows)


def compare_against_original(fresh: pd.DataFrame, original: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = fresh.merge(
        original[["policy_instance", "eval_seed", "outcome", "success"]],
        on=["policy_instance", "eval_seed"], suffixes=("_fresh", "_original"),
    )
    merged["outcome_match"] = merged["outcome_fresh"] == merged["outcome_original"]
    summary = merged.groupby("policy_instance").agg(
        n=("outcome_match", "size"),
        n_matching=("outcome_match", "sum"),
        fresh_success_rate=("success_fresh", "mean"),
        original_success_rate=("success_original", "mean"),
    ).reset_index()
    summary["exact_match_rate"] = summary["n_matching"] / summary["n"]
    summary["config"] = label
    return summary


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: checkpoint hash check ===", flush=True)
    hash_rows = check_checkpoint_hashes()
    for r in hash_rows:
        print(f"  {r['policy_instance']:>16s}: match={r['match']}", flush=True)
    if not all(r["match"] for r in hash_rows):
        raise RuntimeError("Checkpoint hash mismatch detected -- STOP, do not proceed with reproduction claims.")
    pd.DataFrame(hash_rows).to_csv(OUT_DIR / "checkpoint_hash_check.csv", index=False)

    print("=== Step 2: load original recorded episode results (seeds 20000-20049 subset) ===", flush=True)
    original = pd.read_csv(FINAL_NOMINAL_OUTPUT_ROOT / "raw_episode_results.csv")
    original_sample = original[original["eval_seed"].isin(SAMPLE_SEEDS)].copy()
    assert len(original_sample) == 12 * N_SAMPLE, f"expected {12*N_SAMPLE} original rows, got {len(original_sample)}"

    print("=== Step 3: fresh sample WITH CORRECTED config (standby=False, true original semantics) ===", flush=True)
    fresh_corrected = run_sample(standby=False)
    fresh_corrected.to_csv(OUT_DIR / "fresh_sample_standby_false.csv", index=False)
    summary_corrected = compare_against_original(fresh_corrected, original_sample, "standby=False (corrected)")

    print("=== Step 4: fresh sample WITH TODAY'S DRIFTED bare-default config (standby=True) ===", flush=True)
    fresh_drifted = run_sample(standby=True)
    fresh_drifted.to_csv(OUT_DIR / "fresh_sample_standby_true.csv", index=False)
    summary_drifted = compare_against_original(fresh_drifted, original_sample, "standby=True (drifted/unmodified script)")

    combined = pd.concat([summary_corrected, summary_drifted], ignore_index=True)
    combined.to_csv(OUT_DIR / "reproduction_check_summary.csv", index=False)
    print(combined.to_string(index=False), flush=True)

    overall = combined.groupby("config").agg(
        mean_exact_match_rate=("exact_match_rate", "mean"),
        min_exact_match_rate=("exact_match_rate", "min"),
    ).reset_index()
    print("=== overall summary ===", flush=True)
    print(overall.to_string(index=False), flush=True)
    overall.to_csv(OUT_DIR / "reproduction_check_overall.csv", index=False)
