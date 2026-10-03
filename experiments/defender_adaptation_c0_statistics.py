"""
Statistics for the no-obstacle (C0) retention evaluation (journal
"accepted-paper reproduction" sub-task, Phases 4/7/8/9/10/11).

Operates ONLY on the completed `episodes.csv` produced by
`experiments/defender_adaptation_c0_evaluation.py` (new, no-obstacle data)
and, for Phase 9's 4-condition decomposition, the EXISTING
`results/defender_adaptation_evaluation/episodes.csv` (C1-C3 clutter data
from the prior 32,400-episode journal-extension sub-task) -- no new
episodes are ever run here. Reuses `experiments/final_stats.py`'s
hierarchical paired bootstrap UNCHANGED, exactly as
`experiments/defender_adaptation_statistics.py` does for the C1-C3 data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.final_stats import paired_bootstrap_ci, hierarchical_paired_bootstrap_matched_seeds
from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS as C1C3_LEVELS, OUT_DIR as C1C3_OUT_DIR
from experiments.defender_adaptation_c0_evaluation import OUT_DIR as C0_OUT_DIR, EVAL_SEEDS as C0_EVAL_SEEDS

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001  # same convention/value as defender_adaptation_statistics.py (distinct RNG namespace, not an episode seed)
CONFIDENCE_LEVEL = 0.95
NEAR_30_DEG_MARGIN = 2.0  # must match defender_adaptation_evaluation.NEAR_30_DEG_MARGIN

ATTACKER_TYPES = ("A0", "A1-seen-201", "A1-seen-202", "A1-unseen-203")
C0_SEEDS_SORTED = sorted(C0_EVAL_SEEDS)


def load_c0_episodes() -> pd.DataFrame:
    return pd.read_csv(C0_OUT_DIR / "episodes.csv")


def load_c1c3_episodes() -> pd.DataFrame:
    return pd.read_csv(C1C3_OUT_DIR / "episodes.csv")


def _success_matrix(df: pd.DataFrame, architecture: str, generation: str, attacker_type: str,
                     level: str, seeds_sorted: list[int], outcome_col: str = "defender_safe_intercept") -> np.ndarray:
    seeds = sorted(ARCHITECTURES[architecture]["seeds"])
    sub = df[(df.architecture == architecture) & (df.generation == generation)
             & (df.attacker_type == attacker_type) & (df.level == level)]
    mat = np.full((len(seeds), len(seeds_sorted)), np.nan, dtype=np.float64)
    for i, seed in enumerate(seeds):
        seed_df = sub[sub.defender_seed == seed].set_index("eval_seed")
        for j, eval_seed in enumerate(seeds_sorted):
            if eval_seed in seed_df.index:
                mat[i, j] = seed_df.loc[eval_seed, outcome_col]
    if np.isnan(mat).any():
        raise ValueError(f"incomplete C0 success matrix for {architecture}/{generation}/{attacker_type}/{level}")
    return mat


def c0_success_matrix(df: pd.DataFrame, architecture: str, generation: str, attacker_type: str) -> np.ndarray:
    return _success_matrix(df, architecture, generation, attacker_type, "C0", C0_SEEDS_SORTED)


def _delta_report(df: pd.DataFrame, architecture: str, attacker_type: str, generation_a: str, generation_b: str) -> dict:
    mat_a = c0_success_matrix(df, architecture, generation_a, attacker_type)
    mat_b = c0_success_matrix(df, architecture, generation_b, attacker_type)
    observed, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
        mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
    )
    seeds = sorted(ARCHITECTURES[architecture]["seeds"])
    per_seed_paired = {}
    for i, seed in enumerate(seeds):
        obs_i, lo_i, hi_i = paired_bootstrap_ci(mat_a[i], mat_b[i], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        per_seed_paired[seed] = {"observed": obs_i, "ci_low": lo_i, "ci_high": hi_i}
    return {
        "architecture": architecture, "attacker_type": attacker_type, "level": "C0",
        "generation_a": generation_a, "generation_b": generation_b,
        "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b)),
        "observed_diff": observed, "ci_low": lo, "ci_high": hi,
        "per_seed_diffs": {int(s): float(d) for s, d in zip(seeds, per_seed)},
        "per_seed_paired_bootstrap": per_seed_paired,
    }


# =============================================================================
# Phase 4/7: C0 nominal retention vs A0-paper (the paper's own threat model).
# =============================================================================

def c0_scripted_retention(df: pd.DataFrame) -> list[dict]:
    """Against A0 (the original scripted hostile), D0 vs D0+ vs D1 success
    rates under C0 -- does removing obstacles reveal any retention loss
    from clutter training/adversarial continuation relative to each
    architecture's own D0 (clutter-trained) baseline?"""
    rows = []
    for arch in ARCHITECTURES:
        rows.append(_delta_report(df, arch, "A0", "d0_plus", "d0"))
        rows.append(_delta_report(df, arch, "A0", "d1", "d0"))
        rows.append(_delta_report(df, arch, "A0", "d1", "d0_plus"))
    return rows


# =============================================================================
# Phase 8: A1 extension in C0 (does adversarial-adaptation's A1-unseen gain
# -- found in C1-C3 -- persist in a no-obstacle environment?)
# =============================================================================

def c0_primary_endpoint_d1_vs_d0plus_unseen(df: pd.DataFrame) -> list[dict]:
    """The SAME primary endpoint as defender_adaptation_statistics.py's
    `primary_endpoint_d1_vs_d0plus_unseen` (Delta_adapt = D1 - D0+ vs
    A1-unseen-203), computed under C0 instead of C1-C3."""
    return [_delta_report(df, arch, "A1-unseen-203", "d1", "d0_plus") for arch in ARCHITECTURES]


def c0_seen_attacker_analysis(df: pd.DataFrame) -> list[dict]:
    rows = []
    for arch in ARCHITECTURES:
        for attacker_type in ("A1-seen-201", "A1-seen-202"):
            rows.append(_delta_report(df, arch, attacker_type, "d1", "d0_plus"))
    return rows


# =============================================================================
# Phase 9: 4-condition decomposition (A0+no-clutter / A0+clutter /
# A1+no-clutter / A1+clutter), never collapsed into one effect. "Clutter"
# here is the equal-weight C1-C3 aggregate (concatenating each level's own
# 100 matched episodes), matching defender_adaptation_statistics.py's
# `aggregate_across_levels` convention exactly.
# =============================================================================

def four_condition_decomposition(df_c0: pd.DataFrame, df_c1c3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        for generation in ("d0", "d0_plus", "d1"):
            no_clutter_a0 = float(np.nanmean(c0_success_matrix(df_c0, arch, generation, "A0")))
            clutter_a0 = float(np.nanmean(np.concatenate(
                [_success_matrix(df_c1c3, arch, generation, "A0", lv, sorted(set(df_c1c3.eval_seed))) for lv in C1C3_LEVELS],
                axis=1,
            )))
            no_clutter_a1 = float(np.nanmean(c0_success_matrix(df_c0, arch, generation, "A1-unseen-203")))
            clutter_a1 = float(np.nanmean(np.concatenate(
                [_success_matrix(df_c1c3, arch, generation, "A1-unseen-203", lv, sorted(set(df_c1c3.eval_seed))) for lv in C1C3_LEVELS],
                axis=1,
            )))
            rows.append({
                "architecture": arch, "generation": generation,
                "A0_no_clutter_C0": no_clutter_a0, "A0_clutter_C1C3": clutter_a0,
                "A1_unseen_no_clutter_C0": no_clutter_a1, "A1_unseen_clutter_C1C3": clutter_a1,
                "clutter_effect_under_A0": clutter_a0 - no_clutter_a0,
                "clutter_effect_under_A1_unseen": clutter_a1 - no_clutter_a1,
                "attacker_effect_under_no_clutter": no_clutter_a1 - no_clutter_a0,
                "attacker_effect_under_clutter": clutter_a1 - clutter_a0,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Phase 10: LR-PPO residual-angle usage in C0.
# =============================================================================

def lrppo_residual_c0_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df.architecture == "lrppo_clutter_v1") & (df.level == "C0")]
    rows = []
    for generation in ("d0", "d0_plus", "d1"):
        for attacker_type in ATTACKER_TYPES:
            cell = sub[(sub.generation == generation) & (sub.attacker_type == attacker_type)]
            rows.append({
                "generation": generation, "attacker_type": attacker_type,
                "mean_residual_angle_deg": float(cell["mean_residual_angle_deg"].mean()),
                "p95_residual_angle_deg": float(cell["p95_residual_angle_deg"].mean()),
                "fraction_near_30deg": float(cell["fraction_near_30deg"].mean()),
                "n_episodes": int(len(cell)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT = C0_OUT_DIR
    df_c0 = load_c0_episodes()
    df_c1c3 = load_c1c3_episodes()

    retention = c0_scripted_retention(df_c0)
    (OUT / "c0_scripted_retention.json").write_text(json.dumps(retention, indent=2, default=str))

    primary = c0_primary_endpoint_d1_vs_d0plus_unseen(df_c0)
    (OUT / "c0_primary_endpoint_unseen.json").write_text(json.dumps(primary, indent=2, default=str))

    seen = c0_seen_attacker_analysis(df_c0)
    (OUT / "c0_seen_attacker_analysis.json").write_text(json.dumps(seen, indent=2, default=str))

    decomposition = four_condition_decomposition(df_c0, df_c1c3)
    decomposition.to_csv(OUT / "four_condition_decomposition.csv", index=False)

    lrppo_summary = lrppo_residual_c0_summary(df_c0)
    lrppo_summary.to_csv(OUT / "lrppo_residual_c0_summary.csv", index=False)

    print("=== C0 scripted (A0) retention: D1 vs D0 ===")
    for r in retention:
        if r["generation_a"] == "d1" and r["generation_b"] == "d0":
            print(f"  {r['architecture']:>22s}: D0={r['success_b_mean']:.3f} D1={r['success_a_mean']:.3f} "
                  f"diff={r['observed_diff']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")

    print("=== C0 primary endpoint (D1-D0+) vs A1-unseen-203 ===")
    for r in primary:
        print(f"  {r['architecture']:>22s}: D0+={r['success_b_mean']:.3f} D1={r['success_a_mean']:.3f} "
              f"diff={r['observed_diff']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")

    print("=== 4-condition decomposition ===")
    print(decomposition.to_string(index=False))

    print("=== LR-PPO residual-angle usage in C0 ===")
    print(lrppo_summary.to_string(index=False))
