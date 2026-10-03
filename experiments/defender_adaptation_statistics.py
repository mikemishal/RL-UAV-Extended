"""
Statistical analysis for the defender-adaptation held-out evaluation
(journal extension, 38-phase defender-adaptation-EVALUATION sub-task,
Phases 18-33). Operates ONLY on the completed `episodes.csv` produced by
`experiments/defender_adaptation_evaluation.py` -- no new episodes are
ever run here.

PRIMARY ENDPOINT (Phase 18): safe interception against A1-unseen-203.
    Delta_adapt = Success(D1) - Success(D0+)
This isolates adversarial-opponent adaptation from equal-budget continued
ordinary training. `D1 - D0` is explicitly NOT the primary comparison
(Phase 18/19 distinction; see Delta_total below, reported as secondary).

All paired comparisons reuse the repository's validated bootstrap
functions from `experiments/final_stats.py` UNCHANGED:
    - `paired_bootstrap_ci`: per-defender-seed paired bootstrap (fixed
      single checkpoint, matched held-out episode seeds).
    - `hierarchical_paired_bootstrap_matched_seeds`: architecture-level
      estimate, resampling BOTH the 3 defender-training-seed axis AND the
      100 matched-episode axis (Phase 26), since the 3 architectures each
      have exactly 3 independently-trained defender seeds with matched
      labels across D0/D0+/D1 (seed 71 D0+ and seed 71 D1 share the SAME
      training-seed label and the SAME 100 held-out episode seeds).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.final_stats import paired_bootstrap_ci, hierarchical_paired_bootstrap_matched_seeds
from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS, EVAL_SEEDS, OUT_DIR

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001  # distinct date-coded seed, same convention as the repo's other BOOTSTRAP_SEED constants
CONFIDENCE_LEVEL = 0.95
NEAR_30_DEG_MARGIN = 2.0  # must match experiments/defender_adaptation_evaluation.NEAR_30_DEG_MARGIN

ATTACKER_TYPES = ("A0", "A1-seen-201", "A1-seen-202", "A1-unseen-203")
SEEDS_SORTED = sorted(EVAL_SEEDS)


def load_episodes(path: Path | None = None) -> pd.DataFrame:
    path = path if path is not None else (OUT_DIR / "episodes.csv")
    return pd.read_csv(path)


def success_matrix(df: pd.DataFrame, architecture: str, generation: str, attacker_type: str, level: str,
                    outcome_col: str = "defender_safe_intercept") -> np.ndarray:
    """Shape (3, 100): rows = the architecture's 3 defender training seeds
    (sorted ascending), columns = the 100 held-out eval seeds (sorted
    ascending, SAME order for every call -- Phase 4 common-random-numbers
    alignment)."""
    seeds = sorted(ARCHITECTURES[architecture]["seeds"])
    sub = df[(df.architecture == architecture) & (df.generation == generation)
             & (df.attacker_type == attacker_type) & (df.level == level)]
    mat = np.full((len(seeds), len(SEEDS_SORTED)), np.nan, dtype=np.float64)
    for i, seed in enumerate(seeds):
        seed_df = sub[sub.defender_seed == seed].set_index("eval_seed")
        for j, eval_seed in enumerate(SEEDS_SORTED):
            if eval_seed in seed_df.index:
                mat[i, j] = seed_df.loc[eval_seed, outcome_col]
    if np.isnan(mat).any():
        raise ValueError(f"incomplete success matrix for {architecture}/{generation}/{attacker_type}/{level}")
    return mat


def aggregate_across_levels(matrices: list[np.ndarray]) -> np.ndarray:
    """Equal-weight C1-C3 aggregate: concatenate episode columns (each
    level contributes its own 100 matched episodes; the 3 defender-seed
    rows stay aligned)."""
    return np.concatenate(matrices, axis=1)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Standard BH step-up FDR-adjusted p-values (Phase 27)."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out.tolist()


# =============================================================================
# PHASE 18-21: primary/secondary/extra-training/retention endpoints.
# =============================================================================

def _delta_report(df: pd.DataFrame, architecture: str, attacker_type: str, level: str | None,
                   generation_a: str, generation_b: str) -> dict:
    """Delta = Success(generation_a) - Success(generation_b), per
    architecture, for a single level OR the equal-weight C1-C3 aggregate
    (level=None), using the hierarchical paired bootstrap (Phase 26)."""
    if level is not None:
        mat_a = success_matrix(df, architecture, generation_a, attacker_type, level)
        mat_b = success_matrix(df, architecture, generation_b, attacker_type, level)
    else:
        mat_a = aggregate_across_levels([success_matrix(df, architecture, generation_a, attacker_type, lv) for lv in EVAL_LEVELS])
        mat_b = aggregate_across_levels([success_matrix(df, architecture, generation_b, attacker_type, lv) for lv in EVAL_LEVELS])
    observed, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
        mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
    )
    seeds = sorted(ARCHITECTURES[architecture]["seeds"])
    per_seed_paired = {}
    for i, seed in enumerate(seeds):
        obs_i, lo_i, hi_i = paired_bootstrap_ci(mat_a[i], mat_b[i], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        per_seed_paired[seed] = {"observed": obs_i, "ci_low": lo_i, "ci_high": hi_i}
    return {
        "architecture": architecture, "attacker_type": attacker_type, "level": level or "aggregate_C1C3",
        "generation_a": generation_a, "generation_b": generation_b,
        "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b)),
        "observed_diff": observed, "ci_low": lo, "ci_high": hi,
        "per_seed_diffs": {int(s): float(d) for s, d in zip(seeds, per_seed)},
        "per_seed_paired_bootstrap": per_seed_paired,
    }


def primary_endpoint_d1_vs_d0plus_unseen(df: pd.DataFrame) -> list[dict]:
    """Phase 18/23: Delta_adapt = Success(D1) - Success(D0+) vs A1-unseen-203,
    per architecture, per C1/C2/C3, and the equal-weight aggregate."""
    rows = []
    for arch in ARCHITECTURES:
        for level in list(EVAL_LEVELS) + [None]:
            rows.append(_delta_report(df, arch, "A1-unseen-203", level, "d1", "d0_plus"))
    return rows


def secondary_total_gain(df: pd.DataFrame) -> list[dict]:
    """Phase 19: Delta_total = Success(D1) - Success(D0) vs A1-unseen-203."""
    rows = []
    for arch in ARCHITECTURES:
        for level in list(EVAL_LEVELS) + [None]:
            rows.append(_delta_report(df, arch, "A1-unseen-203", level, "d1", "d0"))
    return rows


def extra_training_control(df: pd.DataFrame) -> list[dict]:
    """Phase 20: Delta_extra = Success(D0+) - Success(D0) vs A1-unseen-203
    (isolates the effect of simply training 425,984 more steps with NO
    opponent-mixture change)."""
    rows = []
    for arch in ARCHITECTURES:
        for level in list(EVAL_LEVELS) + [None]:
            rows.append(_delta_report(df, arch, "A1-unseen-203", level, "d0_plus", "d0"))
    return rows


def scripted_threat_retention(df: pd.DataFrame) -> list[dict]:
    """Phase 21: against A0, D0/D0+/D1 success rates and
    Delta_retention = Success(D1) - Success(D0+). No arbitrary
    non-inferiority margin is imposed -- the actual change and its CI are
    reported as-is."""
    rows = []
    for arch in ARCHITECTURES:
        for level in list(EVAL_LEVELS) + [None]:
            rows.append(_delta_report(df, arch, "A0", level, "d1", "d0_plus"))
    return rows


def seen_attacker_analysis(df: pd.DataFrame) -> list[dict]:
    """Phase 22: seeds 201/202 individually, plus an equal-weight
    A1-seen aggregate (concatenating the two seeds' episode columns --
    NOT double-counting a seed, each contributes its own 100 episodes)."""
    rows = []
    for arch in ARCHITECTURES:
        for attacker_type in ("A1-seen-201", "A1-seen-202"):
            for level in list(EVAL_LEVELS) + [None]:
                rows.append(_delta_report(df, arch, attacker_type, level, "d1", "d0_plus"))
        # Equal-weight A1-seen aggregate (across the two seeds AND C1-C3).
        for level in list(EVAL_LEVELS) + [None]:
            levels = [level] if level is not None else list(EVAL_LEVELS)
            mat_a = aggregate_across_levels([
                success_matrix(df, arch, "d1", at, lv) for at in ("A1-seen-201", "A1-seen-202") for lv in levels
            ])
            mat_b = aggregate_across_levels([
                success_matrix(df, arch, "d0_plus", at, lv) for at in ("A1-seen-201", "A1-seen-202") for lv in levels
            ])
            observed, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
                mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
            )
            seeds = sorted(ARCHITECTURES[arch]["seeds"])
            rows.append({
                "architecture": arch, "attacker_type": "A1-seen-aggregate", "level": level or "aggregate_C1C3",
                "generation_a": "d1", "generation_b": "d0_plus",
                "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b)),
                "observed_diff": observed, "ci_low": lo, "ci_high": hi,
                "per_seed_diffs": {int(s): float(d) for s, d in zip(seeds, per_seed)},
            })
    return rows


# =============================================================================
# PHASE 25: defender-seed-level summary (never report only one checkpoint).
# =============================================================================

def defender_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        for generation in ("d0", "d0_plus", "d1"):
            for attacker_type in ATTACKER_TYPES:
                for level in EVAL_LEVELS:
                    mat = success_matrix(df, arch, generation, attacker_type, level)
                    per_seed_means = np.nanmean(mat, axis=1)
                    seeds = sorted(ARCHITECTURES[arch]["seeds"])
                    rows.append({
                        "architecture": arch, "generation": generation, "attacker_type": attacker_type, "level": level,
                        "mean": float(per_seed_means.mean()), "sd": float(per_seed_means.std(ddof=1)),
                        "min": float(per_seed_means.min()), "max": float(per_seed_means.max()),
                        **{f"seed_{s}": float(v) for s, v in zip(seeds, per_seed_means)},
                    })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE 28-29: balanced/worst-case robustness.
# =============================================================================

def robustness_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        for generation in ("d0", "d0_plus", "d1"):
            for level in EVAL_LEVELS:
                a0 = float(np.nanmean(success_matrix(df, arch, generation, "A0", level)))
                a1_201 = float(np.nanmean(success_matrix(df, arch, generation, "A1-seen-201", level)))
                a1_202 = float(np.nanmean(success_matrix(df, arch, generation, "A1-seen-202", level)))
                a1_seen_mean = (a1_201 + a1_202) / 2.0
                a1_unseen = float(np.nanmean(success_matrix(df, arch, generation, "A1-unseen-203", level)))
                balanced = (a0 + a1_seen_mean + a1_unseen) / 3.0
                worst_case = min(a0, a1_seen_mean, a1_unseen)
                rows.append({
                    "architecture": arch, "generation": generation, "level": level,
                    "A0": a0, "A1_seen_mean": a1_seen_mean, "A1_unseen": a1_unseen,
                    "BalancedRobustness": balanced, "WorstCase": worst_case,
                })
    df_rows = pd.DataFrame(rows)
    agg_rows = []
    for arch in ARCHITECTURES:
        for generation in ("d0", "d0_plus", "d1"):
            sub = df_rows[(df_rows.architecture == arch) & (df_rows.generation == generation)]
            agg_rows.append({
                "architecture": arch, "generation": generation, "level": "aggregate_C1C3",
                "A0": sub["A0"].mean(), "A1_seen_mean": sub["A1_seen_mean"].mean(), "A1_unseen": sub["A1_unseen"].mean(),
                "BalancedRobustness": sub["BalancedRobustness"].mean(), "WorstCase": sub["WorstCase"].mean(),
            })
    return pd.concat([df_rows, pd.DataFrame(agg_rows)], ignore_index=True)


# =============================================================================
# PHASE 30: safety-layer dependence.
# =============================================================================

def safety_layer_dependence(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["architecture", "generation", "attacker_type"]).agg(
        n=("eval_seed", "count"),
        mean_intervention_fraction=("defender_safety_intervention_fraction", "mean"),
        mean_override_angle_deg=("mean_override_angle_deg", "mean"),
        p95_override_angle_deg=("p95_override_angle_deg", lambda x: float(np.percentile(x, 95))),
        max_override_angle_deg=("max_override_angle_deg", "max"),
    ).reset_index()


# =============================================================================
# PHASE 17 (publication sub-task): interception/path efficiency, SUCCESSFUL
# interceptions only (defender_safe_intercept == 1) -- mission success
# (safe-interception RATE) is reported separately (see success_matrix /
# the primary-endpoint functions above); this function characterizes the
# COST of achieving that success, not whether it was achieved.
# =============================================================================

def interception_efficiency_summary(df: pd.DataFrame) -> pd.DataFrame:
    successful = df[df.defender_safe_intercept == 1]
    return successful.groupby(["architecture", "generation", "attacker_type"]).agg(
        n_successful=("eval_seed", "count"),
        mean_interception_time=("interception_time", "mean"),
        median_interception_time=("interception_time", "median"),
        mean_episode_length=("episode_length", "mean"),
        mean_defender_path_length=("defender_path_length", "mean"),
        mean_min_defender_attacker_distance=("min_defender_attacker_distance", "mean"),
    ).reset_index()


# =============================================================================
# PHASE 31: LR-PPO residual-angle structural analysis.
# =============================================================================

def lr_ppo_residual_analysis(df: pd.DataFrame, by_level: bool = False) -> pd.DataFrame:
    lrppo = df[df.architecture == "lrppo_clutter_v1"].copy()
    if lrppo.empty or "mean_residual_angle_deg" not in lrppo.columns:
        return pd.DataFrame()
    rows = []
    levels = list(EVAL_LEVELS) if by_level else [None]
    for generation in ("d0", "d0_plus", "d1"):
        for attacker_type in ATTACKER_TYPES:
            for level in levels:
                sub = lrppo[(lrppo.generation == generation) & (lrppo.attacker_type == attacker_type)]
                if level is not None:
                    sub = sub[sub.level == level]
                if sub.empty:
                    continue
                all_means = sub["mean_residual_angle_deg"].dropna()
                rows.append({
                    "generation": generation, "attacker_type": attacker_type,
                    "level": level or "aggregate_C1C3", "n_episodes": len(sub),
                    "mean_residual_angle_deg": float(all_means.mean()),
                    "median_residual_angle_deg": float(all_means.median()),
                    "p95_residual_angle_deg": float(np.percentile(sub["p95_residual_angle_deg"].dropna(), 95)) if sub["p95_residual_angle_deg"].notna().any() else float("nan"),
                    "max_residual_angle_deg": float(sub["max_residual_angle_deg"].max()),
                    "mean_fraction_near_30deg": float(sub["fraction_near_30deg"].mean()),
                    "near_30deg_margin_deg": NEAR_30_DEG_MARGIN,
                })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE 32: outcome decomposition.
# =============================================================================

OUTCOME_COLUMNS = (
    "defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
)


def outcome_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["architecture", "generation", "attacker_type", "level"])[list(OUTCOME_COLUMNS)].mean().reset_index()


# =============================================================================
# PHASE 27: pre-specified comparison family + BH correction for exploratory cells.
# =============================================================================

def run_full_statistical_analysis(df: pd.DataFrame) -> dict:
    primary = primary_endpoint_d1_vs_d0plus_unseen(df)
    secondary_total = secondary_total_gain(df)
    extra_training = extra_training_control(df)
    retention = scripted_threat_retention(df)
    seen = seen_attacker_analysis(df)

    # Exploratory cell-level p-values (approximate, from the CI's
    # bootstrap distribution: p = 2*min(frac(diffs<=0), frac(diffs>=0))
    # is NOT reconstructed here from CI alone -- use a direct binomial
    # McNemar-style approximation per cell for the BH-correction family
    # instead, which IS exact and available.
    from experiments.final_stats import mcnemar_exact
    exploratory_pvalues = []
    exploratory_labels = []
    for arch in ARCHITECTURES:
        for generation_a, generation_b in (("d1", "d0_plus"), ("d1", "d0"), ("d0_plus", "d0")):
            for attacker_type in ATTACKER_TYPES:
                for level in EVAL_LEVELS:
                    mat_a = success_matrix(df, arch, generation_a, attacker_type, level)
                    mat_b = success_matrix(df, arch, generation_b, attacker_type, level)
                    n01 = int(np.sum((mat_a == 0) & (mat_b == 1)))
                    n10 = int(np.sum((mat_a == 1) & (mat_b == 0)))
                    p = mcnemar_exact(n01, n10)
                    exploratory_pvalues.append(p)
                    exploratory_labels.append(f"{arch}|{generation_a}-{generation_b}|{attacker_type}|{level}")
    adjusted = benjamini_hochberg(exploratory_pvalues)
    exploratory_table = [
        {"label": label, "p_value": p, "p_value_bh_adjusted": adj}
        for label, p, adj in zip(exploratory_labels, exploratory_pvalues, adjusted)
    ]

    return {
        "primary_d1_vs_d0plus_unseen": primary,
        "secondary_d1_vs_d0_unseen": secondary_total,
        "extra_training_d0plus_vs_d0_unseen": extra_training,
        "scripted_retention_d1_vs_d0plus": retention,
        "seen_attacker_d1_vs_d0plus": seen,
        "exploratory_cellwise_mcnemar_bh": exploratory_table,
    }
