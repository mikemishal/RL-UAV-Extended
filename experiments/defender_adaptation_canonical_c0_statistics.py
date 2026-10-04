"""
Statistics for the CANONICAL no-obstacle (C0, defender_standby_until_detection
=True) LR-PPO-semantics evaluation (journal "first publication evaluation"
sub-task). Operates ONLY on the completed `episodes.csv` produced by
`experiments/defender_adaptation_canonical_c0_evaluation.py` and, for the
C0-vs-clutter comparison (Phase 9) and four-condition decomposition
(Phase 13), the EXISTING frozen `results/defender_adaptation_evaluation/episodes.csv`
(32,400-episode C1-C3 dataset) -- no new episodes are ever run here, and the
legacy C0 dataset (`results/defender_adaptation_c0_evaluation/`, tag
`journal-paper-reproduction-v1`) is read ONLY for the explicit legacy-vs-
canonical comparison in Phase 11, never as a substitute for the canonical
data itself.

Reuses `experiments/final_stats.py`'s hierarchical paired bootstrap
UNCHANGED, exactly as `experiments/defender_adaptation_statistics.py` does
for the C1-C3 data (10,000 replicates, resample training seeds then
matched eval seeds).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.final_stats import paired_bootstrap_ci, hierarchical_paired_bootstrap_matched_seeds
from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS as C1C3_LEVELS, OUT_DIR as C1C3_OUT_DIR
from experiments.defender_adaptation_canonical_c0_evaluation import OUT_DIR as C0_OUT_DIR, EVAL_SEEDS as C0_EVAL_SEEDS

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001  # same convention/value used throughout this sub-task family
CONFIDENCE_LEVEL = 0.95
NEAR_30_DEG_MARGIN = 2.0  # must match defender_adaptation_evaluation.NEAR_30_DEG_MARGIN

ATTACKER_TYPES = ("A0", "A1-seen-201", "A1-seen-202", "A1-unseen-203")
C0_SEEDS_SORTED = sorted(C0_EVAL_SEEDS)
ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}
GENERATIONS = ("d0", "d0_plus", "d1")


def load_canonical_c0_episodes() -> pd.DataFrame:
    return pd.read_csv(C0_OUT_DIR / "episodes.csv")


def load_c1c3_episodes() -> pd.DataFrame:
    return pd.read_csv(C1C3_OUT_DIR / "episodes.csv")


def load_legacy_c0_episodes() -> pd.DataFrame:
    """Phase 11 ONLY: the earlier (non-canonical-framing) C0 dataset, read
    strictly for an explicit, clearly-labeled legacy-vs-canonical comparison."""
    legacy_path = Path(C0_OUT_DIR).parent / "defender_adaptation_c0_evaluation" / "episodes.csv"
    return pd.read_csv(legacy_path)


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
        raise ValueError(f"incomplete success matrix for {architecture}/{generation}/{attacker_type}/{level}")
    return mat


def c0_success_matrix(df: pd.DataFrame, architecture: str, generation: str, attacker_type: str) -> np.ndarray:
    return _success_matrix(df, architecture, generation, attacker_type, "C0", C0_SEEDS_SORTED)


def _delta_report(df: pd.DataFrame, architecture: str, attacker_type: str, generation_a: str, generation_b: str,
                   matrix_fn=c0_success_matrix) -> dict:
    mat_a = matrix_fn(df, architecture, generation_a, attacker_type)
    mat_b = matrix_fn(df, architecture, generation_b, attacker_type)
    observed, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
        mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
    )
    seeds = sorted(ARCHITECTURES[architecture]["seeds"])
    per_seed_paired = {}
    for i, seed in enumerate(seeds):
        obs_i, lo_i, hi_i = paired_bootstrap_ci(mat_a[i], mat_b[i], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        per_seed_paired[seed] = {"observed": obs_i, "ci_low": lo_i, "ci_high": hi_i}
    per_seed_rates_a = mat_a.mean(axis=1)
    per_seed_rates_b = mat_b.mean(axis=1)
    return {
        "architecture": architecture, "attacker_type": attacker_type, "level": "C0_canonical",
        "generation_a": generation_a, "generation_b": generation_b,
        "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b)),
        "observed_diff": observed, "ci_low": lo, "ci_high": hi,
        "per_seed_diffs": {int(s): float(d) for s, d in zip(seeds, per_seed)},
        "per_seed_paired_bootstrap": per_seed_paired,
        "per_seed_rate_a": {int(s): float(v) for s, v in zip(seeds, per_seed_rates_a)},
        "per_seed_rate_b": {int(s): float(v) for s, v in zip(seeds, per_seed_rates_b)},
        "seed_rate_a_mean": float(per_seed_rates_a.mean()), "seed_rate_a_sd": float(per_seed_rates_a.std(ddof=1)),
        "seed_rate_a_min": float(per_seed_rates_a.min()), "seed_rate_a_max": float(per_seed_rates_a.max()),
        "seed_rate_b_mean": float(per_seed_rates_b.mean()), "seed_rate_b_sd": float(per_seed_rates_b.std(ddof=1)),
        "seed_rate_b_min": float(per_seed_rates_b.min()), "seed_rate_b_max": float(per_seed_rates_b.max()),
    }


# =============================================================================
# Phase 4: A0 canonical retention (D0 vs D0+ vs D1, catastrophic-forgetting check).
# =============================================================================

def a0_canonical_retention(df: pd.DataFrame) -> list[dict]:
    rows = []
    for arch in ARCHITECTURES:
        rows.append(_delta_report(df, arch, "A0", "d0_plus", "d0"))   # Q2: extra training alone
        rows.append(_delta_report(df, arch, "A0", "d1", "d0"))        # Q3/Q4: full adaptation vs original D0
        rows.append(_delta_report(df, arch, "A0", "d1", "d0_plus"))   # Q4: adaptation vs equal-budget control
    return rows


# =============================================================================
# Phase 5/6/10/11: primary endpoint (D1-D0+ vs A1-unseen-203) + A1-seen.
# =============================================================================

def canonical_primary_endpoint_unseen(df: pd.DataFrame) -> list[dict]:
    return [_delta_report(df, arch, "A1-unseen-203", "d1", "d0_plus") for arch in ARCHITECTURES]


def canonical_seen_attacker_analysis(df: pd.DataFrame) -> list[dict]:
    rows = []
    for arch in ARCHITECTURES:
        for attacker_type in ("A1-seen-201", "A1-seen-202"):
            rows.append(_delta_report(df, arch, attacker_type, "d1", "d0_plus"))
        mat_a = np.concatenate([c0_success_matrix(df, arch, "d1", at) for at in ("A1-seen-201", "A1-seen-202")], axis=1)
        mat_b = np.concatenate([c0_success_matrix(df, arch, "d0_plus", at) for at in ("A1-seen-201", "A1-seen-202")], axis=1)
        observed, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
            mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
        )
        seeds = sorted(ARCHITECTURES[arch]["seeds"])
        rows.append({
            "architecture": arch, "attacker_type": "A1-seen-mean", "level": "C0_canonical",
            "generation_a": "d1", "generation_b": "d0_plus",
            "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b)),
            "observed_diff": observed, "ci_low": lo, "ci_high": hi,
            "per_seed_diffs": {int(s): float(d) for s, d in zip(seeds, per_seed)},
        })
    return rows


# =============================================================================
# Phase 9: canonical C0 vs frozen C1/C2/C3 clutter comparison.
# =============================================================================

def c0_vs_clutter_table(df_c0: pd.DataFrame, df_c1c3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        for attacker_type in ATTACKER_TYPES:
            for generation in GENERATIONS:
                row = {"architecture": ARCH_LABELS[arch], "attacker_type": attacker_type, "generation": generation}
                row["C0"] = float(np.nanmean(c0_success_matrix(df_c0, arch, generation, attacker_type)))
                for level in C1C3_LEVELS:
                    row[level] = float(np.nanmean(_success_matrix(df_c1c3, arch, generation, attacker_type, level, sorted(set(df_c1c3.eval_seed)))))
                rows.append(row)
    return pd.DataFrame(rows)


def d1_minus_d0plus_by_condition(df_c0: pd.DataFrame, df_c1c3: pd.DataFrame) -> pd.DataFrame:
    """D1-D0+ vs A1-unseen-203, computed identically for C0(canonical)/C1/C2/C3."""
    rows = []
    for arch in ARCHITECTURES:
        c0_result = _delta_report(df_c0, arch, "A1-unseen-203", "d1", "d0_plus")
        rows.append({"architecture": ARCH_LABELS[arch], "condition": "C0_canonical",
                      "diff": c0_result["observed_diff"], "ci_low": c0_result["ci_low"], "ci_high": c0_result["ci_high"]})
        for level in C1C3_LEVELS:
            mat_a = _success_matrix(df_c1c3, arch, "d1", "A1-unseen-203", level, sorted(set(df_c1c3.eval_seed)))
            mat_b = _success_matrix(df_c1c3, arch, "d0_plus", "A1-unseen-203", level, sorted(set(df_c1c3.eval_seed)))
            observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"architecture": ARCH_LABELS[arch], "condition": level, "diff": observed, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# =============================================================================
# Phase 11: explicit legacy-vs-canonical comparison (PPO-C question).
# =============================================================================

def legacy_vs_canonical_ppo_c(df_canonical: pd.DataFrame, df_legacy: pd.DataFrame) -> dict:
    canonical = _delta_report(df_canonical, "ppo_clutter_v1", "A1-unseen-203", "d1", "d0_plus")

    def _legacy_matrix(architecture, generation, attacker_type):
        seeds = sorted(ARCHITECTURES[architecture]["seeds"])
        sub = df_legacy[(df_legacy.architecture == architecture) & (df_legacy.generation == generation)
                         & (df_legacy.attacker_type == attacker_type) & (df_legacy.level == "C0")]
        legacy_seeds_sorted = sorted(set(df_legacy.eval_seed))
        mat = np.full((len(seeds), len(legacy_seeds_sorted)), np.nan, dtype=np.float64)
        for i, seed in enumerate(seeds):
            seed_df = sub[sub.defender_seed == seed].set_index("eval_seed")
            for j, eval_seed in enumerate(legacy_seeds_sorted):
                if eval_seed in seed_df.index:
                    mat[i, j] = seed_df.loc[eval_seed, "defender_safe_intercept"]
        return mat

    mat_a = _legacy_matrix("ppo_clutter_v1", "d1", "A1-unseen-203")
    mat_b = _legacy_matrix("ppo_clutter_v1", "d0_plus", "A1-unseen-203")
    observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(mat_a, mat_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    legacy = {"observed_diff": observed, "ci_low": lo, "ci_high": hi,
              "success_a_mean": float(np.nanmean(mat_a)), "success_b_mean": float(np.nanmean(mat_b))}
    return {"canonical_NON_LEGACY": canonical, "legacy_NON_CANONICAL": legacy}


# =============================================================================
# Phase 12: LR-PPO residual-angle summary under canonical C0.
# =============================================================================

def lrppo_residual_canonical_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df.architecture == "lrppo_clutter_v1") & (df.level == "C0")]
    rows = []
    for generation in GENERATIONS:
        for attacker_type in ATTACKER_TYPES:
            cell = sub[(sub.generation == generation) & (sub.attacker_type == attacker_type)]
            rows.append({
                "generation": generation, "attacker_type": attacker_type,
                "mean_residual_angle_deg": float(cell["mean_residual_angle_deg"].mean()),
                "median_residual_angle_deg": float(cell["mean_residual_angle_deg"].median()),
                "p95_residual_angle_deg": float(cell["p95_residual_angle_deg"].mean()),
                "max_residual_angle_deg": float(cell["max_residual_angle_deg"].max()),
                "fraction_near_30deg": float(cell["fraction_near_30deg"].mean()),
                "success_rate": float(cell["defender_safe_intercept"].mean()),
                "n_episodes": int(len(cell)),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Phase 13: four-condition decomposition (A0/A1-unseen x C0(canonical)/clutter).
# =============================================================================

def four_condition_decomposition(df_c0: pd.DataFrame, df_c1c3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        for generation in GENERATIONS:
            a0_c0 = float(np.nanmean(c0_success_matrix(df_c0, arch, generation, "A0")))
            a0_clutter = float(np.nanmean(np.concatenate(
                [_success_matrix(df_c1c3, arch, generation, "A0", lv, sorted(set(df_c1c3.eval_seed))) for lv in C1C3_LEVELS], axis=1)))
            a1_c0 = float(np.nanmean(c0_success_matrix(df_c0, arch, generation, "A1-unseen-203")))
            a1_clutter = float(np.nanmean(np.concatenate(
                [_success_matrix(df_c1c3, arch, generation, "A1-unseen-203", lv, sorted(set(df_c1c3.eval_seed))) for lv in C1C3_LEVELS], axis=1)))
            rows.append({
                "architecture": ARCH_LABELS[arch], "generation": generation,
                "A0_C0": a0_c0, "A0_clutter": a0_clutter, "A1_unseen_C0": a1_c0, "A1_unseen_clutter": a1_clutter,
                "clutter_effect_under_A0": a0_clutter - a0_c0, "clutter_effect_under_A1_unseen": a1_clutter - a1_c0,
                "attacker_effect_under_C0": a1_c0 - a0_c0, "attacker_effect_under_clutter": a1_clutter - a0_clutter,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Phase 14: outcome decomposition.
# =============================================================================

OUTCOME_COLUMNS = (
    "defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
)


def outcome_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["architecture", "generation", "attacker_type"])[list(OUTCOME_COLUMNS)].mean().reset_index()


# =============================================================================
# Phase 15: efficiency (successful interceptions only).
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


if __name__ == "__main__":
    OUT = C0_OUT_DIR
    df_c0 = load_canonical_c0_episodes()
    df_c1c3 = load_c1c3_episodes()

    retention = a0_canonical_retention(df_c0)
    (OUT / "canonical_a0_retention.json").write_text(json.dumps(retention, indent=2, default=str))

    primary = canonical_primary_endpoint_unseen(df_c0)
    (OUT / "canonical_primary_endpoint_unseen.json").write_text(json.dumps(primary, indent=2, default=str))

    seen = canonical_seen_attacker_analysis(df_c0)
    (OUT / "canonical_seen_attacker_analysis.json").write_text(json.dumps(seen, indent=2, default=str))

    c0_vs_clutter = c0_vs_clutter_table(df_c0, df_c1c3)
    c0_vs_clutter.to_csv(OUT / "c0_vs_clutter_table.csv", index=False)

    d1_d0plus_by_cond = d1_minus_d0plus_by_condition(df_c0, df_c1c3)
    d1_d0plus_by_cond.to_csv(OUT / "d1_minus_d0plus_by_condition.csv", index=False)

    decomposition = four_condition_decomposition(df_c0, df_c1c3)
    decomposition.to_csv(OUT / "four_condition_decomposition.csv", index=False)

    outcome_decomp = outcome_decomposition(df_c0)
    outcome_decomp.to_csv(OUT / "outcome_decomposition.csv", index=False)
    max_collision = outcome_decomp[[c for c in OUTCOME_COLUMNS if "collision" in c]].to_numpy().max()
    if max_collision != 0:
        raise RuntimeError(f"CONFIGURATION ERROR: nonzero obstacle-collision rate ({max_collision}) found in canonical C0 outcome decomposition")

    efficiency = interception_efficiency_summary(df_c0)
    efficiency.to_csv(OUT / "interception_efficiency_summary.csv", index=False)

    lrppo_summary = lrppo_residual_canonical_summary(df_c0)
    lrppo_summary.to_csv(OUT / "lrppo_residual_canonical_summary.csv", index=False)

    try:
        df_legacy = pd.read_csv(Path(C0_OUT_DIR).parent / "defender_adaptation_c0_evaluation" / "episodes.csv")
        legacy_cmp = legacy_vs_canonical_ppo_c(df_c0, df_legacy)
        (OUT / "legacy_vs_canonical_ppo_c.json").write_text(json.dumps(legacy_cmp, indent=2, default=str))
        print("=== PPO-C: canonical vs legacy C0 ===")
        print(json.dumps(legacy_cmp, indent=2, default=str))
    except FileNotFoundError:
        print("legacy C0 episodes.csv not found -- skipping Phase 11 explicit comparison")

    print("=== A0 canonical retention (D1 vs D0) ===")
    for r in retention:
        if r["generation_a"] == "d1" and r["generation_b"] == "d0":
            print(f"  {r['architecture']:>22s}: D0={r['success_b_mean']:.3f} D1={r['success_a_mean']:.3f} "
                  f"diff={r['observed_diff']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")

    print("=== Canonical primary endpoint (D1-D0+) vs A1-unseen-203 ===")
    for r in primary:
        print(f"  {r['architecture']:>22s}: D0+={r['success_b_mean']:.3f} D1={r['success_a_mean']:.3f} "
              f"diff={r['observed_diff']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")

    print("=== C0 vs clutter table ===")
    print(c0_vs_clutter.to_string(index=False))

    print("=== Four-condition decomposition ===")
    print(decomposition.to_string(index=False))

    print("=== Outcome decomposition (max collision rate must be 0) ===")
    print(outcome_decomp.to_string(index=False))
    print("max_collision_rate:", max_collision)

    print("=== LR-PPO residual usage (canonical C0) ===")
    print(lrppo_summary.to_string(index=False))

    print("=== Efficiency (successful interceptions only) ===")
    print(efficiency.to_string(index=False))
