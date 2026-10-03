"""
Publication-ready tables for the defender-adaptation held-out evaluation
(journal extension, 38-phase defender-adaptation-EVALUATION sub-task,
Phase 34). Pure post-processing of `episodes.csv` via
`experiments/defender_adaptation_statistics.py` -- no new episodes run
here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS, OUT_DIR
from experiments.defender_adaptation_statistics import (
    load_episodes, success_matrix, aggregate_across_levels, robustness_scores, safety_layer_dependence,
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
)
from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds

TABLES_DIR = OUT_DIR / "tables"

ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}


def _success_row(df: pd.DataFrame, architecture: str, attacker_type: str, level: str | list[str]) -> dict:
    levels = [level] if isinstance(level, str) else level
    means = {}
    matrices = {}
    for gen in ("d0", "d0_plus", "d1"):
        mats = [success_matrix(df, architecture, gen, attacker_type, lv) for lv in levels]
        mat = aggregate_across_levels(mats) if len(mats) > 1 else mats[0]
        matrices[gen] = mat
        means[gen] = float(mat.mean())
    _, ci_low, ci_high, _ = hierarchical_paired_bootstrap_matched_seeds(
        matrices["d1"], matrices["d0_plus"], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
    )
    return {
        "Architecture": ARCH_LABELS[architecture],
        "D0": means["d0"], "D0+": means["d0_plus"], "D1": means["d1"],
        "D1-D0+": means["d1"] - means["d0_plus"],
        "95% CI": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
    }


def table1_adaptive_threat_generalization(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: D0/D0+/D1 safe-interception rate against A1-unseen-203, by
    architecture, C1/C2/C3 and the equal-weight aggregate."""
    rows = []
    for level in list(EVAL_LEVELS) + ["aggregate_C1C3"]:
        lv_arg = level if level != "aggregate_C1C3" else list(EVAL_LEVELS)
        for arch in ARCHITECTURES:
            row = _success_row(df, arch, "A1-unseen-203", lv_arg)
            row["Clutter"] = level
            rows.append(row)
    return pd.DataFrame(rows)[["Clutter", "Architecture", "D0", "D0+", "D1", "D1-D0+", "95% CI"]]


def table2_scripted_threat_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: same structure, against A0."""
    rows = []
    for level in list(EVAL_LEVELS) + ["aggregate_C1C3"]:
        lv_arg = level if level != "aggregate_C1C3" else list(EVAL_LEVELS)
        for arch in ARCHITECTURES:
            row = _success_row(df, arch, "A0", lv_arg)
            row["Clutter"] = level
            rows.append(row)
    return pd.DataFrame(rows)[["Clutter", "Architecture", "D0", "D0+", "D1", "D1-D0+", "95% CI"]]


def table3_seen_attackers(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: same structure, equal-weight A1-seen-201/202 aggregate."""
    import numpy as np
    rows = []
    for level in list(EVAL_LEVELS) + ["aggregate_C1C3"]:
        lv_arg = level if level != "aggregate_C1C3" else list(EVAL_LEVELS)
        levels = [lv_arg] if isinstance(lv_arg, str) else lv_arg
        for arch in ARCHITECTURES:
            means = {}
            matrices = {}
            for gen in ("d0", "d0_plus", "d1"):
                mats = [success_matrix(df, arch, gen, at, lv) for at in ("A1-seen-201", "A1-seen-202") for lv in levels]
                mat = aggregate_across_levels(mats)
                matrices[gen] = mat
                means[gen] = float(mat.mean())
            _, ci_low, ci_high, _ = hierarchical_paired_bootstrap_matched_seeds(
                matrices["d1"], matrices["d0_plus"], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
            )
            rows.append({
                "Clutter": level, "Architecture": ARCH_LABELS[arch],
                "D0": means["d0"], "D0+": means["d0_plus"], "D1": means["d1"],
                "D1-D0+": means["d1"] - means["d0_plus"],
                "95% CI": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
            })
    return pd.DataFrame(rows)[["Clutter", "Architecture", "D0", "D0+", "D1", "D1-D0+", "95% CI"]]


def table4_robustness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: BalancedRobustness, WorstCase, safety-intervention
    fraction, per architecture/generation (aggregate_C1C3 row). No
    rankings are computed (Phase 29: 'do not optimize or rank controllers
    solely by this value')."""
    robustness = robustness_scores(df)
    robustness = robustness[robustness.level == "aggregate_C1C3"]
    safety = safety_layer_dependence(df)
    safety_agg = safety.groupby(["architecture", "generation"])["mean_intervention_fraction"].mean().reset_index()
    merged = robustness.merge(safety_agg, on=["architecture", "generation"], how="left")
    merged["Architecture"] = merged["architecture"].map(ARCH_LABELS)
    return merged[["Architecture", "generation", "A0", "A1_seen_mean", "A1_unseen",
                   "BalancedRobustness", "WorstCase", "mean_intervention_fraction"]].rename(
        columns={"generation": "Generation", "mean_intervention_fraction": "SafetyInterventionFraction"}
    )


def write_all_tables(df: pd.DataFrame | None = None) -> dict[str, Path]:
    df = df if df is not None else load_episodes()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, builder in (
        ("table1_adaptive_threat_generalization", table1_adaptive_threat_generalization),
        ("table2_scripted_threat_retention", table2_scripted_threat_retention),
        ("table3_seen_attackers", table3_seen_attackers),
        ("table4_robustness_summary", table4_robustness_summary),
    ):
        table = builder(df)
        path = TABLES_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


if __name__ == "__main__":
    paths = write_all_tables()
    for name, path in paths.items():
        print(f"wrote {name} -> {path}")
