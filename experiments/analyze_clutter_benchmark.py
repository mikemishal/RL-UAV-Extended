"""
Statistical analysis (Phase 10) for the clutter benchmark: per-level success
rates + CIs, failure-mode breakdown, degradation-from-C0, and paired
LR-PPO-vs-Lead / LR-PPO-vs-RMPC comparisons.

Usage:
    python -m experiments.analyze_clutter_benchmark --run-name main_50
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments.clutter_benchmark_protocol import CLUTTER_LEVEL_ORDER, LEARNED_ROOTS, PROJECT_ROOT
from experiments.final_stats import (
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_matched_seeds,
    hierarchical_paired_bootstrap_vs_scripted,
    paired_bootstrap_ci,
    wilson_interval,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "clutter_benchmark"
N_BOOT = 10_000
BOOT_SEED = 20260917
DETERMINISTIC_FAMILIES = ("Greedy", "PN", "Lead", "RMPC")
LEARNED_FAMILIES = ("PPO", "PPO-Kalman", "LR-PPO")
OUTCOME_COLS = ("success", "soldier_caught", "unsafe_intercept", "timeout")


def _load(run_name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_ROOT / run_name / "all_episodes.csv")
    return df


def _family_success_vector(df: pd.DataFrame, level: str, family: str) -> np.ndarray:
    """(n_episodes,) success vector for a deterministic (single-instance) family."""
    sub = df[(df.level == level) & (df.family == family) & (df.feasible == True)]  # noqa: E712
    sub = sub.sort_values("seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _family_success_matrix(df: pd.DataFrame, level: str, family: str) -> np.ndarray:
    """(n_seeds, n_episodes) success matrix for a learned (multi-instance) family."""
    rows = []
    for seed in LEARNED_ROOTS[family]:
        instance_id = {
            "PPO": f"PPO-{seed}", "PPO-Kalman": f"PPOK-{seed}", "LR-PPO": f"LRPPO-{seed}",
        }[family]
        sub = df[(df.level == level) & (df.instance_id == instance_id) & (df.feasible == True)]  # noqa: E712
        sub = sub.sort_values("seed")
        rows.append(sub["success"].to_numpy(dtype=np.float64))
    return np.stack(rows, axis=0)


def success_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in CLUTTER_LEVEL_ORDER:
        for family in DETERMINISTIC_FAMILIES:
            vec = _family_success_vector(df, level, family)
            n = len(vec)
            successes = int(vec.sum())
            lo, hi = wilson_interval(successes, n)
            rows.append(dict(level=level, family=family, n=n, success_rate=successes / n if n else float("nan"),
                              ci_low=lo, ci_high=hi))
        for family in LEARNED_FAMILIES:
            mat = _family_success_matrix(df, level, family)
            observed, lo, hi = hierarchical_bootstrap_track(mat, N_BOOT, BOOT_SEED)
            rows.append(dict(level=level, family=family, n=mat.shape[1] * mat.shape[0],
                              success_rate=observed, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def failure_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in CLUTTER_LEVEL_ORDER:
        level_df = df[df.level == level]
        for family in DETERMINISTIC_FAMILIES + LEARNED_FAMILIES:
            sub = level_df[level_df.family == family]
            n_total = len(sub)
            n_infeasible = int((~sub.feasible).sum())
            feasible_sub = sub[sub.feasible == True]  # noqa: E712
            n = len(feasible_sub)
            row = dict(level=level, family=family, n_requested=n_total, n_infeasible=n_infeasible, n_feasible=n)
            for col in OUTCOME_COLS:
                row[f"{col}_rate"] = float(feasible_sub[col].mean()) if n else float("nan")
            row["defender_obstacle_collision_rate"] = (
                float(feasible_sub["defender_obstacle_collision"].mean()) if n else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def geometry_timing_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "detection_time", "intercept_time", "episode_duration_s",
        "min_defender_obstacle_clearance", "min_enemy_obstacle_clearance",
        "min_defender_enemy_dist", "min_enemy_soldier_dist",
        "defender_path_length", "enemy_path_length",
        "mean_accel_utilization", "mean_turn_rate_utilization",
    ]
    rows = []
    for level in CLUTTER_LEVEL_ORDER:
        for family in DETERMINISTIC_FAMILIES + LEARNED_FAMILIES:
            sub = df[(df.level == level) & (df.family == family) & (df.feasible == True)]  # noqa: E712
            row = dict(level=level, family=family, n=len(sub))
            for col in cols:
                if col in sub.columns:
                    valid = sub[col].replace([-1], np.nan) if col in ("detection_time", "intercept_time") else sub[col]
                    row[f"mean_{col}"] = float(valid.mean())
                    row[f"median_{col}"] = float(valid.median())
            if "mean_residual_angle_deg" in sub.columns and sub["mean_residual_angle_deg"].notna().any():
                row["mean_residual_angle_deg"] = float(sub["mean_residual_angle_deg"].mean())
                row["max_residual_angle_deg"] = float(sub["max_residual_angle_deg"].max())
            rows.append(row)
    return pd.DataFrame(rows)


def degradation_from_c0_table(df: pd.DataFrame) -> pd.DataFrame:
    """Delta_success(Ck) = success(Ck) - success(C0), paired on matched seeds."""
    rows = []
    for level in ("C1", "C2", "C3"):
        for family in DETERMINISTIC_FAMILIES:
            vec_c0 = _family_success_vector(df, "C0", family)
            vec_ck = _family_success_vector(df, level, family)
            n = min(len(vec_c0), len(vec_ck))
            diff, lo, hi = paired_bootstrap_ci(vec_ck[:n], vec_c0[:n], N_BOOT, BOOT_SEED)
            rows.append(dict(level=level, family=family, delta_success=diff, ci_low=lo, ci_high=hi,
                              significant=bool(lo > 0 or hi < 0)))
        for family in LEARNED_FAMILIES:
            mat_c0 = _family_success_matrix(df, "C0", family)
            mat_ck = _family_success_matrix(df, level, family)
            diff, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(mat_ck, mat_c0, N_BOOT, BOOT_SEED)
            rows.append(dict(level=level, family=family, delta_success=diff, ci_low=lo, ci_high=hi,
                              significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def lr_ppo_vs_comparisons_table(df: pd.DataFrame) -> pd.DataFrame:
    """LR-PPO vs Lead and LR-PPO vs RMPC at each clutter level (paired, matched seeds)."""
    rows = []
    for level in CLUTTER_LEVEL_ORDER:
        lr_ppo_mat = _family_success_matrix(df, level, "LR-PPO")
        for comparator in ("Lead", "RMPC"):
            comp_vec = _family_success_vector(df, level, comparator)
            n = min(lr_ppo_mat.shape[1], len(comp_vec))
            diff, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
                lr_ppo_mat[:, :n], comp_vec[:n], N_BOOT, BOOT_SEED
            )
            rows.append(dict(level=level, comparison=f"LR-PPO - {comparator}", diff=diff, ci_low=lo, ci_high=hi,
                              significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def run_analysis(run_name: str) -> dict:
    df = _load(run_name)
    out_dir = RESULTS_ROOT / run_name / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "success_rate": success_rate_table(df),
        "failure_mode": failure_mode_table(df),
        "geometry_timing": geometry_timing_table(df),
        "degradation_from_c0": degradation_from_c0_table(df),
        "lr_ppo_vs_comparisons": lr_ppo_vs_comparisons_table(df),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[{run_name}] wrote {out_dir / f'{name}.csv'} ({len(table)} rows)")

    infeasible_total = int((~df.feasible).sum())
    summary = {
        "run_name": run_name,
        "n_episode_rows": len(df),
        "n_infeasible_placement": infeasible_total,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    run_analysis(args.run_name)


if __name__ == "__main__":
    main()
