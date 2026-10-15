"""Phase 5/8/11 statistics for the architecture-mechanism diagnostic.
Pure post-processing of the mechanism_diagnostic.py / mechanism_diagnostic_
authority.py outputs. Reuses experiments/final_stats.py's hierarchical
paired bootstrap UNCHANGED (n_training_seeds=3 per architecture --
Phase 11's explicit "account for only 3 training seeds, avoid
pseudo-replication" requirement)."""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds, paired_bootstrap_ci
from experiments.mechanism_diagnostic import OUT_DIR

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001
CONFIDENCE_LEVEL = 0.95

RESIDUAL_BINS = [0, 5, 10, 15, 20, 25, 30]
RESIDUAL_BIN_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30"]


def _parse_tuple_col(series: pd.Series) -> np.ndarray:
    """Parses the stringified 3-tuples written by mechanism_diagnostic.py's
    CSV output back into an (n, 3) array using ast.literal_eval (never
    eval()) -- these CSVs are generated exclusively by our own trusted code."""
    return np.array([ast.literal_eval(v) if isinstance(v, str) else v for v in series])


def _seed_matrix(df: pd.DataFrame, value_col: str, seed_col: str = "defender_seed") -> tuple[np.ndarray, list]:
    """shape (n_seeds, n_episodes_per_seed); seeds sorted ascending; requires
    equal episode counts per seed (true by construction -- matched eval_seed
    grid)."""
    seeds = sorted(df[seed_col].unique())
    per_seed = [df[df[seed_col] == s][value_col].to_numpy(dtype=np.float64) for s in seeds]
    n = min(len(a) for a in per_seed)
    mat = np.stack([a[:n] for a in per_seed])
    return mat, seeds


def _matched_pair_matrices(df: pd.DataFrame, group_a: str, group_b: str, split_col: str, value_col: str,
                            seed_col: str = "defender_seed") -> tuple[np.ndarray, np.ndarray, list]:
    """Inner-merges rows of `df` where split_col==group_a vs ==group_b on
    (defender_seed, eval_seed), guaranteeing IDENTICAL matched-pair shapes
    for hierarchical_paired_bootstrap_matched_seeds (avoids the
    independent-dropna shape mismatch)."""
    a = df[df[split_col] == group_a][[seed_col, "eval_seed", value_col]].rename(columns={value_col: "a"})
    b = df[df[split_col] == group_b][[seed_col, "eval_seed", value_col]].rename(columns={value_col: "b"})
    merged = a.merge(b, on=[seed_col, "eval_seed"], how="inner").dropna(subset=["a", "b"])
    seeds = sorted(merged[seed_col].unique())
    per_seed_a = [merged[merged[seed_col] == s]["a"].to_numpy(dtype=np.float64) for s in seeds]
    per_seed_b = [merged[merged[seed_col] == s]["b"].to_numpy(dtype=np.float64) for s in seeds]
    n = min(len(x) for x in per_seed_a)
    mat_a = np.stack([x[:n] for x in per_seed_a])
    mat_b = np.stack([x[:n] for x in per_seed_b])
    return mat_a, mat_b, seeds


# ---------------------------------------------------------------------------
# Phase 3: Lead error A0 vs A1, D0 vs D0+ vs D1, success vs failure, C0 vs C3
# ---------------------------------------------------------------------------

def lead_error_a0_vs_a1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in sorted(df.level.unique()):
        for gen in ("d0", "d0_plus", "d1"):
            sub = df[(df.level == level) & (df.generation == gen)]
            if sub.empty:
                continue
            mat_a1, mat_a0, seeds = _matched_pair_matrices(
                sub, "A1-unseen-203", "A0", "attacker_type", "mean_angular_lead_error_deg")
            if mat_a1.size == 0:
                continue
            obs, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
                mat_a1, mat_a0, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            rows.append({
                "level": level, "generation": gen,
                "mean_lead_error_a0": float(np.nanmean(mat_a0)), "mean_lead_error_a1": float(np.nanmean(mat_a1)),
                "diff_a1_minus_a0": obs, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


def lead_error_success_vs_failure(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in sorted(df.level.unique()):
        for gen in ("d0", "d0_plus", "d1"):
            for attacker in ("A0", "A1-unseen-203"):
                sub = df[(df.level == level) & (df.generation == gen) & (df.attacker_type == attacker)]
                sub = sub.dropna(subset=["mean_angular_lead_error_deg"])
                succ = sub[sub.defender_safe_intercept == 1]["mean_angular_lead_error_deg"]
                fail = sub[sub.defender_safe_intercept == 0]["mean_angular_lead_error_deg"]
                if len(succ) < 2 or len(fail) < 2:
                    continue
                rows.append({
                    "level": level, "generation": gen, "attacker_type": attacker,
                    "n_success": len(succ), "n_failure": len(fail),
                    "mean_lead_error_success": float(succ.mean()), "mean_lead_error_failure": float(fail.mean()),
                    "median_lead_error_success": float(succ.median()), "median_lead_error_failure": float(fail.median()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 5: residual-magnitude binning (D1 only, canonical 30deg controller)
# ---------------------------------------------------------------------------

def residual_bin_analysis(steps_df: pd.DataFrame, generation: str = "d1") -> pd.DataFrame:
    sub = steps_df[(steps_df.generation == generation) & steps_df.residual_angle_deg.notna()].copy()
    sub["bin"] = pd.cut(sub["residual_angle_deg"], bins=RESIDUAL_BINS, labels=RESIDUAL_BIN_LABELS, include_lowest=True)
    rows = []
    for b in RESIDUAL_BIN_LABELS:
        bsub = sub[sub["bin"] == b]
        if bsub.empty:
            continue
        rows.append({
            "residual_bin_deg": b,
            "n_steps": len(bsub),
            "frequency": len(bsub) / len(sub),
            "mean_lead_error_deg": float(bsub["angular_lead_error_deg"].mean(skipna=True)),
            "mean_maneuver_intensity": float(bsub["maneuver_intensity"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def residual_bin_success_association(episodes_df: pd.DataFrame, steps_df: pd.DataFrame, generation: str = "d1") -> pd.DataFrame:
    """Per-episode mean residual angle, binned, vs that episode's outcome --
    avoids step-level pseudo-replication (one row per episode)."""
    ep = episodes_df[episodes_df.generation == generation].copy()
    ep["bin"] = pd.cut(ep["mean_residual_angle_deg"], bins=RESIDUAL_BINS, labels=RESIDUAL_BIN_LABELS, include_lowest=True)
    rows = []
    for b in RESIDUAL_BIN_LABELS:
        bsub = ep[ep["bin"] == b]
        if bsub.empty:
            continue
        rows.append({
            "residual_bin_deg": b, "n_episodes": len(bsub),
            "success_rate": float(bsub["defender_safe_intercept"].mean()),
            "mean_episode_lead_error_deg": float(bsub["mean_angular_lead_error_deg"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 6/7: behavioral metrics from step-level trajectories
# ---------------------------------------------------------------------------

def _path_curvature(xs: np.ndarray) -> float:
    """Mean turn-angle per step (degrees) along a position trajectory."""
    if len(xs) < 3:
        return 0.0
    v = np.diff(xs, axis=0)
    norms = np.linalg.norm(v, axis=1)
    valid = norms > 1e-6
    angles = []
    for i in range(len(v) - 1):
        if valid[i] and valid[i + 1]:
            cos_a = np.clip(np.dot(v[i], v[i + 1]) / (norms[i] * norms[i + 1]), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_a)))
    return float(np.mean(angles)) if angles else 0.0


def _action_reversals(actions: np.ndarray, threshold_deg: float = 90.0) -> int:
    """Count of consecutive-step action-direction changes exceeding threshold_deg."""
    count = 0
    for i in range(1, len(actions)):
        a, b = actions[i - 1], actions[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6:
            cos_a = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            if np.degrees(np.arccos(cos_a)) > threshold_deg:
                count += 1
    return count


def behavioral_summary(steps_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (architecture, defender_seed, generation, attacker_type,
    level, eval_seed) episode: path curvature, action-reversal count, mean
    action magnitude, closing-velocity trend."""
    rows = []
    group_cols = ["architecture", "defender_seed", "generation", "attacker_type", "level", "eval_seed"]
    for keys, g in steps_df.groupby(group_cols):
        g = g.sort_values("step")
        pos = _parse_tuple_col(g["defender_pos"])
        actions = _parse_tuple_col(g["final_action"])
        def_pos = pos
        att_pos = _parse_tuple_col(g["attacker_pos_truth"])
        closing_dist = np.linalg.norm(def_pos - att_pos, axis=1)
        row = dict(zip(group_cols, keys))
        row["path_curvature_deg"] = _path_curvature(pos)
        row["n_action_reversals"] = _action_reversals(actions)
        row["mean_action_magnitude"] = float(np.mean(np.linalg.norm(actions, axis=1)))
        row["closing_velocity_trend"] = float(closing_dist[0] - closing_dist[-1]) / max(len(closing_dist), 1)
        row["n_steps"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2: objective matched-case selection (A: both succeed, B: D0+ succeeds/
# D1 fails, C: D0+ fails/D1 succeeds, D: both fail). Selection is BY RULE
# (the case closest to the median within its category, on a fixed
# tie-break metric), never hand-picked.
# ---------------------------------------------------------------------------

def select_matched_cases(
    episodes_df: pd.DataFrame, arch_filter: str, attacker_type: str, level: str,
    gen_a: str = "d0_plus", gen_b: str = "d1", rank_metric: str = "episode_length",
) -> dict:
    """Returns {"A": row_or_None, "B": ..., "C": ..., "D": ...} using matched
    (defender_seed, eval_seed) pairs present in BOTH generations. Within each
    category, picks the case whose `rank_metric` is closest to that
    category's median (an OBJECTIVE, reproducible tie-break -- never a
    visual/manual pick)."""
    sub = episodes_df[(episodes_df.architecture == arch_filter) & (episodes_df.attacker_type == attacker_type)
                       & (episodes_df.level == level)]
    a = sub[sub.generation == gen_a].set_index(["defender_seed", "eval_seed"])
    b = sub[sub.generation == gen_b].set_index(["defender_seed", "eval_seed"])
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]

    cats = {
        "A_both_succeed": shared[(a.defender_safe_intercept == 1) & (b.defender_safe_intercept == 1)],
        "B_gen_a_succeeds_gen_b_fails": shared[(a.defender_safe_intercept == 1) & (b.defender_safe_intercept == 0)],
        "C_gen_a_fails_gen_b_succeeds": shared[(a.defender_safe_intercept == 0) & (b.defender_safe_intercept == 1)],
        "D_both_fail": shared[(a.defender_safe_intercept == 0) & (b.defender_safe_intercept == 0)],
    }
    result = {}
    for cat, idx in cats.items():
        if len(idx) == 0:
            result[cat] = None
            continue
        metric_vals = b.loc[idx, rank_metric] if rank_metric in b.columns else a.loc[idx, rank_metric]
        median_val = metric_vals.median()
        closest_idx = (metric_vals - median_val).abs().idxmin()
        result[cat] = {"defender_seed": closest_idx[0], "eval_seed": closest_idx[1], "n_in_category": len(idx)}
    return result


# ---------------------------------------------------------------------------
# Phase 11: hierarchical-bootstrap CIs for the key D0+ vs D1 behavioral
# metrics (closing-velocity trend, action reversals, path curvature) --
# the primary quantitative evidence behind the Phase 14/15 mechanism
# decisions. n_training_seeds=3 throughout (no pseudo-replication).
# ---------------------------------------------------------------------------

BEHAVIORAL_METRICS = ("closing_velocity_trend", "n_action_reversals", "path_curvature_deg", "mean_action_magnitude")


def behavioral_d0plus_vs_d1(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in sorted(behavioral_df.level.unique()):
        sub = behavioral_df[behavioral_df.level == level]
        for metric in BEHAVIORAL_METRICS:
            mat_d1, mat_d0plus, seeds = _matched_pair_matrices(sub, "d1", "d0_plus", "generation", metric)
            if mat_d1.size == 0:
                continue
            obs, lo, hi, per_seed = hierarchical_paired_bootstrap_matched_seeds(
                mat_d1, mat_d0plus, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            rows.append({
                "level": level, "metric": metric,
                "mean_d0plus": float(np.nanmean(mat_d0plus)), "mean_d1": float(np.nanmean(mat_d1)),
                "diff_d1_minus_d0plus": obs, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lrppo_ep = pd.read_csv(OUT_DIR / "lrppo_lead_episodes.csv")

    a0_vs_a1 = lead_error_a0_vs_a1(lrppo_ep)
    a0_vs_a1.to_csv(OUT_DIR / "lead_error_a0_vs_a1.csv", index=False)
    print("=== Lead error: A1 vs A0 (hierarchical paired bootstrap) ===")
    print(a0_vs_a1.to_string(index=False))

    succ_fail = lead_error_success_vs_failure(lrppo_ep)
    succ_fail.to_csv(OUT_DIR / "lead_error_success_vs_failure.csv", index=False)
    print("\n=== Lead error: success vs failure ===")
    print(succ_fail.to_string(index=False))

    lrppo_steps = pd.read_csv(OUT_DIR / "lrppo_lead_steps.csv.gz")
    bin_analysis = residual_bin_analysis(lrppo_steps, generation="d1")
    bin_analysis.to_csv(OUT_DIR / "residual_bin_analysis_d1.csv", index=False)
    print("\n=== Residual-magnitude bin analysis (D1) ===")
    print(bin_analysis.to_string(index=False))

    bin_success = residual_bin_success_association(lrppo_ep, lrppo_steps, generation="d1")
    bin_success.to_csv(OUT_DIR / "residual_bin_success_association_d1.csv", index=False)
    print("\n=== Residual-magnitude bin vs episode success (D1) ===")
    print(bin_success.to_string(index=False))

    for tag, arch_label in (("ppo_kalman_mechanism", "PPO-Kalman-C"), ("ppo_c_regression", "PPO-C")):
        steps = pd.read_csv(OUT_DIR / f"{tag}_steps.csv.gz")
        beh = behavioral_summary(steps)
        beh.to_csv(OUT_DIR / f"{tag}_behavioral_summary.csv", index=False)
        print(f"\n=== {arch_label} behavioral summary (n={len(beh)} episodes) ===")
        print(beh.groupby(["generation", "level"])[
            ["path_curvature_deg", "n_action_reversals", "mean_action_magnitude", "closing_velocity_trend"]
        ].mean().to_string())

        beh_ci = behavioral_d0plus_vs_d1(beh)
        beh_ci.to_csv(OUT_DIR / f"{tag}_behavioral_d0plus_vs_d1_ci.csv", index=False)
        print(f"\n=== {arch_label}: D1-D0+ behavioral metrics (hierarchical bootstrap CI) ===")
        print(beh_ci.to_string(index=False))
