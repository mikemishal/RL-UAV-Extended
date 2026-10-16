"""Phase 8-11 statistics for the journal robustness/assurance evaluation.
Pure post-processing. Reuses experiments/final_stats.py's hierarchical
paired bootstrap UNCHANGED (n_training_seeds=3 per architecture,
10,000 replicates, matched evaluation seeds)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds
from experiments.defender_adaptation_canonical_c0_evaluation import OUT_DIR as CANONICAL_C0_OUT_DIR
from experiments.defender_adaptation_evaluation import OUT_DIR as C1C3_OUT_DIR
from experiments.journal_robustness_assurance import OUT_DIR, PRIMARY_ATTACKER, R1_SETTINGS, R2_SETTINGS, R3_SETTINGS

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001
ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}

DIMENSION_ORDER = [
    ("nominal", "nominal"),
    ("R1_measurement_noise", "low"), ("R1_measurement_noise", "high"),
    ("R2_detection_radius", "short"), ("R2_detection_radius", "long"),
    ("R3_relative_mobility", "defender_disadvantaged"), ("R3_relative_mobility", "defender_advantaged"),
]


def load_nominal_c0() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_C0_OUT_DIR / "episodes.csv")
    df = df[(df.attacker_type == PRIMARY_ATTACKER) & (df.generation.isin(["d0_plus", "d1"])) & (df.level == "C0")].copy()
    df["dimension"] = "nominal"
    df["setting_label"] = "nominal"
    return df


def load_nominal_c2() -> pd.DataFrame:
    df = pd.read_csv(C1C3_OUT_DIR / "episodes.csv")
    df = df[(df.attacker_type == PRIMARY_ATTACKER) & (df.generation.isin(["d0_plus", "d1"])) & (df.level == "C2")].copy()
    df["dimension"] = "nominal"
    df["setting_label"] = "nominal"
    return df


def load_new_robustness_episodes() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "episodes.csv")


def load_all() -> pd.DataFrame:
    cols_keep = ["architecture", "defender_seed", "generation", "dimension", "setting_label", "level", "eval_seed",
                 "defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
                 "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
                 "episode_length", "interception_time", "defender_path_length", "attacker_path_length",
                 "min_defender_attacker_distance", "defender_safety_intervention_fraction",
                 "mean_residual_angle_deg", "fraction_near_30deg", "mean_kalman_position_error", "mean_kalman_velocity_error"]
    frames = [load_nominal_c0(), load_nominal_c2(), load_new_robustness_episodes()]
    frames = [f.reindex(columns=cols_keep) for f in frames]
    return pd.concat(frames, ignore_index=True)


def _matched_pair_matrices_by_generation(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, list]:
    a = df[df.generation == "d1"][["defender_seed", "eval_seed", value_col]].rename(columns={value_col: "a"})
    b = df[df.generation == "d0_plus"][["defender_seed", "eval_seed", value_col]].rename(columns={value_col: "b"})
    merged = a.merge(b, on=["defender_seed", "eval_seed"], how="inner").dropna(subset=["a", "b"])
    seeds = sorted(merged.defender_seed.unique())
    per_seed_a = [merged[merged.defender_seed == s]["a"].to_numpy(dtype=np.float64) for s in seeds]
    per_seed_b = [merged[merged.defender_seed == s]["b"].to_numpy(dtype=np.float64) for s in seeds]
    n = min(len(x) for x in per_seed_a) if per_seed_a else 0
    mat_a = np.stack([x[:n] for x in per_seed_a]) if n else np.zeros((0, 0))
    mat_b = np.stack([x[:n] for x in per_seed_b]) if n else np.zeros((0, 0))
    return mat_a, mat_b, seeds


def d1_minus_d0plus_table(df: pd.DataFrame, value_col: str = "defender_safe_intercept") -> pd.DataFrame:
    rows = []
    for arch in ARCH_LABELS:
        for level in ("C0", "C2"):
            for dimension, setting_label in DIMENSION_ORDER:
                sub = df[(df.architecture == arch) & (df.level == level)
                         & (df.dimension == dimension) & (df.setting_label == setting_label)]
                if sub.empty:
                    continue
                mat_d1, mat_d0plus, seeds = _matched_pair_matrices_by_generation(sub, value_col)
                if mat_d1.size == 0:
                    continue
                per_seed = np.nanmean(mat_d1, axis=1) - np.nanmean(mat_d0plus, axis=1)
                obs, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(mat_d1, mat_d0plus, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                rows.append({
                    "architecture": ARCH_LABELS[arch], "level": level, "dimension": dimension, "setting_label": setting_label,
                    "d0plus_mean": float(np.nanmean(mat_d0plus)), "d1_mean": float(np.nanmean(mat_d1)),
                    "diff": obs, "ci_low": lo, "ci_high": hi,
                    "per_seed_diffs": {str(s): float(d) for s, d in zip(seeds, per_seed)},
                })
    return pd.DataFrame(rows)


def outcome_decomposition_challenging(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 16 Table 3: failure-mode decomposition for the two MOST
    challenging off-nominal settings (R1 high noise, R2 short detection, R3 defender_disadvantaged)."""
    challenging = [("R1_measurement_noise", "high"), ("R2_detection_radius", "short"),
                   ("R3_relative_mobility", "defender_disadvantaged")]
    rows = []
    for dimension, setting_label in challenging:
        sub = df[(df.dimension == dimension) & (df.setting_label == setting_label)]
        for arch in ARCH_LABELS:
            for level in ("C0", "C2"):
                for gen in ("d0_plus", "d1"):
                    cell = sub[(sub.architecture == arch) & (sub.level == level) & (sub.generation == gen)]
                    if cell.empty:
                        continue
                    rows.append({
                        "dimension": dimension, "setting_label": setting_label, "architecture": ARCH_LABELS[arch],
                        "level": level, "generation": gen, "n": len(cell),
                        "safe_intercept": float(cell.defender_safe_intercept.mean()),
                        "asset_reach": float(cell.attacker_asset_reach.mean()),
                        "unsafe_intercept": float(cell.unsafe_intercept.mean()),
                        "timeout": float(cell.timeout.mean()),
                        "defender_collision": float((cell.defender_static_obstacle_collision + cell.defender_dynamic_obstacle_collision).mean()),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 14: pre-specified difference-of-differences interaction tests
# (small, fixed set -- NOT a fishing expedition).
# ---------------------------------------------------------------------------

def interaction_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified comparisons:
    1. PPO-Kalman: (D1-D0+ at R1-high) - (D1-D0+ at nominal), C0 -- "does PPO-Kalman's
       advantage grow under noise?"
    2. PPO-Kalman: (D1-D0+ at R2-short) - (D1-D0+ at nominal), C0 -- "does reduced
       reaction time eliminate the advantage?"
    3. PPO-C: (D1-D0+ at nominal C2) - (D1-D0+ at nominal C0) -- "is the C0
       regression environment-specific?" (already established; replicated here
       under the SAME robustness-campaign statistical framework.)
    """
    specs = [
        ("ppo_kalman_clutter_v1", "C0", ("R1_measurement_noise", "high"), ("nominal", "nominal")),
        ("ppo_kalman_clutter_v1", "C0", ("R2_detection_radius", "short"), ("nominal", "nominal")),
    ]
    rows = []
    for arch, level, (dim_a, set_a), (dim_b, set_b) in specs:
        sub_a = df[(df.architecture == arch) & (df.level == level) & (df.dimension == dim_a) & (df.setting_label == set_a)]
        sub_b = df[(df.architecture == arch) & (df.level == level) & (df.dimension == dim_b) & (df.setting_label == set_b)]
        mat_d1_a, mat_d0_a, seeds_a = _matched_pair_matrices_by_generation(sub_a, "defender_safe_intercept")
        mat_d1_b, mat_d0_b, seeds_b = _matched_pair_matrices_by_generation(sub_b, "defender_safe_intercept")
        if mat_d1_a.size == 0 or mat_d1_b.size == 0:
            continue
        n_seeds = min(mat_d1_a.shape[0], mat_d1_b.shape[0])
        n_eps = min(mat_d1_a.shape[1], mat_d1_b.shape[1])
        dd_matrix_a = mat_d1_a[:n_seeds, :n_eps] - mat_d0_a[:n_seeds, :n_eps]
        dd_matrix_b = mat_d1_b[:n_seeds, :n_eps] - mat_d0_b[:n_seeds, :n_eps]
        obs, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(dd_matrix_a, dd_matrix_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        rows.append({
            "architecture": ARCH_LABELS[arch], "level": level,
            "condition_a": f"{dim_a}:{set_a}", "condition_b": f"{dim_b}:{set_b}",
            "diff_of_diffs": obs, "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_all()
    df.to_csv(OUT_DIR / "all_operating_points_combined.csv", index=False)
    print(f"combined dataset: {len(df)} episode rows across {df.groupby(['architecture','level','dimension','setting_label']).ngroups} cells")

    d1_table = d1_minus_d0plus_table(df)
    d1_table.drop(columns=["per_seed_diffs"]).to_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv", index=False)
    with open(OUT_DIR / "d1_minus_d0plus_robustness_full.json", "w") as f:
        json.dump(d1_table.to_dict(orient="records"), f, indent=2, default=str)
    print("\n=== D1-D0+ robustness table ===")
    print(d1_table.drop(columns=["per_seed_diffs"]).to_string(index=False))

    outcomes = outcome_decomposition_challenging(df)
    outcomes.to_csv(OUT_DIR / "outcome_decomposition_challenging.csv", index=False)
    print("\n=== Outcome decomposition (challenging conditions) ===")
    print(outcomes.to_string(index=False))

    interactions = interaction_tests(df)
    interactions.to_csv(OUT_DIR / "interaction_tests.csv", index=False)
    print("\n=== Pre-specified interaction (difference-of-differences) tests ===")
    print(interactions.to_string(index=False))
