"""
Phase R24: paired statistical analysis for the Adaptive Attacker v1
evaluation. Scripted and each learned-attacker seed condition are
evaluated on IDENTICAL held-out seeds per (level, defender_instance) --
true matched/paired data (common random numbers) -- so this reuses the
EXISTING, validated `experiments/final_stats.py::paired_bootstrap_ci`
function (never reimplemented), with this repo's established
BOOTSTRAP_REPLICATES=10_000 / BOOTSTRAP_SEED=20260816 convention.

Usage:
    python -m experiments.adaptive_attacker_statistics
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.final_stats import paired_bootstrap_ci

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_evaluation"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def compute_paired_comparisons(episodes_csv_path: Path) -> pd.DataFrame:
    """Per (level, defender_instance, learned_attacker_condition): paired
    bootstrap CI on the defender_safe_intercept rate DIFFERENCE
    (learned - scripted), using ONLY the seeds common to both conditions
    (always all of them, by construction -- held_out_seeds() is
    deterministic and identical for every condition)."""
    df = pd.read_csv(episodes_csv_path)
    rows = []
    for (level, defender_instance), group in df.groupby(["level", "defender_instance"]):
        scripted = group[group["attacker_condition"] == "scripted"].set_index("seed")
        if scripted.empty:
            continue
        defender_family = group["defender_family"].iloc[0]
        seen_in_training = bool(group["seen_in_training"].iloc[0])
        for condition in sorted(c for c in group["attacker_condition"].unique() if c != "scripted"):
            learned = group[group["attacker_condition"] == condition].set_index("seed")
            common_seeds = sorted(set(scripted.index) & set(learned.index))
            if not common_seeds:
                continue
            scripted_arr = scripted.loc[common_seeds, "defender_safe_intercept"].to_numpy()
            learned_arr = learned.loc[common_seeds, "defender_safe_intercept"].to_numpy()
            observed, ci_lo, ci_hi = paired_bootstrap_ci(
                learned_arr, scripted_arr, n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            )
            rows.append({
                "level": level, "defender_family": defender_family, "defender_instance": defender_instance,
                "seen_in_training": seen_in_training, "attacker_condition": condition, "n": len(common_seeds),
                "delta_defender_success": observed, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "significant_at_95": not (ci_lo <= 0.0 <= ci_hi),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = compute_paired_comparisons(EVAL_DIR / "episodes.csv")
    df.to_csv(EVAL_DIR / "paired_bootstrap_comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    n_sig = int(df["significant_at_95"].sum())
    print(f"\n{n_sig}/{len(df)} (defender_instance, learned_seed, level) comparisons show a "
          f"statistically significant (95% paired bootstrap CI excludes 0) defender degradation or "
          f"improvement.")
