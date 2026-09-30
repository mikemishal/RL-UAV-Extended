"""
Phase R23: journal figures for the Adaptive Attacker v1 evaluation,
generated ONLY from the final canonical (100%-complete, post-recovery)
dataset in results/adaptive_attacker_evaluation/{episodes.csv,summary.csv,
degradation_matrix.csv,paired_bootstrap_comparisons.csv}.

Usage:
    python -m experiments.plot_adaptive_attacker_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_evaluation"
OUT_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_figures"

LEVELS = ("C1", "C2", "C3")
ATTACKER_SEEDS = (201, 202, 203)


def fig1_scripted_vs_learned_success(summary: pd.DataFrame, out_dir: Path):
    """Fig 1: defender safe-intercept rate, scripted vs. mean-across-
    learned-seeds, per defender family (pooled across levels, n-weighted)."""
    families = sorted(summary["defender_family"].unique())
    scripted_rates, learned_rates = [], []
    for fam in families:
        fam_df = summary[summary["defender_family"] == fam]
        scripted = fam_df[fam_df["attacker_condition"] == "scripted"]
        learned = fam_df[fam_df["attacker_condition"] != "scripted"]
        scripted_rates.append((scripted["defender_safe_intercept_rate"] * scripted["n"]).sum() / scripted["n"].sum())
        learned_rates.append((learned["defender_safe_intercept_rate"] * learned["n"]).sum() / learned["n"].sum())
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.18, scripted_rates, width=0.36, label="scripted attacker", color="tab:blue")
    ax.bar(x + 0.18, learned_rates, width=0.36, label="Adaptive Attacker v1 (mean of 3 seeds)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("defender safe-intercept rate")
    ax.set_title("Fig 1: Defender success -- scripted vs. learned attacker (pooled across C1-C3)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_scripted_vs_learned_success.png", dpi=150)
    plt.close(fig)


def fig2_degradation_by_family(degradation: pd.DataFrame, out_dir: Path):
    """Fig 2: delta_defender_success by family, one bar cluster per
    learned-attacker seed."""
    families = sorted(degradation["defender_family"].unique())
    x = np.arange(len(families))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, seed in enumerate(ATTACKER_SEEDS):
        cond = f"learned_seed_{seed}"
        vals = [
            degradation[(degradation.defender_family == fam) & (degradation.attacker_condition == cond)]
            ["delta_defender_success"].mean() for fam in families
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("delta_defender_success\n(learned - scripted safe-intercept rate)")
    ax.set_title("Fig 2: Defender degradation by controller family and attacker seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_degradation_by_family.png", dpi=150)
    plt.close(fig)


def fig3_attacker_target_reach_rate(summary: pd.DataFrame, out_dir: Path):
    """Fig 3: attacker_soldier_caught_rate (target-reach) by attacker
    condition, pooled across all defenders/levels."""
    conditions = ["scripted"] + [f"learned_seed_{s}" for s in ATTACKER_SEEDS]
    rates = []
    for cond in conditions:
        cdf = summary[summary["attacker_condition"] == cond]
        rates.append((cdf["attacker_soldier_caught_rate"] * cdf["n"]).sum() / cdf["n"].sum())
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue"] + ["tab:red"] * 3
    ax.bar(conditions, rates, color=colors)
    ax.set_ylabel("attacker target-reach (soldier_caught) rate")
    ax.set_title("Fig 3: Attacker target-reach rate, pooled across all defenders/levels")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_attacker_target_reach_rate.png", dpi=150)
    plt.close(fig)


def fig4_outcomes_by_level(summary: pd.DataFrame, out_dir: Path):
    """Fig 4: mutually-exclusive outcome-rate decomposition (mean across
    learned seeds vs scripted) at each clutter level, pooled across
    defenders."""
    outcome_cols = [
        "defender_safe_intercept_rate", "attacker_soldier_caught_rate", "unsafe_intercept_rate",
        "attacker_environmental_failure_rate", "defender_environmental_failure_rate", "timeout_rate",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, is_scripted in zip(axes, (True, False)):
        cond_mask = summary["attacker_condition"] == "scripted" if is_scripted else summary["attacker_condition"] != "scripted"
        data = []
        for level in LEVELS:
            ldf = summary[(summary["level"] == level) & cond_mask]
            data.append([(ldf[c] * ldf["n"]).sum() / ldf["n"].sum() for c in outcome_cols])
        data = np.array(data).T  # (outcomes, levels)
        bottom = np.zeros(len(LEVELS))
        for row, label in zip(data, outcome_cols):
            ax.bar(LEVELS, row, bottom=bottom, label=label.replace("_rate", ""))
            bottom += row
        ax.set_title("scripted attacker" if is_scripted else "Adaptive Attacker v1 (all seeds)")
        ax.set_ylabel("outcome fraction")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Fig 4: Outcome decomposition by clutter level")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_outcomes_by_level.png", dpi=150)
    plt.close(fig)


def fig5_seen_vs_unseen_generalization(degradation: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    """Fig 5: mean delta_defender_success, seen-in-training vs unseen
    defender families, per learned-attacker seed."""
    family_to_seen = summary.drop_duplicates("defender_family").set_index("defender_family")["seen_in_training"]
    degradation = degradation.copy()
    degradation["seen_in_training"] = degradation["defender_family"].map(family_to_seen)
    x = np.arange(len(ATTACKER_SEEDS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, seen in enumerate((True, False)):
        vals = []
        for seed in ATTACKER_SEEDS:
            cond = f"learned_seed_{seed}"
            sub = degradation[(degradation.attacker_condition == cond) & (degradation.seen_in_training == seen)]
            vals.append(sub["delta_defender_success"].mean())
        ax.bar(x + (i - 0.5) * width, vals, width, label=("seen (training pool)" if seen else "unseen (withheld)"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in ATTACKER_SEEDS])
    ax.set_ylabel("mean delta_defender_success")
    ax.set_title("Fig 5: Seen vs. unseen defender generalization")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_seen_vs_unseen_generalization.png", dpi=150)
    plt.close(fig)


def fig6_attacker_seed_variability(degradation: pd.DataFrame, out_dir: Path):
    """Fig 6: box/scatter of delta_defender_success across the 3 learned
    seeds, one point per defender family, showing spread."""
    families = sorted(degradation["defender_family"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [degradation[degradation.defender_family == fam]["delta_defender_success"].values for fam in families]
    ax.boxplot(data, tick_labels=families, showmeans=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("delta_defender_success across 3 learned-attacker seeds")
    ax.set_title("Fig 6: Attacker seed-to-seed variability in defender degradation")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_attacker_seed_variability.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(EVAL_DIR / "summary.csv")
    degradation = pd.read_csv(EVAL_DIR / "degradation_matrix.csv")

    fig1_scripted_vs_learned_success(summary, OUT_DIR)
    fig2_degradation_by_family(degradation, OUT_DIR)
    fig3_attacker_target_reach_rate(summary, OUT_DIR)
    fig4_outcomes_by_level(summary, OUT_DIR)
    fig5_seen_vs_unseen_generalization(degradation, summary, OUT_DIR)
    fig6_attacker_seed_variability(degradation, OUT_DIR)
    print(f"wrote figures 1-6 to {OUT_DIR}")


if __name__ == "__main__":
    main()
