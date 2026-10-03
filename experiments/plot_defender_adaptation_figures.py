"""
Figures for the defender-adaptation held-out evaluation (journal
extension, 38-phase defender-adaptation-EVALUATION sub-task, Phase 35).
Pure post-processing of `episodes.csv` / the statistics module's outputs
-- no new episodes run here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS, OUT_DIR
from experiments.defender_adaptation_statistics import (
    success_matrix, load_episodes, primary_endpoint_d1_vs_d0plus_unseen,
    seen_attacker_analysis, robustness_scores, safety_layer_dependence,
    defender_seed_summary, outcome_decomposition, lr_ppo_residual_analysis,
)

FIGURES_DIR = OUT_DIR / "figures"
ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}
GEN_LABELS = {"d0": "D0", "d0_plus": "D0+", "d1": "D1"}
GEN_COLORS = {"d0": "#888888", "d0_plus": "#4C72B0", "d1": "#DD8452"}


def fig1_unseen_attacker_generations(df: pd.DataFrame, out_dir: Path):
    """Fig 1: D0/D0+/D1 safe-interception rate against A1-unseen-203, per architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        means = [float(np.mean([success_matrix(df, arch, gen, "A1-unseen-203", lv).mean() for lv in EVAL_LEVELS]))
                 for gen in ("d0", "d0_plus", "d1")]
        ax.bar(["D0", "D0+", "D1"], means, color=[GEN_COLORS[g] for g in ("d0", "d0_plus", "d1")])
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate vs A1-unseen-203")
    fig.suptitle("Defender generations vs held-out learned attacker (A1-unseen-203)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_unseen_attacker_generations.png", dpi=150)
    plt.close(fig)


def fig2_adaptation_gain_by_clutter(df: pd.DataFrame, out_dir: Path):
    """Fig 2: D1-D0+ adaptation gain by architecture and clutter level (never pre-averaged)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(EVAL_LEVELS))
    for i, arch in enumerate(ARCHITECTURES):
        gains = [
            float(success_matrix(df, arch, "d1", "A1-unseen-203", lv).mean()
                   - success_matrix(df, arch, "d0_plus", "A1-unseen-203", lv).mean())
            for lv in EVAL_LEVELS
        ]
        ax.bar(x + i * width, gains, width, label=ARCH_LABELS[arch])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(EVAL_LEVELS)
    ax.set_ylabel("Delta_adapt = Success(D1) - Success(D0+) vs A1-unseen-203")
    ax.set_title("Adaptation gain by architecture and clutter density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_adaptation_gain_by_clutter.png", dpi=150)
    plt.close(fig)


def fig3_a0_retention(df: pd.DataFrame, out_dir: Path):
    """Fig 3: scripted-threat (A0) retention across generations."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        means = [float(np.mean([success_matrix(df, arch, gen, "A0", lv).mean() for lv in EVAL_LEVELS]))
                 for gen in ("d0", "d0_plus", "d1")]
        ax.bar(["D0", "D0+", "D1"], means, color=[GEN_COLORS[g] for g in ("d0", "d0_plus", "d1")])
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate vs A0 (scripted)")
    fig.suptitle("Scripted-threat retention across defender generations")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_a0_retention.png", dpi=150)
    plt.close(fig)


def fig4_seen_vs_unseen(df: pd.DataFrame, out_dir: Path):
    """Fig 4: A1-seen (201/202 aggregate) vs A1-unseen (203), D1 generation, by architecture."""
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(ARCHITECTURES))
    seen_means, unseen_means = [], []
    for arch in ARCHITECTURES:
        seen = float(np.mean([success_matrix(df, arch, "d1", at, lv).mean()
                               for at in ("A1-seen-201", "A1-seen-202") for lv in EVAL_LEVELS]))
        unseen = float(np.mean([success_matrix(df, arch, "d1", "A1-unseen-203", lv).mean() for lv in EVAL_LEVELS]))
        seen_means.append(seen)
        unseen_means.append(unseen)
    ax.bar(x - width / 2, seen_means, width, label="A1-seen (201/202 aggregate)")
    ax.bar(x + width / 2, unseen_means, width, label="A1-unseen-203")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHITECTURES])
    ax.set_ylabel("D1 safe-interception rate")
    ax.set_title("D1: seen vs held-out learned attacker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_seen_vs_unseen.png", dpi=150)
    plt.close(fig)


def fig5_balanced_robustness(df: pd.DataFrame, out_dir: Path):
    robustness = robustness_scores(df)
    agg = robustness[robustness.level == "aggregate_C1C3"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(ARCHITECTURES))
    for i, gen in enumerate(("d0", "d0_plus", "d1")):
        vals = [float(agg[(agg.architecture == arch) & (agg.generation == gen)]["BalancedRobustness"].iloc[0]) for arch in ARCHITECTURES]
        ax.bar(x + i * width, vals, width, label=GEN_LABELS[gen], color=GEN_COLORS[gen])
    ax.set_xticks(x + width)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHITECTURES])
    ax.set_ylabel("BalancedRobustness = (A0 + A1_seen_mean + A1_unseen) / 3")
    ax.set_title("Balanced robustness (equal-weight; descriptive only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_balanced_robustness.png", dpi=150)
    plt.close(fig)


def fig6_defender_seed_variability(df: pd.DataFrame, out_dir: Path):
    summary = defender_seed_summary(df)
    agg = summary[summary.attacker_type == "A1-unseen-203"].groupby(["architecture", "generation"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        sub = agg[agg.architecture == arch]
        for gen in ("d0", "d0_plus", "d1"):
            row = sub[sub.generation == gen]
            if row.empty:
                continue
            seed_cols = [c for c in row.columns if c.startswith("seed_")]
            seed_vals = row[seed_cols].values.flatten()
            ax.scatter([GEN_LABELS[gen]] * len(seed_vals), seed_vals, alpha=0.7)
        ax.set_title(ARCH_LABELS[arch])
    axes[0].set_ylabel("Per-seed safe-interception rate vs A1-unseen-203")
    fig.suptitle("Defender-seed variability (3 seeds per architecture)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_defender_seed_variability.png", dpi=150)
    plt.close(fig)


def fig7_safety_intervention_dependence(df: pd.DataFrame, out_dir: Path):
    safety = safety_layer_dependence(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(ARCHITECTURES))
    for i, gen in enumerate(("d0", "d0_plus", "d1")):
        vals = []
        for arch in ARCHITECTURES:
            sub = safety[(safety.architecture == arch) & (safety.generation == gen)]
            vals.append(float(sub["mean_intervention_fraction"].mean()) if not sub.empty else 0.0)
        ax.bar(x + i * width, vals, width, label=GEN_LABELS[gen], color=GEN_COLORS[gen])
    ax.set_xticks(x + width)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHITECTURES])
    ax.set_ylabel("Mean safety-intervention fraction")
    ax.set_title("Safety-layer dependence across generations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_safety_intervention_dependence.png", dpi=150)
    plt.close(fig)


def fig8_outcome_decomposition(df: pd.DataFrame, out_dir: Path):
    decomp = outcome_decomposition(df)
    agg = decomp[decomp.attacker_type == "A1-unseen-203"].groupby(["architecture", "generation"]).mean(numeric_only=True).reset_index()
    outcome_cols = ["defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
                    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
                    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        sub = agg[agg.architecture == arch].set_index("generation").reindex(["d0", "d0_plus", "d1"])
        bottom = np.zeros(3)
        for col in outcome_cols:
            vals = sub[col].values
            ax.bar(["D0", "D0+", "D1"], vals, bottom=bottom, label=col)
            bottom += vals
        ax.set_title(ARCH_LABELS[arch])
    axes[0].set_ylabel("Outcome fraction vs A1-unseen-203")
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Outcome decomposition vs A1-unseen-203")
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_outcome_decomposition.png", dpi=150)
    plt.close(fig)


def fig10_lr_ppo_residual_distribution(df: pd.DataFrame, out_dir: Path):
    residual = lr_ppo_residual_analysis(df)
    if residual.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    gens = ["d0", "d0_plus", "d1"]
    x = np.arange(len(gens))
    width = 0.2
    for i, attacker_type in enumerate(("A0", "A1-seen-201", "A1-seen-202", "A1-unseen-203")):
        vals = [float(residual[(residual.generation == g) & (residual.attacker_type == attacker_type)]["mean_residual_angle_deg"].mean()) for g in gens]
        ax.bar(x + i * width, vals, width, label=attacker_type)
    ax.axhline(30.0, color="red", linestyle="--", linewidth=1, label="30 deg bound")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([GEN_LABELS[g] for g in gens])
    ax.set_ylabel("Mean residual angle (deg)")
    ax.set_title("LR-PPO residual-angle distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig10_lr_ppo_residual_distribution.png", dpi=150)
    plt.close(fig)


def generate_all_figures(df: pd.DataFrame | None = None) -> list[Path]:
    df = df if df is not None else load_episodes()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fns = [
        fig1_unseen_attacker_generations, fig2_adaptation_gain_by_clutter, fig3_a0_retention,
        fig4_seen_vs_unseen, fig5_balanced_robustness, fig6_defender_seed_variability,
        fig7_safety_intervention_dependence, fig8_outcome_decomposition, fig10_lr_ppo_residual_distribution,
    ]
    paths = []
    for fn in fns:
        fn(df, FIGURES_DIR)
        paths.append(FIGURES_DIR)
    return sorted(FIGURES_DIR.glob("*.png"))


if __name__ == "__main__":
    paths = generate_all_figures()
    for p in paths:
        print(f"wrote {p}")
