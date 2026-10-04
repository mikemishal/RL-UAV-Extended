"""Phase 17: publication figures for the canonical C0 LR-PPO-semantics
evaluation. Pure post-processing of episodes.csv / statistics outputs --
no new episodes run here. Uses 95% bootstrap CIs already computed by
defender_adaptation_canonical_c0_statistics.py; y-axes for rate plots are
always the full [0,1] range (never truncated)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.defender_adaptation_evaluation import ARCHITECTURES, EVAL_LEVELS as C1C3_LEVELS
from experiments.defender_adaptation_canonical_c0_evaluation import OUT_DIR
from experiments.defender_adaptation_canonical_c0_statistics import (
    load_canonical_c0_episodes, load_c1c3_episodes, c0_success_matrix, _success_matrix,
)

FIGURES_DIR = OUT_DIR / "figures"
ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}
GEN_LABELS = {"d0": "D0", "d0_plus": "D0+", "d1": "D1"}
GEN_COLORS = {"d0": "#888888", "d0_plus": "#4C72B0", "d1": "#DD8452"}


def fig1_a0_generations(df_c0: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        means = [float(c0_success_matrix(df_c0, arch, gen, "A0").mean()) for gen in ("d0", "d0_plus", "d1")]
        ax.bar(["D0", "D0+", "D1"], means, color=[GEN_COLORS[g] for g in ("d0", "d0_plus", "d1")])
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate vs A0 (canonical C0)")
    fig.suptitle("Canonical C0: D0/D0+/D1 vs original scripted hostile A0")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_canonical_c0_a0_generations.png", dpi=150)
    plt.close(fig)


def fig2_a1_unseen_generations(df_c0: pd.DataFrame, out_dir: Path, primary_path: Path):
    primary = {r["architecture"]: r for r in json.loads(primary_path.read_text())}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        d0 = float(c0_success_matrix(df_c0, arch, "d0", "A1-unseen-203").mean())
        d0plus = primary[arch]["success_b_mean"]
        d1 = primary[arch]["success_a_mean"]
        ax.bar(["D0", "D0+", "D1"], [d0, d0plus, d1], color=[GEN_COLORS[g] for g in ("d0", "d0_plus", "d1")])
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate vs A1-unseen-203 (canonical C0)")
    fig.suptitle("Canonical C0: D0/D0+/D1 vs held-out learned attacker")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_canonical_c0_a1_unseen_generations.png", dpi=150)
    plt.close(fig)


def fig3_adaptation_effect_by_condition(d1_d0plus_path: Path, out_dir: Path):
    df = pd.read_csv(d1_d0plus_path)
    conditions = ["C0_canonical"] + list(C1C3_LEVELS)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(conditions))
    for i, arch_label in enumerate(df["architecture"].unique()):
        sub = df[df.architecture == arch_label].set_index("condition").reindex(conditions)
        yerr = np.vstack([sub["diff"] - sub["ci_low"], sub["ci_high"] - sub["diff"]])
        ax.bar(x + i * width, sub["diff"], width, yerr=yerr, capsize=3, label=arch_label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("D1-D0+ (vs A1-unseen-203)")
    ax.set_title("Adaptation effect: canonical C0 vs clutter C1-C3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_adaptation_effect_by_condition.png", dpi=150)
    plt.close(fig)


def fig4_a0_vs_a1_unseen_across_conditions(df_c0: pd.DataFrame, df_c1c3: pd.DataFrame, out_dir: Path):
    conditions = ["C0"] + list(C1C3_LEVELS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        a0_vals, a1_vals = [], []
        for cond in conditions:
            if cond == "C0":
                a0_vals.append(float(c0_success_matrix(df_c0, arch, "d1", "A0").mean()))
                a1_vals.append(float(c0_success_matrix(df_c0, arch, "d1", "A1-unseen-203").mean()))
            else:
                seeds_sorted = sorted(set(df_c1c3.eval_seed))
                a0_vals.append(float(_success_matrix(df_c1c3, arch, "d1", "A0", cond, seeds_sorted).mean()))
                a1_vals.append(float(_success_matrix(df_c1c3, arch, "d1", "A1-unseen-203", cond, seeds_sorted).mean()))
        x = np.arange(len(conditions))
        ax.bar(x - 0.2, a0_vals, 0.4, label="A0", color="#4C72B0")
        ax.bar(x + 0.2, a1_vals, 0.4, label="A1-unseen-203", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate (D1)")
    axes[0].legend()
    fig.suptitle("A0 vs A1-unseen-203 across C0(canonical)-C3")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_a0_vs_a1unseen_across_conditions.png", dpi=150)
    plt.close(fig)


def fig5_ppo_kalman_seed_effects(primary_path: Path, out_dir: Path):
    primary = {r["architecture"]: r for r in json.loads(primary_path.read_text())}
    r = primary["ppo_kalman_clutter_v1"]
    seeds = sorted(int(s) for s in r["per_seed_diffs"].keys())
    diffs = [r["per_seed_diffs"][str(s)] if str(s) in r["per_seed_diffs"] else r["per_seed_diffs"][s] for s in seeds]
    cis = r["per_seed_paired_bootstrap"]
    lo = [cis[str(s)]["ci_low"] if str(s) in cis else cis[s]["ci_low"] for s in seeds]
    hi = [cis[str(s)]["ci_high"] if str(s) in cis else cis[s]["ci_high"] for s in seeds]
    yerr = np.vstack([np.array(diffs) - np.array(lo), np.array(hi) - np.array(diffs)])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([str(s) for s in seeds], diffs, yerr=yerr, capsize=4, color="#55A868")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PPO-Kalman-C defender training seed")
    ax.set_ylabel("D1-D0+ (vs A1-unseen-203), canonical C0")
    ax.set_title("PPO-Kalman-C adaptation effect by defender seed (canonical C0)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_ppo_kalman_seed_effects.png", dpi=150)
    plt.close(fig)


def fig6_ppo_c_canonical_vs_legacy(legacy_cmp_path: Path, out_dir: Path):
    if not legacy_cmp_path.exists():
        return
    cmp = json.loads(legacy_cmp_path.read_text())
    canonical = cmp["canonical_NON_LEGACY"]
    legacy = cmp["legacy_NON_CANONICAL"]
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Canonical C0\n(this task)", "Legacy C0\n(non-canonical,\nhistorical)"]
    diffs = [canonical["observed_diff"], legacy["observed_diff"]]
    lo = [canonical["ci_low"], legacy["ci_low"]]
    hi = [canonical["ci_high"], legacy["ci_high"]]
    yerr = np.vstack([np.array(diffs) - np.array(lo), np.array(hi) - np.array(diffs)])
    ax.bar(labels, diffs, yerr=yerr, capsize=4, color=["#4C72B0", "#888888"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("PPO-C: D1-D0+ (vs A1-unseen-203)")
    ax.set_title("PPO-C: canonical vs legacy (historical) C0 comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_ppo_c_canonical_vs_legacy.png", dpi=150)
    plt.close(fig)


def fig7_lrppo_residual_usage(residual_summary_path: Path, out_dir: Path):
    df = pd.read_csv(residual_summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for gen, color in GEN_COLORS.items():
        sub = df[df.generation == gen]
        axes[0].plot(sub["attacker_type"], sub["mean_residual_angle_deg"], marker="o", label=GEN_LABELS[gen], color=color)
        axes[1].plot(sub["attacker_type"], sub["fraction_near_30deg"], marker="o", label=GEN_LABELS[gen], color=color)
    axes[0].set_ylabel("Mean residual angle (deg)")
    axes[0].set_title("LR-PPO-C mean residual angle (canonical C0)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("Fraction within 2 deg of 30 deg bound")
    axes[1].set_title("LR-PPO-C near-bound fraction (canonical C0)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_lrppo_residual_usage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_c0 = load_canonical_c0_episodes()
    df_c1c3 = load_c1c3_episodes()
    fig1_a0_generations(df_c0, FIGURES_DIR)
    fig2_a1_unseen_generations(df_c0, FIGURES_DIR, OUT_DIR / "canonical_primary_endpoint_unseen.json")
    fig3_adaptation_effect_by_condition(OUT_DIR / "d1_minus_d0plus_by_condition.csv", FIGURES_DIR)
    fig4_a0_vs_a1_unseen_across_conditions(df_c0, df_c1c3, FIGURES_DIR)
    fig5_ppo_kalman_seed_effects(OUT_DIR / "canonical_primary_endpoint_unseen.json", FIGURES_DIR)
    fig6_ppo_c_canonical_vs_legacy(OUT_DIR / "legacy_vs_canonical_ppo_c.json", FIGURES_DIR)
    fig7_lrppo_residual_usage(OUT_DIR / "lrppo_residual_canonical_summary.csv", FIGURES_DIR)
    print(f"wrote figures to {FIGURES_DIR}")
