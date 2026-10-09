"""Consolidated meeting figures for the 2026-10 professor meeting.

Pure post-processing of ALREADY-GENERATED results. No training, no new
evaluation episodes are run here. Every plotted number is read from an
existing summary CSV/JSON (or, for Figure 2's seed-level error bars, from
the frozen `episodes.csv` of the canonical C0 evaluation via a simple
groupby aggregation -- not a re-run of the evaluation).

Inputs used (see README.md for the full per-figure mapping):
  results/defender_adaptation_canonical_c0_evaluation/d1_minus_d0plus_by_condition.csv
  results/defender_adaptation_canonical_c0_evaluation/c0_vs_clutter_table.csv
  results/defender_adaptation_canonical_c0_evaluation/episodes.csv
  results/defender_adaptation_canonical_c0_evaluation/lrppo_residual_canonical_summary.csv
  results/defender_adaptation_canonical_c0_evaluation/four_condition_decomposition.csv
  results/clutter_retraining_evaluation/recovery_deltas.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CANON = PROJECT_ROOT / "results" / "defender_adaptation_canonical_c0_evaluation"
RETRAIN = PROJECT_ROOT / "results" / "clutter_retraining_evaluation"
OUT_DIR = Path(__file__).resolve().parent

ARCHES = ["PPO-C", "PPO-Kalman-C", "LR-PPO-C"]
ARCH_KEY = {"PPO-C": "ppo_clutter_v1", "PPO-Kalman-C": "ppo_kalman_clutter_v1", "LR-PPO-C": "lrppo_clutter_v1"}
ARCH_COLORS = {"PPO-C": "#4C72B0", "PPO-Kalman-C": "#55A868", "LR-PPO-C": "#C44E52"}
GEN_LABELS = ["D0", "D0+", "D1"]
GEN_COLORS = {"D0": "#888888", "D0+": "#4C72B0", "D1": "#DD8452"}
CONDITIONS = ["C0_canonical", "C1", "C2", "C3"]
CONDITION_LABELS = ["C0", "C1", "C2", "C3"]

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
})


def _save(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: adaptation effect (D1-D0+ vs A1-unseen-203) across C0-C3
# ---------------------------------------------------------------------------

def fig1():
    df = pd.read_csv(CANON / "d1_minus_d0plus_by_condition.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arch in ARCHES:
        sub = df[df.architecture == arch].set_index("condition").loc[CONDITIONS]
        y = sub["diff"].to_numpy() * 100
        lo = (sub["diff"] - sub["ci_low"]).to_numpy() * 100
        hi = (sub["ci_high"] - sub["diff"]).to_numpy() * 100
        ax.errorbar(CONDITION_LABELS, y, yerr=[lo, hi], marker="o", markersize=8,
                    linewidth=2.2, capsize=5, color=ARCH_COLORS[arch], label=arch)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("\u0394 Safe-interception rate, D1\u2212D0+ (pp)")
    ax.set_title("Adversarial-adaptation effect vs held-out attacker\n(A1-unseen-203), across C0\u2013C3")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig1_adaptation_effect_c0_c3")

    out = df[df.condition.isin(CONDITIONS)].copy()
    out["condition"] = out["condition"].replace({"C0_canonical": "C0"})
    out["diff_pp"] = out["diff"] * 100
    out["ci_low_pp"] = out["ci_low"] * 100
    out["ci_high_pp"] = out["ci_high"] * 100
    out[["architecture", "condition", "diff_pp", "ci_low_pp", "ci_high_pp"]].to_csv(
        OUT_DIR / "fig1_source.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 2: A0 retention vs A1-unseen generalization (canonical C0, D0/D0+/D1)
# ---------------------------------------------------------------------------

def _seed_stats(df_ep: pd.DataFrame, arch_key: str, generation: str, attacker_type: str):
    sub = df_ep[(df_ep.architecture == arch_key) & (df_ep.generation == generation)
                & (df_ep.attacker_type == attacker_type) & (df_ep.level == "C0")]
    per_seed = sub.groupby("defender_seed")["defender_safe_intercept"].mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1))


def fig2():
    df_ep = pd.read_csv(CANON / "episodes.csv")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, attacker_type, title in zip(axes, ["A0", "A1-unseen-203"],
                                        ["Original scripted attacker (A0)", "Held-out learned attacker (A1-unseen-203)"]):
        for arch in ARCHES:
            means, stds = [], []
            for gen in ["d0", "d0_plus", "d1"]:
                m, s = _seed_stats(df_ep, ARCH_KEY[arch], gen, attacker_type)
                means.append(m)
                stds.append(s)
                rows.append({"architecture": arch, "attacker_type": attacker_type, "generation": gen,
                             "mean_safe_intercept_rate": m, "seed_sd": s})
            ax.errorbar(GEN_LABELS, means, yerr=stds, marker="o", markersize=8, linewidth=2.2,
                        capsize=5, color=ARCH_COLORS[arch], label=arch)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Safe-interception rate (canonical C0)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("A0 retention vs A1-unseen generalization across adaptation generations")
    fig.tight_layout()
    _save(fig, "fig2_a0_retention_vs_a1_generalization")
    pd.DataFrame(rows).to_csv(OUT_DIR / "fig2_source.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 3: LR-PPO clutter strength vs adaptation limitation (3 panels)
# ---------------------------------------------------------------------------

def fig3():
    recovery = pd.read_csv(RETRAIN / "recovery_deltas.csv")
    lrppo_recovery = recovery[recovery.family == "LR-PPO"].set_index("level").loc[["C1", "C2", "C3"]]

    diffs = pd.read_csv(CANON / "d1_minus_d0plus_by_condition.csv")
    lrppo_diffs = diffs[diffs.architecture == "LR-PPO-C"].set_index("condition").loc[CONDITIONS]

    residual = pd.read_csv(CANON / "lrppo_residual_canonical_summary.csv")
    residual_by_gen = residual.groupby("generation")["mean_residual_angle_deg"].agg(["mean", "std"])
    residual_by_gen = residual_by_gen.loc[["d0", "d0_plus", "d1"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    bar_colors = ["#55A868" if lvl == "C3" else "#9BBF9B" for lvl in lrppo_recovery.index]
    bars = ax.bar(lrppo_recovery.index, lrppo_recovery["safe_intercept_rate_delta"] * 100, color=bar_colors)
    ax.axhline(0, color="black", linewidth=1)
    c3_val = float(lrppo_recovery.loc["C3", "safe_intercept_rate_delta"] * 100)
    ax.annotate(f"+{c3_val:.1f} pp", xy=(2, c3_val), xytext=(2, c3_val - 1.4),
                ha="center", va="top", fontsize=12, fontweight="bold")
    ax.set_title("A. Obstacle-aware retraining\nrecovery (LR-PPO)")
    ax.set_ylim(top=max(lrppo_recovery["safe_intercept_rate_delta"] * 100) + 2.5)
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("\u0394 Safe-interception rate,\ntrained\u2212frozen (pp)")

    ax = axes[1]
    y = lrppo_diffs["diff"].to_numpy() * 100
    lo = (lrppo_diffs["diff"] - lrppo_diffs["ci_low"]).to_numpy() * 100
    hi = (lrppo_diffs["ci_high"] - lrppo_diffs["diff"]).to_numpy() * 100
    ax.errorbar(CONDITION_LABELS, y, yerr=[lo, hi], marker="o", markersize=8, linewidth=2.2,
                capsize=5, color=ARCH_COLORS["LR-PPO-C"])
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title("B. Adversarial-adaptation effect\n(D1\u2212D0+ vs A1-unseen-203)")
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("\u0394 Safe-interception rate (pp)")

    ax = axes[2]
    ax.errorbar(GEN_LABELS, residual_by_gen["mean"], yerr=residual_by_gen["std"], marker="o",
                markersize=8, linewidth=2.2, capsize=5, color=ARCH_COLORS["LR-PPO-C"])
    ax.set_title("C. Residual steering-correction\nmagnitude (canonical C0)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean residual angle (deg)")

    fig.suptitle("LR-PPO-C: clutter-navigation strength vs. adversarial-adaptation limitation", y=1.04)
    fig.tight_layout()
    _save(fig, "fig3_lrppo_clutter_vs_adaptation")

    part_a = lrppo_recovery.reset_index()[["level", "frozen_safe_intercept_rate", "trained_safe_intercept_rate", "safe_intercept_rate_delta"]]
    part_a.insert(0, "part", "A_recovery")
    part_b = lrppo_diffs.reset_index()[["condition", "diff", "ci_low", "ci_high"]]
    part_b.insert(0, "part", "B_adaptation_effect")
    part_c = residual_by_gen.reset_index()
    part_c.insert(0, "part", "C_residual_angle")
    with open(OUT_DIR / "fig3_source.csv", "w") as f:
        f.write("# Part A: obstacle-aware retraining recovery (frozen vs trained), LR-PPO, C1-C3\n")
        part_a.to_csv(f, index=False)
        f.write("\n# Part B: D1-D0+ adaptation effect vs A1-unseen-203, LR-PPO-C, C0-C3\n")
        part_b.to_csv(f, index=False)
        f.write("\n# Part C: mean residual-correction angle by generation, canonical C0 (avg over 4 attacker conditions)\n")
        part_c.to_csv(f, index=False)


# ---------------------------------------------------------------------------
# Figure 4: PPO-C environment-specific regression
# ---------------------------------------------------------------------------

def fig4():
    df = pd.read_csv(CANON / "d1_minus_d0plus_by_condition.csv")
    sub = df[df.architecture == "PPO-C"].set_index("condition").loc[CONDITIONS]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = sub["diff"].to_numpy() * 100
    lo = (sub["diff"] - sub["ci_low"]).to_numpy() * 100
    hi = (sub["ci_high"] - sub["diff"]).to_numpy() * 100
    bar_colors = ["#C44E52" if c == "C0_canonical" else "#D9A0A2" for c in sub.index]
    ax.bar(CONDITION_LABELS, y, yerr=[lo, hi], capsize=6, color=bar_colors)
    ax.axhline(0, color="black", linewidth=1)
    c0_val = float(sub.loc["C0_canonical", "diff"] * 100)
    c0_lo = float(sub.loc["C0_canonical", "ci_low"] * 100)
    c0_hi = float(sub.loc["C0_canonical", "ci_high"] * 100)
    ax.annotate(f"{c0_val:.1f} pp\n95% CI [{c0_lo:.1f}, {c0_hi:.1f}]", xy=(0, c0_val),
                xytext=(0.15, c0_val - 18), fontsize=11, fontweight="bold")
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("\u0394 Safe-interception rate, D1\u2212D0+ (pp)")
    ax.set_title("PPO-C: adversarial-adaptation effect is\nsignificantly negative only in open-air (C0)")
    fig.tight_layout()
    _save(fig, "fig4_ppoc_c0_regression")

    out = sub.reset_index()
    out["condition"] = out["condition"].replace({"C0_canonical": "C0"})
    out["diff_pp"] = out["diff"] * 100
    out["ci_low_pp"] = out["ci_low"] * 100
    out["ci_high_pp"] = out["ci_high"] * 100
    out[["condition", "diff_pp", "ci_low_pp", "ci_high_pp"]].to_csv(OUT_DIR / "fig4_source.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 5 (optional): four-condition decomposition at D1
# ---------------------------------------------------------------------------

def fig5():
    df = pd.read_csv(CANON / "four_condition_decomposition.csv")
    d1 = df[df.generation == "d1"].copy()
    arch_label_map = {"PPO-C": "PPO-C", "PPO-Kalman-C": "PPO-Kalman-C", "LR-PPO-C": "LR-PPO-C"}
    d1["architecture"] = d1["architecture"]
    conds = ["A0_C0", "A0_clutter", "A1_unseen_C0", "A1_unseen_clutter"]
    cond_labels = ["A0, C0", "A0, clutter", "A1-unseen, C0", "A1-unseen, clutter"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_arch = len(ARCHES)
    n_cond = len(conds)
    width = 0.2
    x = np.arange(n_cond)
    for i, arch in enumerate(ARCHES):
        row = d1[d1.architecture == arch].iloc[0]
        vals = [row[c] * 100 for c in conds]
        ax.bar(x + (i - (n_arch - 1) / 2) * width, vals, width=width, color=ARCH_COLORS[arch], label=arch)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Safe-interception rate (%)")
    ax.set_title("D1 four-condition decomposition: attacker type \u00d7 environment")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig5_four_condition_decomposition")

    out = d1[["architecture"] + conds].copy()
    out.columns = ["architecture"] + cond_labels
    out.to_csv(OUT_DIR / "fig5_source.csv", index=False)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("wrote figures to", OUT_DIR)
