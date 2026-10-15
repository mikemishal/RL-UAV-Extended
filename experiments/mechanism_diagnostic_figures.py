"""Phase 9/10 figures for the architecture-mechanism diagnostic. Pure
post-processing of mechanism_diagnostic*.py outputs -- no new rollouts."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.mechanism_diagnostic import OUT_DIR
from experiments.mechanism_diagnostic_stats import (
    _parse_tuple_col, select_matched_cases, lead_error_a0_vs_a1, residual_bin_analysis,
    RESIDUAL_BIN_LABELS,
)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARCH_COLORS = {"PPO-C": "#4C72B0", "PPO-Kalman-C": "#55A868", "LR-PPO-C": "#C44E52"}
GEN_COLORS = {"d0": "#888888", "d0_plus": "#4C72B0", "d1": "#DD8452"}

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 150,
})


def _save(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 10.1: Lead prediction error, A0 vs A1
# ---------------------------------------------------------------------------

def fig_lead_error_a0_vs_a1():
    df = pd.read_csv(OUT_DIR / "lead_error_a0_vs_a1.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for level, marker in zip(sorted(df.level.unique()), ("o", "s")):
        sub = df[df.level == level].set_index("generation").loc[["d0", "d0_plus", "d1"]]
        y = sub["diff_a1_minus_a0"].to_numpy()
        lo = (sub["diff_a1_minus_a0"] - sub["ci_low"]).to_numpy()
        hi = (sub["ci_high"] - sub["diff_a1_minus_a0"]).to_numpy()
        ax.errorbar(["D0", "D0+", "D1"], y, yerr=[lo, hi], marker=marker, markersize=9,
                    linewidth=2, capsize=5, label=level)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("\u0394 mean angular Lead error,\nA1-unseen \u2212 A0 (deg)")
    ax.set_xlabel("Generation")
    ax.set_title("LR-PPO: Lead-prediction error under adaptive vs scripted attacker")
    ax.legend(frameon=False, title="Level")
    fig.tight_layout()
    _save(fig, "fig_lead_error_a0_vs_a1")


# ---------------------------------------------------------------------------
# Phase 10.2 & 10.3: Lead error vs residual magnitude / success-failure
# ---------------------------------------------------------------------------

def fig_lead_error_vs_residual():
    df = pd.read_csv(OUT_DIR / "residual_bin_analysis_d1.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.bar(df["residual_bin_deg"], df["mean_lead_error_deg"], color="#C44E52", alpha=0.85)
    ax1.set_xlabel("Residual-correction magnitude bin (deg)")
    ax1.set_ylabel("Mean angular Lead error (deg)", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(df["residual_bin_deg"], df["frequency"] * 100, color="#4C72B0", marker="o", linewidth=2)
    ax2.set_ylabel("Frequency of bin (%)", color="#4C72B0")
    ax1.set_title("LR-PPO D1: Lead error and step frequency by residual-magnitude bin")
    fig.tight_layout()
    _save(fig, "fig_lead_error_vs_residual")


def fig_lead_error_vs_success():
    df = pd.read_csv(OUT_DIR / "lead_error_success_vs_failure.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["mean_lead_error_success"], width, label="Success (intercepted)", color="#55A868")
    ax.bar(x + width / 2, df["mean_lead_error_failure"], width, label="Failure", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.level}\n{r.generation}\n{r.attacker_type}" for r in df.itertuples()], fontsize=9)
    ax.set_ylabel("Mean angular Lead error (deg)")
    ax.set_title("LR-PPO: Lead error by episode outcome")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig_lead_error_vs_success")


# ---------------------------------------------------------------------------
# Phase 10.4: 15/30/45/60 authority diagnostic
# ---------------------------------------------------------------------------

def fig_authority_diagnostic():
    path = OUT_DIR / "lrppo_authority_diagnostic_episodes.csv"
    if not path.exists():
        print("skipping fig_authority_diagnostic: authority episodes not yet generated")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, level in zip(axes, ("C0", "C3")):
        sub = df[df.level == level].groupby("inference_residual_max_angle_deg")["defender_safe_intercept"]
        means = sub.mean()
        sds = sub.std()
        ns = sub.count()
        se = sds / np.sqrt(ns)
        colors = ["#DD8452" if a == 30.0 else "#4C72B0" for a in means.index]
        ax.bar([str(int(a)) for a in means.index], means.values, yerr=1.96 * se.values, capsize=6, color=colors)
        ax.set_title(f"{level}")
        ax.set_xlabel("Inference-time residual authority (deg)")
    axes[0].set_ylabel("Safe-interception rate")
    fig.suptitle("LR-PPO D1 (trained at 30\u00b0): inference-time authority diagnostic\n"
                 "(orange = canonical training-time bound; NOT retrained policies)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_authority_diagnostic")


# ---------------------------------------------------------------------------
# Phase 10.5 / 10.6: behavioral metric comparisons
# ---------------------------------------------------------------------------

def fig_behavioral_comparison(tag: str, title: str, out_name: str):
    path = OUT_DIR / f"{tag}_behavioral_summary.csv"
    if not path.exists():
        print(f"skipping {out_name}: {path} not found")
        return
    df = pd.read_csv(path)
    metrics = ["path_curvature_deg", "n_action_reversals", "mean_action_magnitude", "closing_velocity_trend"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, metric in zip(axes, metrics):
        grouped = df.groupby(["level", "generation"])[metric].mean().unstack("generation")
        grouped = grouped[["d0_plus", "d1"]]
        grouped.plot(kind="bar", ax=ax, color=[GEN_COLORS["d0_plus"], GEN_COLORS["d1"]], legend=False)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    axes[0].legend(["D0+", "D1"], frameon=False)
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    _save(fig, out_name)


# ---------------------------------------------------------------------------
# Phase 9: representative matched trajectory figures
# ---------------------------------------------------------------------------

def _plot_trajectory(ax, steps_sub: pd.DataFrame, title: str):
    def_pos = _parse_tuple_col(steps_sub["defender_pos"])
    att_pos = _parse_tuple_col(steps_sub["attacker_pos_truth"])
    ax.plot(def_pos[:, 0], def_pos[:, 1], color="#4C72B0", linewidth=2, label="Defender")
    ax.plot(att_pos[:, 0], att_pos[:, 1], color="#C44E52", linewidth=2, label="Attacker")
    ax.scatter(*def_pos[0, :2], color="#4C72B0", marker="o", s=60, zorder=5)
    ax.scatter(*att_pos[0, :2], color="#C44E52", marker="o", s=60, zorder=5)
    ax.scatter(*def_pos[-1, :2], color="#4C72B0", marker="X", s=90, zorder=5)
    ax.scatter(*att_pos[-1, :2], color="#C44E52", marker="X", s=90, zorder=5)
    detected = steps_sub[steps_sub.enemy_detected]
    if len(detected):
        dp = _parse_tuple_col(detected["defender_pos"]).reshape(-1, 3)[0]
        ax.scatter(dp[0], dp[1], facecolors="none", edgecolors="black", s=150, linewidths=2, zorder=6, label="Detection")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def fig_matched_trajectories(tag: str, arch_filter: str, attacker_type: str, level: str, out_name: str, title: str,
                              category: str, gen_a: str = "d0_plus", gen_b: str = "d1"):
    steps_path = OUT_DIR / f"{tag}_steps.csv.gz"
    ep_path = OUT_DIR / f"{tag}_episodes.csv"
    if not steps_path.exists():
        print(f"skipping {out_name}: {steps_path} not found")
        return
    steps = pd.read_csv(steps_path)
    episodes = pd.read_csv(ep_path)
    cases = select_matched_cases(episodes, arch_filter, attacker_type, level, gen_a=gen_a, gen_b=gen_b)

    case = cases.get(category)
    if case is None:
        print(f"skipping {out_name}: no {category} case found")
        return
    ds, es = case["defender_seed"], case["eval_seed"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, gen in zip(axes, (gen_a, gen_b)):
        sub = steps[(steps.defender_seed == ds) & (steps.eval_seed == es) & (steps.generation == gen)
                    & (steps.attacker_type == attacker_type) & (steps.level == level)]
        _plot_trajectory(ax, sub, f"{gen.upper()} (seed {ds}, eval_seed {es})")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, out_name)


def fig_lrppo_lead_error_cases():
    """Phase 9 items 3/4: LR-PPO D1 failure with high Lead error/large
    residual, and (optional) a successful case with low Lead error --
    selected OBJECTIVELY (extreme/typical by mean_angular_lead_error_deg
    within its outcome category), not hand-picked."""
    steps_path = OUT_DIR / "lrppo_lead_steps.csv.gz"
    ep_path = OUT_DIR / "lrppo_lead_episodes.csv"
    if not steps_path.exists():
        print("skipping fig_lrppo_lead_error_cases: lrppo_lead_steps.csv.gz not found")
        return
    steps = pd.read_csv(steps_path)
    episodes = pd.read_csv(ep_path)
    d1 = episodes[(episodes.generation == "d1") & (episodes.attacker_type == "A1-unseen-203")
                  & (episodes.level == "C0")].dropna(subset=["mean_angular_lead_error_deg"])
    failures = d1[d1.defender_safe_intercept == 0]
    successes = d1[d1.defender_safe_intercept == 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    if len(failures):
        worst = failures.loc[failures["mean_angular_lead_error_deg"].idxmax()]
        sub = steps[(steps.defender_seed == worst.defender_seed) & (steps.eval_seed == worst.eval_seed)
                    & (steps.generation == "d1") & (steps.attacker_type == "A1-unseen-203") & (steps.level == "C0")]
        _plot_trajectory(axes[0], sub, f"Failure, high Lead error\n(mean={worst.mean_angular_lead_error_deg:.1f}\u00b0, "
                                        f"seed {worst.defender_seed}, eval_seed {worst.eval_seed})")
    if len(successes):
        best = successes.loc[successes["mean_angular_lead_error_deg"].idxmin()]
        sub = steps[(steps.defender_seed == best.defender_seed) & (steps.eval_seed == best.eval_seed)
                    & (steps.generation == "d1") & (steps.attacker_type == "A1-unseen-203") & (steps.level == "C0")]
        _plot_trajectory(axes[1], sub, f"Success, low Lead error\n(mean={best.mean_angular_lead_error_deg:.1f}\u00b0, "
                                        f"seed {best.defender_seed}, eval_seed {best.eval_seed})")
    fig.suptitle("LR-PPO D1 vs A1-unseen-203 (C0): Lead-error extremes", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_traj_lrppo_lead_error_cases")


if __name__ == "__main__":
    fig_lead_error_a0_vs_a1()
    fig_lead_error_vs_residual()
    fig_lead_error_vs_success()
    fig_authority_diagnostic()
    fig_behavioral_comparison("ppo_kalman_mechanism", "PPO-Kalman-C: D0+ vs D1 response/smoothness", "fig_ppo_kalman_behavior")
    fig_behavioral_comparison("ppo_c_regression", "PPO-C: D0+ vs D1 behavioral differences", "fig_ppo_c_behavior")

    fig_matched_trajectories("ppo_kalman_mechanism", "ppo_kalman_clutter_v1", "A1-unseen-203", "C0",
                              "fig_traj_ppo_kalman_d0plus_fail_d1_success",
                              "PPO-Kalman-C: D0+ failure vs D1 success (matched seed)",
                              category="C_gen_a_fails_gen_b_succeeds")
    fig_matched_trajectories("ppo_c_regression", "ppo_clutter_v1", "A1-unseen-203", "C0",
                              "fig_traj_ppo_c_d0plus_success_d1_fail",
                              "PPO-C: D0+ success vs D1 failure in C0 (matched seed)",
                              category="B_gen_a_succeeds_gen_b_fails")
    fig_lrppo_lead_error_cases()
    print("=== mechanism diagnostic figures written to", FIG_DIR, "===")
