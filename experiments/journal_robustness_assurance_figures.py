"""Phase 15 publication figures for the journal robustness/assurance
evaluation. Styling matches paper/meeting_figures/2026-10-prof-meeting/
(same architecture color palette)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.journal_robustness_assurance import OUT_DIR, R1_SETTINGS, R2_SETTINGS, R3_SETTINGS, NOMINAL_POINT

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARCHES = ["PPO-C", "PPO-Kalman-C", "LR-PPO-C"]
ARCH_COLORS = {"PPO-C": "#4C72B0", "PPO-Kalman-C": "#55A868", "LR-PPO-C": "#C44E52"}

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 150,
})


def _save(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _dim_table(d1_table: pd.DataFrame, dimension: str, level: str) -> pd.DataFrame:
    return d1_table[(d1_table.dimension == dimension) & (d1_table.level == level)]


def fig_r1_measurement_noise():
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    x_order = [("nominal", "nominal"), ("R1_measurement_noise", "low"), ("R1_measurement_noise", "high")]
    x_labels = [f"nominal\n(var={NOMINAL_POINT['measurement_var']})", f"low\n(var={R1_SETTINGS['low']})", f"high\n(var={R1_SETTINGS['high']})"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, level in zip(axes, ("C0", "C2")):
        for arch in ARCHES:
            sub = df[(df.architecture == arch) & (df.level == level)]
            d0plus, d1 = [], []
            for dim, setting in x_order:
                row = sub[(sub.dimension == dim) & (sub.setting_label == setting)]
                d0plus.append(float(row.d0plus_mean.iloc[0]) if len(row) else np.nan)
                d1.append(float(row.d1_mean.iloc[0]) if len(row) else np.nan)
            ax.plot(x_labels, d0plus, linestyle="--", marker="s", color=ARCH_COLORS[arch], alpha=0.6, label=f"{arch} D0+")
            ax.plot(x_labels, d1, linestyle="-", marker="o", color=ARCH_COLORS[arch], label=f"{arch} D1")
        ax.set_title(level)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Measurement-noise level")
    axes[0].set_ylabel("Safe-interception rate")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("R1: Measurement-noise robustness (D0+ vs D1)")
    fig.tight_layout()
    _save(fig, "figR1_measurement_noise")


def fig_r2_detection_radius():
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    x_order = [("R2_detection_radius", "short"), ("nominal", "nominal"), ("R2_detection_radius", "long")]
    x_labels = [f"short\n(r={R2_SETTINGS['short']})", f"nominal\n(r={NOMINAL_POINT['detection_radius']})", f"long\n(r={R2_SETTINGS['long']})"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, level in zip(axes, ("C0", "C2")):
        for arch in ARCHES:
            sub = df[(df.architecture == arch) & (df.level == level)]
            d0plus, d1 = [], []
            for dim, setting in x_order:
                row = sub[(sub.dimension == dim) & (sub.setting_label == setting)]
                d0plus.append(float(row.d0plus_mean.iloc[0]) if len(row) else np.nan)
                d1.append(float(row.d1_mean.iloc[0]) if len(row) else np.nan)
            ax.plot(x_labels, d0plus, linestyle="--", marker="s", color=ARCH_COLORS[arch], alpha=0.6, label=f"{arch} D0+")
            ax.plot(x_labels, d1, linestyle="-", marker="o", color=ARCH_COLORS[arch], label=f"{arch} D1")
        ax.set_title(level)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Detection radius")
    axes[0].set_ylabel("Safe-interception rate")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("R2: Detection-radius robustness (D0+ vs D1)")
    fig.tight_layout()
    _save(fig, "figR2_detection_radius")


def fig_r3_relative_mobility():
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    x_order = [("R3_relative_mobility", "defender_disadvantaged"), ("nominal", "nominal"), ("R3_relative_mobility", "defender_advantaged")]
    ratios = {"defender_disadvantaged": R3_SETTINGS["defender_disadvantaged"][0] / R3_SETTINGS["defender_disadvantaged"][1],
              "nominal": NOMINAL_POINT["v_d"] / NOMINAL_POINT["v_e"],
              "defender_advantaged": R3_SETTINGS["defender_advantaged"][0] / R3_SETTINGS["defender_advantaged"][1]}
    x_labels = [f"disadv.\n(ratio={ratios['defender_disadvantaged']:.2f})", f"nominal\n(ratio={ratios['nominal']:.2f})",
                f"adv.\n(ratio={ratios['defender_advantaged']:.2f})"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, level in zip(axes, ("C0", "C2")):
        for arch in ARCHES:
            sub = df[(df.architecture == arch) & (df.level == level)]
            d0plus, d1 = [], []
            for dim, setting in x_order:
                row = sub[(sub.dimension == dim) & (sub.setting_label == setting)]
                d0plus.append(float(row.d0plus_mean.iloc[0]) if len(row) else np.nan)
                d1.append(float(row.d1_mean.iloc[0]) if len(row) else np.nan)
            ax.plot(x_labels, d0plus, linestyle="--", marker="s", color=ARCH_COLORS[arch], alpha=0.6, label=f"{arch} D0+")
            ax.plot(x_labels, d1, linestyle="-", marker="o", color=ARCH_COLORS[arch], label=f"{arch} D1")
        ax.set_title(level)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Relative mobility (v_d/v_e)")
    axes[0].set_ylabel("Safe-interception rate")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("R3: Relative-mobility robustness (D0+ vs D1)")
    fig.tight_layout()
    _save(fig, "figR3_relative_mobility")


def fig_r4_consolidated_effect():
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    cells = [("nominal", "nominal", "Nominal"), ("R1_measurement_noise", "high", "High noise"),
             ("R2_detection_radius", "short", "Short detection"), ("R3_relative_mobility", "defender_disadvantaged", "Defender disadv.")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, level in zip(axes, ("C0", "C2")):
        x = np.arange(len(cells))
        width = 0.25
        for i, arch in enumerate(ARCHES):
            sub = df[(df.architecture == arch) & (df.level == level)]
            vals, lo_err, hi_err = [], [], []
            for dim, setting, _ in cells:
                row = sub[(sub.dimension == dim) & (sub.setting_label == setting)]
                if len(row):
                    diff = float(row["diff"].iloc[0]) * 100
                    lo = float(row.ci_low.iloc[0]) * 100
                    hi = float(row.ci_high.iloc[0]) * 100
                    vals.append(diff)
                    lo_err.append(diff - lo)
                    hi_err.append(hi - diff)
                else:
                    vals.append(np.nan); lo_err.append(0); hi_err.append(0)
            ax.bar(x + (i - 1) * width, vals, width=width, yerr=[lo_err, hi_err], capsize=4, color=ARCH_COLORS[arch], label=arch)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in cells], fontsize=9)
        ax.set_title(level)
    axes[0].set_ylabel("\u0394 Safe-interception rate, D1\u2212D0+ (pp)")
    axes[0].legend(frameon=False)
    fig.suptitle("R4: Consolidated D1\u2212D0+ effect, nominal vs challenging conditions")
    fig.tight_layout()
    _save(fig, "figR4_consolidated_effect")


if __name__ == "__main__":
    fig_r1_measurement_noise()
    fig_r2_detection_radius()
    fig_r3_relative_mobility()
    fig_r4_consolidated_effect()
    print("=== robustness figures written to", FIG_DIR, "===")
