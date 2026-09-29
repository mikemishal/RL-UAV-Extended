"""
Phase 22-24 (journal extension, scale-up sub-task): publication figures
summarizing the vectorized-training scale-up and the resulting held-out
evaluation. Reads ONLY already-computed result artifacts (no new
training/evaluation runs) from:
    - results/clutter_training_400k/*/seed_*/training_manifest.json
    - results/vectorization_benchmark/phase9_throughput_benchmark.json
    - results/clutter_retraining_evaluation/*.csv
    - results/lrppo_residual_saturation_400k/residual_saturation_summary.json

Usage:
    python -m experiments.plot_clutter_scaleup_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k"
BENCH_ROOT = PROJECT_ROOT / "results" / "vectorization_benchmark"
EVAL_ROOT = PROJECT_ROOT / "results" / "clutter_retraining_evaluation"
SAT_ROOT = PROJECT_ROOT / "results" / "lrppo_residual_saturation_400k"
OUT_DIR = PROJECT_ROOT / "results" / "clutter_scaleup_figures"

ARCH_SEEDS = {
    "ppo_clutter_v1": (71, 72, 73),
    "ppo_kalman_clutter_v1": (74, 75, 76),
    "lrppo_clutter_v1": (77, 78, 79),
}
ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPOK-C", "lrppo_clutter_v1": "LRPPO-C"}
ARCH_COLORS = {"ppo_clutter_v1": "tab:blue", "ppo_kalman_clutter_v1": "tab:cyan", "lrppo_clutter_v1": "tab:red"}
LEVEL_ORDER = ("C1", "C2", "C3")
FAMILY_ORDER = ("PPO", "PPO-Kalman", "LR-PPO")
FAMILY_TO_ARCH = {"PPO": "ppo_clutter_v1", "PPO-Kalman": "ppo_kalman_clutter_v1", "LR-PPO": "lrppo_clutter_v1"}


def fig_training_curves(out_dir: Path):
    """Fig 1: periodic (same-distribution) eval success rate vs timesteps,
    one subplot per architecture, one line per seed -- Phase 13 stability."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (arch, seeds) in zip(axes, ARCH_SEEDS.items()):
        for seed in seeds:
            manifest_path = CAMPAIGN_ROOT / arch / f"seed_{seed}" / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            hist = manifest["eval_history"]
            xs = [h["timesteps"] for h in hist]
            ys = [h["success_rate"] for h in hist]
            ax.plot(xs, ys, marker="o", markersize=3, label=f"seed {seed}")
        ax.set_title(ARCH_LABELS[arch])
        ax.set_xlabel("training timesteps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("periodic eval success rate\n(same-distribution, NOT held-out)")
    fig.suptitle("Phase 10/13: 400k training curves (3 seeds x 3 architectures)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_training_curves.png", dpi=150)
    plt.close(fig)


def fig_throughput_scaling(out_dir: Path):
    """Fig 2: measured env-steps/sec vs num_envs (Phase 9)."""
    data = json.loads((BENCH_ROOT / "phase9_throughput_benchmark.json").read_text())
    num_envs = sorted(int(k) for k in data.keys())
    steps_per_sec = [data[str(n)]["env_steps_per_sec"] for n in num_envs]
    speedup = [data[str(n)]["speedup_vs_num_envs_1"] for n in num_envs]
    efficiency = [data[str(n)]["scaling_efficiency"] for n in num_envs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(num_envs, steps_per_sec, marker="o", color="tab:blue")
    ax1.set_xlabel("num_envs")
    ax1.set_ylabel("env-steps/sec (measured)")
    ax1.set_title("Throughput vs num_envs")
    ax1.grid(alpha=0.3)
    for n, s in zip(num_envs, steps_per_sec):
        ax1.annotate(f"{s:.0f}", (n, s), textcoords="offset points", xytext=(0, 6), fontsize=8)

    ax2.plot(num_envs, speedup, marker="o", color="tab:green", label="speedup")
    ax2.plot(num_envs, num_envs, linestyle="--", color="gray", label="ideal linear speedup")
    ax2.set_xlabel("num_envs")
    ax2.set_ylabel("speedup vs num_envs=1")
    ax2.set_title(f"Speedup (chosen production num_envs=16, efficiency={efficiency[num_envs.index(16)]:.1%})"
                  if 16 in num_envs else "Speedup")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_throughput_scaling.png", dpi=150)
    plt.close(fig)


def fig_recovery_deltas(out_dir: Path):
    """Fig 3: frozen-vs-clutter-trained safe-intercept-rate delta by level/family (Phase 17)."""
    df = pd.read_csv(EVAL_ROOT / "recovery_deltas.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(LEVEL_ORDER))
    for i, family in enumerate(FAMILY_ORDER):
        sub = df[df.family == family].set_index("level").reindex(LEVEL_ORDER)
        ax.bar(x + (i - 1) * width, sub.safe_intercept_rate_delta, width, label=family)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_ylabel("safe_intercept_rate delta\n(clutter-trained - frozen)")
    ax.set_title("Phase 17: clutter-training recovery effect (N=100 held-out episodes/controller)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_recovery_deltas.png", dpi=150)
    plt.close(fig)


def fig_success_by_level_with_ci(out_dir: Path):
    """Fig 4: frozen vs clutter-trained safe-intercept rate with Wilson 95% CI, per family/level."""
    frozen = pd.read_csv(EVAL_ROOT / "frozen_and_classical_summary.csv")
    trained_raw = pd.read_csv(EVAL_ROOT / "clutter_trained_episodes.csv")
    # Aggregate clutter-trained across seeds (pooled, not per-seed) for a like-for-like CI vs frozen's
    # pooled-across-seeds CI.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        frozen_sub = frozen[frozen.family == family].set_index("level").reindex(LEVEL_ORDER)
        arch = FAMILY_TO_ARCH[family]
        trained_sub = trained_raw[trained_raw.family == arch]
        levels_x = np.arange(len(LEVEL_ORDER))
        frozen_rate = frozen_sub["safe_intercept_rate"].values
        frozen_err = np.vstack([
            frozen_rate - frozen_sub["safe_intercept_rate_ci95_lo"].values,
            frozen_sub["safe_intercept_rate_ci95_hi"].values - frozen_rate,
        ])
        trained_rate, trained_err_lo, trained_err_hi = [], [], []
        for level in LEVEL_ORDER:
            n = trained_sub[trained_sub.level == level]
            successes = n["safe_intercept"].sum()
            total = len(n)
            p = successes / total if total else float("nan")
            trained_rate.append(p)
            from experiments.clutter_retraining_evaluation import wilson_ci
            lo, hi = wilson_ci(successes, total)
            trained_err_lo.append(p - lo)
            trained_err_hi.append(hi - p)
        ax.errorbar(levels_x - 0.08, frozen_rate, yerr=frozen_err, marker="o", capsize=3, label="frozen")
        ax.errorbar(levels_x + 0.08, trained_rate, yerr=[trained_err_lo, trained_err_hi], marker="s",
                    capsize=3, label="clutter-trained")
        ax.set_xticks(levels_x)
        ax.set_xticklabels(LEVEL_ORDER)
        ax.set_title(family)
        ax.set_xlabel("clutter level")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("safe_intercept_rate (95% Wilson CI)\nN=100 held-out episodes/controller/level")
    fig.suptitle("Phase 15-17: frozen vs clutter-trained held-out performance")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_success_by_level_ci.png", dpi=150)
    plt.close(fig)


def fig_seed_robustness(out_dir: Path):
    """Fig 5: seed-to-seed variability (mean +/- std across 3 seeds) per architecture/level (Phase 18)."""
    df = pd.read_csv(EVAL_ROOT / "clutter_trained_seed_robustness.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(LEVEL_ORDER))
    for i, arch in enumerate(ARCH_SEEDS):
        sub = df[df.family == arch].set_index("level").reindex(LEVEL_ORDER)
        ax.bar(x + (i - 1) * width, sub.mean_safe_intercept_rate, width,
               yerr=sub.std_safe_intercept_rate, capsize=3, label=ARCH_LABELS[arch],
               color=ARCH_COLORS[arch])
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_ylabel("mean safe_intercept_rate +/- std across 3 seeds")
    ax.set_title("Phase 18: seed-to-seed robustness of clutter-trained models")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_seed_robustness.png", dpi=150)
    plt.close(fig)


def fig_safety_layer_intervention(out_dir: Path):
    """Fig 6: safety-layer intervention frequency (avoidance_fraction) and
    magnitude (mean_override_angle) by family/level -- Phase 20 proxy for
    safety-layer dependence (see report for the documented reason a
    stronger disabled-safety-layer ablation was not attempted)."""
    frozen = pd.read_csv(EVAL_ROOT / "frozen_and_classical_summary.csv")
    trained = pd.read_csv(EVAL_ROOT / "clutter_trained_summary.csv")
    trained_agg = trained.groupby(["level", "family"]).agg(
        mean_avoidance_fraction=("mean_avoidance_fraction", "mean"),
        mean_override_angle=("mean_override_angle", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(LEVEL_ORDER))
    width = 0.12
    offset = -1.5
    for family in FAMILY_ORDER:
        sub = frozen[frozen.family == family].set_index("level").reindex(LEVEL_ORDER)
        axes[0].bar(x + offset * width, sub.mean_avoidance_fraction, width, label=f"{family} (frozen)")
        offset += 1
    for arch in ARCH_SEEDS:
        sub = trained_agg[trained_agg.family == arch].set_index("level").reindex(LEVEL_ORDER)
        axes[0].bar(x + offset * width, sub.mean_avoidance_fraction, width,
                    label=f"{ARCH_LABELS[arch]} (trained)", hatch="//")
        offset += 1
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LEVEL_ORDER)
    axes[0].set_ylabel("mean fraction of steps with safety-layer intervention")
    axes[0].set_title("Safety-layer intervention frequency")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3, axis="y")

    offset = -1.5
    for family in FAMILY_ORDER:
        sub = frozen[frozen.family == family].set_index("level").reindex(LEVEL_ORDER)
        axes[1].bar(x + offset * width, sub.mean_override_angle, width, label=f"{family} (frozen)")
        offset += 1
    for arch in ARCH_SEEDS:
        sub = trained_agg[trained_agg.family == arch].set_index("level").reindex(LEVEL_ORDER)
        axes[1].bar(x + offset * width, sub.mean_override_angle, width,
                    label=f"{ARCH_LABELS[arch]} (trained)", hatch="//")
        offset += 1
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LEVEL_ORDER)
    axes[1].set_ylabel("mean override angle (deg) when intervening")
    axes[1].set_title("Safety-layer correction magnitude")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Phase 20: safety-layer dependence proxy (frequency/magnitude of correction)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_safety_layer_intervention.png", dpi=150)
    plt.close(fig)


def fig_residual_saturation(out_dir: Path):
    """Fig 7: LR-PPO residual-saturation summary (Phase 19/21) -- all
    P(near_theta_max) values are 0.0 at full scale; shown as a simple bar
    for completeness/traceability rather than omitted."""
    data = json.loads((SAT_ROOT / "residual_saturation_summary.json").read_text())
    pooled = data["pooled"]
    keys = [k for k in pooled if k.startswith("P(")]
    labels = [k.replace("P(near_theta_max)_", "") for k in keys]
    values = [pooled[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="tab:red")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(|theta_residual| >= 0.9 * theta_max)")
    ax.set_title(f"Phase 19: LR-PPO residual saturation, full scale "
                 f"(pooled n={pooled['n_step_samples']} steps, 3 seeds x 200 episodes)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_residual_saturation.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_training_curves(OUT_DIR)
    fig_throughput_scaling(OUT_DIR)
    fig_recovery_deltas(OUT_DIR)
    fig_success_by_level_with_ci(OUT_DIR)
    fig_seed_robustness(OUT_DIR)
    fig_safety_layer_intervention(OUT_DIR)
    fig_residual_saturation(OUT_DIR)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
