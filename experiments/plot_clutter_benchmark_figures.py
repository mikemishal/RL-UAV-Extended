"""
Publication-draft figures (Phase 11) for the clutter benchmark:
    1. success rate vs clutter level (all controllers)
    2. obstacle-collision rate vs clutter level
    3. success degradation relative to C0
    4. representative C1/C2/C3 trajectories
    5. failure-mode composition by controller at high clutter (C3)

Usage:
    python -m experiments.plot_clutter_benchmark_figures --run-name main_50
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_defend.envs import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

from experiments.clutter_benchmark_protocol import (
    CLUTTER_LEVEL_ORDER,
    FAMILY_ORDER,
    PROJECT_ROOT,
    SMOKE_SEEDS,
    build_env_config,
    controller_instances,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "clutter_benchmark"
FAMILY_COLORS = {
    "Greedy": "tab:gray", "PN": "tab:brown", "Lead": "tab:orange", "RMPC": "tab:purple",
    "PPO": "tab:blue", "PPO-Kalman": "tab:cyan", "LR-PPO": "tab:red",
}


def fig_success_rate_vs_clutter(analysis_dir, out_dir):
    df = pd.read_csv(analysis_dir / "success_rate.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for family in FAMILY_ORDER:
        sub = df[df.family == family].set_index("level").reindex(CLUTTER_LEVEL_ORDER)
        yerr = np.vstack([sub.success_rate - sub.ci_low, sub.ci_high - sub.success_rate])
        ax.errorbar(CLUTTER_LEVEL_ORDER, sub.success_rate, yerr=yerr, marker="o", capsize=3,
                    label=family, color=FAMILY_COLORS[family])
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("Success rate (safe interception)")
    ax.set_title("Success rate vs. clutter level")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_success_rate_vs_clutter.png", dpi=300)
    plt.close(fig)


def fig_collision_rate_vs_clutter(analysis_dir, out_dir):
    df = pd.read_csv(analysis_dir / "failure_mode.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for family in FAMILY_ORDER:
        sub = df[df.family == family].set_index("level").reindex(CLUTTER_LEVEL_ORDER)
        ax.plot(CLUTTER_LEVEL_ORDER, sub.defender_obstacle_collision_rate, marker="o",
                label=family, color=FAMILY_COLORS[family])
    ax.set_xlabel("Clutter level")
    ax.set_ylabel("Defender-obstacle collision rate (diagnostic)")
    ax.set_title("Obstacle-collision rate vs. clutter level\n(diagnostic only -- never blocking/terminal, see Phase 3 audit)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_collision_rate_vs_clutter.png", dpi=300)
    plt.close(fig)


def fig_degradation_from_c0(analysis_dir, out_dir):
    df = pd.read_csv(analysis_dir / "degradation_from_c0.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = ("C1", "C2", "C3")
    width = 0.1
    x = np.arange(len(levels))
    for i, family in enumerate(FAMILY_ORDER):
        sub = df[df.family == family].set_index("level").reindex(levels)
        yerr = np.vstack([sub.delta_success - sub.ci_low, sub.ci_high - sub.delta_success])
        ax.bar(x + (i - 3) * width, sub.delta_success, width=width, yerr=yerr, capsize=2,
               label=family, color=FAMILY_COLORS[family])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(r"$\Delta$success $= $ success(Ck) $-$ success(C0)")
    ax.set_title("Success degradation relative to C0")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_degradation_from_c0.png", dpi=300)
    plt.close(fig)


def _rollout_trajectory(config, policy, seed):
    env = SoldierEnv(config=config)
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError:
        return None
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    defender_xy, enemy_xy, soldier_xy = [info["defender_pos"][:2].copy()], [info["enemy_pos"][:2].copy()], [info["soldier_pos"][:2].copy()]
    done = False
    while not done:
        action = policy.act(obs, build_policy_info(info, estimator_mode))
        obs, reward, done, truncated, info = env.step(action)
        defender_xy.append(info["defender_pos"][:2].copy())
        enemy_xy.append(info["enemy_pos"][:2].copy())
        soldier_xy.append(info["soldier_pos"][:2].copy())
        if done or truncated:
            break
    return dict(
        defender=np.array(defender_xy), enemy=np.array(enemy_xy), soldier=np.array(soldier_xy),
        obstacles=info["obstacle_layout"], outcome=info["outcome"], L=config.L,
    )


def fig_representative_trajectories(out_dir):
    from uav_defend.policies.registry import get_policy

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, level in zip(axes, ("C1", "C2", "C3")):
        config = build_env_config(level, "measurement")
        policy = get_policy("lead", config=config)
        traj = None
        for seed in SMOKE_SEEDS:
            traj = _rollout_trajectory(config, policy, seed)
            if traj is not None:
                break
        if traj is None:
            ax.set_title(f"{level}: no feasible seed found")
            continue
        L = traj["L"]
        for obs in traj["obstacles"]["obstacles"]:
            min_c, max_c = obs["min_corner"], obs["max_corner"]
            ax.add_patch(plt.Rectangle((min_c[0], min_c[1]), max_c[0] - min_c[0], max_c[1] - min_c[1],
                                        color="dimgray", alpha=0.5))
        ax.plot(traj["defender"][:, 0], traj["defender"][:, 1], color="tab:orange", label="Defender (Lead)")
        ax.plot(traj["enemy"][:, 0], traj["enemy"][:, 1], color="tab:red", label="Hostile")
        ax.plot(traj["soldier"][:, 0], traj["soldier"][:, 1], color="tab:green", label="Soldier")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_title(f"{level} (outcome={traj['outcome']})")
        if level == "C1":
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Representative trajectories (Lead controller, top-down)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_representative_trajectories.png", dpi=300)
    plt.close(fig)


def fig_failure_mode_composition_high_clutter(analysis_dir, out_dir):
    df = pd.read_csv(analysis_dir / "failure_mode.csv")
    sub = df[df.level == "C3"].set_index("family").reindex(FAMILY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(FAMILY_ORDER))
    for col, label, color in (
        ("success_rate", "Safe interception", "tab:green"),
        ("soldier_caught_rate", "Soldier caught", "tab:red"),
        ("unsafe_intercept_rate", "Unsafe intercept", "tab:orange"),
        ("timeout_rate", "Timeout", "tab:gray"),
    ):
        ax.bar(FAMILY_ORDER, sub[col], bottom=bottoms, label=label, color=color)
        bottoms += sub[col].to_numpy()
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Failure-mode composition at C3 (high clutter)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_failure_mode_composition_c3.png", dpi=300)
    plt.close(fig)


def generate_all_figures(run_name: str) -> None:
    analysis_dir = RESULTS_ROOT / run_name / "analysis"
    out_dir = RESULTS_ROOT / run_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_success_rate_vs_clutter(analysis_dir, out_dir)
    fig_collision_rate_vs_clutter(analysis_dir, out_dir)
    fig_degradation_from_c0(analysis_dir, out_dir)
    fig_representative_trajectories(out_dir)
    fig_failure_mode_composition_high_clutter(analysis_dir, out_dir)
    print(f"[{run_name}] wrote 5 figures to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    generate_all_figures(args.run_name)


if __name__ == "__main__":
    main()
