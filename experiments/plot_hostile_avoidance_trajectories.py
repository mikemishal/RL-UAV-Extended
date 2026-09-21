"""
Mission-progress validation trajectory plots (Section 18): attacker ->
obstacle -> protected asset, showing the attacker recognizes a blocked
path, deviates, clears it, and resumes pursuit -- for representative
static-building, tree-like, crossing-dynamic-bird, and multiple-obstacle
scenarios.

Usage:
    python -m experiments.plot_hostile_avoidance_trajectories
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "hostile_avoidance" / "trajectories"


def _rollout(config: EnvConfig, seed: int, max_steps: int = 400):
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    enemy_xy = [info["enemy_pos"][:2].copy()]
    soldier_xy = [info["soldier_pos"][:2].copy()]
    avoidance_active = [info["enemy_avoidance_active"]]
    for _ in range(max_steps):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        enemy_xy.append(info["enemy_pos"][:2].copy())
        soldier_xy.append(info["soldier_pos"][:2].copy())
        avoidance_active.append(info["enemy_avoidance_active"])
        if done or truncated:
            break
    return dict(
        enemy=np.array(enemy_xy), soldier=np.array(soldier_xy), avoidance_active=np.array(avoidance_active),
        obstacle_layout=info["obstacle_layout"], outcome=info["outcome"], L=config.L,
        dynamic_obstacles_enabled=config.dynamic_obstacles_enabled,
    )


def _plot_case(ax, traj, title):
    L = traj["L"]
    for obs in traj["obstacle_layout"]["obstacles"]:
        min_c, max_c = obs["min_corner"], obs["max_corner"]
        ax.add_patch(plt.Rectangle((min_c[0], min_c[1]), max_c[0] - min_c[0], max_c[1] - min_c[1],
                                    color="dimgray", alpha=0.5))
    enemy = traj["enemy"]
    active = traj["avoidance_active"]
    ax.plot(enemy[:, 0], enemy[:, 1], color="tab:red", linewidth=1.0, label="Hostile path")
    if active.any():
        ax.scatter(enemy[active, 0], enemy[active, 1], color="tab:orange", s=8, zorder=5, label="Avoidance active")
    ax.plot(traj["soldier"][:, 0], traj["soldier"][:, 1], color="tab:green", linewidth=1.0, label="Soldier")
    ax.scatter([enemy[0, 0]], [enemy[0, 1]], color="black", marker="^", s=40, zorder=6, label="Hostile start")
    ax.scatter([traj["soldier"][0, 0]], [traj["soldier"][0, 1]], color="black", marker="*", s=60, zorder=6)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n(outcome={traj['outcome']})", fontsize=9)


def _find_feasible(config: EnvConfig, seeds) -> dict | None:
    for seed in seeds:
        try:
            return _rollout(config, seed)
        except RuntimeError:
            continue
    return None


def generate_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = range(80000, 80030)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    base = EnvConfig()
    env = SoldierEnv(config=base)
    _, info0 = env.reset(seed=80000)
    enemy0, soldier0 = info0["enemy_pos"], info0["soldier_pos"]
    midpoint = (enemy0 + soldier0) / 2.0
    cx = float(np.clip(midpoint[0], -base.L + 10, base.L - 10))
    cy = float(np.clip(midpoint[1], -base.L + 10, base.L - 10))

    building_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((cx, cy, 10.0, 8.0, 8.0, 10.0),),
        enemy_obstacle_avoidance_enabled=True,
    )
    _plot_case(axes[0], _find_feasible(building_cfg, seeds), "Static building avoidance")

    tree_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((cx, cy, 15.0, 1.0, 1.0, 10.0),),
        enemy_obstacle_avoidance_enabled=True,
    )
    _plot_case(axes[1], _find_feasible(tree_cfg, seeds), "Tree-like obstacle avoidance")

    bird_cfg = EnvConfig(
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=1,
        dynamic_obstacle_min_speed=2.0, dynamic_obstacle_max_speed=2.0,
        enemy_obstacle_avoidance_enabled=True,
    )
    _plot_case(axes[2], _find_feasible(bird_cfg, seeds), "Crossing dynamic bird")

    multi_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=8,
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=4,
        enemy_obstacle_avoidance_enabled=True,
    )
    _plot_case(axes[3], _find_feasible(multi_cfg, seeds), "Multiple obstacles (static + dynamic)")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Hostile-UAS predictive obstacle avoidance -- representative trajectories")
    fig.tight_layout()
    out_path = OUT_DIR / "mission_progress_validation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    generate_all()
