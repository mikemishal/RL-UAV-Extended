"""
Level B / Level C / mixed deterministic navigation scenarios (Sections
19-21): attacker -> obstacle(s) -> target, demonstrating perception-gated
avoidance (map-known obstacles known immediately; unmapped/dynamic
obstacles only after local sensing) without relying on hidden ground-truth
knowledge.

Usage:
    python -m experiments.plot_hostile_perception_scenarios
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
OUT_DIR = PROJECT_ROOT / "results" / "hostile_perception" / "scenarios"


def _rollout(config: EnvConfig, seed: int, max_steps: int = 400):
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    enemy_xy = [info["enemy_pos"][:2].copy()]
    soldier_xy = [info["soldier_pos"][:2].copy()]
    perceived_flags = [info["enemy_total_perceived_obstacles"] > 0]
    active_flags = [info["enemy_obstacle_avoidance_active"]]
    first_detection_step = None
    for step in range(1, max_steps + 1):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        enemy_xy.append(info["enemy_pos"][:2].copy())
        soldier_xy.append(info["soldier_pos"][:2].copy())
        perceived_flags.append(info["enemy_total_perceived_obstacles"] > 0)
        active_flags.append(info["enemy_obstacle_avoidance_active"])
        if first_detection_step is None and info["enemy_new_obstacle_detections"] > 0:
            first_detection_step = step
        if done or truncated:
            break
    return dict(
        enemy=np.array(enemy_xy), soldier=np.array(soldier_xy),
        perceived=np.array(perceived_flags), active=np.array(active_flags),
        obstacle_layout=info["obstacle_layout"], outcome=info["outcome"],
        first_detection_step=first_detection_step, L=config.L,
    )


def _plot(ax, traj, title):
    L = traj["L"]
    for obs in traj["obstacle_layout"]["obstacles"]:
        min_c, max_c = obs["min_corner"], obs["max_corner"]
        color = "tab:blue" if obs.get("map_known") else "dimgray"
        ax.add_patch(plt.Rectangle((min_c[0], min_c[1]), max_c[0] - min_c[0], max_c[1] - min_c[1],
                                    color=color, alpha=0.4))
    enemy = traj["enemy"]
    active = traj["active"]
    ax.plot(enemy[:, 0], enemy[:, 1], color="tab:red", linewidth=1.0, label="Hostile path")
    if active.any():
        ax.scatter(enemy[active, 0], enemy[active, 1], color="tab:orange", s=8, zorder=5, label="Avoidance active")
    if traj["first_detection_step"] is not None:
        p = enemy[traj["first_detection_step"]]
        ax.scatter([p[0]], [p[1]], color="black", marker="x", s=80, zorder=6, label="First detection")
    ax.plot(traj["soldier"][:, 0], traj["soldier"][:, 1], color="tab:green", linewidth=1.0, label="Target (soldier)")
    ax.scatter([enemy[0, 0]], [enemy[0, 1]], color="black", marker="^", s=40, zorder=6)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n(outcome={traj['outcome']}, 1st detect step={traj['first_detection_step']})", fontsize=8)


def _find_feasible(config: EnvConfig, seeds):
    for seed in seeds:
        try:
            return _rollout(config, seed)
        except RuntimeError:
            continue
    return None


def generate_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = range(80000, 80030)

    base = EnvConfig()
    env = SoldierEnv(config=base)
    _, info0 = env.reset(seed=80000)
    midpoint = (info0["enemy_pos"] + info0["soldier_pos"]) / 2.0
    cx = float(np.clip(midpoint[0], -base.L + 8, base.L - 8))
    cy = float(np.clip(midpoint[1], -base.L + 8, base.L - 8))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Scenario 1 (Section 19): Level B, unmapped building.
    level_b_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((cx, cy, 9.0, 6.0, 6.0, 8.0),), obstacle_fixed_map_known=(False,),
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        enemy_obstacle_sensor_range=12.0,
    )
    _plot(axes[0], _find_feasible(level_b_cfg, seeds), "Level B: unmapped building")

    # Scenario 2 (Section 20): Level C, mapped building cluster.
    level_c_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(
            (cx, cy, 9.0, 5.0, 5.0, 8.0),
            (cx + 12.0, cy, 9.0, 5.0, 5.0, 8.0),
        ),
        obstacle_fixed_map_known=(True, True),
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="map_local",
        enemy_obstacle_sensor_range=12.0,
    )
    _plot(axes[1], _find_feasible(level_c_cfg, seeds), "Level C: mapped building cluster")

    # Scenario 3 (Section 21): mixed -- mapped building + unmapped tree + bird.
    mixed_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=(
            (cx, cy, 9.0, 6.0, 6.0, 8.0),
            (cx - 12.0, cy + 4.0, 15.0, 1.0, 1.0, 10.0),
        ),
        obstacle_fixed_map_known=(True, False),
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=1,
        dynamic_obstacle_min_speed=2.0, dynamic_obstacle_max_speed=2.0,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="map_local",
        enemy_obstacle_sensor_range=12.0, enemy_dynamic_obstacle_sensor_range=12.0,
    )
    _plot(axes[2], _find_feasible(mixed_cfg, seeds), "Mixed: mapped + unmapped + bird")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Hostile-UAS perception-gated obstacle avoidance (blue=mapped, gray=unmapped)")
    fig.tight_layout()
    out_path = OUT_DIR / "perception_scenarios.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    generate_all()
