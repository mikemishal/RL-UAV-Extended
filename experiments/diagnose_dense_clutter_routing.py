"""
Phase 2 diagnostic: quantify EXACTLY why the current (pre-route-planner)
single-obstacle reactive avoidance produces ~120-140x detour ratios at C3.

Runs representative C2/C3 episodes with Level B perception + the existing
reactive `ObstacleAvoidancePlanner` (no route planner yet -- this measures
the BASELINE being replaced), using the same "null defender" isolation as
prior diagnostic studies, and records the full metric set requested in
Phase 2.

Usage:
    python -m experiments.diagnose_dense_clutter_routing
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "hostile_route_planning" / "phase2_diagnosis"
SEEDS = tuple(range(80000, 80030))  # 30 episodes per level for this diagnosis
STALL_WINDOW = 20
STALL_PROGRESS_THRESHOLD = 1.0  # m; negligible net goal-distance reduction over the window
REVISIT_RADIUS = 3.0  # m
REVISIT_MIN_STEP_GAP = 30  # ignore recent history when checking revisits


def _run_diagnostic_episode(config: EnvConfig, seed: int, max_steps: int = 2000) -> dict | None:
    env = SoldierEnv(config=config)
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError:
        return None

    start_pos = info["enemy_pos"].copy()
    positions = [info["enemy_pos"][:2].copy()]
    goal_distances = [float(np.linalg.norm(info["enemy_pos"] - info["soldier_pos"]))]
    path_length = 0.0
    prev_pos = info["enemy_pos"].copy()

    active_steps = 0
    obstacle_id_sequence = []
    mode_sequence = []
    perceived_counts = []
    step = 0
    done = False
    while not done and step < max_steps:
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        step += 1
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev_pos))
        prev_pos = info["enemy_pos"].copy()
        positions.append(info["enemy_pos"][:2].copy())
        goal_distances.append(float(np.linalg.norm(info["enemy_pos"] - info["soldier_pos"])))
        perceived_counts.append(info["enemy_total_perceived_obstacles"])
        if info["enemy_obstacle_avoidance_active"]:
            active_steps += 1
            obstacle_id_sequence.append(info["enemy_avoidance_obstacle_id"])
            mode_sequence.append(info["enemy_obstacle_avoidance_mode"])
        if done or truncated:
            break

    positions = np.array(positions)
    goal_distances = np.array(goal_distances)

    # Obstacle-ID switches: consecutive distinct entries in the (compressed) sequence.
    compressed_ids = [v for i, v in enumerate(obstacle_id_sequence) if i == 0 or v != obstacle_id_sequence[i - 1]]
    obstacle_id_switches = max(0, len(compressed_ids) - 1)
    # Repeated encounters with the same obstacle: an id appearing in >1 separate activation runs.
    from collections import Counter
    repeat_counts = Counter(compressed_ids)
    repeated_obstacle_encounters = sum(1 for c in repeat_counts.values() if c > 1)
    # Bypass-choice (mode) changes.
    compressed_modes = [v for i, v in enumerate(mode_sequence) if i == 0 or v != mode_sequence[i - 1]]
    mode_changes = max(0, len(compressed_modes) - 1)

    # Progress-stall steps: sliding window where goal-distance reduction is negligible.
    stall_steps = 0
    for i in range(STALL_WINDOW, len(goal_distances)):
        if goal_distances[i - STALL_WINDOW] - goal_distances[i] < STALL_PROGRESS_THRESHOLD:
            stall_steps += 1

    # Route revisits: thinned position samples returning near an earlier (non-recent) sample.
    revisit_count = 0
    thinned = positions[::5]
    for i in range(len(thinned)):
        earlier = thinned[: max(0, i - REVISIT_MIN_STEP_GAP // 5)]
        if len(earlier) == 0:
            continue
        dists = np.linalg.norm(earlier - thinned[i], axis=1)
        if np.any(dists < REVISIT_RADIUS):
            revisit_count += 1

    direct_distance = float(np.linalg.norm(info["soldier_pos"] - start_pos))
    initial_goal_distance = goal_distances[0]
    final_goal_distance = goal_distances[-1]
    progress_efficiency = max(0.0, initial_goal_distance - final_goal_distance) / path_length if path_length > 1e-6 else 0.0

    return dict(
        seed=seed, feasible=True, outcome=info["outcome"], episode_length=step,
        path_length=path_length, direct_distance=direct_distance,
        detour_ratio=path_length / direct_distance if direct_distance > 1e-6 else float("nan"),
        n_avoidance_activations=len(compressed_ids),
        avoidance_active_fraction=active_steps / step if step else 0.0,
        obstacle_id_switches=obstacle_id_switches,
        bypass_mode_changes=mode_changes,
        repeated_obstacle_encounters=repeated_obstacle_encounters,
        net_goal_distance_reduction=initial_goal_distance - final_goal_distance,
        goal_distance_reduction_per_step=(initial_goal_distance - final_goal_distance) / step if step else 0.0,
        progress_efficiency=progress_efficiency,
        progress_stall_steps=stall_steps,
        route_revisit_count=revisit_count,
        mean_local_obstacle_density=float(np.mean(perceived_counts)) if perceived_counts else 0.0,
        positions=positions,
    )


def run_diagnosis() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    trajectories = {}
    for level in ("C2", "C3"):
        config = EnvConfig(
            **CLUTTER_LEVELS[level], dynamic_obstacles_enabled=False,
            enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        )
        for seed in SEEDS:
            result = _run_diagnostic_episode(config, seed)
            if result is None:
                continue
            result["level"] = level
            trajectories[(level, seed)] = result.pop("positions")
            rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "phase2_episode_metrics.csv", index=False)

    summary = df.groupby("level").agg(
        n=("seed", "count"),
        mean_detour_ratio=("detour_ratio", "mean"),
        median_detour_ratio=("detour_ratio", "median"),
        mean_avoidance_activations=("n_avoidance_activations", "mean"),
        mean_avoidance_fraction=("avoidance_active_fraction", "mean"),
        mean_obstacle_id_switches=("obstacle_id_switches", "mean"),
        mean_bypass_mode_changes=("bypass_mode_changes", "mean"),
        mean_repeated_encounters=("repeated_obstacle_encounters", "mean"),
        mean_progress_efficiency=("progress_efficiency", "mean"),
        mean_stall_steps=("progress_stall_steps", "mean"),
        mean_revisit_count=("route_revisit_count", "mean"),
        mean_obstacle_density=("mean_local_obstacle_density", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "phase2_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Plot the 5-10 most pathological (highest detour ratio) trajectories.
    worst = df.sort_values("detour_ratio", ascending=False).head(8)
    n = len(worst)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, row) in zip(axes, worst.iterrows()):
        pos = trajectories[(row["level"], row["seed"])]
        ax.plot(pos[:, 0], pos[:, 1], linewidth=0.7, color="tab:red")
        ax.scatter([pos[0, 0]], [pos[0, 1]], color="black", marker="^", s=30)
        ax.set_title(f"{row['level']} seed={row['seed']}\ndetour={row['detour_ratio']:.1f}x, revisits={row['route_revisit_count']}", fontsize=8)
        ax.set_aspect("equal")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Phase 2: representative pathological trajectories (pre-route-planner)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pathological_trajectories.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'pathological_trajectories.png'}")
    return summary


if __name__ == "__main__":
    run_diagnosis()
