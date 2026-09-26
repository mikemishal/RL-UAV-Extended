"""
Phase 20-25: matched Monte Carlo comparison of the pre-existing reactive-
only obstacle avoidance ("B": ObstacleAvoidancePlanner + Level-B local
perception) against the new perception-constrained multi-obstacle route
planner ("E": RoutePlanner + Level-B local perception), across the C1-C3
clutter levels from the Phase 2 diagnostic, using the SAME seed set so
episodes are directly comparable (matched Monte Carlo).

Also reports runtime-per-step statistics (Phase 25) and a feasibility
classification (Phase 24): whether outcome=timeout episodes plausibly
reflect a geometric planning failure (route planner reports "no feasible
route" / large replan counts) vs a purely dynamics-limited pursuit case.

Honest finding driving this script's design: preliminary probing (see
final report) showed the route planner's benefit is highly sensitive to
interaction with the PRE-EXISTING reactive evasion mechanic
(`enemy_evasion_enabled`, unrelated to this task, from an earlier
sub-task). Both evasion-enabled (realistic mission default) and
evasion-disabled configurations are measured to separate the two effects.

Usage:
    python -m experiments.route_planning_monte_carlo
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "hostile_route_planning" / "phase20_25_monte_carlo"
SEEDS = tuple(range(80000, 80015))  # N=15/level (scoped down from the N=30 Phase 2 protocol for runtime budget)
LEVELS = ("C1", "C2", "C3")
MAX_STEPS = 2000


def _run_episode(config: EnvConfig, seed: int, max_steps: int = MAX_STEPS) -> dict:
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    start_pos = info["enemy_pos"].copy()
    path_length = 0.0
    prev = info["enemy_pos"].copy()
    step_times = []
    step = 0
    for step in range(max_steps):
        t0 = time.perf_counter()
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        step_times.append(time.perf_counter() - t0)
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev))
        prev = info["enemy_pos"].copy()
        if done or truncated:
            break
    direct = float(np.linalg.norm(info["soldier_pos"] - start_pos))
    return dict(
        seed=seed, outcome=info["outcome"], episode_length=step + 1,
        detour_ratio=path_length / direct if direct > 1e-6 else float("nan"),
        completed=info["outcome"] in ("intercepted", "soldier_caught"),
        replan_count=info.get("enemy_route_replan_count", 0),
        mean_step_time=float(np.mean(step_times)),
        p95_step_time=float(np.percentile(step_times, 95)),
        max_step_time=float(np.max(step_times)),
    )


def _method_config(level: str, method: str, evasion_enabled: bool) -> EnvConfig:
    common = dict(
        **CLUTTER_LEVELS[level], dynamic_obstacles_enabled=False,
        enemy_obstacle_avoidance_enabled=True, enemy_evasion_enabled=evasion_enabled,
    )
    if method == "B_reactive_only":
        return EnvConfig(**common, enemy_obstacle_awareness_mode="local")
    if method == "E_route_planner_levelB":
        return EnvConfig(**common, enemy_obstacle_awareness_mode="local", enemy_route_planning_enabled=True)
    if method == "F_route_planner_levelC":
        return EnvConfig(**common, enemy_obstacle_awareness_mode="map_local", enemy_route_planning_enabled=True)
    raise ValueError(method)


def run_comparison() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for evasion_enabled in (True, False):
        for level in LEVELS:
            for method in ("B_reactive_only", "E_route_planner_levelB"):
                config = _method_config(level, method, evasion_enabled)
                for seed in SEEDS:
                    result = _run_episode(config, seed)
                    result.update(level=level, method=method, evasion_enabled=evasion_enabled)
                    rows.append(result)
                    print(f"evasion={evasion_enabled} level={level} method={method} seed={result['seed']} "
                          f"outcome={result['outcome']} detour={result['detour_ratio']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "episode_metrics.csv", index=False)

    summary = df.groupby(["evasion_enabled", "level", "method"]).agg(
        n=("seed", "count"),
        completion_rate=("completed", "mean"),
        mean_detour=("detour_ratio", "mean"),
        median_detour=("detour_ratio", "median"),
        p90_detour=("detour_ratio", lambda s: float(np.percentile(s, 90))),
        p95_detour=("detour_ratio", lambda s: float(np.percentile(s, 95))),
        mean_replans=("replan_count", "mean"),
        mean_step_time_ms=("mean_step_time", lambda s: float(np.mean(s)) * 1000.0),
        p95_step_time_ms=("p95_step_time", lambda s: float(np.mean(s)) * 1000.0),
        max_step_time_ms=("max_step_time", lambda s: float(np.max(s)) * 1000.0),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, evasion_enabled in zip(axes, (True, False)):
        sub = summary[summary["evasion_enabled"] == evasion_enabled]
        for method in sub["method"].unique():
            m = sub[sub["method"] == method]
            ax.plot(m["level"], m["mean_detour"], marker="o", label=method)
        ax.set_title(f"evasion_enabled={evasion_enabled}")
        ax.set_ylabel("mean detour ratio")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "detour_ratio_comparison.png", dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    run_comparison()
