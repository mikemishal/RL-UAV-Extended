"""
Phase 9/14: matched Monte Carlo comparison of the four hostile-navigation
architectures defined in the behavior-arbitration follow-up task:

    A - baseline: no route planner at all (reactive-only avoidance)
    B - existing (pre-arbitration) persistent route planner: route is
        always active/replanned regardless of defender evasion state
    C - hierarchical arbitration: route suspended while evasion is
        active (+ hysteresis hold), rejoined (not blindly replanned) on
        resuming; planning_range=40m (same horizon as B)
    D - short-horizon hierarchical arbitration: same as C but with a
        much smaller planning_range (15m), targeting a local
        mission-direction corridor rather than a long-lived route toward
        the (possibly distant) moving target

All four share Level-B local perception and the SAME seed set per level
(matched Monte Carlo), evaluated under both evasion-enabled (realistic
default) and evasion-disabled configurations, across C1-C3.

Usage:
    python -m experiments.behavior_arbitration_monte_carlo
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
OUT_DIR = PROJECT_ROOT / "results" / "hostile_route_planning" / "phase9_14_arbitration_monte_carlo"
SEEDS = tuple(range(80000, 80008))  # N=8/level (reduced from N=15 given the runtime budget for 4 methods x 3 levels)
LEVELS = ("C1", "C2", "C3")
METHODS = ("A_baseline_no_route", "B_persistent_no_arbitration", "C_hierarchical_arbitration", "D_short_horizon_arbitration")
MAX_STEPS = 2000


def _run_episode(config: EnvConfig, seed: int, max_steps: int = MAX_STEPS) -> dict:
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    start_pos = info["enemy_pos"].copy()
    path_length = 0.0
    prev = info["enemy_pos"].copy()
    step_times = []
    max_replans_while_evading = 0
    max_waypoint_invalidations = 0
    max_mode_switches = 0
    step = 0
    for step in range(max_steps):
        t0 = time.perf_counter()
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        step_times.append(time.perf_counter() - t0)
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev))
        prev = info["enemy_pos"].copy()
        max_replans_while_evading = info.get("enemy_replans_while_evading", 0)
        max_waypoint_invalidations = info.get("enemy_waypoint_invalidations_from_evasion", 0)
        max_mode_switches = info.get("enemy_behavior_mode_switch_count", 0)
        if done or truncated:
            break
    direct = float(np.linalg.norm(info["soldier_pos"] - start_pos))
    return dict(
        seed=seed, outcome=info["outcome"], episode_length=step + 1,
        detour_ratio=path_length / direct if direct > 1e-6 else float("nan"),
        completed=info["outcome"] in ("intercepted", "soldier_caught"),
        replan_count=info.get("enemy_route_replan_count", 0),
        replans_while_evading=max_replans_while_evading,
        waypoint_invalidations_from_evasion=max_waypoint_invalidations,
        behavior_mode_switch_count=max_mode_switches,
        mean_step_time=float(np.mean(step_times)),
        p95_step_time=float(np.percentile(step_times, 95)),
        max_step_time=float(np.max(step_times)),
    )


def _method_config(level: str, method: str, evasion_enabled: bool) -> EnvConfig:
    common = dict(
        **CLUTTER_LEVELS[level], dynamic_obstacles_enabled=False,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        enemy_evasion_enabled=evasion_enabled,
    )
    if method == "A_baseline_no_route":
        return EnvConfig(**common)
    if method == "B_persistent_no_arbitration":
        return EnvConfig(**common, enemy_route_planning_enabled=True, enemy_route_arbitration_enabled=False)
    if method == "C_hierarchical_arbitration":
        return EnvConfig(**common, enemy_route_planning_enabled=True, enemy_route_arbitration_enabled=True,
                          enemy_route_planning_range=40.0)
    if method == "D_short_horizon_arbitration":
        return EnvConfig(**common, enemy_route_planning_enabled=True, enemy_route_arbitration_enabled=True,
                          enemy_route_planning_range=15.0)
    raise ValueError(method)


def run_comparison(evasion_settings=(True, False)) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for evasion_enabled in evasion_settings:
        for level in LEVELS:
            for method in METHODS:
                config = _method_config(level, method, evasion_enabled)
                for seed in SEEDS:
                    result = _run_episode(config, seed)
                    result.update(level=level, method=method, evasion_enabled=evasion_enabled)
                    rows.append(result)
                    print(f"evasion={evasion_enabled} level={level} method={method} seed={result['seed']} "
                          f"outcome={result['outcome']} detour={result['detour_ratio']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    episode_csv = OUT_DIR / "episode_metrics.csv"
    df.to_csv(episode_csv, index=False, mode="a" if episode_csv.exists() else "w", header=not episode_csv.exists())

    summary = df.groupby(["evasion_enabled", "level", "method"]).agg(
        n=("seed", "count"),
        completion_rate=("completed", "mean"),
        mean_detour=("detour_ratio", "mean"),
        median_detour=("detour_ratio", "median"),
        p90_detour=("detour_ratio", lambda s: float(np.percentile(s, 90))),
        p95_detour=("detour_ratio", lambda s: float(np.percentile(s, 95))),
        mean_replans=("replan_count", "mean"),
        mean_replans_while_evading=("replans_while_evading", "mean"),
        mean_waypoint_invalidations=("waypoint_invalidations_from_evasion", "mean"),
        mean_mode_switches=("behavior_mode_switch_count", "mean"),
        mean_step_time_ms=("mean_step_time", lambda s: float(np.mean(s)) * 1000.0),
        max_step_time_ms=("max_step_time", lambda s: float(np.max(s)) * 1000.0),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False, mode="a" if (OUT_DIR / "summary.csv").exists() else "w",
                   header=not (OUT_DIR / "summary.csv").exists())
    print(summary.to_string(index=False))

    present_evasion_settings = sorted(summary["evasion_enabled"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(present_evasion_settings), figsize=(7 * len(present_evasion_settings), 5), squeeze=False)
    for ax, evasion_enabled in zip(axes[0], present_evasion_settings):
        sub = summary[summary["evasion_enabled"] == evasion_enabled]
        for method in METHODS:
            m = sub[sub["method"] == method]
            ax.plot(m["level"], m["mean_detour"], marker="o", label=method)
        ax.set_title(f"evasion_enabled={evasion_enabled}")
        ax.set_ylabel("mean detour ratio")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"detour_ratio_comparison_{'_'.join(str(e) for e in present_evasion_settings)}.png", dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "evasion_false_only":
        run_comparison(evasion_settings=(False,))
    elif len(sys.argv) > 1 and sys.argv[1] == "evasion_true_only":
        run_comparison(evasion_settings=(True,))
    else:
        run_comparison()
