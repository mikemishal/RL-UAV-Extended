"""
Section 23 (Level B vs Level C matched Monte Carlo) and Section 24
(Level-B sensor-range sensitivity) diagnostic studies.

Reuses the same "null defender" (zero action every step) isolation
approach as experiments/hostile_avoidance_diagnostic_study.py, and the
C1-C3 static-clutter parameter table from
experiments/clutter_benchmark_protocol.py.

Usage:
    python -m experiments.hostile_perception_diagnostic_study
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "hostile_perception" / "diagnostic_study"
SEEDS = tuple(range(80000, 80100))  # N=100/level
DYNAMIC_KWARGS = dict(dynamic_obstacles_enabled=True, dynamic_obstacle_count=3)
SENSOR_RANGES = {"short": 6.0, "medium": 15.0, "long": 30.0}  # domain L=50 -> up to 60% of half-domain


def _run_episode(config: EnvConfig, seed: int) -> dict:
    env = SoldierEnv(config=config)
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError as e:
        return {"seed": seed, "feasible": False, "infeasibility_reason": str(e)}

    path_length = 0.0
    prev_pos = info["enemy_pos"].copy()
    start_pos = info["enemy_pos"].copy()
    min_clearance = info["enemy_actual_min_clearance"]
    n_avoidance_steps = 0
    static_collision = False
    dynamic_collision = False
    first_avoidance_step = None
    step = 0
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        step += 1
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev_pos))
        prev_pos = info["enemy_pos"].copy()
        min_clearance = min(min_clearance, info["enemy_actual_min_clearance"])
        if info["enemy_obstacle_avoidance_active"]:
            n_avoidance_steps += 1
            if first_avoidance_step is None:
                first_avoidance_step = step
        static_collision = static_collision or info["enemy_obstacle_collision"]
        dynamic_collision = dynamic_collision or info["enemy_dynamic_obstacle_collision"]
        if done or truncated:
            break

    direct_path_length = float(np.linalg.norm(info["soldier_pos"] - start_pos))
    return {
        "seed": seed, "feasible": True,
        "outcome": info["outcome"],
        "reached_target": 1 if info["outcome"] in ("soldier_caught", "intercepted") else 0,
        "static_obstacle_collision": int(static_collision),
        "dynamic_obstacle_collision": int(dynamic_collision),
        "min_obstacle_clearance": float(min_clearance) if np.isfinite(min_clearance) else float("nan"),
        "path_length": path_length,
        "detour_ratio": path_length / direct_path_length if direct_path_length > 1e-6 else float("nan"),
        "avoidance_fraction": n_avoidance_steps / step if step else 0.0,
        "first_avoidance_step": first_avoidance_step if first_avoidance_step is not None else -1,
        "episode_length": step,
    }


def run_level_b_vs_c_study() -> pd.DataFrame:
    rows = []
    for level in ("C1", "C2", "C3"):
        for mode in ("local", "map_local"):
            config = EnvConfig(
                **CLUTTER_LEVELS[level], **DYNAMIC_KWARGS,
                obstacle_unmapped_fraction=0.5,
                enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode=mode,
            )
            for seed in SEEDS:
                result = _run_episode(config, seed)
                result["level"] = level
                result["mode"] = mode
                rows.append(result)
    df = pd.DataFrame(rows)
    out_dir = OUT_DIR / "level_b_vs_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episodes.csv", index=False)

    summary_rows = []
    for level in ("C1", "C2", "C3"):
        for mode in ("local", "map_local"):
            sub = df[(df.level == level) & (df["mode"] == mode) & (df.feasible == True)]  # noqa: E712
            summary_rows.append(dict(
                level=level, mode=mode, n=len(sub),
                reach_target_rate=float(sub.reached_target.mean()),
                static_collision_rate=float(sub.static_obstacle_collision.mean()),
                dynamic_collision_rate=float(sub.dynamic_obstacle_collision.mean()),
                mean_path_length=float(sub.path_length.mean()),
                mean_detour_ratio=float(sub.detour_ratio.mean()),
                mean_min_clearance=float(sub.min_obstacle_clearance.mean()),
                mean_avoidance_fraction=float(sub.avoidance_fraction.mean()),
                mean_first_avoidance_step=float(sub[sub.first_avoidance_step >= 0].first_avoidance_step.mean()),
            ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("=== Level B vs Level C ===")
    print(summary.to_string(index=False))
    return summary


def run_sensor_range_sensitivity() -> pd.DataFrame:
    rows = []
    for range_name, sensor_range in SENSOR_RANGES.items():
        for level in ("C1", "C2", "C3"):
            config = EnvConfig(
                **CLUTTER_LEVELS[level], **DYNAMIC_KWARGS,
                enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
                enemy_obstacle_sensor_range=sensor_range, enemy_dynamic_obstacle_sensor_range=sensor_range,
            )
            for seed in SEEDS:
                result = _run_episode(config, seed)
                result["level"] = level
                result["sensor_range_name"] = range_name
                result["sensor_range_m"] = sensor_range
                rows.append(result)
    df = pd.DataFrame(rows)
    out_dir = OUT_DIR / "sensor_range_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episodes.csv", index=False)

    summary_rows = []
    for range_name in SENSOR_RANGES:
        for level in ("C1", "C2", "C3"):
            sub = df[(df.sensor_range_name == range_name) & (df.level == level) & (df.feasible == True)]  # noqa: E712
            summary_rows.append(dict(
                sensor_range_name=range_name, sensor_range_m=SENSOR_RANGES[range_name], level=level, n=len(sub),
                reach_target_rate=float(sub.reached_target.mean()),
                static_collision_rate=float(sub.static_obstacle_collision.mean()),
                mean_detour_ratio=float(sub.detour_ratio.mean()),
                mean_first_avoidance_step=float(sub[sub.first_avoidance_step >= 0].first_avoidance_step.mean()),
            ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("=== Sensor-range sensitivity (Level B) ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_level_b_vs_c_study()
    run_sensor_range_sensitivity()
