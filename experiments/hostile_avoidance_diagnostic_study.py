"""
Section 19 diagnostic Monte Carlo: hostile-UAS obstacle-avoidance
DISABLED (A) vs ENABLED (B), same C1-C3 static-clutter layouts/seeds, same
modest dynamic-obstacle population in both arms (only the avoidance flag
differs). The defender supplies the zero action every step (a
non-interfering "null" defender, per Section 19's "no defender interferes"
requirement), isolating the hostile's OWN navigation competence.

No defender retraining; no controller changes.

Usage:
    python -m experiments.hostile_avoidance_diagnostic_study
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "hostile_avoidance" / "diagnostic_study"
SEEDS = tuple(range(80000, 80100))  # 100 episodes/level/arm
DYNAMIC_KWARGS = dict(dynamic_obstacles_enabled=True, dynamic_obstacle_count=4)


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
    max_angle = 0.0
    static_collision = False
    dynamic_collision = False
    step = 0
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        step += 1
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev_pos))
        prev_pos = info["enemy_pos"].copy()
        min_clearance = min(min_clearance, info["enemy_actual_min_clearance"])
        if info["enemy_avoidance_active"]:
            n_avoidance_steps += 1
            max_angle = max(max_angle, info["enemy_avoidance_angle_deg"])
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
        "max_avoidance_angle_deg": max_angle,
        "episode_length": step,
    }


def run_study() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in ("C1", "C2", "C3"):
        for arm, avoidance_enabled in (("A_disabled", False), ("B_enabled", True)):
            config = EnvConfig(
                **CLUTTER_LEVELS[level], **DYNAMIC_KWARGS,
                enemy_obstacle_avoidance_enabled=avoidance_enabled,
            )
            for seed in SEEDS:
                result = _run_episode(config, seed)
                result["level"] = level
                result["arm"] = arm
                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "episodes.csv", index=False)

    summary_rows = []
    for level in ("C1", "C2", "C3"):
        for arm in ("A_disabled", "B_enabled"):
            sub = df[(df.level == level) & (df.arm == arm) & (df.feasible == True)]  # noqa: E712
            summary_rows.append(dict(
                level=level, arm=arm, n=len(sub),
                static_collision_rate=float(sub.static_obstacle_collision.mean()),
                dynamic_collision_rate=float(sub.dynamic_obstacle_collision.mean()),
                reach_target_rate=float(sub.reached_target.mean()),
                mean_min_clearance=float(sub.min_obstacle_clearance.mean()),
                mean_path_length=float(sub.path_length.mean()),
                mean_detour_ratio=float(sub.detour_ratio.mean()),
                mean_avoidance_fraction=float(sub.avoidance_fraction.mean()),
            ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_study()
