"""
Phase 22-26: frozen-controller diagnostic benchmark for the NEW shared
defender obstacle-awareness/avoidance layer.

Conditions (identical clutter level/seed/hostile behavior across all
three -- Phase 26 fairness):
    A_disabled -- defender_obstacle_avoidance_enabled=False (conference
        baseline, exact legacy behavior).
    B_level_b  -- defender_obstacle_avoidance_enabled=True,
        defender_obstacle_awareness_mode="local" (Level B).
    C_level_c  -- defender_obstacle_avoidance_enabled=True,
        defender_obstacle_awareness_mode="map_local" (Level C).

Reuses the EXISTING, canonical controller inventory
(experiments/clutter_benchmark_protocol.controller_instances(),
build_env_config()) -- no controller is duplicated or modified. All 7
FAMILY_ORDER families are covered (Greedy, PN, Lead, RMPC, PPO x3 seeds,
PPO-Kalman x3 seeds, LR-PPO x3 seeds = 13 instances), using the SAME
reduced N for every instance (Phase 23: "no asymmetric-sample-size
complication").

RUNTIME NOTE (reusing the already-documented Phase-8 finding from
clutter_benchmark_protocol.py): RMPC's CEM optimizer costs ~24-58s/episode,
~150-400x slower than every other controller. N is therefore kept small
(matched across ALL instances, not just RMPC) rather than silently
dropping RMPC -- see clutter_benchmark_protocol.py's own Phase-8 runtime
note for the precedent.

Usage:
    python -m experiments.defender_obstacle_benchmark
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

from experiments.clutter_benchmark_protocol import (
    CLUTTER_SEED_START,
    build_env_config,
    controller_instances,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "defender_obstacle_awareness" / "phase22_26_benchmark"
LEVELS = ("C1", "C2", "C3")
CONDITIONS = ("A_disabled", "B_level_b", "C_level_c")
N_SEEDS = 3  # reduced diagnostic subset (RMPC-runtime-justified, matched across ALL instances)
SEEDS = tuple(range(CLUTTER_SEED_START, CLUTTER_SEED_START + N_SEEDS))


def _apply_condition(config, condition: str):
    if condition == "A_disabled":
        return dataclasses.replace(config, defender_obstacle_avoidance_enabled=False)
    if condition == "B_level_b":
        return dataclasses.replace(
            config, defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        )
    if condition == "C_level_c":
        return dataclasses.replace(
            config, defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="map_local",
        )
    raise ValueError(condition)


def run_episode(env, policy, seed: int) -> dict:
    """Run one episode, returning both the standard clutter-benchmark
    metrics and the NEW defender-avoidance diagnostics (Phase 15/16/24)."""
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError as e:
        return {"seed": seed, "feasible": False, "infeasibility_reason": str(e)}

    policy.reset()
    estimator_mode = estimator_mode_for_env(env)

    avoidance_active_steps = 0
    override_angles = []
    min_static_clearance = info["defender_min_obstacle_clearance"]
    min_dynamic_clearance = info["defender_min_dynamic_obstacle_clearance"]
    path_length = 0.0
    prev_pos = info["defender_pos"].copy()
    step_times = []

    step = 0
    done = False
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        t0 = time.perf_counter()
        obs, reward, done, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - t0)
        step += 1

        path_length += float(np.linalg.norm(info["defender_pos"] - prev_pos))
        prev_pos = info["defender_pos"].copy()
        min_static_clearance = min(min_static_clearance, info["defender_min_obstacle_clearance"])
        min_dynamic_clearance = min(min_dynamic_clearance, info["defender_min_dynamic_obstacle_clearance"])
        if info["defender_avoidance_active"]:
            avoidance_active_steps += 1
            override_angles.append(info["defender_avoidance_angle_deg"])

        if done or truncated:
            break

    return {
        "seed": seed, "feasible": True, "outcome": info["outcome"],
        "success": 1 if info["outcome"] == "intercepted" else 0,
        "soldier_caught": 1 if info["outcome"] == "soldier_caught" else 0,
        "unsafe_intercept": 1 if info["outcome"] == "unsafe_intercept" else 0,
        "timeout": 1 if info["outcome"] == "timeout" else 0,
        "episode_length": step,
        "defender_obstacle_collision": 1 if info["defender_ever_collided"] else 0,
        "defender_dynamic_obstacle_collision": 1 if info["defender_dynamic_ever_collided"] else 0,
        "min_static_clearance": min_static_clearance,
        "min_dynamic_clearance": min_dynamic_clearance,
        "path_length": path_length,
        "avoidance_intervention_count": avoidance_active_steps,
        "avoidance_fraction": avoidance_active_steps / step if step else 0.0,
        "mean_override_angle": float(np.mean(override_angles)) if override_angles else 0.0,
        "max_override_angle": float(np.max(override_angles)) if override_angles else 0.0,
        "mean_step_time_ms": float(np.mean(step_times)) * 1000.0,
    }


def run_benchmark() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instances = controller_instances()
    rows = []
    t_start = time.time()

    for level in LEVELS:
        for condition in CONDITIONS:
            for inst in instances:
                base_config = build_env_config(level, inst.estimator)
                config = _apply_condition(base_config, condition)
                from uav_defend.envs import SoldierEnv
                env = SoldierEnv(config=config)
                policy = inst.build(config)

                t0 = time.time()
                for seed in SEEDS:
                    metrics = run_episode(env, policy, seed)
                    metrics.update(level=level, condition=condition, family=inst.family, instance_id=inst.instance_id)
                    rows.append(metrics)
                elapsed = time.time() - t0
                print(f"level={level} condition={condition} instance={inst.instance_id:>10s} "
                      f"({elapsed:.1f}s, {elapsed/len(SEEDS):.2f}s/ep)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "episode_metrics.csv", index=False)

    feasible = df[df["feasible"]]
    summary = feasible.groupby(["level", "condition", "family"]).agg(
        n=("seed", "count"),
        intercept_rate=("success", "mean"),
        soldier_caught_rate=("soldier_caught", "mean"),
        unsafe_intercept_rate=("unsafe_intercept", "mean"),
        timeout_rate=("timeout", "mean"),
        mean_static_collisions=("defender_obstacle_collision", "mean"),
        mean_dynamic_collisions=("defender_dynamic_obstacle_collision", "mean"),
        mean_min_static_clearance=("min_static_clearance", "mean"),
        mean_path_length=("path_length", "mean"),
        mean_avoidance_fraction=("avoidance_fraction", "mean"),
        mean_override_angle=("mean_override_angle", "mean"),
        max_override_angle=("max_override_angle", "max"),
        mean_step_time_ms=("mean_step_time_ms", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print(f"total wall time: {time.time() - t_start:.1f}s")
    return summary


if __name__ == "__main__":
    run_benchmark()
