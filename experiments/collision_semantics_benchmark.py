"""
Phase 12-14: frozen-controller diagnostic benchmark, RE-RUN after physical
obstacle-collision semantics were introduced (this sub-task). Reuses the
exact same conditions/levels/controller inventory as the prior
`defender_obstacle_benchmark.py` (Phase 22-26 of the previous sub-task),
but with a larger N (20 seeds, per this phase's "modest diagnostic N of
20-30" instruction) and NEW metrics that separate:
    - defender mission success (interception) from
    - defender physical failure (obstacle collision) from
    - attacker environmental failure (obstacle collision) from
    - attacker reaching the soldier (mission failure via geometry).

Conditions (identical clutter level/seed/hostile behavior across all
three -- fairness preserved):
    A_disabled -- defender_obstacle_avoidance_enabled=False (legacy/no
        safety layer; physical collisions are now consequential).
    B_level_b  -- defender_obstacle_avoidance_enabled=True,
        defender_obstacle_awareness_mode="local" (Level B).
    C_level_c  -- defender_obstacle_avoidance_enabled=True,
        defender_obstacle_awareness_mode="map_local" (Level C).

Reuses the EXISTING, canonical controller inventory
(experiments/clutter_benchmark_protocol.controller_instances(),
build_env_config()) -- no controller is duplicated or modified.

RUNTIME NOTE: RMPC remains ~150-400x slower than every other controller
(see clutter_benchmark_protocol.py's own Phase-8 runtime note); N=20 is
used (the lower end of the requested 20-30 range) to keep this a
"modest diagnostic" run, not a full statistical study.

Usage:
    python -m experiments.collision_semantics_benchmark
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
OUT_DIR = PROJECT_ROOT / "results" / "physical_collision_semantics" / "phase12_14_benchmark"
LEVELS = ("C1", "C2", "C3")
CONDITIONS = ("A_disabled", "B_level_b", "C_level_c")
N_SEEDS = 20  # "modest diagnostic" per this phase's instruction (range 20-30)
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
    """Run one episode, returning outcome/mission/collision/safety metrics
    (Phase 13's metric list), with defender mission success kept
    structurally separate from attacker environmental failure."""
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
    defender_path_length = 0.0
    enemy_path_length = 0.0
    prev_defender_pos = info["defender_pos"].copy()
    prev_enemy_pos = info["enemy_pos"].copy()
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

        defender_path_length += float(np.linalg.norm(info["defender_pos"] - prev_defender_pos))
        enemy_path_length += float(np.linalg.norm(info["enemy_pos"] - prev_enemy_pos))
        prev_defender_pos = info["defender_pos"].copy()
        prev_enemy_pos = info["enemy_pos"].copy()
        min_static_clearance = min(min_static_clearance, info["defender_min_obstacle_clearance"])
        min_dynamic_clearance = min(min_dynamic_clearance, info["defender_min_dynamic_obstacle_clearance"])
        if info["defender_avoidance_active"]:
            avoidance_active_steps += 1
            override_angles.append(info["defender_avoidance_angle_deg"])

        if done or truncated:
            break

    outcome = info["outcome"]
    return {
        "seed": seed, "feasible": True, "outcome": outcome,
        # Mission outcomes (unchanged geometry-based categories).
        "success": 1 if outcome == "intercepted" else 0,
        "soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        # NEW physical-collision outcome categories (mutually exclusive
        # with the mission outcomes above and with each other).
        "defender_static_collision": 1 if outcome == "defender_static_obstacle_collision" else 0,
        "defender_dynamic_collision": 1 if outcome == "defender_dynamic_obstacle_collision" else 0,
        "enemy_static_collision": 1 if outcome == "enemy_static_obstacle_collision" else 0,
        "enemy_dynamic_collision": 1 if outcome == "enemy_dynamic_obstacle_collision" else 0,
        # Aggregate: any physical collision terminated this episode.
        "defender_physical_failure": 1 if outcome in (
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision") else 0,
        "attacker_environmental_failure": 1 if outcome in (
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision") else 0,
        "episode_length": step,
        "interception_time": step if outcome == "intercepted" else None,
        "min_static_clearance": min_static_clearance,
        "min_dynamic_clearance": min_dynamic_clearance,
        "defender_path_length": defender_path_length,
        "enemy_path_length": enemy_path_length,
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
        defender_static_collision_rate=("defender_static_collision", "mean"),
        defender_dynamic_collision_rate=("defender_dynamic_collision", "mean"),
        enemy_static_collision_rate=("enemy_static_collision", "mean"),
        enemy_dynamic_collision_rate=("enemy_dynamic_collision", "mean"),
        defender_physical_failure_rate=("defender_physical_failure", "mean"),
        attacker_environmental_failure_rate=("attacker_environmental_failure", "mean"),
        mean_min_static_clearance=("min_static_clearance", "mean"),
        mean_defender_path_length=("defender_path_length", "mean"),
        mean_enemy_path_length=("enemy_path_length", "mean"),
        mean_avoidance_fraction=("avoidance_fraction", "mean"),
        mean_override_angle=("mean_override_angle", "mean"),
        max_override_angle=("max_override_angle", "max"),
        mean_step_time_ms=("mean_step_time_ms", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 240)
    print(summary.to_string(index=False))
    print(f"total wall time: {time.time() - t_start:.1f}s")
    return summary


if __name__ == "__main__":
    run_benchmark()
