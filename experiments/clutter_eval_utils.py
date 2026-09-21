"""
Episode runner and per-episode metrics for the clutter benchmark (Phase 6
metrics; Phase 7 feasibility accounting).

Extends the existing experiments/eval_utils.py convention (policy.act(obs,
policy_info) / policy.reset(), sanitized info via build_policy_info) with:
    - obstacle-collision/clearance diagnostics,
    - path-length and control-effort/turn-rate utilization,
    - LR-PPO residual-magnitude/angle capture (best-effort, only for
      LeadResidualPPOPolicyWrapper instances),
    - explicit PLACEMENT-INFEASIBLE scenario handling: if the requested
      clutter level's obstacle layout cannot be placed for a given seed
      (uav_defend/obstacles/layout.py raises RuntimeError), the episode is
      reported with feasible=False and no other metrics are fabricated.

This module runs NO training and modifies NO policy/environment logic.
"""

from __future__ import annotations

import numpy as np

from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env


def run_clutter_episode(env, policy, seed: int) -> dict:
    """Run one episode and return a flat dict of clutter-benchmark metrics.

    On obstacle-layout placement failure (RuntimeError from env.reset()),
    returns {"seed": seed, "feasible": False, "infeasibility_reason": str(e)}
    and nothing else -- callers must branch on "feasible" before reading
    any other key.
    """
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError as e:
        return {"seed": seed, "feasible": False, "infeasibility_reason": str(e)}

    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    config = env.config

    total_reward = 0.0
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    min_defender_enemy_dist = info["defender_enemy_dist"]
    min_defender_obstacle_clearance = info["defender_min_obstacle_clearance"]
    min_enemy_obstacle_clearance = info["enemy_min_obstacle_clearance"]
    detection_time = None
    intercept_time = None

    defender_path_length = 0.0
    enemy_path_length = 0.0
    prev_defender_pos = info["defender_pos"].copy()
    prev_enemy_pos = info["enemy_pos"].copy()

    accel_utilizations = []
    turn_utilizations = []
    residual_angles_deg = []

    has_lead_reference = hasattr(policy, "_lead_policy")

    step = 0
    done = False
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)

        if has_lead_reference:
            lead_dir = np.asarray(policy._lead_policy.act(obs, policy_info), dtype=np.float64)
            act_dir = np.asarray(action, dtype=np.float64)
            ln, an = np.linalg.norm(lead_dir), np.linalg.norm(act_dir)
            if ln > 1e-8 and an > 1e-8:
                cos_angle = np.clip(np.dot(lead_dir, act_dir) / (ln * an), -1.0, 1.0)
                residual_angles_deg.append(float(np.degrees(np.arccos(cos_angle))))

        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward

        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])
        min_defender_enemy_dist = min(min_defender_enemy_dist, info["defender_enemy_dist"])
        min_defender_obstacle_clearance = min(min_defender_obstacle_clearance, info["defender_min_obstacle_clearance"])
        min_enemy_obstacle_clearance = min(min_enemy_obstacle_clearance, info["enemy_min_obstacle_clearance"])

        defender_path_length += float(np.linalg.norm(info["defender_pos"] - prev_defender_pos))
        enemy_path_length += float(np.linalg.norm(info["enemy_pos"] - prev_enemy_pos))
        prev_defender_pos = info["defender_pos"].copy()
        prev_enemy_pos = info["enemy_pos"].copy()

        if config.defender_max_accel > 0:
            accel_utilizations.append(info["defender_accel"] / config.defender_max_accel)
        if config.defender_max_turn_rate_deg > 0:
            turn_utilizations.append(info["defender_turn_rate"] / config.defender_max_turn_rate_deg)

        if info["enemy_detected"] and detection_time is None:
            detection_time = step

        if done or truncated:
            break

    if info["outcome"] == "intercepted":
        intercept_time = step

    metrics = {
        "seed": seed,
        "feasible": True,
        "outcome": info["outcome"],
        "success": 1 if info["outcome"] == "intercepted" else 0,
        "soldier_caught": 1 if info["outcome"] == "soldier_caught" else 0,
        "unsafe_intercept": 1 if info["outcome"] == "unsafe_intercept" else 0,
        "timeout": 1 if info["outcome"] == "timeout" else 0,
        "episode_length": step,
        "episode_duration_s": step * config.dt,
        "total_reward": total_reward,
        "detected": 1 if detection_time is not None else 0,
        "detection_time": detection_time if detection_time is not None else -1,
        "intercept_time": intercept_time if intercept_time is not None else -1,
        "min_enemy_soldier_dist": min_enemy_soldier_dist,
        "min_defender_enemy_dist": min_defender_enemy_dist,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        # Obstacle diagnostics (see clutter_benchmark_protocol module docstring:
        # DIAGNOSTIC ONLY, never blocking -- collision does not affect outcome).
        "obstacles_enabled": info["obstacles_enabled"],
        "obstacle_count": info["obstacle_count"],
        "defender_obstacle_collision": 1 if info["defender_ever_collided"] else 0,
        "enemy_obstacle_collision": 1 if info["enemy_ever_collided"] else 0,
        "defender_collision_count": info["defender_collision_count"],
        "enemy_collision_count": info["enemy_collision_count"],
        "min_defender_obstacle_clearance": (
            float(min_defender_obstacle_clearance) if np.isfinite(min_defender_obstacle_clearance) else float("nan")
        ),
        "min_enemy_obstacle_clearance": (
            float(min_enemy_obstacle_clearance) if np.isfinite(min_enemy_obstacle_clearance) else float("nan")
        ),
        "soldier_obstacle_block_count": info["soldier_obstacle_block_count"],
        # Efficiency / control-effort metrics.
        "defender_path_length": defender_path_length,
        "enemy_path_length": enemy_path_length,
        "mean_accel_utilization": float(np.mean(accel_utilizations)) if accel_utilizations else float("nan"),
        "mean_turn_rate_utilization": float(np.mean(turn_utilizations)) if turn_utilizations else float("nan"),
    }
    if residual_angles_deg:
        metrics["mean_residual_angle_deg"] = float(np.mean(residual_angles_deg))
        metrics["max_residual_angle_deg"] = float(np.max(residual_angles_deg))
    return metrics
