"""
Phase R22: representative trajectory analysis for Adaptive Attacker v1.
Re-runs 5 SPECIFIC, already-identified (level, defender_instance, seed)
matched scenarios (scripted vs. the relevant learned-attacker seed),
recording full position trajectories (not stored during the main N=15
sweep, which only records summary metrics per episode). Produces one
trajectory plot per scenario (Fig 7 material) plus a JSON of purely
descriptive, metric-backed observations -- no intent/strategy is claimed
beyond what the recorded positions/metrics show.

Usage:
    python -m experiments.adaptive_attacker_trajectory_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.adaptive_attacker_env import build_attacker_observation
from experiments.adaptive_attacker_evaluation import (
    build_seen_defender_instances,
    build_unseen_defender_instances,
    _attacker_checkpoint_path,
)

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_evaluation"
OUT_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_figures"

# Phase R22: 5 representative scenarios, chosen from the actual completed
# episode dataset (results/adaptive_attacker_evaluation/episodes.csv) --
# not hand-invented. See adaptive_attacker_trajectory_selection_notes.md
# equivalent reasoning inline below each entry.
SCENARIOS = [
    {
        "name": "scripted_vs_strong_defender",
        "description": "Scripted attacker vs. LR-PPO-C (a 'seen' defender with a high scripted-arm "
                        "safe-intercept rate at C1, ~0.89) -- baseline reference.",
        "level": "C1", "defender_instance": "LRPPO-C-77", "seed": 190000, "learned_seed": 201,
    },
    {
        "name": "learned_succeeds_where_scripted_failed",
        "description": "Matched seed where the SAME defender (Greedy) safely intercepted the scripted "
                        "attacker but was reached (soldier_caught) by the learned attacker (seed 201).",
        "level": "C1", "defender_instance": "Greedy", "seed": 190003, "learned_seed": 201,
    },
    {
        "name": "learned_still_intercepted",
        "description": "A matched scenario where the learned attacker (seed 202) was still safely "
                        "intercepted by a strong defender (LR-PPO-C) -- the attacker does not win every case. "
                        "Verified against the completed episode dataset before selection.",
        "level": "C1", "defender_instance": "LRPPO-C-77", "seed": 190002, "learned_seed": 202,
    },
    {
        "name": "high_clutter_c3_behavior",
        "description": "High-clutter (C3) behavior of the strongest learned-attacker seed (202) against "
                        "a clutter-trained defender (PPO-Kalman-C).",
        "level": "C3", "defender_instance": "PPOK-C-74", "seed": 190000, "learned_seed": 202,
    },
    {
        "name": "unseen_defender_generalization",
        "description": "Learned attacker (seed 203) vs. Greedy -- a defender family NEVER present in "
                        "the attacker's training pool -- demonstrating generalization beyond seen opponents.",
        "level": "C1", "defender_instance": "Greedy", "seed": 190004, "learned_seed": 203,
    },
]


def _find_defender(instance_id: str):
    for d in build_seen_defender_instances() + build_unseen_defender_instances():
        if d.instance_id == instance_id:
            return d
    raise ValueError(f"unknown defender instance {instance_id!r}")


def _run_episode_with_trajectory(env, defender_policy, defender, attacker_model, seed):
    pending = {"action": np.zeros(3, dtype=np.float32)}
    if attacker_model is not None:
        env.enemy_tactical_override = lambda: pending["action"]
    else:
        env.enemy_tactical_override = None
    obs, info = env.reset(seed=seed)
    defender_policy.reset()
    enemy_traj, defender_traj, soldier_traj = [env._enemy_pos.copy()], [env._defender_pos.copy()], [env._soldier_pos.copy()]
    avoidance_flags = [bool(info.get("enemy_obstacle_avoidance_active", False))]
    steps = 0
    done = False
    while not done:
        if attacker_model is not None:
            attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
            action, _ = attacker_model.predict(attacker_obs, deterministic=True)
            pending["action"] = np.asarray(action, dtype=np.float32)
        defender_obs = env._get_obs()
        defender_info = build_policy_info(env._get_info(), defender.estimator)
        if defender.needs_sector_features:
            sector = compute_sector_features(
                defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
            )
            if defender.is_lr_ppo:
                defender_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
            else:
                defender_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)
        else:
            defender_action = defender_policy.act(defender_obs, defender_info)
        obs, reward, terminated, truncated, info = env.step(np.asarray(defender_action, dtype=np.float32))
        steps += 1
        enemy_traj.append(env._enemy_pos.copy())
        defender_traj.append(env._defender_pos.copy())
        soldier_traj.append(env._soldier_pos.copy())
        avoidance_flags.append(bool(info.get("enemy_obstacle_avoidance_active", False)))
        done = terminated or truncated
    enemy_traj = np.array(enemy_traj)
    defender_traj = np.array(defender_traj)
    soldier_traj = np.array(soldier_traj)
    path_length = float(np.sum(np.linalg.norm(np.diff(enemy_traj, axis=0), axis=1)))
    return {
        "outcome": info["outcome"], "steps": steps,
        "enemy_trajectory": enemy_traj, "defender_trajectory": defender_traj, "soldier_trajectory": soldier_traj,
        "avoidance_fraction": float(np.mean(avoidance_flags)),
        "path_length": path_length,
        "altitude_range": [float(enemy_traj[:, 2].min()), float(enemy_traj[:, 2].max())],
        "lateral_extent_xy": [float(np.ptp(enemy_traj[:, 0])), float(np.ptp(enemy_traj[:, 1]))],
    }


def _plot_scenario(scenario_name, scripted_result, learned_result, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, result, label in zip(axes, (scripted_result, learned_result), ("scripted", "learned")):
        et, dt, st = result["enemy_trajectory"], result["defender_trajectory"], result["soldier_trajectory"]
        ax.plot(et[:, 0], et[:, 1], color="tab:red", label="attacker (enemy)", linewidth=1.5)
        ax.scatter(et[0, 0], et[0, 1], color="tab:red", marker="o", s=40, zorder=5)
        ax.scatter(et[-1, 0], et[-1, 1], color="tab:red", marker="x", s=60, zorder=5)
        ax.plot(dt[:, 0], dt[:, 1], color="tab:blue", label="defender", linewidth=1.5)
        ax.scatter(dt[0, 0], dt[0, 1], color="tab:blue", marker="o", s=40, zorder=5)
        ax.plot(st[:, 0], st[:, 1], color="tab:green", label="soldier (asset)", linewidth=1.0, linestyle="--")
        ax.set_title(f"{label}: outcome={result['outcome']}, steps={result['steps']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Fig 7 ({scenario_name}): representative scripted vs. learned trajectories (XY plane)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig7_trajectory_{scenario_name}.png", dpi=150)
    plt.close(fig)


def run_trajectory_analysis() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in SCENARIOS:
        defender = _find_defender(scenario["defender_instance"])
        base_config = build_canonical_env_config(scenario["level"], defender.estimator)
        defender_policy = defender.build(base_config)
        env = SoldierEnv(base_config)

        scripted_result = _run_episode_with_trajectory(env, defender_policy, defender, None, scenario["seed"])
        model = PPO.load(str(_attacker_checkpoint_path(scenario["learned_seed"])))
        learned_result = _run_episode_with_trajectory(env, defender_policy, defender, model, scenario["seed"])

        _plot_scenario(scenario["name"], scripted_result, learned_result, OUT_DIR)

        results[scenario["name"]] = {
            "description": scenario["description"],
            "level": scenario["level"], "defender_instance": scenario["defender_instance"],
            "seed": scenario["seed"], "learned_seed": scenario["learned_seed"],
            "scripted": {k: v for k, v in scripted_result.items() if not k.endswith("_trajectory")},
            "learned": {k: v for k, v in learned_result.items() if not k.endswith("_trajectory")},
        }
        print(scenario["name"], "scripted outcome=", scripted_result["outcome"],
              "learned outcome=", learned_result["outcome"], flush=True)

    (OUT_DIR / "trajectory_analysis_summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run_trajectory_analysis()
