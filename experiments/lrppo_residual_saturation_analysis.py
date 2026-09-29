"""
Phase 20: LR-PPO residual-saturation analysis (journal extension). Answers
whether the fixed 30-degree residual bound becomes restrictive in clutter
by measuring, over held-out episodes:

    P(|theta_residual| >= threshold), for threshold near theta_max,

conditioned on: overall, near obstacles (low ground-truth clearance),
during safety-layer intervention, during failed interception, and during
collision. Does NOT change theta_max -- read-only ablation/diagnostic.

Usage:
    python -m experiments.lrppo_residual_saturation_analysis <model_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from experiments.journal_clutter_protocol import CLUTTER_EVAL_SEED_START
from experiments.lr_ppo_clutter_training_env import LeadResidualClutterTrainingEnv, LRPPOSectorObservationWrapper
from experiments.journal_clutter_protocol import build_randomized_training_config

NEAR_THETA_MAX_FRACTION = 0.9  # "near 30 deg" := >= 90% of theta_max


def run_analysis(model_path: str, n_episodes: int = 30, theta_max: float = 30.0) -> pd.DataFrame:
    model = PPO.load(model_path)
    rows = []
    for i in range(n_episodes):
        seed = CLUTTER_EVAL_SEED_START + i
        config = build_randomized_training_config(np.random.default_rng(seed), "measurement")
        inner = LeadResidualClutterTrainingEnv(config=config, root_seed=seed, residual_max_angle_deg=theta_max)
        env = LRPPOSectorObservationWrapper(inner)
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            lead_dir = np.asarray(inner._last_lead_direction, dtype=np.float64)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            se = env._soldier_env()
            hybrid = np.asarray(se._defender_desired_command, dtype=np.float64)
            ln, hn = np.linalg.norm(lead_dir), np.linalg.norm(hybrid)
            theta_residual = 0.0
            if ln > 1e-8 and hn > 1e-8:
                cos_a = np.clip(np.dot(lead_dir, hybrid) / (ln * hn), -1.0, 1.0)
                theta_residual = float(np.degrees(np.arccos(cos_a)))
            rows.append({
                "seed": seed, "theta_residual": theta_residual,
                "near_theta_max": theta_residual >= NEAR_THETA_MAX_FRACTION * theta_max,
                "safety_active": bool(info.get("defender_avoidance_active", False)),
                "min_clearance": info.get("defender_actual_min_clearance", float("inf")),
                "outcome": info["outcome"],
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    near_clearance = df["min_clearance"] < 5.0
    failed = df["outcome"].isin(("soldier_caught", "unsafe_intercept", "timeout"))
    collided = df["outcome"].str.contains("obstacle_collision")
    return {
        "P(near_theta_max)_overall": float(df["near_theta_max"].mean()),
        "P(near_theta_max)_near_obstacles": float(df.loc[near_clearance, "near_theta_max"].mean()) if near_clearance.any() else float("nan"),
        "P(near_theta_max)_during_safety_intervention": float(df.loc[df["safety_active"], "near_theta_max"].mean()) if df["safety_active"].any() else float("nan"),
        "P(near_theta_max)_during_failed_interception": float(df.loc[failed, "near_theta_max"].mean()) if failed.any() else float("nan"),
        "P(near_theta_max)_during_collision": float(df.loc[collided, "near_theta_max"].mean()) if collided.any() else float("nan"),
        "n_step_samples": int(len(df)),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/clutter_diagnostic_training/lrppo_clutter_v1/seed_77/best_model.zip"
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    df = run_analysis(path, n_episodes=n_episodes)
    print(summarize(df))
