"""
Phase 21-25: primary comparison evaluation for the obstacle-aware clutter
retraining study (journal extension). Compares, under IDENTICAL canonical
clutter scenarios (Phase 2 protocol, held-out seeds -- Phase 23):

    - classical baselines (Greedy, PN, Lead, RMPC) + shared safety layer
      (Phase 22 -- they never receive the new RL obstacle observation);
    - FROZEN conference PPO/PPO-Kalman/LR-PPO (16-D / 19-D observation,
      unchanged) + shared safety layer (Phase 21 zero-shot-degradation arm);
    - CLUTTER-TRAINED PPO-C/PPOK-C/LRPPO-C (36-D / 39-D observation with
      sector features) + shared safety layer (Phase 21 recovery arm).

Every controller is evaluated with the SAME held-out seed set (never used
in training -- see `journal_clutter_protocol.CLUTTER_EVAL_SEED_START`) at
the SAME canonical clutter levels (Phase 2), so results are directly
comparable (Phase 26 fairness precedent).

RUNTIME NOTE: RMPC remains ~150-400x slower than every other controller
(documented in `clutter_benchmark_protocol.py`); `RMPC_N_EPISODES` may use
a smaller matched subset, reported separately (Phase 23's own allowance).

Usage:
    python -m experiments.clutter_retraining_evaluation
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_clutter_protocol import (
    CANONICAL_CLUTTER_LEVEL_ORDER,
    CLUTTER_EVAL_SEED_START,
    build_canonical_env_config,
)
from experiments.clutter_benchmark_protocol import controller_instances as frozen_controller_instances

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "clutter_retraining_evaluation"

EVAL_LEVELS = ("C1", "C2", "C3")
N_EPISODES_DEFAULT = 30  # Phase 23 target is >=500; see module docstring
RMPC_N_EPISODES = 10


def held_out_seeds(n_episodes: int) -> tuple[int, ...]:
    return tuple(range(CLUTTER_EVAL_SEED_START, CLUTTER_EVAL_SEED_START + n_episodes))


def _run_episode(env, policy, seed: int, obs_slice: slice | None) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset()
    estimator_mode = "kalman" if env.unwrapped.config.use_kalman_tracking else "measurement"
    avoidance_steps = 0
    override_angles = []
    min_static_clearance = info["defender_min_obstacle_clearance"]
    step = 0
    done = False
    while not done:
        policy_obs = obs[obs_slice] if obs_slice is not None else obs
        action = np.asarray(policy.act(policy_obs, build_policy_info(info, estimator_mode)), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        min_static_clearance = min(min_static_clearance, info["defender_min_obstacle_clearance"])
        if info.get("defender_avoidance_active"):
            avoidance_steps += 1
            override_angles.append(info["defender_avoidance_angle_deg"])
    outcome = info["outcome"]
    return {
        "seed": seed, "outcome": outcome,
        "safe_intercept": 1 if outcome == "intercepted" else 0,
        "defender_collision": 1 if outcome in (
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision") else 0,
        "attacker_environmental_collision": 1 if outcome in (
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision") else 0,
        "soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        "episode_length": step,
        "intercept_time": step if outcome == "intercepted" else None,
        "min_clearance": min_static_clearance,
        "avoidance_fraction": avoidance_steps / step if step else 0.0,
        "mean_override_angle": float(np.mean(override_angles)) if override_angles else 0.0,
        "p95_override_angle": float(np.percentile(override_angles, 95)) if override_angles else 0.0,
    }


def evaluate_frozen_and_classical(n_episodes: int = N_EPISODES_DEFAULT, rmpc_n_episodes: int = RMPC_N_EPISODES) -> pd.DataFrame:
    """Phase 21 (frozen arm) + Phase 22 (classical baselines): every
    instance from the existing `controller_instances()` inventory, run
    under the canonical clutter protocol with held-out seeds. Frozen
    learned policies receive obs[:16] (their original observation slice);
    classical policies receive the same slice for API compatibility (they
    never use the appended sector features)."""
    from uav_defend.envs import SoldierEnv
    from experiments.clutter_training_env import build_ppo_clutter_env

    rows = []
    instances = frozen_controller_instances()
    for level in EVAL_LEVELS:
        for inst in instances:
            n = rmpc_n_episodes if inst.family == "RMPC" else n_episodes
            seeds = held_out_seeds(n)
            base_config = build_canonical_env_config(level, inst.estimator)
            env = build_ppo_clutter_env(inst.estimator, randomized=False, fixed_config=base_config)
            policy = inst.build(base_config)
            t0 = time.time()
            for seed in seeds:
                metrics = _run_episode(env, policy, seed, obs_slice=slice(0, 16))
                metrics.update(level=level, family=inst.family, instance_id=inst.instance_id, arm="frozen_or_classical")
                rows.append(metrics)
            print(f"level={level} instance={inst.instance_id:>10s} n={n} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def evaluate_clutter_trained(clutter_checkpoints: dict[str, str], n_episodes: int = N_EPISODES_DEFAULT) -> pd.DataFrame:
    """Phase 21 (recovery arm): clutter-trained PPO-C/PPOK-C/LRPPO-C,
    receiving the FULL augmented observation. `clutter_checkpoints` maps
    {"ppo_clutter_v1": path, "ppo_kalman_clutter_v1": path, "lrppo_clutter_v1": path}."""
    from stable_baselines3 import PPO
    from experiments.clutter_training_env import build_ppo_clutter_env
    from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env

    class _SB3PolicyAdapter:
        def __init__(self, model):
            self.model = model
        def reset(self):
            pass
        def act(self, obs, info):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

    rows = []
    for level in EVAL_LEVELS:
        for arch, path in clutter_checkpoints.items():
            model = PPO.load(path)
            policy = _SB3PolicyAdapter(model)
            estimator = "kalman" if arch == "ppo_kalman_clutter_v1" else "measurement"
            if arch == "lrppo_clutter_v1":
                base_config = build_canonical_env_config(level, "measurement")
                env = build_lr_ppo_clutter_env(root_seed=CLUTTER_EVAL_SEED_START)
            else:
                base_config = build_canonical_env_config(level, estimator)
                env = build_ppo_clutter_env(estimator, randomized=False, fixed_config=base_config)
            seeds = held_out_seeds(n_episodes)
            t0 = time.time()
            for seed in seeds:
                metrics = _run_episode(env, policy, seed, obs_slice=None)
                metrics.update(level=level, family=arch, instance_id=arch, arm="clutter_trained")
                rows.append(metrics)
            print(f"level={level} instance={arch:>18s} n={n_episodes} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["level", "family", "arm"]).agg(
        n=("seed", "count"),
        safe_intercept_rate=("safe_intercept", "mean"),
        defender_collision_rate=("defender_collision", "mean"),
        attacker_environmental_collision_rate=("attacker_environmental_collision", "mean"),
        soldier_caught_rate=("soldier_caught", "mean"),
        unsafe_intercept_rate=("unsafe_intercept", "mean"),
        timeout_rate=("timeout", "mean"),
        mean_min_clearance=("min_clearance", "mean"),
        mean_avoidance_fraction=("avoidance_fraction", "mean"),
        mean_override_angle=("mean_override_angle", "mean"),
        p95_override_angle=("p95_override_angle", "mean"),
    ).reset_index()


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = evaluate_frozen_and_classical()
    df.to_csv(OUT_DIR / "frozen_and_classical_episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT_DIR / "frozen_and_classical_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
