"""
Shared training runner for the obstacle-aware clutter retraining campaign
(journal extension, Phase 12/16/17/18/19). ONE implementation used for all
three architectures (PPO, PPO-Kalman, LR-PPO) so training budgets/
hyperparameters/evaluation cadence are identical across them (Phase 16
fairness) -- only the environment-construction function differs per
architecture.

Observation-versioning (Phase 12): every run writes a `training_manifest.json`
recording the observation dimension/feature names/order/normalization/
awareness mode/sensor range/clutter training distribution/seed/git commit/
config hash, so old conference checkpoints (which never wrote these fields)
remain unambiguously distinguishable and independently loadable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from experiments import ppo_hyperparams as H
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, SectorObservationSpec

PROJECT_ROOT = Path(__file__).parent.parent

# Phase 4: deterministic, non-overlapping per-worker seed stride. Worker i
# (0-indexed) in an N-worker vectorized run uses `seed + i * SEED_STRIDE`
# for BOTH its env construction (architectures whose seed is baked in at
# construction, e.g. LR-PPO's root_seed) and its first `reset(seed=...)`
# call (architectures whose seed drives reset-time resampling, e.g.
# PPO/PPO-Kalman's randomized-clutter wrapper). 10_000 matches the stride
# already used elsewhere in this repo for seed-namespace separation
# (e.g. VALIDATION_SEED_OFFSET) and comfortably exceeds any plausible
# num_envs, so worker seed ranges never collide with each other.
SEED_STRIDE = 10_000

CLUTTER_HYPERPARAMS = dict(
    learning_rate=H.LEARNING_RATE, gamma=H.GAMMA, batch_size=H.BATCH_SIZE, n_steps=H.N_STEPS,
    n_epochs=H.N_EPOCHS, clip_range=H.CLIP_RANGE, ent_coef=H.ENT_COEF, vf_coef=H.VF_COEF,
    max_grad_norm=H.MAX_GRAD_NORM, gae_lambda=H.GAE_LAMBDA,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class _LightweightEvalCallback(BaseCallback):
    """Periodic light-weight evaluation (Phase 19 metrics), run every
    `eval_freq` timesteps on a FRESH held-out-style env instance (built by
    the same `env_fn`, i.e. randomized clutter, so this is a same-
    distribution sanity eval, not the held-out Phase 23 evaluation)."""

    def __init__(self, env_fn: Callable[[int], object], eval_seed: int, n_episodes: int, eval_freq: int,
                 best_model_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self._env_fn = env_fn
        self._eval_seed = eval_seed
        self._n_episodes = n_episodes
        self._eval_freq = eval_freq
        self._best_model_path = best_model_path
        self._best_success_rate = -1.0
        self._last_eval_step = 0
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self._eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        env = self._env_fn(self._eval_seed)
        outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0,
                    "defender_static_obstacle_collision": 0, "defender_dynamic_obstacle_collision": 0,
                    "enemy_static_obstacle_collision": 0, "enemy_dynamic_obstacle_collision": 0}
        avoidance_steps = 0
        total_steps = 0
        for i in range(self._n_episodes):
            obs, info = env.reset(seed=90_000_000 + i)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_steps += 1
                if info.get("defender_avoidance_active"):
                    avoidance_steps += 1
            outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
        success_rate = outcomes["intercepted"] / self._n_episodes
        snapshot = {
            "timesteps": self.num_timesteps, "success_rate": success_rate,
            "outcomes": dict(outcomes),
            "avoidance_fraction": avoidance_steps / total_steps if total_steps else 0.0,
        }
        self.history.append(snapshot)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] success_rate={success_rate:.3f} outcomes={outcomes}", flush=True)
        if success_rate >= self._best_success_rate:
            self._best_success_rate = success_rate
            self.model.save(str(self._best_model_path))
        return True


def run_clutter_training(
    architecture: str,
    env_fn: Callable[[int], object],
    seed: int,
    output_dir: Path,
    total_timesteps: int,
    observation_spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
    eval_freq: int = 10_000,
    eval_episodes: int = 20,
    checkpoint_freq: int = 25_000,
    training_config_summary: dict | None = None,
    verbose: int = 1,
    progress_bar: bool = False,
    num_envs: int = 1,
    vec_env_cls: str = "auto",
) -> dict:
    """Train ONE clutter-aware PPO model (architecture in
    {"ppo_clutter_v1","ppo_kalman_clutter_v1","lrppo_clutter_v1"}) using
    the SAME hyperparameters/eval-cadence/checkpoint-cadence for every
    architecture (Phase 16).

    `env_fn(worker_seed)` must construct ONE environment instance; for
    architectures whose seed is applied at construction (e.g. LR-PPO's
    `root_seed`) `env_fn` should use `worker_seed` directly, and for
    architectures whose seed is applied via `reset(seed=...)` (e.g.
    PPO/PPO-Kalman's randomized-clutter wrapper) `env_fn` may ignore its
    argument -- this runner ALWAYS also calls `env.reset(seed=worker_seed)`
    after construction, so either convention (or both) works correctly.

    `num_envs` (Phase 3): number of parallel environment workers.
    `num_envs=1` reproduces the exact prior single-environment behavior
    (DummyVecEnv, `worker_seed == seed`). `num_envs>1` uses
    `SubprocVecEnv` by default (`vec_env_cls="auto"`; pass "dummy" to
    force `DummyVecEnv` for testing/debugging), with DETERMINISTIC,
    NON-OVERLAPPING per-worker seeds `seed + worker_id * SEED_STRIDE`
    (Phase 4) -- worker 0 always uses exactly `seed`, so `num_envs=1`
    and the first worker of any `num_envs>1` run start from the SAME
    per-worker seed (though overall training dynamics still differ once
    >1 worker contributes transitions to the same rollout buffer).
    """
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    monitor_path = output_dir / "monitor"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)

    worker_seeds = [seed + i * SEED_STRIDE for i in range(num_envs)]

    def _make(worker_seed: int, worker_idx: int):
        def _init():
            env = env_fn(worker_seed)
            env.reset(seed=worker_seed)
            return Monitor(env, str(monitor_path / f"worker_{worker_idx}"))
        return _init

    monitor_path.mkdir(parents=True, exist_ok=True)
    env_fns = [_make(ws, i) for i, ws in enumerate(worker_seeds)]
    if num_envs == 1:
        train_env = DummyVecEnv(env_fns)
    elif vec_env_cls == "dummy":
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)

    model = PPO(policy="MlpPolicy", env=train_env, seed=seed, verbose=verbose, **CLUTTER_HYPERPARAMS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // num_envs, 1), save_path=str(checkpoints_dir),
        name_prefix=f"{architecture}_seed{seed}", save_replay_buffer=False,
    )
    eval_cb = _LightweightEvalCallback(
        env_fn=env_fn, eval_seed=seed, n_episodes=eval_episodes, eval_freq=eval_freq,
        best_model_path=output_dir / "best_model", verbose=verbose,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=CallbackList([checkpoint_cb, eval_cb]),
                progress_bar=progress_bar)
    elapsed = time.time() - t0

    train_env.close()
    model.save(str(output_dir / "final_model"))
    if not (output_dir / "best_model.zip").exists():
        model.save(str(output_dir / "best_model"))

    manifest = {
        "architecture": architecture,
        "seed": seed,
        "num_envs": num_envs,
        "worker_seeds": worker_seeds,
        "git_commit": _git_commit(),
        "total_timesteps": total_timesteps,
        # Phase 5 accounting: SB3's `num_timesteps` is the CUMULATIVE count
        # of env-interactions across ALL parallel workers (one VecEnv
        # `.step()` call == `num_envs` interactions, incrementing
        # `num_timesteps` by `num_envs`) -- so `total_timesteps` means the
        # SAME total environment experience regardless of `num_envs`; more
        # workers only change how that experience is split into rollouts,
        # never the total (verified by test, see test_vectorized_clutter_training.py).
        # NOTE: SB3 always completes a FULL rollout (`n_steps * num_envs`
        # transitions) before checking the stop condition, so the ACTUAL
        # number of transitions collected (`actual_total_timesteps`) can
        # OVERSHOOT the requested `total_timesteps` by up to one rollout's
        # worth -- use `actual_total_timesteps`, not `total_timesteps`, for
        # any throughput/steps-per-second calculation.
        "actual_total_timesteps": int(model.num_timesteps),
        "rollout_n_steps_per_worker": CLUTTER_HYPERPARAMS["n_steps"],
        "effective_rollout_batch": CLUTTER_HYPERPARAMS["n_steps"] * num_envs,
        "total_transitions": int(model.num_timesteps),
        "n_gradient_update_iterations": model.num_timesteps // (CLUTTER_HYPERPARAMS["n_steps"] * num_envs),
        "hyperparameters": CLUTTER_HYPERPARAMS,
        "observation_spec": observation_spec.to_dict(),
        "training_config_summary": training_config_summary or {},
        "config_hash": _config_hash(training_config_summary or {}),
        "eval_history": eval_cb.history,
        "wall_time_s": elapsed,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest

