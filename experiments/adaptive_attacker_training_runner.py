"""
Adaptive Attacker v1 training runner (journal extension, Phase 4/16/17).
A SEPARATE pipeline from the defender's `clutter_training_runner.py`
(Phase 4 explicit instruction: "Create a separate attacker training
pipeline. Do NOT modify defender PPO implementations."), reusing the SAME
vectorization APPROACH (SEED_STRIDE, DummyVecEnv @ num_envs=1 /
SubprocVecEnv @ num_envs>1) validated by the defender clutter-training
scale-up sub-task, applied to `AdaptiveAttackerTrainingEnv` instead.

Success metric here is SYMMETRIC-BUT-OPPOSITE to the defender's own
training runner: `success_rate = P(soldier_caught)` (the ATTACKER's
mission-success condition), NOT `P(intercepted)` (that would be the
DEFENDER's success metric and must never be conflated here).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from experiments import ppo_hyperparams as H
from experiments.adaptive_attacker_env import AdaptiveAttackerTrainingEnv, DefenderPoolMember, build_defender_pool

PROJECT_ROOT = Path(__file__).parent.parent

# Phase 15: SAME deterministic per-worker seed-stride scheme validated for
# defender clutter training (experiments/clutter_training_runner.SEED_STRIDE).
SEED_STRIDE = 10_000

ATTACKER_HYPERPARAMS = dict(
    learning_rate=H.LEARNING_RATE, gamma=H.GAMMA, batch_size=H.BATCH_SIZE, n_steps=H.N_STEPS,
    n_epochs=H.N_EPOCHS, clip_range=H.CLIP_RANGE, ent_coef=H.ENT_COEF, vf_coef=H.VF_COEF,
    max_grad_norm=H.MAX_GRAD_NORM, gae_lambda=H.GAE_LAMBDA,
)


def _config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


class _AttackerLightweightEvalCallback(BaseCallback):
    """Periodic same-distribution eval (Phase 18 in-training checkpoint
    selection -- NOT the Phase 19+ held-out evaluation). success_rate =
    P(soldier_caught): the ATTACKER's own mission-success condition."""

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
        outcomes: dict[str, int] = {}
        total_steps = 0
        total_progress = 0.0
        for i in range(self._n_episodes):
            obs, info = env.reset(seed=self._eval_seed + i)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_steps += 1
            outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
            total_progress += info["asset_progress_per_step"]
        success_rate = outcomes.get("soldier_caught", 0) / self._n_episodes
        snapshot = {
            "timesteps": self.num_timesteps, "success_rate": success_rate,
            "outcomes": dict(outcomes),
            "mean_asset_progress_per_step": total_progress / self._n_episodes,
        }
        self.history.append(snapshot)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] success_rate={success_rate:.3f} outcomes={outcomes}", flush=True)
        if success_rate >= self._best_success_rate:
            self._best_success_rate = success_rate
            self.model.save(str(self._best_model_path))
        return True


def run_adaptive_attacker_training(
    env_fn: Callable[[int], object],
    seed: int,
    output_dir: Path,
    total_timesteps: int,
    eval_freq: int = 10_000,
    eval_episodes: int = 20,
    eval_seed: int | None = None,
    checkpoint_freq: int = 25_000,
    training_config_summary: dict | None = None,
    opponent_pool_spec: list[dict] | None = None,
    verbose: int = 1,
    progress_bar: bool = False,
    num_envs: int = 1,
    vec_env_cls: str = "auto",
) -> dict:
    """Trains ONE Adaptive Attacker v1 PPO model. Mirrors
    `clutter_training_runner.run_clutter_training`'s vectorization
    approach exactly (num_envs=1 -> DummyVecEnv, num_envs>1 ->
    SubprocVecEnv with worker_seed = seed + i*SEED_STRIDE), but is a
    SEPARATE function/module per Phase 4's explicit "separate attacker
    training pipeline" instruction, with an attacker-appropriate
    (soldier_caught-based) success metric."""
    from experiments.adaptive_attacker_env import ATTACKER_VALIDATION_SEED_START

    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir / "monitor"
    monitor_path.mkdir(parents=True, exist_ok=True)

    worker_seeds = [seed + i * SEED_STRIDE for i in range(num_envs)]

    def _make(worker_seed: int, worker_idx: int):
        def _init():
            env = env_fn(worker_seed)
            env.reset(seed=worker_seed)
            return Monitor(env, str(monitor_path / f"worker_{worker_idx}"))
        return _init

    env_fns = [_make(ws, i) for i, ws in enumerate(worker_seeds)]
    if num_envs == 1:
        train_env = DummyVecEnv(env_fns)
    elif vec_env_cls == "dummy":
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)

    model = PPO(policy="MlpPolicy", env=train_env, seed=seed, verbose=verbose, **ATTACKER_HYPERPARAMS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // num_envs, 1), save_path=str(checkpoints_dir),
        name_prefix=f"adaptive_attacker_v1_seed{seed}", save_replay_buffer=False,
    )
    eval_cb = _AttackerLightweightEvalCallback(
        env_fn=env_fn, eval_seed=eval_seed if eval_seed is not None else ATTACKER_VALIDATION_SEED_START,
        n_episodes=eval_episodes, eval_freq=eval_freq,
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
        "architecture": "adaptive_attacker_v1",
        "seed": seed,
        "num_envs": num_envs,
        "worker_seeds": worker_seeds,
        "git_commit": _git_commit(),
        "total_timesteps": total_timesteps,
        "actual_total_timesteps": int(model.num_timesteps),
        "rollout_n_steps_per_worker": ATTACKER_HYPERPARAMS["n_steps"],
        "effective_rollout_batch": ATTACKER_HYPERPARAMS["n_steps"] * num_envs,
        "hyperparameters": ATTACKER_HYPERPARAMS,
        "training_config_summary": training_config_summary or {},
        "opponent_pool_spec": opponent_pool_spec or [],
        "config_hash": _config_hash(training_config_summary or {}),
        "eval_history": eval_cb.history,
        "wall_time_s": elapsed,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
