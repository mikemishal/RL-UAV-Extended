"""
Continuation-aware, restart-safe training runner for the defender-
adaptation campaign (journal extension, 34-phase defender-adaptation
sub-task, Phase 8/13/16/17/18).

Unlike `experiments/clutter_training_runner.py::run_clutter_training`
(which always constructs a FRESH `PPO(...)`), this runner ALWAYS continues
an EXISTING checkpoint via `PPO.load(path, env=...)` +
`model.learn(..., reset_num_timesteps=False)` -- used for BOTH D0+
(`generation="d0_plus"`, 100% scripted opponent, identical env_fn to the
original D0 training) and D1 (`generation="d1"`, 50/25/25 opponent
mixture, see `experiments/defender_adaptation_env.py`). Nothing else may
differ between a matched D0+/D1 pair except `env_fn` (Phase 17/18
guarantee is enforced by the CALLER passing identical seed/budget/
hyperparameters/vectorization/cadence to both).

`total_timesteps` semantics (Phase 8/13: "exactly 400,000 ADDITIONAL
transitions, architecture-independent"): SB3's `reset_num_timesteps=False`
adds the model's CURRENT `num_timesteps` to whatever `total_timesteps` is
passed for THIS `learn()` call (see `BaseAlgorithm._setup_learn`). To keep
this invariant exact ACROSS an arbitrary number of restart/resume events
(Phase 8 restart-safety), this runner tracks `starting_num_timesteps` (the
D0 baseline, fixed once and persisted in `resume_manifest.json`) and always
passes `remaining = total_additional_timesteps - completed_so_far` as
`total_timesteps`, recomputed fresh on every invocation (whether this is
the very first call for a job or a resume after interruption) -- NEVER the
full target unconditionally (that would double-count on resume).

Restart safety (Phase 8): every `checkpoint_freq` environment steps, in
addition to the existing named-snapshot `CheckpointCallback` (audit trail,
unchanged pattern from `clutter_training_runner.py`), a
`resume_latest.zip` + `resume_manifest.json` pair is written (atomically:
model saved first, manifest second) recording: starting checkpoint path +
SHA-256, starting num_timesteps, target additional transitions, opponent-
pool spec, git commit, env/config hash, worker seeds, and completion state.
Re-invoking this runner against the SAME `output_dir` resumes from
`resume_latest.zip` rather than restarting from the D0 checkpoint,
verifying the recorded starting-checkpoint SHA-256 still matches before
trusting the resume state (refuses to resume against a mismatched
lineage).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, SectorObservationSpec

from experiments.clutter_training_runner import (
    SEED_STRIDE, CLUTTER_HYPERPARAMS, _LightweightEvalCallback, _git_commit, _config_hash,
)

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTIVE_ATTACKER_HASHES_PATH = PROJECT_ROOT / "results" / "adaptive_attacker_400k" / "model_hashes.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_frozen_attacker_hashes(opponent_pool_spec_dict: dict | None) -> None:
    """Phase 3 defense-in-depth: if `opponent_pool_spec_dict` references any
    frozen A1-seen attacker checkpoints, reverify their ON-DISK SHA-256
    against the recorded baseline (`results/adaptive_attacker_400k/model_hashes.json`)
    -- proving this training run never wrote to / modified those files.
    Raises `RuntimeError` on any mismatch (never silently continues)."""
    if not opponent_pool_spec_dict:
        return
    seen_seeds = [m["seed"] for m in opponent_pool_spec_dict.get("members", []) if m.get("kind") == "learned"]
    if not seen_seeds:
        return
    if not ADAPTIVE_ATTACKER_HASHES_PATH.exists():
        raise RuntimeError(f"expected frozen-attacker hash baseline missing: {ADAPTIVE_ATTACKER_HASHES_PATH}")
    recorded = {entry["seed"]: entry["sha256"] for entry in json.loads(ADAPTIVE_ATTACKER_HASHES_PATH.read_text())}
    for seed in seen_seeds:
        run_dir = PROJECT_ROOT / "results" / "adaptive_attacker_400k" / f"seed_{seed}"
        path = run_dir / "best_model.zip"
        if not path.exists():
            path = run_dir / "final_model.zip"
        actual = _sha256_file(path)
        expected = recorded.get(seed)
        if expected is None:
            raise RuntimeError(f"no recorded baseline hash for frozen attacker seed_{seed}")
        if actual != expected:
            raise RuntimeError(
                f"FROZEN ATTACKER CHECKPOINT MODIFIED: seed_{seed} at {path} has sha256={actual}, "
                f"expected {expected}. This must NEVER happen (attacker models are inference-only "
                f"during defender-adaptation training)."
            )


class _RestartSafeCheckpointCallback(BaseCallback):
    """Every `checkpoint_freq` steps: saves `resume_latest.zip` and writes
    `resume_manifest.json` (Phase 8 full-state restart-safe checkpoint)."""

    def __init__(self, checkpoints_dir: Path, manifest_base: dict, starting_num_timesteps: int,
                 total_additional_timesteps: int, checkpoint_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self._checkpoints_dir = checkpoints_dir
        self._manifest_base = manifest_base
        self._starting_num_timesteps = starting_num_timesteps
        self._total_additional_timesteps = total_additional_timesteps
        self._checkpoint_freq = checkpoint_freq
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved < self._checkpoint_freq:
            return True
        self._last_saved = self.num_timesteps
        self.model.save(str(self._checkpoints_dir / "resume_latest"))
        manifest = dict(self._manifest_base)
        manifest.update({
            "starting_num_timesteps": self._starting_num_timesteps,
            "total_additional_timesteps_target": self._total_additional_timesteps,
            "model_num_timesteps": int(self.num_timesteps),
            "completed_additional_timesteps": int(self.num_timesteps) - self._starting_num_timesteps,
            "completed": False,
        })
        (self._checkpoints_dir / "resume_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return True


def run_defender_adaptation_training(
    architecture: str,
    generation: str,
    env_fn: Callable[[int], gym.Env],
    seed: int,
    starting_checkpoint_path: Path,
    output_dir: Path,
    total_additional_timesteps: int,
    opponent_pool_spec_dict: dict | None = None,
    observation_spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
    eval_freq: int = 20_000,
    eval_episodes: int = 20,
    checkpoint_freq: int = 50_000,
    training_config_summary: dict | None = None,
    verbose: int = 0,
    num_envs: int = 16,
    vec_env_cls: str = "auto",
) -> dict:
    """Continue training ONE defender checkpoint for exactly
    `total_additional_timesteps` MORE environment transitions (Phase 8/13),
    restart-safely. `generation` must be "d0_plus" or "d1". `env_fn` is the
    ONLY thing that may differ between a matched D0+/D1 pair (Phase 18)."""
    if generation not in ("d0_plus", "d1"):
        raise ValueError(f"generation must be 'd0_plus' or 'd1', got {generation!r}")

    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    starting_checkpoint_path = Path(starting_checkpoint_path)
    starting_sha256 = _sha256_file(starting_checkpoint_path)

    set_random_seed(seed)
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

    resume_manifest_path = checkpoints_dir / "resume_manifest.json"
    resume_latest_path = checkpoints_dir / "resume_latest.zip"

    resumed_from_interrupt = False
    if resume_manifest_path.exists() and resume_latest_path.exists():
        prior = json.loads(resume_manifest_path.read_text())
        if prior.get("starting_checkpoint_sha256") != starting_sha256:
            raise RuntimeError(
                f"resume manifest at {resume_manifest_path} was recorded against starting checkpoint "
                f"sha256={prior.get('starting_checkpoint_sha256')}, but the checkpoint requested now "
                f"({starting_checkpoint_path}) has sha256={starting_sha256} -- refusing to resume "
                f"against a mismatched lineage."
            )
        model = PPO.load(str(resume_latest_path), env=train_env)
        starting_num_timesteps = int(prior["starting_num_timesteps"])
        completed_so_far = int(model.num_timesteps) - starting_num_timesteps
        remaining = max(total_additional_timesteps - completed_so_far, 0)
        resumed_from_interrupt = True
    else:
        model = PPO.load(str(starting_checkpoint_path), env=train_env)
        starting_num_timesteps = int(model.num_timesteps)
        remaining = total_additional_timesteps

    manifest_base = {
        "architecture": architecture,
        "generation": generation,
        "seed": seed,
        "starting_checkpoint_path": str(starting_checkpoint_path),
        "starting_checkpoint_sha256": starting_sha256,
        "opponent_pool_spec": opponent_pool_spec_dict,
        "num_envs": num_envs,
        "worker_seeds": worker_seeds,
        "git_commit": _git_commit(),
        "env_config_hash": _config_hash(training_config_summary or {}),
    }

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // num_envs, 1), save_path=str(checkpoints_dir),
        name_prefix=f"{architecture}_{generation}_seed{seed}", save_replay_buffer=False,
    )
    eval_cb = _LightweightEvalCallback(
        env_fn=env_fn, eval_seed=seed, n_episodes=eval_episodes, eval_freq=eval_freq,
        best_model_path=output_dir / "best_model", verbose=verbose,
    )
    resume_cb = _RestartSafeCheckpointCallback(
        checkpoints_dir=checkpoints_dir, manifest_base=manifest_base,
        starting_num_timesteps=starting_num_timesteps, total_additional_timesteps=total_additional_timesteps,
        checkpoint_freq=max(checkpoint_freq // num_envs, 1), verbose=verbose,
    )

    t0 = time.time()
    if remaining > 0:
        model.learn(
            total_timesteps=remaining, reset_num_timesteps=False,
            callback=CallbackList([checkpoint_cb, eval_cb, resume_cb]),
        )
    elapsed = time.time() - t0

    train_env.close()
    model.save(str(output_dir / "final_model"))
    if not (output_dir / "best_model.zip").exists():
        model.save(str(output_dir / "best_model"))

    model.save(str(resume_latest_path.with_suffix("")))
    final_resume_manifest = dict(manifest_base)
    final_resume_manifest.update({
        "starting_num_timesteps": starting_num_timesteps,
        "total_additional_timesteps_target": total_additional_timesteps,
        "model_num_timesteps": int(model.num_timesteps),
        "completed_additional_timesteps": int(model.num_timesteps) - starting_num_timesteps,
        "completed": True,
    })
    resume_manifest_path.write_text(json.dumps(final_resume_manifest, indent=2, default=str))

    _verify_frozen_attacker_hashes(opponent_pool_spec_dict)

    manifest = {
        **manifest_base,
        "starting_num_timesteps": starting_num_timesteps,
        "total_additional_timesteps_target": total_additional_timesteps,
        "actual_additional_timesteps": int(model.num_timesteps) - starting_num_timesteps,
        "model_num_timesteps": int(model.num_timesteps),
        "hyperparameters": CLUTTER_HYPERPARAMS,
        "observation_spec": observation_spec.to_dict(),
        "training_config_summary": training_config_summary or {},
        "eval_history": eval_cb.history,
        "wall_time_s": elapsed,
        "resumed_from_interrupt": resumed_from_interrupt,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
