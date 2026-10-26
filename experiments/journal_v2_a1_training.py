"""Canonical V2 A1 training runner (`a1_v2_training_v1`, reward `a1_v2_reward_v1`).

    python -m experiments.journal_v2_a1_training --seed 201 --dry-run     # build + verify, no PPO, no output
    python -m experiments.journal_v2_a1_training --seed 201 --train       # the actual campaign (next task)

`--dry-run` stops immediately BEFORE PPO construction. Training always asserts the canonical V2 benchmark
(`require_canonical_v2` semantics: one environment per run is reset and checked, and every episode config is
asserted again), writes only inside the V2 result namespace, never overwrites a completed run or a V1
checkpoint, and gives every saved model an atomic sidecar manifest. See docs/journal_v2_a1_training_protocol.md.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.journal_v2_a1 import (
    A1_V2_ACTION_DIM,
    A1_V2_CHECKPOINT_FREQ,
    A1_V2_CLUTTER_LEVELS,
    A1_V2_CLUTTER_PROBABILITIES,
    A1_V2_DEFENDER_FAMILIES,
    A1_V2_DEFENDER_PROBABILITIES,
    A1_V2_EVAL_EPISODES,
    A1_V2_EVAL_SEED_START,
    A1_V2_NUM_ENVS,
    A1_V2_OBSERVATION_DIM,
    A1_V2_OBSERVATION_SCHEMA,
    A1_V2_OBSERVATION_VERSION,
    A1_V2_POLICY,
    A1_V2_PPO_HYPERPARAMS,
    A1_V2_PPO_SEEDS,
    A1_V2_REWARD_VERSION,
    A1_V2_SCENARIO_STREAM_TAG,
    A1_V2_TOTAL_TIMESTEPS,
    A1_V2_TRAIN_SCENARIO_SEED_HIGH,
    A1_V2_TRAIN_SCENARIO_SEED_LOW,
    A1_V2_TRAINING_PROTOCOL,
    A1_V2_VALIDATION_EPISODES,
    A1_V2_VALIDATION_FREQ,
    A1_V2_VALIDATION_SEED_START,
    A1_V2_V1_OBSERVATION_DIM,
    REWARD_TERMINAL_FAILURE,
    REWARD_TERMINAL_SUCCESS,
    SEED_STRIDE,
    W_PATH,
    W_PROGRESS,
    W_TIME,
    A1V2DefenderBank,
    A1V2TrainingEnv,
    sample_training_scenario,
    training_scenario_stream,
    validation_scenarios,
)
from experiments.journal_v2_protocol import (
    CANONICAL_V2_ASSET_MOTION_MODEL,
    CANONICAL_V2_ENDURANCE_PATH_FACTOR,
    V2_RESULTS_DIR,
    assert_canonical_v2_env_fn,
    build_v2_env_config,
    canonical_v2_config_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
A1_V2_RESULTS_ROOT = V2_RESULTS_DIR / "a1_training" / A1_V2_TRAINING_PROTOCOL
CANONICAL_A0_TAG = "journal-v2-a0-canonical-v1"
CANONICAL_A0_COMMIT = "bdbee45dc08a583a0da5ca882dc6b1269207a8cd"

# Every V1 training namespace: no V2 run may write inside any of them.
V1_PROTECTED_ROOTS: tuple[Path, ...] = tuple(
    (PROJECT_ROOT / "results" / name).resolve() for name in (
        "adaptive_attacker_400k", "adaptive_attacker_diagnostic", "clutter_training_400k",
        "defender_adaptation_400k", "defender_adaptation", "training_400k", "training_post_detection_400k",
    )
)


class RunConflictError(RuntimeError):
    """The output location is a V1 namespace, outside the V2 namespace, or holds an incompatible run."""


class RunAlreadyCompleteError(RunConflictError):
    """The run already finished; completed canonical models are never overwritten."""


# ----------------------------------------------------------------------------- paths and isolation
@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    summaries: Path
    manifests: Path

    @property
    def final_model(self) -> Path:
        return self.checkpoints / "final_model.zip"

    @property
    def best_validation_model(self) -> Path:
        return self.checkpoints / "best_validation.zip"

    @property
    def resume_model(self) -> Path:
        return self.checkpoints / "resume_latest.zip"

    @property
    def resume_state(self) -> Path:
        return self.manifests / "resume_state.json"

    @property
    def run_manifest(self) -> Path:
        return self.manifests / "run_manifest.json"

    def make_dirs(self) -> None:
        for directory in (self.checkpoints, self.logs, self.summaries, self.manifests):
            directory.mkdir(parents=True, exist_ok=True)


def assert_output_isolated(path: Path) -> Path:
    """No V1 namespace, and nothing under the repository `results/` tree outside the V2 namespace."""
    resolved = Path(path).resolve()
    for protected in V1_PROTECTED_ROOTS:
        if resolved == protected or protected in resolved.parents:
            raise RunConflictError(f"{resolved} is inside the V1 namespace {protected}; V1 artifacts are never written.")
    results_root = (PROJECT_ROOT / "results").resolve()
    v2_root = V2_RESULTS_DIR.resolve()
    if results_root in resolved.parents and v2_root not in resolved.parents and resolved != v2_root:
        raise RunConflictError(f"{resolved} is under results/ but outside the V2 namespace {v2_root}.")
    return resolved


def run_paths(ppo_seed: int, root: Path | None = None) -> RunPaths:
    if int(ppo_seed) not in A1_V2_PPO_SEEDS:
        raise ValueError(f"PPO seed must be one of {A1_V2_PPO_SEEDS}, got {ppo_seed}")
    base = assert_output_isolated(A1_V2_RESULTS_ROOT if root is None else Path(root))
    seed_dir = base / f"seed_{int(ppo_seed)}"
    return RunPaths(seed_dir, seed_dir / "checkpoints", seed_dir / "logs", seed_dir / "summaries", seed_dir / "manifests")


def decide_run_state(paths: RunPaths) -> str:
    """'fresh' | 'resume'; raises for a completed or inconsistent directory (never overwrites)."""
    if paths.final_model.exists():
        raise RunAlreadyCompleteError(f"{paths.final_model} exists: this run is complete and is never overwritten.")
    if paths.resume_model.exists():
        if not paths.resume_state.exists():
            raise RunConflictError(f"{paths.resume_model} has no {paths.resume_state.name}; refusing to guess a resume point.")
        return "resume"
    if paths.root.exists() and any(p.is_file() for p in paths.root.rglob("*")):
        raise RunConflictError(f"{paths.root} already contains files but no resumable state; refusing to continue.")
    return "fresh"


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- provenance / manifests
def config_fingerprints() -> dict[str, str]:
    """SHA-256 of every resolved canonical V2 config this protocol can instantiate (level x estimator)."""
    return {f"{level}/{estimator}": canonical_v2_config_sha256(build_v2_env_config(level, estimator))
            for level in A1_V2_CLUTTER_LEVELS for estimator in ("measurement", "kalman")}


def observation_schema_sha256() -> str:
    return hashlib.sha256(json.dumps(A1_V2_OBSERVATION_SCHEMA).encode()).hexdigest()


def protocol_description() -> dict:
    """Everything that defines the protocol; embedded in every manifest."""
    return {
        "training_protocol": A1_V2_TRAINING_PROTOCOL,
        "reward_version": A1_V2_REWARD_VERSION,
        "observation_version": A1_V2_OBSERVATION_VERSION,
        "observation_dim": A1_V2_OBSERVATION_DIM,
        "observation_schema_sha256": observation_schema_sha256(),
        "v1_observation_prefix_dim": A1_V2_V1_OBSERVATION_DIM,
        "action_dim": A1_V2_ACTION_DIM,
        "reward": {"terminal_success": REWARD_TERMINAL_SUCCESS, "terminal_failure": REWARD_TERMINAL_FAILURE,
                   "w_progress": W_PROGRESS, "w_path": W_PATH, "w_time": W_TIME, "gamma": A1_V2_PPO_HYPERPARAMS["gamma"]},
        "ppo_policy": A1_V2_POLICY,
        "ppo_hyperparameters": dict(A1_V2_PPO_HYPERPARAMS),
        "requested_total_timesteps": A1_V2_TOTAL_TIMESTEPS,
        "num_envs": A1_V2_NUM_ENVS,
        "defender_sampling": dict(zip(A1_V2_DEFENDER_FAMILIES, A1_V2_DEFENDER_PROBABILITIES)),
        "clutter_sampling": dict(zip(A1_V2_CLUTTER_LEVELS, A1_V2_CLUTTER_PROBABILITIES)),
        "endurance_enabled": True,
        "endurance_path_factor": CANONICAL_V2_ENDURANCE_PATH_FACTOR,
        "asset_motion_model": CANONICAL_V2_ASSET_MOTION_MODEL,
        "canonical_a0_tag": CANONICAL_A0_TAG,
        "canonical_a0_commit": CANONICAL_A0_COMMIT,
        "scenario_seed_namespaces": {
            "training_scenarios": [A1_V2_TRAIN_SCENARIO_SEED_LOW, A1_V2_TRAIN_SCENARIO_SEED_HIGH],
            "validation": [A1_V2_VALIDATION_SEED_START, A1_V2_VALIDATION_SEED_START + A1_V2_VALIDATION_EPISODES - 1],
            "evaluation": [A1_V2_EVAL_SEED_START, A1_V2_EVAL_SEED_START + A1_V2_EVAL_EPISODES - 1],
            "scenario_stream_entropy": [A1_V2_SCENARIO_STREAM_TAG, "ppo_seed", "worker_index", "resume_index"],
        },
    }


def build_model_manifest(*, ppo_seed: int, model_path: Path, num_timesteps: int, role: str, canonical: bool,
                         extra: dict | None = None) -> dict:
    return {
        **protocol_description(),
        "ppo_seed": int(ppo_seed),
        "model_file": Path(model_path).name,
        "model_role": role,
        "canonical": bool(canonical),
        "num_timesteps": int(num_timesteps),
        "research_commit": _git("rev-parse", "HEAD"),
        "research_tree_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": config_fingerprints(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(model_path),
        **(extra or {}),
    }


def save_model_with_manifest(model, path: Path, *, ppo_seed: int, role: str, canonical: bool,
                             allow_replace: bool = False, extra: dict | None = None) -> dict:
    """Save a model (never replacing a canonical or periodic file) and write its sidecar manifest atomically."""
    path = Path(path)
    assert_output_isolated(path)
    if path.exists() and not allow_replace:
        raise RunConflictError(f"{path} already exists; models are never overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path.with_suffix("")))
    manifest = build_model_manifest(ppo_seed=ppo_seed, model_path=path, num_timesteps=int(model.num_timesteps),
                                    role=role, canonical=canonical, extra=extra)
    write_json_atomic(path.with_suffix(".manifest.json"), manifest)
    return manifest


# ----------------------------------------------------------------------------- environments
def make_a1_v2_env(ppo_seed: int, worker_index: int, resume_index: int) -> A1V2TrainingEnv:
    return A1V2TrainingEnv(ppo_seed, worker_index, resume_index)


def make_monitored_a1_v2_env(ppo_seed: int, worker_index: int, resume_index: int):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_a1_v2_env(ppo_seed, worker_index, resume_index))


def worker_env_fns(ppo_seed: int, num_envs: int, resume_index: int = 0) -> list:
    """Picklable per-worker env constructors; worker i draws its scenarios from its own stream."""
    return [functools.partial(make_a1_v2_env, ppo_seed, i, resume_index) for i in range(num_envs)]


def worker_env_seeds(ppo_seed: int, num_envs: int) -> list[int]:
    return [ppo_seed + i * SEED_STRIDE for i in range(num_envs)]


# ----------------------------------------------------------------------------- dry run (stops before PPO)
def dry_run(ppo_seed: int, *, num_envs: int = A1_V2_NUM_ENVS, root: Path | None = None, probe_resets: int = 40,
            probe_steps: int = 3) -> dict:
    """Everything up to, and excluding, PPO construction: isolation, run state, canonical guard, 37-D / 3-D
    spaces, scenario sampling, reward wrapper. Writes nothing."""
    paths = run_paths(ppo_seed, root)
    state = decide_run_state(paths)
    env_fns = worker_env_fns(ppo_seed, num_envs)
    assert_canonical_v2_env_fn(lambda seed: env_fns[0](), ppo_seed)  # the training guard, one reset

    probe = env_fns[0]()
    assert probe.observation_space.shape == (A1_V2_OBSERVATION_DIM,)
    assert probe.action_space.shape == (A1_V2_ACTION_DIM,)
    observation, _ = probe.reset(seed=worker_env_seeds(ppo_seed, num_envs)[0])
    assert observation.shape == (A1_V2_OBSERVATION_DIM,)
    reward_probe = []
    for _ in range(probe_steps):
        observation, reward, terminated, truncated, info = probe.step(np.full(3, 3 ** -0.5, dtype=np.float32))
        reward_probe.append({**info["a1_reward_components"], "total": reward})
        if terminated or truncated:
            break

    levels, families, seeds = {}, {}, []
    for worker in range(min(num_envs, 4)):
        stream = training_scenario_stream(ppo_seed, worker)
        for _ in range(probe_resets):
            level, family, scenario_seed = sample_training_scenario(stream)
            levels[level] = levels.get(level, 0) + 1
            families[family] = families.get(family, 0) + 1
            seeds.append(scenario_seed)
    return {
        "ppo_seed": ppo_seed, "run_state": state, "output_root": str(paths.root),
        "observation_dim": A1_V2_OBSERVATION_DIM, "action_dim": A1_V2_ACTION_DIM,
        "canonical_guard": "passed", "num_envs": num_envs, "worker_env_seeds": worker_env_seeds(ppo_seed, num_envs),
        "sampled_levels": levels, "sampled_defenders": families, "scenario_seed_min": min(seeds), "scenario_seed_max": max(seeds),
        "reward_probe": reward_probe, "ppo_constructed": False,
    }


# ----------------------------------------------------------------------------- training (explicit --train only)
def _make_callbacks(paths: RunPaths, ppo_seed: int, resume_index: int):
    from stable_baselines3.common.callbacks import BaseCallback

    from experiments.journal_v2_a1_evaluation import run_scenarios, summarize

    class EpisodeLog(BaseCallback):
        """Per-episode JSONL (outcome, reward components, endurance, path-excess) + rolling summaries."""

        def __init__(self):
            super().__init__(0)
            self.window: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("a1_episode")
                if episode is None:
                    continue
                with open(paths.logs / "episodes.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**episode, "timesteps": self.num_timesteps}, default=str) + "\n")
                self.window.append(episode)
                if len(self.window) >= 200:
                    outcomes: dict[str, int] = {}
                    for item in self.window:
                        outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
                    summary = {
                        "timesteps": self.num_timesteps, "episodes": len(self.window), "outcomes": outcomes,
                        "capture_rate": outcomes.get("soldier_caught", 0) / len(self.window),
                        "endurance_exhaustion_rate": outcomes.get("enemy_endurance_exhausted", 0) / len(self.window),
                        "mean_return": sum(i["return_undiscounted"] for i in self.window) / len(self.window),
                        "mean_components": {k: sum(i["components"][k] for i in self.window) / len(self.window)
                                            for k in ("terminal", "progress", "path", "time")},
                        "mean_path_excess": sum(i["path_excess"] for i in self.window if i["path_excess"] == i["path_excess"]) / len(self.window),
                    }
                    with open(paths.logs / "training_summary.jsonl", "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(summary) + "\n")
                    self.window = []
            return True

    class Checkpoints(BaseCallback):
        def __init__(self, start_timesteps: int = 0):
            super().__init__(0)
            self._last = start_timesteps - (start_timesteps % A1_V2_CHECKPOINT_FREQ)

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last >= A1_V2_CHECKPOINT_FREQ:
                self._last = self.num_timesteps - (self.num_timesteps % A1_V2_CHECKPOINT_FREQ)
                path = paths.checkpoints / f"a1_v2_seed{ppo_seed}_{self.num_timesteps}_steps.zip"
                save_model_with_manifest(self.model, path, ppo_seed=ppo_seed, role="periodic_checkpoint", canonical=False)
                self.model.save(str(paths.resume_model.with_suffix("")))
                write_json_atomic(paths.resume_state, {"num_timesteps": int(self.num_timesteps), "ppo_seed": ppo_seed,
                                                       "completed_resume_index": resume_index})
            return True

    class Validation(BaseCallback):
        """Fixed 40-episode stratified validation (seeds 150000-150039); diagnostics + a replaceable
        `best_validation.zip` (not canonical). The canonical model is the FINAL model."""

        def __init__(self, start_timesteps: int = 0):
            super().__init__(0)
            self._last = start_timesteps - (start_timesteps % A1_V2_VALIDATION_FREQ)
            self._best = -1.0
            self._bank = A1V2DefenderBank()

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last < A1_V2_VALIDATION_FREQ:
                return True
            self._last = self.num_timesteps - (self.num_timesteps % A1_V2_VALIDATION_FREQ)
            records = run_scenarios(self.model, f"A1-{ppo_seed}", validation_scenarios(), bank=self._bank)
            summary = {"timesteps": int(self.num_timesteps), **summarize(records)}
            with open(paths.summaries / "validation.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, default=str) + "\n")
            if summary["capture_rate"] >= self._best:
                self._best = summary["capture_rate"]
                save_model_with_manifest(self.model, paths.best_validation_model, ppo_seed=ppo_seed,
                                         role="best_validation_diagnostic", canonical=False, allow_replace=True,
                                         extra={"validation_summary": summary})
            return True

    return EpisodeLog, Checkpoints, Validation


def _resume_index(paths: RunPaths) -> int:
    if paths.resume_state.exists():
        return int(json.loads(paths.resume_state.read_text()).get("completed_resume_index", 0))
    return 0


def run_a1_v2_training(ppo_seed: int, *, num_envs: int = A1_V2_NUM_ENVS, root: Path | None = None,
                       total_timesteps: int = A1_V2_TOTAL_TIMESTEPS, vec_env_cls: str = "subproc") -> dict:
    """The canonical A1 V2 campaign for ONE PPO seed. Only reached through `--train`."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    paths = run_paths(ppo_seed, root)
    state = decide_run_state(paths)
    resume_index = (_resume_index(paths) + 1) if state == "resume" else 0
    env_fns = worker_env_fns(ppo_seed, num_envs, resume_index)
    assert_canonical_v2_env_fn(lambda seed: env_fns[0](), ppo_seed)  # fail fast, before anything is created
    paths.make_dirs()

    monitored = [functools.partial(make_monitored_a1_v2_env, ppo_seed, i, resume_index) for i in range(num_envs)]
    train_env = DummyVecEnv(monitored) if (num_envs == 1 or vec_env_cls == "dummy") else SubprocVecEnv(monitored)
    tensorboard = None
    try:
        import tensorboard as _tb  # noqa: F401

        tensorboard = str(paths.logs / "tensorboard")
    except ImportError:
        pass  # per-episode / rolling JSONL logs are always written

    started = time.time()
    if state == "resume":
        model = PPO.load(str(paths.resume_model.with_suffix("")), env=train_env)
        start_timesteps = int(model.num_timesteps)
        remaining = max(total_timesteps - start_timesteps, 0)
    else:
        model = PPO(policy=A1_V2_POLICY, env=train_env, seed=ppo_seed, verbose=0, tensorboard_log=tensorboard,
                    **A1_V2_PPO_HYPERPARAMS)
        start_timesteps, remaining = 0, total_timesteps
    write_json_atomic(paths.run_manifest, {**protocol_description(), "ppo_seed": ppo_seed, "status": "running",
                                           "resume_index": resume_index, "started_utc": datetime.now(timezone.utc).isoformat(),
                                           "config_sha256": config_fingerprints(), "research_commit": _git("rev-parse", "HEAD")})
    EpisodeLog, Checkpoints, Validation = _make_callbacks(paths, ppo_seed, resume_index)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                    callback=CallbackList([EpisodeLog(), Checkpoints(start_timesteps), Validation(start_timesteps)]))
    train_env.close()
    manifest = save_model_with_manifest(model, paths.final_model, ppo_seed=ppo_seed, role="final", canonical=True,
                                        extra={"wall_time_s": time.time() - started, "resume_index": resume_index})
    write_json_atomic(paths.run_manifest, {**protocol_description(), "ppo_seed": ppo_seed, "status": "complete",
                                           "final_model_sha256": manifest["model_sha256"], "num_timesteps": manifest["num_timesteps"],
                                           "completed_utc": datetime.now(timezone.utc).isoformat(),
                                           "config_sha256": config_fingerprints(), "research_commit": manifest["research_commit"]})
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=A1_V2_PPO_SEEDS)
    parser.add_argument("--num-envs", type=int, default=A1_V2_NUM_ENVS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="verify everything up to PPO construction; write nothing")
    mode.add_argument("--train", action="store_true", help="run the canonical campaign for this seed")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args.seed, num_envs=args.num_envs), indent=2, default=str))
        return 0
    run_a1_v2_training(args.seed, num_envs=args.num_envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
