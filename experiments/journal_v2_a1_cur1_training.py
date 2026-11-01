"""Power-loss-safe mastery-based PPO runner for curriculum Stage 1 (`a1_cur1_training_v1`).

    python -m experiments.journal_v2_a1_cur1_training status
    python -m experiments.journal_v2_a1_cur1_training run --seeds 201 202 203      # safe to re-run after ANY interruption
    python -m experiments.journal_v2_a1_cur1_training resume-test                  # real-SB3 interrupted vs uninterrupted check

Protocol (frozen before training; see docs/journal_v2_a1_curriculum/stage1/):
* one PPO rollout = 16 envs x 2048 steps = 32,768 steps = ONE `learn(32768, reset_num_timesteps=False)` call; at most 8 rollouts (262,144 steps) per seed;
* before every rollout the env scenario streams are re-keyed to the rollout index and every env is reset (`model._last_obs = None`), so a rollout is a pure function
  of (model + optimizer state, torch RNG state, rollout index): an interrupted+resumed run reproduces the uninterrupted run;
* after every rollout: atomic commit (model ZIP incl. optimizer state, RNG pickle, episode log, then the MANIFEST is renamed LAST), mirror to OneDrive, then a fixed
  held-out validation (rollout k: set A if k is odd, set B if even; deterministic policy) written to its own hashed file, then the mastery gate;
* mastery = the gate passes on two consecutive rollouts (hence one A and one B evaluation); the seed stops at the second pass; no mastery by rollout 8 = FAIL;
* a seed is never retrained: on restart the highest fully verified rollout is located, temp/partial files are discarded, missing validation files are recomputed
  from the committed model (cheap), and training continues only if the seed is neither mastered nor exhausted. A per-seed lock prevents duplicate trainers.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_r2_training as T  # reuses the tested file primitives only
from experiments.journal_v2_a1_eg_state import CellLock

PROJECT_ROOT = T.PROJECT_ROOT
SCHEMA = "a1_cur1_rollout_manifest/1"
VALIDATION_SCHEMA = "a1_cur1_validation/1"
DEFAULT_SCRATCH = Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage1")
RunConflictError = T.RunConflictError
SimulatedPowerLoss = T.SimulatedPowerLoss


@dataclass(frozen=True)
class Spec:
    n_envs: int = C.N_ENVS
    n_steps: int = A.A1_V2_PPO_HYPERPARAMS["n_steps"]
    max_rollouts: int = C.MAX_ROLLOUTS

    @property
    def steps_per_rollout(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def max_timesteps(self) -> int:
        return self.max_rollouts * self.steps_per_rollout


CANONICAL_SPEC = Spec()
assert CANONICAL_SPEC.steps_per_rollout == 32_768 and CANONICAL_SPEC.max_timesteps == 262_144


def ppo_kwargs_hash(kwargs: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(kwargs if kwargs is not None else A.A1_V2_PPO_HYPERPARAMS, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- layout
def select_scratch_root(override: str | os.PathLike | None = None) -> Path:
    candidates = [Path(override)] if override else ([Path(os.environ["A1_SCRATCH_ROOT"])] if os.environ.get("A1_SCRATCH_ROOT") else [])
    candidates += [DEFAULT_SCRATCH, Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "journal_v2_a1_curriculum" / "stage1",
                   Path(tempfile.gettempdir()) / "journal_v2_a1_curriculum" / "stage1"]
    for candidate in candidates:
        if not T._is_synced(candidate) and T._writable_roundtrip(candidate):
            return candidate.resolve()
    raise RunConflictError(f"no writable local non-synced scratch directory among {[str(c) for c in candidates]}")


@dataclass(frozen=True)
class Layout:
    seed: int
    scratch: Path
    canonical: Path

    @property
    def scratch_rollouts(self) -> Path:
        return self.scratch / "rollouts"

    @property
    def canonical_rollouts(self) -> Path:
        return self.canonical / "rollouts"

    @property
    def accepted_model(self) -> Path:
        return self.canonical / "checkpoints" / "accepted_model.zip"

    @property
    def seed_manifest(self) -> Path:
        return self.canonical / "checkpoints" / "seed_manifest.json"

    @property
    def lock_base(self) -> Path:
        return self.scratch / "train"


def assert_isolated(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = C.RESULTS_ROOT.resolve()
    results = (PROJECT_ROOT / "results").resolve()
    if results in resolved.parents and root not in resolved.parents and resolved != root:
        raise RunConflictError(f"{resolved} is under results/ but outside the curriculum namespace {root}")
    return resolved


def make_layout(seed: int, scratch_root: Path, results_root: Path | None = None) -> Layout:
    results_root = Path(results_root) if results_root is not None else C.RESULTS_ROOT
    layout = Layout(int(seed), Path(scratch_root) / f"seed_{seed}", results_root / f"seed_{seed}")
    assert_isolated(layout.canonical)
    return layout


def names(k: int) -> dict[str, str]:
    return {"model": f"rollout_{k:02d}.zip", "rng": f"rollout_{k:02d}.rng.pkl", "episodes": f"rollout_{k:02d}.episodes.jsonl",
            "manifest": f"rollout_{k:02d}.manifest.json", "validation": f"rollout_{k:02d}.validation.json"}


# ----------------------------------------------------------------------------- state hash (policy + optimizer tensors)
def state_sha256(model) -> str:
    import torch

    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    opt = model.policy.optimizer.state_dict()["state"]
    for index in sorted(opt):
        for key in sorted(opt[index]):
            value = opt[index][key]
            if isinstance(value, torch.Tensor):
                digest.update(f"{index}/{key}".encode())
                digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize_training(episodes: list[dict]) -> dict:
    if not episodes:
        return {"episodes": 0}
    n = len(episodes)
    outcome = [e["outcome"] for e in episodes]
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    return {"episodes": n, "capture_rate": outcome.count("soldier_caught") / n, "exhaustion_rate": outcome.count("enemy_endurance_exhausted") / n,
            "timeout_rate": outcome.count("timeout") / n, "median_path_excess": float(np.nanmedian(pe)), "mean_return": float(np.mean([e["return_undiscounted"] for e in episodes])),
            "mean_return_discounted": float(np.mean([e["return_discounted"] for e in episodes])), "mean_duration_s": float(np.mean([e["duration_s"] for e in episodes]))}


# ----------------------------------------------------------------------------- rollout commit / verify / recover
def verify_rollout(directory: Path, k: int, *, seed: int, spec: Spec) -> dict | None:
    n = names(k)
    try:
        manifest = json.loads((Path(directory) / n["manifest"]).read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("rollout") != k or manifest.get("ppo_seed") != seed:
            return None
        if manifest.get("num_timesteps") != k * spec.steps_per_rollout or manifest.get("training_protocol") != C.TRAINING_PROTOCOL:
            return None
        for part in ("model", "rng", "episodes"):
            if T.sha256_file(Path(directory) / n[part]) != manifest[f"{part}_sha256"]:
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def commit_rollout(layout: Layout, k: int, model, episodes: list[dict], *, spec: Spec, ppo_kwargs: dict, fault_hook=None) -> dict:
    n = names(k)
    directory = layout.scratch_rollouts
    buffer = io.BytesIO()
    model.save(buffer)
    model_bytes = buffer.getvalue()
    rng_bytes = T.capture_rng()
    episodes_bytes = "".join(json.dumps({**e, "rollout": k}, default=str) + "\n" for e in episodes).encode()
    T.atomic_write_bytes(directory / n["model"], model_bytes)
    _hook(fault_hook, "after_model_written", k)
    T.atomic_write_bytes(directory / n["rng"], rng_bytes)
    T.atomic_write_bytes(directory / n["episodes"], episodes_bytes)
    _hook(fault_hook, "before_manifest", k)
    manifest = {
        "schema": SCHEMA, "rollout": k, "ppo_seed": layout.seed, "num_timesteps": int(model.num_timesteps), "n_updates": int(model._n_updates),
        "n_envs": spec.n_envs, "n_steps": spec.n_steps, "model_file": n["model"], "model_sha256": T.sha256_bytes(model_bytes), "state_sha256": state_sha256(model),
        "rng_file": n["rng"], "rng_sha256": T.sha256_bytes(rng_bytes), "episodes_file": n["episodes"], "episodes_sha256": T.sha256_bytes(episodes_bytes),
        "n_episodes": len(episodes), "training_summary": summarize_training(episodes), "training_protocol": C.TRAINING_PROTOCOL, "reward_version": C.REWARD_VERSION,
        "action_version": C.ACTION_VERSION, "observation_version": C.OBSERVATION_VERSION, "env_config_sha256": C.config_sha256(), "ppo_hyperparameters_sha256": ppo_kwargs_hash(ppo_kwargs),
        "scenario_namespace": {"stream_tag": hex(C.SCENARIO_STREAM_TAG), "ppo_seed": layout.seed, "rollout_index": k, "workers": spec.n_envs},
        "research_commit": T._git("rev-parse", "HEAD"), "created_utc": datetime.now(timezone.utc).isoformat(), "scratch_dir": str(layout.scratch),
    }
    T.atomic_write_json(directory / n["manifest"], manifest)
    _hook(fault_hook, "after_manifest", k)
    return manifest


def _hook(fault_hook, stage: str, k: int) -> None:
    if fault_hook is not None:
        fault_hook(stage, k)


def mirror_rollout(layout: Layout, k: int) -> bool:
    n = names(k)
    try:
        for part in ("model", "rng", "episodes", "validation", "manifest"):   # manifest last
            source = layout.scratch_rollouts / n[part]
            if not source.exists():
                continue
            destination = layout.canonical_rollouts / n[part]
            if destination.exists() and T.sha256_file(destination) == T.sha256_file(source):
                continue
            T.atomic_copy(source, destination)
        return True
    except OSError as error:
        print(f"[mirror] rollout {k}: {error!r}; will retry later", flush=True)
        return False


def recover(layout: Layout, *, spec: Spec) -> tuple[int, dict | None]:
    """Highest verified rollout (0 = none). Removes temp files and every partial / unverifiable rollout artifact (validation files are kept if they belong to a verified rollout)."""
    layout.scratch_rollouts.mkdir(parents=True, exist_ok=True)
    removed = T.cleanup_temporaries(layout.scratch_rollouts) + T.cleanup_temporaries(layout.canonical_rollouts)
    best_k, best_manifest = 0, None
    for k in range(spec.max_rollouts, 0, -1):
        manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)
        if manifest is None and layout.canonical_rollouts.exists() and verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec) is not None:
            for part, name in names(k).items():
                if (layout.canonical_rollouts / name).exists():
                    T.atomic_copy(layout.canonical_rollouts / name, layout.scratch_rollouts / name)
            manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)
        if manifest is not None:
            best_k, best_manifest = k, manifest
            break
    for path in sorted(layout.scratch_rollouts.glob("rollout_*")):
        number = int(path.name.split("_")[1].split(".")[0])
        if number > best_k:
            path.unlink()
            removed.append(path.name)
        elif number <= best_k and verify_rollout(layout.scratch_rollouts, number, seed=layout.seed, spec=spec) is None:
            path.unlink()   # a hole below the best rollout cannot exist by construction (rollouts are committed in order); remove anything inconsistent
            removed.append(path.name)
    if removed:
        print(f"[recover] discarded incomplete/temp files: {removed}", flush=True)
    return best_k, best_manifest


# ----------------------------------------------------------------------------- validation + mastery
def validation_set_for(k: int) -> str:
    return C.VALIDATION_ROLLOUT_SET[k % 2 == 1]


def verify_validation(directory: Path, k: int, manifest: dict) -> dict | None:
    try:
        payload = json.loads((Path(directory) / names(k)["validation"]).read_text())
        body = json.dumps(payload["metrics"], sort_keys=True).encode()
        if (payload.get("schema") != VALIDATION_SCHEMA or payload.get("rollout") != k or payload.get("model_sha256") != manifest["model_sha256"]
                or payload.get("content_sha256") != T.sha256_bytes(body) or payload.get("validation_set") != validation_set_for(k)
                or payload.get("gate_spec") != C.GATE):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_validation(model, k: int, manifest: dict) -> dict:
    name = validation_set_for(k)
    seeds = list(C.validation_seeds(name))
    episodes = C.evaluate(C.model_actor(model), seeds)
    m = C.metrics(episodes)
    gate = C.gate_status(m)
    body = json.dumps(m, sort_keys=True).encode()
    return {"schema": VALIDATION_SCHEMA, "rollout": k, "ppo_seed": manifest["ppo_seed"], "model_sha256": manifest["model_sha256"], "num_timesteps": manifest["num_timesteps"],
            "validation_set": name, "seeds": [seeds[0], seeds[-1]], "metrics": m, "gate": gate, "passed": gate["passed"], "gate_spec": C.GATE,
            "content_sha256": T.sha256_bytes(body), "created_utc": datetime.now(timezone.utc).isoformat()}


def ensure_validation(layout: Layout, k: int, manifest: dict, model=None) -> dict:
    payload = verify_validation(layout.scratch_rollouts, k, manifest)
    if payload is None:
        if model is None:
            from stable_baselines3 import PPO

            model = PPO.load(str(layout.scratch_rollouts / manifest["model_file"]).removesuffix(".zip"), device="cpu")
        payload = run_validation(model, k, manifest)
        T.atomic_write_json(layout.scratch_rollouts / names(k)["validation"], payload)
    mirror_rollout(layout, k)
    return payload


def mastery_from_passes(passes: list[bool]) -> int | None:
    """1-based rollout index at which two consecutive validations have passed (the second one), else None."""
    for k in range(2, len(passes) + 1):
        if passes[k - 1] and passes[k - 2]:
            return k
    return None


def collect_validations(layout: Layout, completed: int, spec: Spec) -> list[dict]:
    out = []
    for k in range(1, completed + 1):
        manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)
        out.append(ensure_validation(layout, k, manifest))
    return out


# ----------------------------------------------------------------------------- environments / PPO
def worker_env_fns(ppo_seed: int, n_envs: int) -> list:
    return [functools.partial(C.make_env, ppo_seed, i, 0) for i in range(n_envs)]


def default_env_factory(seed: int, spec: Spec):
    def factory():
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        fns = worker_env_fns(seed, spec.n_envs)
        return DummyVecEnv(fns) if spec.n_envs == 1 else SubprocVecEnv(fns)

    return factory


def collector():
    from stable_baselines3.common.callbacks import BaseCallback

    class Collect(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.episodes: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("a1_episode")
                if episode is not None:
                    self.episodes.append({**episode, "timesteps": int(self.num_timesteps)})
            return True

    return Collect()


def one_rollout(model, env, k: int, spec: Spec) -> list[dict]:
    """Exactly one collect+update iteration of rollout `k`: re-key the scenario streams to k, force a fresh reset of every env, learn 32,768 steps."""
    env.env_method("reseed", k)
    model._last_obs = None            # SB3 resets the vector env (and episode starts) at the beginning of learn()
    cb = collector()
    model.learn(total_timesteps=spec.steps_per_rollout, reset_num_timesteps=False, callback=cb, progress_bar=False)
    if int(model.num_timesteps) != k * spec.steps_per_rollout:
        raise RunConflictError(f"num_timesteps {model.num_timesteps} != {k * spec.steps_per_rollout} after rollout {k}")
    T.assert_model_finite(model)
    return cb.episodes


# ----------------------------------------------------------------------------- the seed driver
def train_seed(seed: int, *, scratch_root: Path, results_root: Path | None = None, spec: Spec = CANONICAL_SPEC, env_factory=None, ppo_kwargs: dict | None = None,
               validate: bool = True, stop_after: int | None = None, fault_hook=None) -> dict:
    """Train ONE PPO seed rollout by rollout until mastery or `spec.max_rollouts`; resumes automatically. Idempotent. `stop_after` (tests) stops after that many committed rollouts."""
    from stable_baselines3 import PPO

    layout = make_layout(seed, scratch_root, results_root)
    layout.canonical.mkdir(parents=True, exist_ok=True)
    layout.scratch.mkdir(parents=True, exist_ok=True)
    kwargs = dict(ppo_kwargs) if ppo_kwargs is not None else dict(A.A1_V2_PPO_HYPERPARAMS)
    lock = CellLock(layout.lock_base)
    if not lock.acquire():
        raise RunConflictError(f"another training process owns seed {seed} (lock {lock.path})")
    started = time.time()
    try:
        if layout.seed_manifest.exists():
            return json.loads(layout.seed_manifest.read_text())
        completed, manifest = recover(layout, spec=spec)
        validations = collect_validations(layout, completed, spec) if (validate and completed) else []
        mastered = mastery_from_passes([v["passed"] for v in validations]) if validate else None
        if mastered is None and completed < spec.max_rollouts and (stop_after is None or completed < stop_after):
            env = (env_factory or default_env_factory(seed, spec))()
            try:
                if completed:
                    model = PPO.load(str(layout.scratch_rollouts / manifest["model_file"]).removesuffix(".zip"), env=env, device="cpu")
                    if int(model.num_timesteps) != completed * spec.steps_per_rollout:
                        raise RunConflictError(f"restored num_timesteps {model.num_timesteps} != {completed * spec.steps_per_rollout}")
                    T.restore_rng((layout.scratch_rollouts / manifest["rng_file"]).read_bytes())
                    print(f"[seed {seed}] resumed from verified rollout {completed}/{spec.max_rollouts}", flush=True)
                else:
                    model = PPO(policy=A.A1_V2_POLICY, env=env, seed=seed, verbose=0, device="cpu", **kwargs)
                    print(f"[seed {seed}] fresh initialization", flush=True)
                for k in range(completed + 1, spec.max_rollouts + 1):
                    _hook(fault_hook, "before_learn", k)
                    episodes = one_rollout(model, env, k, spec)
                    _hook(fault_hook, "after_learn", k)
                    committed = commit_rollout(layout, k, model, episodes, spec=spec, ppo_kwargs=kwargs, fault_hook=fault_hook)
                    mirror_rollout(layout, k)
                    line = f"[seed {seed}] rollout {k:02d} steps={committed['num_timesteps']} train_capture={committed['training_summary'].get('capture_rate', 0):.3f}"
                    if validate:
                        payload = ensure_validation(layout, k, committed, model)
                        validations.append(payload)
                        mm = payload["metrics"]
                        line += (f" | val{payload['validation_set']} capture={mm['capture']:.3f} exh={mm['exhaustion']:.3f} medPE={mm['median_path_excess']:.3f} "
                                 f"gt2={mm['gt2']:.3f} gt5={mm['gt5']:.3f} path={'PASS' if payload['passed'] else 'fail'}")
                    print(line + f" t={time.time() - started:.0f}s", flush=True)
                    _hook(fault_hook, "after_commit", k)
                    completed = k
                    mastered = mastery_from_passes([v["passed"] for v in validations]) if validate else None
                    if mastered is not None or (stop_after is not None and k >= stop_after):
                        break
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        if not validate or (mastered is None and completed < spec.max_rollouts):
            return {"seed": seed, "status": "INCOMPLETE", "rollouts_complete": completed}
        return finalize(layout, spec=spec, validations=validations, mastered=mastered, kwargs=kwargs, wall_time_s=time.time() - started)
    finally:
        lock.release()


def finalize(layout: Layout, *, spec: Spec, validations: list[dict], mastered: int | None, kwargs: dict, wall_time_s: float | None) -> dict:
    last = len(validations)
    for k in range(1, last + 1):
        for _ in range(3):
            if mirror_rollout(layout, k):
                break
            time.sleep(2)
    manifests = [verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec) for k in range(1, last + 1)]
    if any(m is None for m in manifests):
        raise RunConflictError(f"a rollout of seed {layout.seed} failed verification in the canonical directory; cannot finalize")
    accepted_k = mastered if mastered is not None else last
    accepted = manifests[accepted_k - 1]
    if layout.accepted_model.exists() and T.sha256_file(layout.accepted_model) != accepted["model_sha256"]:
        raise RunConflictError("existing accepted model does not match the accepted rollout; refusing to overwrite")
    if not layout.accepted_model.exists():
        T.atomic_copy(layout.canonical_rollouts / accepted["model_file"], layout.accepted_model)
    sha = T.sha256_file(layout.accepted_model)
    if sha != accepted["model_sha256"]:
        raise RunConflictError("accepted model hash mismatch after copy")
    assembled = b"".join((layout.canonical_rollouts / m["episodes_file"]).read_bytes() for m in manifests)
    (layout.canonical / "logs").mkdir(parents=True, exist_ok=True)
    T.atomic_write_bytes(layout.canonical / "logs" / "training_episodes_all_rollouts.jsonl", assembled)
    payload = {
        "seed": layout.seed, "status": "MASTERED" if mastered is not None else "FAIL", "mastery_rollout": mastered,
        "timesteps_to_mastery": (mastered * spec.steps_per_rollout) if mastered is not None else None, "rollouts_trained": last, "timesteps_trained": last * spec.steps_per_rollout,
        "accepted_rollout": accepted_k, "accepted_model": "checkpoints/accepted_model.zip", "accepted_model_sha256": sha, "accepted_state_sha256": accepted["state_sha256"],
        "model_role": "curriculum initialization checkpoint (NOT a final A1 evaluation model)" if mastered is not None else "failed Stage-1 seed (rollout-8 policy; not accepted)",
        "training_protocol": C.TRAINING_PROTOCOL, "reward_version": C.REWARD_VERSION, "action_version": C.ACTION_VERSION, "observation_version": C.OBSERVATION_VERSION,
        "env_config_sha256": C.config_sha256(), "ppo_hyperparameters": kwargs, "ppo_hyperparameters_sha256": ppo_kwargs_hash(kwargs), "gate": C.GATE,
        "rollouts": [{"rollout": m["rollout"], "num_timesteps": m["num_timesteps"], "model_sha256": m["model_sha256"], "state_sha256": m["state_sha256"],
                      "training_summary": m["training_summary"], "validation_set": v["validation_set"], "validation": v["metrics"], "gate_checks": v["gate"]["checks"],
                      "validation_passed": v["passed"]} for m, v in zip(manifests, validations)],
        "scenario_namespace": {"stream_tag": hex(C.SCENARIO_STREAM_TAG), "heading_tag": hex(C.HEADING_STREAM_TAG)}, "episodes_log_sha256": T.sha256_bytes(assembled),
        "scratch_dir": str(layout.scratch), "research_commit": T._git("rev-parse", "HEAD"), "created_utc": datetime.now(timezone.utc).isoformat(), "wall_time_s_last_segment": wall_time_s,
    }
    if not layout.seed_manifest.exists():
        T.atomic_write_json(layout.seed_manifest, payload)
    return json.loads(layout.seed_manifest.read_text())


def status(scratch_root: Path | None = None, seeds=C.PPO_SEEDS) -> dict:
    scratch_root = scratch_root or select_scratch_root()
    out = {}
    for seed in seeds:
        layout = make_layout(seed, scratch_root)
        if layout.seed_manifest.exists():
            m = json.loads(layout.seed_manifest.read_text())
            out[seed] = {"status": m["status"], "mastery_rollout": m["mastery_rollout"], "timesteps_to_mastery": m["timesteps_to_mastery"]}
            continue
        best = 0
        for k in range(CANONICAL_SPEC.max_rollouts, 0, -1):
            if verify_rollout(layout.scratch_rollouts, k, seed=seed, spec=CANONICAL_SPEC) or verify_rollout(layout.canonical_rollouts, k, seed=seed, spec=CANONICAL_SPEC):
                best = k
                break
        out[seed] = {"status": "IN_PROGRESS" if best else "NOT_STARTED", "verified_rollouts": best}
    return out


# ----------------------------------------------------------------------------- real-SB3 interrupted vs uninterrupted check
def resume_test(work_dir: Path, *, n_envs: int = 4, n_steps: int = 64, rollouts: int = 3, interrupt_after: int = 1, crash_mid_commit: bool = True) -> dict:
    """Tiny REAL Stable-Baselines3 run (SubprocVecEnv, real PPO updates) with the production code path:
    A = uninterrupted `rollouts` rollouts in one process; B = rollout 1, an injected abrupt death inside the commit of rollout 2 (optional), process-level
    destruction of env and model, reload from disk, continue to `rollouts`. Compares the per-rollout policy+optimizer state hashes."""
    work_dir = Path(work_dir)
    spec = Spec(n_envs=n_envs, n_steps=n_steps, max_rollouts=rollouts)
    kwargs = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=n_steps, batch_size=64)
    run_a = {"scratch": work_dir / "a_scratch", "results": work_dir / "a_results"}
    run_b = {"scratch": work_dir / "b_scratch", "results": work_dir / "b_results"}
    for r in (run_a, run_b):
        shutil.rmtree(r["scratch"], ignore_errors=True)
        shutil.rmtree(r["results"], ignore_errors=True)
    t0 = time.time()
    train_seed(201, scratch_root=run_a["scratch"], results_root=run_a["results"], spec=spec, ppo_kwargs=kwargs, validate=False)
    train_seed(201, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False, stop_after=interrupt_after)
    crashed = False
    if crash_mid_commit:
        def hook(stage, k):
            if stage == "before_manifest" and k == interrupt_after + 1:
                raise SimulatedPowerLoss()
        try:
            train_seed(201, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False, fault_hook=hook)
        except SimulatedPowerLoss:
            crashed = True
    # the lock of a dead process is stale: emulate by leaving the lock file as written and relying on take-over of a dead owner
    layout_b = make_layout(201, run_b["scratch"], run_b["results"])
    leftover_before = sorted(p.name for p in layout_b.scratch_rollouts.glob("rollout_*"))
    (layout_b.scratch / "train.lock").write_text("999999")    # a lock owned by a dead PID
    train_seed(201, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False)
    layout_a = make_layout(201, run_a["scratch"], run_a["results"])
    rows = []
    for k in range(1, rollouts + 1):
        ma = verify_rollout(layout_a.scratch_rollouts, k, seed=201, spec=spec)
        mb = verify_rollout(layout_b.scratch_rollouts, k, seed=201, spec=spec)
        rows.append({"rollout": k, "uninterrupted_state_sha256": ma["state_sha256"], "interrupted_state_sha256": mb["state_sha256"], "state_equal": ma["state_sha256"] == mb["state_sha256"],
                     "episodes_equal": ma["episodes_sha256"] == mb["episodes_sha256"], "num_timesteps_equal": ma["num_timesteps"] == mb["num_timesteps"]})
    return {"python": sys.version.split()[0], "platform": platform.platform(), "n_envs": n_envs, "n_steps": n_steps, "rollouts": rollouts, "interrupt_after": interrupt_after,
            "crash_injected_between_model_write_and_manifest": crashed, "partial_files_before_resume": leftover_before, "rows": rows,
            "all_states_equal": all(r["state_equal"] for r in rows), "all_episode_logs_equal": all(r["episodes_equal"] for r in rows),
            "vec_env": "SubprocVecEnv", "wall_time_s": time.time() - t0}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "run", "resume-test"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(C.PPO_SEEDS))
    parser.add_argument("--scratch", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    T.keep_awake()
    scratch = select_scratch_root(args.scratch)
    if args.command == "status":
        print(json.dumps(status(scratch), indent=1))
    elif args.command == "run":
        print(f"[curriculum stage 1] scratch={scratch}", flush=True)
        for seed in args.seeds:
            result = train_seed(seed, scratch_root=scratch)
            print(f"[seed {seed}] {result.get('status')} mastery_rollout={result.get('mastery_rollout')} timesteps_to_mastery={result.get('timesteps_to_mastery')}", flush=True)
    else:
        work = Path(args.work_dir) if args.work_dir else scratch.parent / "resume_test"
        result = resume_test(work)
        if args.out:
            T.atomic_write_json(Path(args.out), result)
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
