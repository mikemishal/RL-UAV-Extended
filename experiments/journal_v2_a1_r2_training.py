"""Power-loss-safe canonical training runner for `a1_v2_training_v2` (reward `a1_v2_reward_v2`, action `a1_v2_action_v2`).

    python -m experiments.journal_v2_a1_r2_training status
    python -m experiments.journal_v2_a1_r2_training preflight
    python -m experiments.journal_v2_a1_r2_training campaign        # seeds 201, 202, 203 in order; safe to re-run after ANY interruption

Design (see docs/journal_v2_a1_training_protocol_v2.md):

* A seed is exactly 13 PPO rollouts of 16 x 2048 = 32,768 environment steps = 425,984 steps. Each rollout is one
  `model.learn(total_timesteps=32768, reset_num_timesteps=False)` call, i.e. exactly one collect+update iteration.
* After every completed rollout (update finished) a recovery checkpoint is committed ATOMICALLY: model ZIP, RNG state and
  the rollout's episode log are each written to a temp file and renamed; the rollout manifest (with the SHA-256 of all
  three) is renamed LAST and is the commit marker. A rollout without a verifying manifest is not complete.
* Active training writes only to a local, non-synced scratch directory; completed rollouts are then mirrored into the
  canonical results directory (atomic copy with retries; a mirror failure never loses data and never stops training).
* On start the highest complete, verified rollout is located (scratch first, then the canonical mirror), temp files and
  partial artifacts are discarded, the model and RNG state are restored and training continues with the next rollout
  until exactly 13 are complete. A seed is restarted from scratch only if no valid checkpoint exists. A completed seed
  is never retrained or overwritten. No `a1_v2_training_v1` artifact is ever read or written.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments.journal_v2_protocol import assert_canonical_v2_env_fn, build_v2_env_config, canonical_v2_config_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_A0_TAG = "journal-v2-a0-canonical-v1"
CANONICAL_A0_COMMIT = "bdbee45dc08a583a0da5ca882dc6b1269207a8cd"
N_ROLLOUTS = 13
SCHEMA = "a1_v2_r2_rollout_manifest/1"


class RunConflictError(RuntimeError):
    """A path or checkpoint would violate output isolation or version consistency."""


class SimulatedPowerLoss(BaseException):
    """Raised only by fault-injection hooks in tests to emulate an abrupt process death."""


@dataclass(frozen=True)
class Spec:
    n_envs: int = A.A1_V2_NUM_ENVS
    n_steps: int = A.A1_V2_PPO_HYPERPARAMS["n_steps"]
    n_rollouts: int = N_ROLLOUTS

    @property
    def steps_per_rollout(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def total_timesteps(self) -> int:
        return self.n_rollouts * self.steps_per_rollout


CANONICAL_SPEC = Spec()
assert CANONICAL_SPEC.total_timesteps == 425_984


# ----------------------------------------------------------------------------- file primitives
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry(operation, attempts: int = 6, delay: float = 1.0):
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """temp file in the same directory -> flush + fsync -> os.replace. Never writes over the destination in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _retry(lambda: os.replace(temporary, path))


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8"))


def atomic_copy(source: Path, destination: Path) -> None:
    atomic_write_bytes(destination, Path(source).read_bytes())


def cleanup_temporaries(directory: Path) -> list[str]:
    removed = []
    if Path(directory).exists():
        for path in Path(directory).rglob(".*.tmp*"):
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                pass
    return removed


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- scratch selection and layout
def _is_synced(path: Path) -> bool:
    """True when `path` lies inside a OneDrive (or other well-known sync) root."""
    resolved = str(Path(path).resolve()).lower()
    markers = ["onedrive"] + [os.environ[k].lower() for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial") if os.environ.get(k)]
    return any(marker in resolved for marker in markers)


def _writable_roundtrip(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".probe"
        atomic_write_bytes(probe, b"x" * 1024)
        ok = probe.read_bytes() == b"x" * 1024
        probe.unlink()
        return ok
    except OSError:
        return False


def select_scratch_root(override: str | os.PathLike | None = None) -> Path:
    """First candidate that is writable, supports atomic rename + fsync, and is NOT under a synced folder.
    Store-installed Python virtualises %LOCALAPPDATA%, so the per-user AppData tree is deliberately not a candidate."""
    candidates = []
    if override:
        candidates.append(Path(override))
    elif os.environ.get("A1_SCRATCH_ROOT"):
        candidates.append(Path(os.environ["A1_SCRATCH_ROOT"]))
    candidates += [Path(r"C:\Users\Public\journal_v2_a1_training_v2"), Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "journal_v2_a1_training_v2",
                   Path(tempfile.gettempdir()) / "journal_v2_a1_training_v2"]
    for candidate in candidates:
        if not _is_synced(candidate) and _writable_roundtrip(candidate):
            return candidate.resolve()
    raise RunConflictError(f"no writable local non-synced scratch directory among {[str(c) for c in candidates]}")


@dataclass(frozen=True)
class Layout:
    seed: int
    scratch: Path     # active training state (local, non-synced)
    canonical: Path   # results/.../a1_training_v2/seed_S (OneDrive): verified mirror + final artifacts

    @property
    def scratch_rollouts(self) -> Path:
        return self.scratch / "rollouts"

    @property
    def canonical_rollouts(self) -> Path:
        return self.canonical / "rollouts"

    @property
    def checkpoints(self) -> Path:
        return self.canonical / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.canonical / "logs"

    @property
    def final_model(self) -> Path:
        return self.checkpoints / "final_model.zip"

    @property
    def final_manifest(self) -> Path:
        return self.checkpoints / "final_model.manifest.json"


def make_layout(seed: int, scratch_root: Path, results_root: Path | None = None) -> Layout:
    if int(seed) not in A.A1_V2_PPO_SEEDS:
        raise ValueError(f"PPO seed must be one of {A.A1_V2_PPO_SEEDS}, got {seed}")
    results_root = Path(results_root) if results_root is not None else R.RESULTS_ROOT
    layout = Layout(int(seed), Path(scratch_root) / f"seed_{seed}", results_root / f"seed_{seed}")
    assert_isolated(layout.canonical)
    assert_isolated(layout.scratch)
    return layout


def assert_isolated(path: Path) -> Path:
    """Never inside the failed-v1 namespace or any other V1 results tree."""
    resolved = Path(path).resolve()
    forbidden = [R.FAILED_V1_RESULTS_ROOT.resolve()] + [(PROJECT_ROOT / "results" / n).resolve() for n in (
        "adaptive_attacker_400k", "adaptive_attacker_diagnostic", "clutter_training_400k", "defender_adaptation_400k",
        "defender_adaptation", "training_400k", "training_post_detection_400k")]
    for root in forbidden:
        if resolved == root or root in resolved.parents:
            raise RunConflictError(f"{resolved} is inside the immutable failed/legacy namespace {root}")
    results = (PROJECT_ROOT / "results").resolve()
    if results in resolved.parents and R.RESULTS_ROOT.resolve() not in resolved.parents and resolved != R.RESULTS_ROOT.resolve():
        raise RunConflictError(f"{resolved} is under results/ but outside {R.RESULTS_ROOT}")
    return resolved


# ----------------------------------------------------------------------------- rollout checkpoint commit / verify
def rollout_names(k: int) -> dict[str, str]:
    return {"model": f"rollout_{k:02d}.zip", "rng": f"rollout_{k:02d}.rng.pkl", "episodes": f"rollout_{k:02d}.episodes.jsonl",
            "manifest": f"rollout_{k:02d}.manifest.json"}


def capture_rng() -> bytes:
    import torch

    return pickle.dumps({"torch": torch.get_rng_state().numpy().tobytes(), "numpy": np.random.get_state(), "python": random.getstate()})


def restore_rng(blob: bytes) -> None:
    import torch

    state = pickle.loads(blob)
    torch.set_rng_state(torch.frombuffer(bytearray(state["torch"]), dtype=torch.uint8).clone())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def summarize_episodes(episodes: list[dict]) -> dict:
    n = len(episodes)
    if n == 0:
        return {"episodes": 0}
    count = lambda *o: sum(e["outcome"] in o for e in episodes) / n  # noqa: E731
    mean = lambda key: float(np.nanmean([e[key] for e in episodes]))  # noqa: E731
    return {"episodes": n, "capture_rate": count("soldier_caught"), "interception_rate": count("intercepted", "unsafe_intercept"),
            "collision_rate": count("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
            "exhaustion_rate": count("enemy_endurance_exhausted"), "timeout_rate": count("timeout"),
            "mean_return": mean("return_undiscounted"), "mean_return_discounted": mean("return_discounted"),
            "mean_path_excess": mean("path_excess"), "mean_residual_angle_deg": mean("residual_angle_mean_deg"),
            "mean_ceiling_occupancy": mean("ceiling_occupancy"), "sustained_hover_rate": float(np.mean([e["sustained_hover"] for e in episodes])),
            "ceiling_hover_rate": float(np.mean([e["ceiling_hover"] for e in episodes])),
            "terminal": float(np.mean([e["components"]["terminal"] for e in episodes])),
            "progress": float(np.mean([e["components"]["progress"] for e in episodes])),
            "path": float(np.mean([e["components"]["path"] for e in episodes])),
            "step": float(np.mean([e["components"]["step"] for e in episodes]))}


def verify_rollout(directory: Path, k: int, *, seed: int, spec: Spec, description: dict) -> dict | None:
    """The rollout's manifest if (and only if) it verifies completely; otherwise None."""
    names = rollout_names(k)
    manifest_path = Path(directory) / names["manifest"]
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("rollout") != k or manifest.get("ppo_seed") != seed:
            return None
        if manifest.get("num_timesteps") != k * spec.steps_per_rollout:
            return None
        for key in ("training_protocol", "reward_version", "action_version", "observation_version"):
            if manifest.get(key) != description[key]:
                return None
        for part in ("model", "rng", "episodes"):
            if sha256_file(Path(directory) / names[part]) != manifest[f"{part}_sha256"]:
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def commit_rollout(layout: Layout, k: int, model, episodes: list[dict], *, spec: Spec, description: dict, resume_index: int,
                   fault_hook=None) -> dict:
    """Atomic recovery checkpoint. The manifest rename is the commit point."""
    names = rollout_names(k)
    directory = layout.scratch_rollouts
    buffer = io.BytesIO()
    model.save(buffer)
    model_bytes = buffer.getvalue()
    rng_bytes = capture_rng()
    episodes_bytes = "".join(json.dumps({**e, "rollout": k}, default=str) + "\n" for e in episodes).encode("utf-8")
    atomic_write_bytes(directory / names["model"], model_bytes)
    _hook(fault_hook, "after_model_written", k)
    atomic_write_bytes(directory / names["rng"], rng_bytes)
    atomic_write_bytes(directory / names["episodes"], episodes_bytes)
    _hook(fault_hook, "before_manifest", k)
    manifest = {
        "schema": SCHEMA, "rollout": k, "num_timesteps": int(model.num_timesteps), "n_envs": spec.n_envs, "n_steps": spec.n_steps,
        "ppo_seed": layout.seed, "resume_index": resume_index, "model_file": names["model"], "model_sha256": sha256_bytes(model_bytes),
        "rng_file": names["rng"], "rng_sha256": sha256_bytes(rng_bytes), "episodes_file": names["episodes"],
        "episodes_sha256": sha256_bytes(episodes_bytes), "n_episodes": len(episodes), "summary": summarize_episodes(episodes),
        "training_protocol": description["training_protocol"], "reward_version": description["reward_version"],
        "action_version": description["action_version"], "observation_version": description["observation_version"],
        "theta_max_deg": description.get("theta_max_deg"), "canonical_a0_tag": CANONICAL_A0_TAG,
        "canonical_a0_commit": CANONICAL_A0_COMMIT, "research_commit": _git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(), "scratch_dir": str(layout.scratch),
    }
    atomic_write_json(directory / names["manifest"], manifest)
    _hook(fault_hook, "after_manifest", k)
    return manifest


def _hook(fault_hook, stage: str, k: int) -> None:
    if fault_hook is not None:
        fault_hook(stage, k)


def mirror_rollout(layout: Layout, k: int) -> bool:
    """Copy a committed rollout into the canonical (OneDrive) directory, manifest last. Returns False on failure."""
    names = rollout_names(k)
    try:
        for part in ("model", "rng", "episodes", "manifest"):
            source = layout.scratch_rollouts / names[part]
            destination = layout.canonical_rollouts / names[part]
            if destination.exists() and sha256_file(destination) == sha256_file(source):
                continue
            atomic_copy(source, destination)
        return True
    except OSError as error:
        print(f"[mirror] rollout {k}: {error!r}; will retry later", flush=True)
        return False


def recover(layout: Layout, *, spec: Spec, description: dict) -> tuple[int, dict | None]:
    """Highest complete verified rollout (0 = none). Removes temp files and any partial/unverifiable rollout artifacts."""
    layout.scratch_rollouts.mkdir(parents=True, exist_ok=True)
    removed = cleanup_temporaries(layout.scratch_rollouts) + cleanup_temporaries(layout.canonical_rollouts)
    best_k, best_manifest = 0, None
    for k in range(spec.n_rollouts, 0, -1):
        manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec, description=description)
        if manifest is None and layout.canonical_rollouts.exists():
            if verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec, description=description) is not None:
                for part, name in rollout_names(k).items():  # scratch was lost: restore from the verified mirror
                    atomic_copy(layout.canonical_rollouts / name, layout.scratch_rollouts / name)
                manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec, description=description)
        if manifest is not None:
            best_k, best_manifest = k, manifest
            break
    keep = {n for part, n in rollout_names(best_k).items()} if best_k else set()
    for path in sorted(layout.scratch_rollouts.glob("rollout_*")):
        number = int(path.name.split("_")[1].split(".")[0])
        if number > best_k or (number == best_k and path.name not in keep):
            path.unlink()
            removed.append(path.name)
    if removed:
        print(f"[recover] discarded incomplete/temp files: {removed}", flush=True)
    return best_k, best_manifest


# ----------------------------------------------------------------------------- environments and PPO
def make_env(ppo_seed: int, worker_index: int, resume_index: int):
    return R.A1V2R2TrainingEnv(ppo_seed, worker_index, resume_index)


def worker_env_fns(ppo_seed: int, n_envs: int, resume_index: int) -> list:
    return [functools.partial(make_env, ppo_seed, i, resume_index) for i in range(n_envs)]


def default_env_factory(layout_seed: int, spec: Spec):
    def factory(resume_index: int):
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        fns = worker_env_fns(layout_seed, spec.n_envs, resume_index)
        assert_canonical_v2_env_fn(lambda seed: fns[0](), layout_seed)
        return DummyVecEnv(fns) if spec.n_envs == 1 else SubprocVecEnv(fns)

    return factory


def _episode_collector():
    from stable_baselines3.common.callbacks import BaseCallback

    class Collector(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.episodes: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("a1_episode")
                if episode is not None:
                    self.episodes.append({**episode, "timesteps": int(self.num_timesteps)})
            return True

    return Collector()


def assert_model_finite(model) -> None:
    import torch

    if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
        raise FloatingPointError("non-finite PPO parameters after an update; the rollout is NOT committed")


def keep_awake() -> None:
    """Ask Windows not to sleep while training (harmless elsewhere)."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass


# ----------------------------------------------------------------------------- the seed driver
def train_seed(seed: int, *, scratch_root: Path, results_root: Path | None = None, spec: Spec = CANONICAL_SPEC,
               env_factory=None, ppo_kwargs: dict | None = None, description: dict | None = None, fault_hook=None) -> dict:
    """Train ONE PPO seed to exactly `spec.n_rollouts` committed rollouts, resuming automatically. Idempotent."""
    from stable_baselines3 import PPO

    description = description if description is not None else R.protocol_description()
    layout = make_layout(seed, scratch_root, results_root)
    layout.canonical.mkdir(parents=True, exist_ok=True)
    layout.scratch.mkdir(parents=True, exist_ok=True)
    if layout.final_model.exists():
        return finalize(layout, spec=spec, description=description, already_final=True)

    completed, manifest = recover(layout, spec=spec, description=description)
    resume_index = (manifest["resume_index"] + 1) if manifest else 0
    started = time.time()
    factory = env_factory if env_factory is not None else default_env_factory(seed, spec)
    env = factory(resume_index)
    kwargs = ppo_kwargs if ppo_kwargs is not None else dict(A.A1_V2_PPO_HYPERPARAMS)
    try:
        if completed:
            model = PPO.load(str(layout.scratch_rollouts / manifest["model_file"]).removesuffix(".zip"), env=env, device="cpu")
            if int(model.num_timesteps) != completed * spec.steps_per_rollout:
                raise RunConflictError(f"restored num_timesteps {model.num_timesteps} != {completed * spec.steps_per_rollout}")
            restore_rng((layout.scratch_rollouts / manifest["rng_file"]).read_bytes())
            print(f"[seed {seed}] resumed from verified rollout {completed}/{spec.n_rollouts} (segment {resume_index})", flush=True)
        else:
            model = PPO(policy=A.A1_V2_POLICY, env=env, seed=seed, verbose=0, device="cpu", **kwargs)
            print(f"[seed {seed}] fresh model (segment {resume_index})", flush=True)
        for k in range(completed + 1, spec.n_rollouts + 1):
            _hook(fault_hook, "before_learn", k)
            collector = _episode_collector()
            model.learn(total_timesteps=spec.steps_per_rollout, reset_num_timesteps=False, callback=collector, progress_bar=False)
            _hook(fault_hook, "after_learn", k)
            if int(model.num_timesteps) != k * spec.steps_per_rollout:
                raise RunConflictError(f"num_timesteps {model.num_timesteps} != {k * spec.steps_per_rollout} after rollout {k}")
            assert_model_finite(model)
            committed = commit_rollout(layout, k, model, collector.episodes, spec=spec, description=description,
                                       resume_index=resume_index, fault_hook=fault_hook)
            mirror_rollout(layout, k)
            _hook(fault_hook, "after_commit", k)
            s = committed["summary"]
            print(f"[seed {seed}] rollout {k:02d}/{spec.n_rollouts} steps={committed['num_timesteps']} episodes={s.get('episodes')} "
                  f"capture={s.get('capture_rate', 0):.3f} exh={s.get('exhaustion_rate', 0):.3f} col={s.get('collision_rate', 0):.3f} "
                  f"ret={s.get('mean_return', float('nan')):.2f} t={time.time() - started:.0f}s", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return finalize(layout, spec=spec, description=description, wall_time_s=time.time() - started)


def finalize(layout: Layout, *, spec: Spec, description: dict, already_final: bool = False, wall_time_s: float | None = None) -> dict:
    """Mirror and verify all rollouts in the canonical directory, then write `final_model.zip` + manifest."""
    for k in range(1, spec.n_rollouts + 1):
        in_scratch = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec, description=description) is not None
        if in_scratch and not already_final:
            for _ in range(3):
                if mirror_rollout(layout, k):
                    break
                time.sleep(2)
        elif not in_scratch and verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec, description=description) is None:
            raise RunConflictError(f"rollout {k} is complete neither in scratch nor in the canonical mirror; cannot finalize seed {layout.seed}")
    manifests = []
    for k in range(1, spec.n_rollouts + 1):
        manifest = verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec, description=description)
        if manifest is None:
            raise RunConflictError(f"rollout {k} failed verification in the canonical results directory")
        manifests.append(manifest)
    last = manifests[-1]
    if last["num_timesteps"] != spec.total_timesteps:
        raise RunConflictError(f"final timesteps {last['num_timesteps']} != {spec.total_timesteps}")
    source = layout.canonical_rollouts / last["model_file"]
    if layout.final_model.exists():
        if sha256_file(layout.final_model) != last["model_sha256"]:
            raise RunConflictError("existing final_model.zip does not match rollout 13; refusing to overwrite")
    else:
        atomic_copy(source, layout.final_model)
    final_sha = sha256_file(layout.final_model)
    if final_sha != last["model_sha256"]:
        raise RunConflictError("final_model.zip hash mismatch after copy")
    # assembled training log (episodes of all rollouts)
    layout.logs.mkdir(parents=True, exist_ok=True)
    assembled = b"".join((layout.canonical_rollouts / m["episodes_file"]).read_bytes() for m in manifests)
    atomic_write_bytes(layout.logs / "episodes_all_rollouts.jsonl", assembled)
    final_manifest = {
        **{k: v for k, v in description.items() if k not in ("gates",)},
        "ppo_seed": layout.seed, "model_file": "final_model.zip", "model_sha256": final_sha, "model_role": "final", "canonical": False,
        "canonical_note": "NOT canonical until the quantitative gates and manual visual inspection pass",
        "num_timesteps": last["num_timesteps"], "rollouts_complete": len(manifests),
        "rollouts": [{"rollout": m["rollout"], "num_timesteps": m["num_timesteps"], "model_sha256": m["model_sha256"],
                      "resume_index": m["resume_index"], "summary": m["summary"]} for m in manifests],
        "resume_segments": sorted({m["resume_index"] for m in manifests}), "scratch_dir": str(layout.scratch),
        "research_commit": last["research_commit"], "canonical_a0_tag": CANONICAL_A0_TAG, "canonical_a0_commit": CANONICAL_A0_COMMIT,
        "config_sha256": {f"{lv}/{est}": canonical_v2_config_sha256(build_v2_env_config(lv, est)) for lv in A.A1_V2_CLUTTER_LEVELS
                          for est in ("measurement", "kalman")},
        "created_utc": datetime.now(timezone.utc).isoformat(), "wall_time_s_last_segment": wall_time_s,
        "episodes_log_sha256": sha256_bytes(assembled),
    }
    if not layout.final_manifest.exists():
        atomic_write_json(layout.final_manifest, final_manifest)
    return json.loads(layout.final_manifest.read_text())


def status(seeds=A.A1_V2_PPO_SEEDS, scratch_root: Path | None = None) -> dict:
    scratch_root = scratch_root or select_scratch_root()
    out = {}
    description = R.protocol_description()
    for seed in seeds:
        layout = make_layout(seed, scratch_root)
        done = sum(verify_rollout(layout.scratch_rollouts, k, seed=seed, spec=CANONICAL_SPEC, description=description) is not None
                   for k in range(1, N_ROLLOUTS + 1)) if layout.scratch_rollouts.exists() else 0
        out[seed] = {"rollouts_verified_in_scratch": done, "final_model": layout.final_model.exists()}
    return out


# ----------------------------------------------------------------------------- campaign + preflight
def compute_environment(scratch_root: Path) -> dict:
    import stable_baselines3
    import torch

    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "ppo_device": "cpu", "cpu_count_logical": os.cpu_count(), "num_vector_envs": A.A1_V2_NUM_ENVS, "vector_env": "SubprocVecEnv",
            "scratch_root": str(scratch_root), "scratch_is_synced": _is_synced(scratch_root), "canonical_results_root": str(R.RESULTS_ROOT)}


def preflight(scratch_root: Path) -> dict:
    """Pre-training safety checks: one env per (level, defender); shapes, finiteness, endurance range, outcome mapping,
    zero-action == A0, canonical guard, distributions. Raises on any failure."""
    from experiments.journal_v2_a1_r2 import A1V2R2TrainingEnv

    bank = A.A1V2DefenderBank()
    rng = np.random.default_rng(7)
    theta = R.load_selected_theta()
    cells, mapping = {}, {}
    for level in A.A1_V2_CLUTTER_LEVELS:
        for family in A.A1_V2_DEFENDER_FAMILIES:
            A.build_episode_config(level, family)
            env = A1V2R2TrainingEnv(201, 0, defender_bank=bank)
            for mode in ("random", "zero"):
                obs, info = env.reset(options={"scenario": (level, family, 3_000_000 + len(cells))})
                assert obs.shape == (37,) and env.action_space.shape == (3,) and np.isfinite(obs).all() and obs[-1] == 1.0
                done, steps = False, 0
                while not done and steps < 2500:
                    action = rng.uniform(-1, 1, 3).astype(np.float32) if mode == "random" else np.zeros(3, np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert np.isfinite(obs).all() and 0.0 <= obs[-1] <= 1.0 and np.isfinite(reward)
                    assert abs(reward - sum(info["a1_reward_components"].values())) < 1e-9
                    assert info["a1_residual_angle_deg"] <= theta + 1e-6
                    done, steps = terminated or truncated, steps + 1
                assert done, (level, family, mode)
                episode = info["a1_episode"]
                assert episode["components"]["terminal"] == R.terminal_reward(episode["outcome"])
                mapping[episode["outcome"]] = episode["components"]["terminal"]
                cells[f"{level}/{family}/{mode}"] = episode["outcome"]
    stream = A.training_scenario_stream(201, 0)
    draws = [A.sample_training_scenario(stream) for _ in range(2000)]
    assert {d[0] for d in draws} == set(A.A1_V2_CLUTTER_LEVELS) and {d[1] for d in draws} == set(A.A1_V2_DEFENDER_FAMILIES)
    return {"cells": len(cells), "terminal_mapping": mapping, "theta_max_deg": theta, "scratch": str(scratch_root)}


def campaign(scratch_root: Path | None = None, seeds=A.A1_V2_PPO_SEEDS) -> None:
    import torch

    torch.set_num_threads(2)
    keep_awake()
    scratch_root = scratch_root or select_scratch_root()
    R.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = R.RESULTS_ROOT / "campaign_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, {
            "campaign": "a1_v2_training_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
            "research_commit": _git("rev-parse", "HEAD"), "compute_environment": compute_environment(scratch_root),
            "description": R.protocol_description(), "preflight": preflight(scratch_root)})
    for seed in seeds:
        result = train_seed(seed, scratch_root=scratch_root)
        print(f"[seed {seed}] COMPLETE steps={result['num_timesteps']} sha={result['model_sha256'][:16]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "preflight", "campaign"))
    parser.add_argument("--scratch", default=None)
    args = parser.parse_args()
    scratch = select_scratch_root(args.scratch)
    if args.command == "status":
        print(json.dumps({"scratch": str(scratch), **status(scratch_root=scratch)}, indent=1))
    elif args.command == "preflight":
        print(json.dumps(preflight(scratch), indent=1))
    else:
        campaign(scratch)
