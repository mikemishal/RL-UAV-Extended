"""Power-loss-safe mastery-based PPO runner for curriculum Stage 2 (`a1_cur2_training_v1`): evade a STATIONARY active defender while reaching a stationary asset.

    python -m experiments.journal_v2_a1_cur2_training status
    python -m experiments.journal_v2_a1_cur2_training run --seeds 201 202 203       # safe to re-run after ANY interruption
    python -m experiments.journal_v2_a1_cur2_training resume-test                   # real-SB3 interrupted vs uninterrupted check

Same machinery and guarantees as the Stage-1 runner (`journal_v2_a1_cur1_training`): one rollout = 16 x 2048 = 32,768 steps = ONE `learn(32768, reset_num_timesteps=False)`; at most 8 rollouts; scenario streams
re-keyed to the rollout index and every env reset before a rollout (a rollout is a pure function of model+optimizer state, torch RNG state and rollout index); after each rollout an atomic commit
(model incl. optimizer, RNG pickle, episode log, MANIFEST renamed LAST), OneDrive mirror, then validation (alternating sets A/B + matched A0 + Stage-1 pursuit anchor) and the mastery / retention rules.

Differences from Stage 1: (1) the PPO model is initialised from the MATCHING Stage-1 BC checkpoint (actor + log_std = -2, FRESH critic, FRESH optimizer); (2) the validation gate includes the improvement over matched A0
and the Stage-1 anchor capture; (3) a seed also stops on PURSUIT RETENTION FAILURE (anchor capture < 0.98 on two consecutive rollouts).
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1_training as T1
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur2 as S
from experiments.journal_v2_a1_eg_state import CellLock

T = T1.T
PROJECT_ROOT = T.PROJECT_ROOT
SCHEMA = "a1_cur2_rollout_manifest/1"
VALIDATION_SCHEMA = "a1_cur2_validation/1"
DEFAULT_SCRATCH = Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage2")
RunConflictError, SimulatedPowerLoss = T.RunConflictError, T.SimulatedPowerLoss
Spec, CANONICAL_SPEC, Layout, names, state_sha256, summarize_training, mastery_from_passes, collector, ppo_kwargs_hash, _hook, one_rollout, mirror_rollout = (
    T1.Spec, T1.CANONICAL_SPEC, T1.Layout, T1.names, T1.state_sha256, T1.summarize_training, T1.mastery_from_passes, T1.collector, T1.ppo_kwargs_hash, T1._hook, T1.one_rollout, T1.mirror_rollout)


def select_scratch_root(override: str | os.PathLike | None = None) -> Path:
    candidates = [Path(override)] if override else ([Path(os.environ["A1_SCRATCH_ROOT"])] if os.environ.get("A1_SCRATCH_ROOT") else [])
    candidates += [DEFAULT_SCRATCH, Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "journal_v2_a1_curriculum" / "stage2",
                   Path(tempfile.gettempdir()) / "journal_v2_a1_curriculum" / "stage2"]
    for candidate in candidates:
        if not T._is_synced(candidate) and T._writable_roundtrip(candidate):
            return candidate.resolve()
    raise RunConflictError(f"no writable local non-synced scratch directory among {[str(c) for c in candidates]}")


def make_layout(seed: int, scratch_root: Path, results_root: Path | None = None) -> Layout:
    results_root = Path(results_root) if results_root is not None else S.RESULTS_ROOT
    layout = Layout(int(seed), Path(scratch_root) / f"seed_{seed}", results_root / f"seed_{seed}")
    resolved, root, results = layout.canonical.resolve(), S.RESULTS_ROOT.resolve(), (PROJECT_ROOT / "results").resolve()
    if results in resolved.parents and root not in resolved.parents and resolved != root:
        raise RunConflictError(f"{resolved} is under results/ but outside the Stage-2 namespace {root}")
    return layout


# ----------------------------------------------------------------------------- BC parent -> PPO initialisation
def bc_parent(seed: int, project_root: Path | None = None) -> dict:
    """Path + SHA-256 of the MATCHING Stage-1 BC checkpoint, from the committed Stage-1 handoff; the file hash is verified."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    handoff = json.loads((root / S.STAGE1_V2_HANDOFF).read_text())
    entry = handoff["checkpoints"][str(seed)]
    path = root / Path(str(entry["path"]).replace("\\", "/"))
    sha = T.sha256_file(path)
    if sha != entry["model_sha256"]:
        raise RunConflictError(f"BC checkpoint {path} hash {sha} != handoff {entry['model_sha256']}")
    return {"seed": int(seed), "attacker_id": S.BC_SEEDS[int(seed)], "path": str(path), "model_sha256": sha, "log_std": handoff["log_std"]}


def initialize_model(seed: int, env, kwargs: dict, *, parent: dict | None = None):
    """PPO(seed=seed) (random actor AND critic from the PPO seed) -> overwrite ONLY the actor trunk/head and log_std from the matching BC model. The critic keeps its PPO-seed initialisation and the optimizer
    is the freshly constructed one (no BC optimizer state, no failed-PPO weights). Returns (model, record)."""
    from stable_baselines3 import PPO

    parent = parent or bc_parent(seed)
    model = PPO(policy=A.A1_V2_POLICY, env=env, seed=seed, verbose=0, device="cpu", **kwargs)
    fresh_value = BC.value_state_sha256(model)
    fresh_actor = BC.tensor_sha256(BC.actor_state(model))
    bc = BC.load_model(Path(parent["path"]))
    BC.load_actor_state(model, BC.actor_state(bc))
    model.policy.log_std.data.copy_(bc.policy.log_std.data)
    if BC.value_state_sha256(model) != fresh_value:
        raise RunConflictError("critic changed during BC->PPO initialisation")
    if BC.tensor_sha256(BC.actor_state(model)) != BC.tensor_sha256(BC.actor_state(bc)) or not np.allclose(model.policy.log_std.detach().numpy(), parent["log_std"]):
        raise RunConflictError("actor / log_std were not copied exactly from the BC parent")
    record = {"seed": int(seed), "parent": parent, "actor_copied": ["mlp_extractor.policy_net", "action_net"], "log_std": [float(x) for x in model.policy.log_std.detach().numpy()],
              "initial_action_std": [float(x) for x in np.exp(model.policy.log_std.detach().numpy())], "fresh_critic_sha256": fresh_value, "fresh_actor_sha256_overwritten": fresh_actor,
              "actor_sha256": BC.tensor_sha256(BC.actor_state(model)), "critic_source": "PPO(seed=%d) default initialisation (not BC, not any earlier PPO model)" % seed,
              "optimizer": "fresh Adam from PPO construction (no BC optimizer state)", "not_loaded": ["failed A1-v1 PPO weights", "Stage-1-v1 failed PPO models"],
              "initial_state_sha256": state_sha256(model)}
    return model, record


# ----------------------------------------------------------------------------- rollout commit / verify / recover
def verify_rollout(directory: Path, k: int, *, seed: int, spec: Spec) -> dict | None:
    n = names(k)
    try:
        manifest = json.loads((Path(directory) / n["manifest"]).read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("rollout") != k or manifest.get("ppo_seed") != seed:
            return None
        if manifest.get("num_timesteps") != k * spec.steps_per_rollout or manifest.get("training_protocol") != S.TRAINING_PROTOCOL:
            return None
        for part in ("model", "rng", "episodes"):
            if T.sha256_file(Path(directory) / n[part]) != manifest[f"{part}_sha256"]:
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def commit_rollout(layout: Layout, k: int, model, episodes: list[dict], *, spec: Spec, ppo_kwargs: dict, parent: dict, fault_hook=None) -> dict:
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
        "schema": SCHEMA, "rollout": k, "ppo_seed": layout.seed, "num_timesteps": int(model.num_timesteps), "n_updates": int(model._n_updates), "n_envs": spec.n_envs, "n_steps": spec.n_steps,
        "model_file": n["model"], "model_sha256": T.sha256_bytes(model_bytes), "state_sha256": state_sha256(model), "rng_file": n["rng"], "rng_sha256": T.sha256_bytes(rng_bytes),
        "episodes_file": n["episodes"], "episodes_sha256": T.sha256_bytes(episodes_bytes), "n_episodes": len(episodes), "training_summary": summarize_training(episodes),
        "training_protocol": S.TRAINING_PROTOCOL, "placement_protocol": S.PLACEMENT_PROTOCOL, "reward_version": S.REWARD_VERSION, "action_version": S.ACTION_VERSION,
        "observation_version": S.OBSERVATION_VERSION, "env_config_sha256": S.config_sha256(), "ppo_hyperparameters_sha256": ppo_kwargs_hash(ppo_kwargs),
        "bc_parent_sha256": parent["model_sha256"], "scenario_namespace": {"stream_tag": hex(S.SCENARIO_STREAM_TAG), "placement_tag": hex(S.PLACEMENT_TAG), "ppo_seed": layout.seed, "rollout_index": k,
                                                                      "workers": spec.n_envs}, "research_commit": T._git("rev-parse", "HEAD"), "created_utc": datetime.now(timezone.utc).isoformat(),
        "scratch_dir": str(layout.scratch),
    }
    T.atomic_write_json(directory / n["manifest"], manifest)
    _hook(fault_hook, "after_manifest", k)
    return manifest


def recover(layout: Layout, *, spec: Spec) -> tuple[int, dict | None]:
    layout.scratch_rollouts.mkdir(parents=True, exist_ok=True)
    removed = T.cleanup_temporaries(layout.scratch_rollouts) + (T.cleanup_temporaries(layout.canonical_rollouts) if layout.canonical_rollouts.exists() else [])
    best_k, best_manifest = 0, None
    for k in range(spec.max_rollouts, 0, -1):
        manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)
        if manifest is None and layout.canonical_rollouts.exists() and verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec) is not None:
            for name in names(k).values():
                if (layout.canonical_rollouts / name).exists():
                    T.atomic_copy(layout.canonical_rollouts / name, layout.scratch_rollouts / name)
            manifest = verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)
        if manifest is not None:
            best_k, best_manifest = k, manifest
            break
    for path in sorted(layout.scratch_rollouts.glob("rollout_*")):
        number = int(path.name.split("_")[1].split(".")[0])
        if number > best_k or verify_rollout(layout.scratch_rollouts, number, seed=layout.seed, spec=spec) is None:
            path.unlink()
            removed.append(path.name)
    if removed:
        print(f"[recover] discarded incomplete/temp files: {removed}", flush=True)
    return best_k, best_manifest


# ----------------------------------------------------------------------------- validation (set A/B + matched A0 + Stage-1 anchor) and the per-seed decision
def validation_set_for(k: int) -> str:
    return S.VALIDATION_ROLLOUT_SET[k % 2 == 1]


def _content(payload_parts: dict) -> str:
    return T.sha256_bytes(json.dumps(payload_parts, sort_keys=True).encode())


def verify_validation(directory: Path, k: int, manifest: dict) -> dict | None:
    try:
        payload = json.loads((Path(directory) / names(k)["validation"]).read_text())
        parts = {"metrics": payload["metrics"], "a0_metrics": payload["a0_metrics"], "anchor_metrics": payload["anchor_metrics"]}
        if (payload.get("schema") != VALIDATION_SCHEMA or payload.get("rollout") != k or payload.get("model_sha256") != manifest["model_sha256"] or payload.get("content_sha256") != _content(parts)
                or payload.get("validation_set") != validation_set_for(k) or payload.get("gate_spec") != S.GATE):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_validation(model, k: int, manifest: dict) -> dict:
    name = validation_set_for(k)
    seeds = list(S.validation_seeds(name))
    act = S.C.model_actor(model)
    m = S.metrics(S.evaluate(act, seeds))
    a0 = S.metrics(S.evaluate(None, seeds, scripted_a0=True))
    anchor = V1V2.metrics(S.evaluate_anchor(act))
    gate = S.gate_status(m, a0_capture=a0["capture"], anchor_capture=anchor["capture"])
    parts = {"metrics": m, "a0_metrics": a0, "anchor_metrics": anchor}
    return {"schema": VALIDATION_SCHEMA, "rollout": k, "ppo_seed": manifest["ppo_seed"], "model_sha256": manifest["model_sha256"], "num_timesteps": manifest["num_timesteps"], "validation_set": name,
            "seeds": [seeds[0], seeds[-1]], **parts, "gate": gate, "passed": gate["passed"], "anchor_capture": anchor["capture"], "gate_spec": S.GATE, "content_sha256": _content(parts),
            "created_utc": datetime.now(timezone.utc).isoformat()}


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


def seed_decision(validations: list[dict]) -> tuple[str | None, int | None]:
    """Scan rollouts in order: ('MASTERED', k) at the second consecutive pass; ('PURSUIT_RETENTION_FAILURE', k) at the second consecutive anchor failure; else (None, None). Mastery wins a tie."""
    for k in range(1, len(validations) + 1):
        passes = [v["passed"] for v in validations[:k]]
        if mastery_from_passes(passes) == k:
            return "MASTERED", k
        if S.retention_failure_rollout([v["anchor_capture"] for v in validations[:k]]) == k:
            return "PURSUIT_RETENTION_FAILURE", k
    return None, None


def collect_validations(layout: Layout, completed: int, spec: Spec) -> list[dict]:
    return [ensure_validation(layout, k, verify_rollout(layout.scratch_rollouts, k, seed=layout.seed, spec=spec)) for k in range(1, completed + 1)]


# ----------------------------------------------------------------------------- environments
def default_env_factory(seed: int, spec: Spec):
    def factory():
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        fns = [functools.partial(S.make_env, seed, i, 0) for i in range(spec.n_envs)]
        return DummyVecEnv(fns) if spec.n_envs == 1 else SubprocVecEnv(fns)

    return factory


# ----------------------------------------------------------------------------- the seed driver
def train_seed(seed: int, *, scratch_root: Path, results_root: Path | None = None, spec: Spec = CANONICAL_SPEC, env_factory=None, ppo_kwargs: dict | None = None, validate: bool = True,
               stop_after: int | None = None, fault_hook=None, parent: dict | None = None) -> dict:
    """Train ONE PPO seed rollout by rollout until mastery / retention failure / `spec.max_rollouts`; resumes automatically; idempotent."""
    from stable_baselines3 import PPO

    layout = make_layout(seed, scratch_root, results_root)
    layout.canonical.mkdir(parents=True, exist_ok=True)
    layout.scratch.mkdir(parents=True, exist_ok=True)
    kwargs = dict(ppo_kwargs) if ppo_kwargs is not None else dict(A.A1_V2_PPO_HYPERPARAMS)
    parent = parent or bc_parent(seed)
    lock = CellLock(layout.lock_base)
    if not lock.acquire():
        raise RunConflictError(f"another training process owns seed {seed} (lock {lock.path})")
    started = time.time()
    try:
        if layout.seed_manifest.exists():
            return json.loads(layout.seed_manifest.read_text())
        completed, manifest = recover(layout, spec=spec)
        validations = collect_validations(layout, completed, spec) if (validate and completed) else []
        status, decided_at = seed_decision(validations) if validate else (None, None)
        if status is None and completed < spec.max_rollouts and (stop_after is None or completed < stop_after):
            env = (env_factory or default_env_factory(seed, spec))()
            try:
                if completed:
                    model = PPO.load(str(layout.scratch_rollouts / manifest["model_file"]).removesuffix(".zip"), env=env, device="cpu")
                    if int(model.num_timesteps) != completed * spec.steps_per_rollout:
                        raise RunConflictError(f"restored num_timesteps {model.num_timesteps} != {completed * spec.steps_per_rollout}")
                    T.restore_rng((layout.scratch_rollouts / manifest["rng_file"]).read_bytes())
                    print(f"[seed {seed}] resumed from verified rollout {completed}/{spec.max_rollouts}", flush=True)
                else:
                    model, init_record = initialize_model(seed, env, kwargs, parent=parent)
                    T.atomic_write_json(layout.scratch / "init_record.json", init_record)
                    print(f"[seed {seed}] initialised from {parent['attacker_id']} ({parent['model_sha256'][:12]}), fresh critic", flush=True)
                for k in range(completed + 1, spec.max_rollouts + 1):
                    _hook(fault_hook, "before_learn", k)
                    episodes = one_rollout(model, env, k, spec)
                    _hook(fault_hook, "after_learn", k)
                    committed = commit_rollout(layout, k, model, episodes, spec=spec, ppo_kwargs=kwargs, parent=parent, fault_hook=fault_hook)
                    mirror_rollout(layout, k)
                    ts = committed["training_summary"]
                    line = f"[seed {seed}] rollout {k:02d} steps={committed['num_timesteps']} train_capture={ts.get('capture_rate', 0):.3f}"
                    if validate:
                        payload = ensure_validation(layout, k, committed, model)
                        validations.append(payload)
                        mm = payload["metrics"]
                        line += (f" | val{payload['validation_set']} capture={mm['capture']:.3f} icpt={mm['interception']:.3f} exh={mm['exhaustion']:.3f} medPE={mm['median_path_excess']:.3f} "
                                 f"anchor={payload['anchor_capture']:.3f} gate={'PASS' if payload['passed'] else 'fail'}")
                    print(line + f" t={time.time() - started:.0f}s", flush=True)
                    _hook(fault_hook, "after_commit", k)
                    completed = k
                    status, decided_at = seed_decision(validations) if validate else (None, None)
                    if status is not None or (stop_after is not None and k >= stop_after):
                        break
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        if not validate or (status is None and completed < spec.max_rollouts):
            return {"seed": seed, "status": "INCOMPLETE", "rollouts_complete": completed}
        return finalize(layout, spec=spec, validations=validations, status=status or "FAIL", decided_at=decided_at, kwargs=kwargs, parent=parent, wall_time_s=time.time() - started)
    finally:
        lock.release()


def finalize(layout: Layout, *, spec: Spec, validations: list[dict], status: str, decided_at: int | None, kwargs: dict, parent: dict, wall_time_s: float | None) -> dict:
    last = len(validations)
    for k in range(1, last + 1):
        for _ in range(3):
            if mirror_rollout(layout, k):
                break
            time.sleep(2)
    manifests = [verify_rollout(layout.canonical_rollouts, k, seed=layout.seed, spec=spec) for k in range(1, last + 1)]
    if any(m is None for m in manifests):
        raise RunConflictError(f"a rollout of seed {layout.seed} failed verification in the canonical directory; cannot finalize")
    accepted_k = decided_at if status == "MASTERED" else last
    accepted = manifests[accepted_k - 1]
    if layout.accepted_model.exists() and T.sha256_file(layout.accepted_model) != accepted["model_sha256"]:
        raise RunConflictError("existing accepted model does not match the accepted rollout; refusing to overwrite")
    if not layout.accepted_model.exists():
        T.atomic_copy(layout.canonical_rollouts / accepted["model_file"], layout.accepted_model)
    sha = T.sha256_file(layout.accepted_model)
    if sha != accepted["model_sha256"]:
        raise RunConflictError("accepted model hash mismatch after copy")
    init_path = layout.scratch / "init_record.json"
    if init_path.exists():
        T.atomic_copy(init_path, layout.canonical / "init_record.json")
    assembled = b"".join((layout.canonical_rollouts / m["episodes_file"]).read_bytes() for m in manifests)
    (layout.canonical / "logs").mkdir(parents=True, exist_ok=True)
    T.atomic_write_bytes(layout.canonical / "logs" / "training_episodes_all_rollouts.jsonl", assembled)
    mastered = status == "MASTERED"
    payload = {
        "seed": layout.seed, "status": status, "mastery_rollout": decided_at if mastered else None, "decided_at_rollout": decided_at,
        "timesteps_to_mastery": (decided_at * spec.steps_per_rollout) if mastered else None, "rollouts_trained": last, "timesteps_trained": last * spec.steps_per_rollout,
        "accepted_rollout": accepted_k, "accepted_model": "checkpoints/accepted_model.zip", "accepted_model_sha256": sha, "accepted_state_sha256": accepted["state_sha256"],
        "model_role": "curriculum Stage-2 checkpoint (NOT a final A1 model)" if mastered else "unmastered Stage-2 seed (final-rollout policy; not accepted)", "bc_parent": parent,
        "training_protocol": S.TRAINING_PROTOCOL, "placement_protocol": S.PLACEMENT_PROTOCOL, "reward_version": S.REWARD_VERSION, "action_version": S.ACTION_VERSION,
        "observation_version": S.OBSERVATION_VERSION, "env_config_sha256": S.config_sha256(), "ppo_hyperparameters": kwargs, "ppo_hyperparameters_sha256": ppo_kwargs_hash(kwargs), "gate": S.GATE,
        "rollouts": [{"rollout": m["rollout"], "num_timesteps": m["num_timesteps"], "model_sha256": m["model_sha256"], "state_sha256": m["state_sha256"], "training_summary": m["training_summary"],
                      "validation_set": v["validation_set"], "validation_metrics": v["metrics"], "a0_metrics": v["a0_metrics"], "anchor_capture": v["anchor_capture"], "gate": v["gate"],
                      "validation_passed": v["passed"]} for m, v in zip(manifests, validations)],
        "scenario_namespace": {"stream_tag": hex(S.SCENARIO_STREAM_TAG), "placement_tag": hex(S.PLACEMENT_TAG), "heading_tag": hex(S.C.HEADING_STREAM_TAG)},
        "episodes_log_sha256": T.sha256_bytes(assembled), "scratch_dir": str(layout.scratch), "research_commit": T._git("rev-parse", "HEAD"), "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s_last_segment": wall_time_s,
    }
    if not layout.seed_manifest.exists():
        T.atomic_write_json(layout.seed_manifest, payload)
    return json.loads(layout.seed_manifest.read_text())


def status(scratch_root: Path | None = None, seeds=S.PPO_SEEDS) -> dict:
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


# ----------------------------------------------------------------------------- real-SB3 interrupted vs uninterrupted check (production code path incl. BC initialisation)
def resume_test(work_dir: Path, *, n_envs: int = 4, n_steps: int = 64, rollouts: int = 3, interrupt_after: int = 1, crash_mid_commit: bool = True, seed: int = 201) -> dict:
    work_dir = Path(work_dir)
    spec = Spec(n_envs=n_envs, n_steps=n_steps, max_rollouts=rollouts)
    kwargs = dict(A.A1_V2_PPO_HYPERPARAMS, n_steps=n_steps, batch_size=64)
    run_a = {"scratch": work_dir / "a_scratch", "results": work_dir / "a_results"}
    run_b = {"scratch": work_dir / "b_scratch", "results": work_dir / "b_results"}
    for r in (run_a, run_b):
        shutil.rmtree(r["scratch"], ignore_errors=True)
        shutil.rmtree(r["results"], ignore_errors=True)
    t0 = time.time()
    train_seed(seed, scratch_root=run_a["scratch"], results_root=run_a["results"], spec=spec, ppo_kwargs=kwargs, validate=False)
    train_seed(seed, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False, stop_after=interrupt_after)
    crashed = False
    if crash_mid_commit:
        def hook(stage, k):
            if stage == "before_manifest" and k == interrupt_after + 1:
                raise SimulatedPowerLoss()
        try:
            train_seed(seed, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False, fault_hook=hook)
        except SimulatedPowerLoss:
            crashed = True
    layout_b = make_layout(seed, run_b["scratch"], run_b["results"])
    leftover_before = sorted(p.name for p in layout_b.scratch_rollouts.glob("rollout_*"))
    (layout_b.scratch / "train.lock").write_text("999999")          # lock owned by a dead PID
    train_seed(seed, scratch_root=run_b["scratch"], results_root=run_b["results"], spec=spec, ppo_kwargs=kwargs, validate=False)
    layout_a = make_layout(seed, run_a["scratch"], run_a["results"])
    rows = []
    for k in range(1, rollouts + 1):
        ma, mb = verify_rollout(layout_a.scratch_rollouts, k, seed=seed, spec=spec), verify_rollout(layout_b.scratch_rollouts, k, seed=seed, spec=spec)
        rows.append({"rollout": k, "uninterrupted_state_sha256": ma["state_sha256"], "interrupted_state_sha256": mb["state_sha256"], "state_equal": ma["state_sha256"] == mb["state_sha256"],
                     "episodes_equal": ma["episodes_sha256"] == mb["episodes_sha256"], "num_timesteps_equal": ma["num_timesteps"] == mb["num_timesteps"]})
    init_equal = (run_a["scratch"] / f"seed_{seed}" / "init_record.json").read_bytes() == (run_b["scratch"] / f"seed_{seed}" / "init_record.json").read_bytes()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "n_envs": n_envs, "n_steps": n_steps, "rollouts": rollouts, "interrupt_after": interrupt_after,
            "crash_injected_between_model_write_and_manifest": crashed, "partial_files_before_resume": leftover_before, "rows": rows, "init_record_equal": init_equal,
            "all_states_equal": all(r["state_equal"] for r in rows), "all_episode_logs_equal": all(r["episodes_equal"] for r in rows), "vec_env": "SubprocVecEnv", "seed": seed,
            "wall_time_s": time.time() - t0}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "run", "resume-test"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(S.PPO_SEEDS))
    parser.add_argument("--scratch", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    T.keep_awake()
    scratch = select_scratch_root(args.scratch)
    if args.command == "status":
        print(json.dumps(status(scratch), indent=1))
    elif args.command == "run":
        print(f"[curriculum stage 2] scratch={scratch}", flush=True)
        for seed in args.seeds:
            result = train_seed(seed, scratch_root=scratch)
            print(f"[seed {seed}] {result.get('status')} mastery_rollout={result.get('mastery_rollout')} timesteps_to_mastery={result.get('timesteps_to_mastery')}", flush=True)
    else:
        work = Path(args.work_dir) if args.work_dir else scratch.parent / "stage2_resume_test"
        result = resume_test(work)
        if args.out:
            T.atomic_write_json(Path(args.out), result)
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
