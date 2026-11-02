"""Curriculum Stage2-v2 (`a1_curriculum_stage2_v2`): a preregistered CONTINUATION of Stage2-v1 that changes ONLY the maximum PPO budget (8 -> 16 complete rollouts).

Stage2-v2 is a preregistered continuation of Stage2-v1. It changes only the maximum PPO budget from 8 to 16 complete rollouts. All environment, reward, initialization, PPO, validation, and mastery
definitions are unchanged.

Mechanics: each seed's verified Stage2-v1 rollout files 1-8 (model + optimizer, RNG, episode log, validation, MANIFEST; hash-verified against the v1 seed manifest) are copied into the Stage2-v2
scratch/results namespace and the unchanged Stage-2 runner (`journal_v2_a1_cur2_training.train_seed`) is called with `max_rollouts = 16`. It recovers rollout 8 exactly as after an interruption (PPO policy, critic,
optimizer, timestep counter and RNG restored; scenario streams / validation alternation keyed by the rollout index), so the extension IS the uninterrupted run. Stage2-v1 files are only READ.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_training as T2
from experiments import journal_v2_a1_r2 as R

VERSION = "a1_curriculum_stage2_v2"
STAGE = "stage2_v2"
V1_MAX_ROLLOUTS, MAX_ROLLOUTS = 8, 16
V1_RESULTS_ROOT = S.RESULTS_ROOT                                   # captured before any activation
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
V1_SCRATCH = Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage2")
DEFAULT_SCRATCH = Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage2_v2")
V1_PARENT_PREFIX = {201: "be8b85e4", 202: "25e67e79", 203: "6472fbee"}
V1_DOCS = "docs/journal_v2_a1_curriculum/stage2/"
_N_STEPS = T2.A.A1_V2_PPO_HYPERPARAMS["n_steps"]
SPEC = T2.Spec(n_envs=S.N_ENVS, n_steps=_N_STEPS, max_rollouts=MAX_ROLLOUTS)
V1_SPEC = T2.Spec(n_envs=S.N_ENVS, n_steps=_N_STEPS, max_rollouts=V1_MAX_ROLLOUTS)
assert SPEC.max_timesteps == 524_288 and V1_SPEC.max_timesteps == 262_144


@contextlib.contextmanager
def activated():
    """Point the Stage-2 runner's output namespace at Stage2-v2 for the duration (in-process only; no file is changed). Stage2-v1 results can then never be written by the runner."""
    previous = S.RESULTS_ROOT
    S.RESULTS_ROOT = RESULTS_ROOT
    try:
        yield
    finally:
        S.RESULTS_ROOT = previous


def select_scratch_root(override=None) -> Path:
    candidates = [Path(override)] if override else []
    candidates += [DEFAULT_SCRATCH, Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "journal_v2_a1_curriculum" / STAGE, Path(tempfile.gettempdir()) / "journal_v2_a1_curriculum" / STAGE]
    for candidate in candidates:
        if not T2.T._is_synced(candidate) and T2.T._writable_roundtrip(candidate):
            return candidate.resolve()
    raise T2.RunConflictError("no writable local non-synced scratch directory")


# ----------------------------------------------------------------------------- protocol
def v1_protocol_sha256(project_root: Path | None = None) -> str:
    root = Path(project_root) if project_root is not None else T2.PROJECT_ROOT
    return json.loads((root / V1_DOCS / "protocol.json").read_text())["protocol_sha256"]


def parent_hashes() -> dict:
    return {str(s): {"attacker_id": S.BC_SEEDS[s], "model_sha256": T2.bc_parent(s)["model_sha256"]} for s in S.PPO_SEEDS}


def protocol_specification() -> dict:
    """The Stage2-v1 protocol with exactly two fields changed (ppo.max_rollouts, ppo.max_timesteps) plus the version / continuation record."""
    spec = copy.deepcopy(S.protocol_specification(parent_hashes()))
    spec["ppo"]["max_rollouts"] = MAX_ROLLOUTS
    spec["ppo"]["max_timesteps"] = MAX_ROLLOUTS * S.N_ENVS * SPEC.n_steps
    spec["identifiers"]["version"] = VERSION
    spec["version"] = {"id": VERSION, "kind": "preregistered budget-only continuation of Stage2-v1", "changed": {"max_rollouts": [V1_MAX_ROLLOUTS, MAX_ROLLOUTS], "max_timesteps": [262_144, 524_288]},
                       "unchanged": "environment, defender placement, observation/action/reward, initialization, PPO hyperparameters, validation sets and alternation, anchor, gates, retention rule",
                       "continuation": "PPO policy, critic, optimizer, timestep counter and RNG/scenario-stream state restored from the verified Stage2-v1 rollout-8 commit (no BC re-initialisation)",
                       "stage2_v1_protocol_sha256": v1_protocol_sha256(), "stop_rule": "fewer than 2/3 seeds mastered by rollout 16 => STOP; no automatic further budget increase"}
    return spec


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()


def differences_from_v1() -> dict:
    """Leaf-level diff between the Stage2-v1 specification and Stage2-v2's (excluding the added `version` record and identifier)."""
    v1, v2 = S.protocol_specification(parent_hashes()), protocol_specification()
    out = {}

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if path == "" and k == "version":
                    continue
                if path == "identifiers" and k == "version":
                    continue
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif a != b:
            out[path] = [a, b]

    walk(v1, v2)
    return out


# ----------------------------------------------------------------------------- Stage2-v1 parents
def v1_seed_manifest(seed: int) -> dict:
    return json.loads((V1_RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())


def v1_rollout_dirs(seed: int) -> list[Path]:
    return [V1_RESULTS_ROOT / f"seed_{seed}" / "rollouts", V1_SCRATCH / f"seed_{seed}" / "rollouts"]


def verify_v1_parent(seed: int) -> dict:
    """The exact verified Stage2-v1 rollout-8 commit of `seed`: manifest hash chain checked against the v1 seed manifest and the pre-registered hash prefix."""
    m = v1_seed_manifest(seed)
    if m["status"] != "FAIL" or m["rollouts_trained"] != V1_MAX_ROLLOUTS or m["training_protocol"] != S.TRAINING_PROTOCOL:
        raise T2.RunConflictError(f"seed {seed}: Stage2-v1 manifest is not an unmastered 8-rollout run")
    r8 = m["rollouts"][V1_MAX_ROLLOUTS - 1]
    if not r8["model_sha256"].startswith(V1_PARENT_PREFIX[seed]) or r8["model_sha256"] != m["accepted_model_sha256"] or r8["rollout"] != V1_MAX_ROLLOUTS:
        raise T2.RunConflictError(f"seed {seed}: rollout-8 hash does not match the pre-registered parent {V1_PARENT_PREFIX[seed]}")
    source = None
    for directory in v1_rollout_dirs(seed):
        if all(T2.verify_rollout(directory, k, seed=seed, spec=V1_SPEC) for k in range(1, V1_MAX_ROLLOUTS + 1)):
            source = directory
            break
    if source is None:
        raise T2.RunConflictError(f"seed {seed}: no directory holds 8 fully verified Stage2-v1 rollouts")
    manifest8 = T2.verify_rollout(source, V1_MAX_ROLLOUTS, seed=seed, spec=V1_SPEC)
    if manifest8["model_sha256"] != r8["model_sha256"] or manifest8["state_sha256"] != r8["state_sha256"]:
        raise T2.RunConflictError(f"seed {seed}: rollout-8 manifest disagrees with the v1 seed manifest")
    return {"seed": seed, "source_dir": str(source), "model_sha256": r8["model_sha256"], "state_sha256_policy_and_optimizer": r8["state_sha256"], "completed_rollout": V1_MAX_ROLLOUTS,
            "num_timesteps": manifest8["num_timesteps"], "rollout8_manifest_sha256": T2.T.sha256_file(source / T2.names(V1_MAX_ROLLOUTS)["manifest"]), "rng_sha256": manifest8["rng_sha256"],
            "v1_seed_manifest_sha256": T2.T.sha256_file(V1_RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json"), "v1_validation_sets_alternation": "odd->A, even->B",
            "v1_final_validation": {"capture": r8["validation_metrics"]["capture"], "interception": r8["validation_metrics"]["interception"], "anchor_capture": r8["anchor_capture"]}}


def seed_from_v1(seed: int, scratch_root: Path) -> dict:
    """Copy the verified v1 rollouts 1-8 (+ init record) into the Stage2-v2 scratch namespace (READ-ONLY on v1). Idempotent; verifies hashes after copying."""
    parent = verify_v1_parent(seed)
    source = Path(parent["source_dir"])
    destination = Path(scratch_root) / f"seed_{seed}" / "rollouts"
    destination.mkdir(parents=True, exist_ok=True)
    for k in range(1, V1_MAX_ROLLOUTS + 1):
        for part in ("model", "rng", "episodes", "validation", "manifest"):          # manifest last
            name = T2.names(k)[part]
            target = destination / name
            if target.exists() and T2.T.sha256_file(target) == T2.T.sha256_file(source / name):
                continue
            T2.T.atomic_copy(source / name, target)
    init = source.parent / "init_record.json"
    if init.exists() and not (destination.parent / "init_record.json").exists():
        T2.T.atomic_copy(init, destination.parent / "init_record.json")
    for k in range(1, V1_MAX_ROLLOUTS + 1):
        manifest = T2.verify_rollout(destination, k, seed=seed, spec=SPEC)
        if manifest is None or T2.verify_validation(destination, k, manifest) is None:
            raise T2.RunConflictError(f"seed {seed}: copied rollout {k} failed verification")
    return parent


# ----------------------------------------------------------------------------- learning-curve description (descriptive only; no model is fitted)
TREND_THRESHOLD = 0.05


def describe_trend(values: list[float], threshold: float = TREND_THRESHOLD) -> dict:
    """Compare the mean of the second half of a series with the mean of its first half (A/B validation sets alternate, so halves average over both): rising / plateau / regressing."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 4:
        return {"n": int(len(v)), "label": "insufficient", "delta": None}
    half = len(v) // 2
    delta = float(v[-half:].mean() - v[:half].mean())
    return {"n": int(len(v)), "first_half_mean": float(v[:half].mean()), "second_half_mean": float(v[-half:].mean()), "delta": delta,
            "label": "rising" if delta > threshold else ("regressing" if delta < -threshold else "plateau")}
