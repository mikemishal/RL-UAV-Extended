"""
Defender-adaptation held-out evaluation harness (journal extension,
38-phase defender-adaptation-EVALUATION sub-task). Pure evaluation: loads
the 9 frozen D0 checkpoints and the 18 completed D0+/D1 continuation
checkpoints (journal-defender-adaptation-training-v1, commit 69596d8),
evaluates them against A0 (scripted) and the 3 Adaptive Attacker v1 seeds
(201/202/203) across C1-C3, N=100 matched held-out episodes per cell.
NO training or model modification happens anywhere in this module.

Reuses the validated Adaptive Attacker v1 episode-evaluation/resume
PATTERN (`experiments/adaptive_attacker_evaluation.py`): immutable
manifest, episode-level JSONL persistence (atomic flush+fsync append),
malformed-final-line tolerance, duplicate-identical-is-no-op/
duplicate-different-raises, skip-already-valid resumability. The key
SCHEMA differs (this evaluation's key has more fields -- see
`EPISODE_KEY_FIELDS` below), so the persistence functions are
reimplemented here (parameterized over THIS module's own key fields)
rather than imported wholesale; `VALID_OUTCOMES` (the environment's fixed
terminal-outcome set, unchanged) is imported directly to avoid drift.

PRIMARY CHECKPOINT SELECTION (Phase 1/2, fixed decision, documented here
per the user's explicit instruction):
    - D0:    the ORIGINAL `best_model.zip` from which adaptation began.
    - D0+:   `final_model.zip` from the scripted-only continuation run.
    - D1:    `final_model.zip` from the adaptive-mixture continuation run.
D0+'s and D1's `best_model.zip` files were each selected by a periodic
in-training evaluation callback using DIFFERENT opponent distributions
(D0+'s validation = 100% A0; D1's validation = the same 50/25/25 mixture
as its own training) -- comparing them would silently compare "best
against A0" to "best against a mixture", an asymmetric model-selection
criterion. Using the FINAL checkpoint from both (same starting D0 + same
+425,984-transition budget + different opponent distribution ONLY) keeps
the comparison clean. `best_model.zip` is retained in the model registry
for a later SECONDARY sensitivity analysis only -- never the primary
result. This is a methodological choice to preserve fairness, not a
failure of the training experiment; no training/validation is rerun or
altered because of it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_composition import DEFAULT_RESIDUAL_MAX_ANGLE_DEG
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.adaptive_attacker_env import ATTACKER_TRAINING_SEEDS, build_attacker_observation
from experiments.adaptive_attacker_evaluation import VALID_OUTCOMES, _attacker_checkpoint_path

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_CODE_COMMIT = "69596d8af8dc27bb080fa1eee37a38ea5090b6ef"  # journal-defender-adaptation-training-v1
D0_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k"
ADAPT_ROOT = PROJECT_ROOT / "results" / "defender_adaptation"
A1_ROOT = PROJECT_ROOT / "results" / "adaptive_attacker_400k"
OUT_DIR = PROJECT_ROOT / "results" / "defender_adaptation_evaluation"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "results" / "defender_adaptation_model_registry.json"

ARCHITECTURES: dict[str, dict] = {
    "ppo_clutter_v1": {"seeds": (71, 72, 73), "estimator": "measurement", "needs_sector_features": True, "is_lr_ppo": False},
    "ppo_kalman_clutter_v1": {"seeds": (74, 75, 76), "estimator": "kalman", "needs_sector_features": True, "is_lr_ppo": False},
    "lrppo_clutter_v1": {"seeds": (77, 78, 79), "estimator": "measurement", "needs_sector_features": True, "is_lr_ppo": True},
}
GENERATIONS = ("d0", "d0_plus", "d1")
EVAL_LEVELS = ("C1", "C2", "C3")

# Phase 3: disjoint held-out evaluation-seed namespace. Decision rule applied:
# (1) audited every seed constant across experiments/*.py; the maximum found
#     is 90_000_000 + (eval_episodes-1) = 90_000_019, from the SHARED
#     periodic in-training validation callback
#     (experiments/clutter_training_runner.py::_LightweightEvalCallback,
#     `env.reset(seed=90_000_000+i)`, eval_episodes=20) used identically by
#     D0's own 400k campaign AND every one of the 18 D0+/D1 continuation
#     runs just completed. All other namespaces found (CLUTTER_EVAL_SEED_START
#     =95_000, ATTACKER_VALIDATION_SEED_START=180_000, ATTACKER_HELDOUT_EVAL_
#     SEED_START=190_000, RESIDUAL_*_SEED_START<=66_000, ROBUSTNESS/SWEEP
#     seed offsets all <100_000, BOOTSTRAP_SEED=202608xx-style RNG seeds for
#     an unrelated purpose) are far lower.
# (2) round upward to the next multiple of 10,000 strictly greater than
#     90_000_019 -> 90_010_000.
# (3) reserve the first 100 seeds beginning there.
EVAL_SEED_START = 90_010_000
N_EVAL_SEEDS = 100
EVAL_SEEDS: tuple[int, ...] = tuple(range(EVAL_SEED_START, EVAL_SEED_START + N_EVAL_SEEDS))

# A1 attacker conditions: "A0" (scripted) + the 3 Adaptive Attacker v1 seeds,
# relabeled per the user's naming (201/202 = seen, 203 = unseen) -- same
# underlying frozen checkpoints as experiments/adaptive_attacker_evaluation.py.
ATTACKER_CONDITIONS: tuple[tuple[str, int | None], ...] = (
    ("A0", None), ("A1-seen-201", 201), ("A1-seen-202", 202), ("A1-unseen-203", 203),
)

EPISODE_KEY_FIELDS = (
    "architecture", "defender_seed", "generation", "checkpoint_role",
    "attacker_type", "attacker_seed", "level", "eval_seed",
)
MANDATORY_NUMERIC_FIELDS = (
    "episode_length", "min_defender_attacker_distance", "min_attacker_asset_distance",
    "defender_path_length", "attacker_path_length", "defender_safety_intervention_fraction",
)


def _eval_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# PHASE 1: audit the 18 campaign outputs, build the frozen model registry.
# =============================================================================

def audit_campaign_outputs() -> list[str]:
    """Programmatically verifies all 18 D0+/D1 training outputs BEFORE any
    registry/evaluation code trusts them. Returns a list of problem
    strings (empty == clean)."""
    problems: list[str] = []
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            for generation, subdir in (("d0_plus", "d0_plus_scripted"), ("d1", "d1_adaptive_mixture")):
                out_dir = ADAPT_ROOT / subdir / arch / f"seed_{seed}"
                manifest_path = out_dir / "training_manifest.json"
                if not manifest_path.exists():
                    problems.append(f"MISSING training_manifest.json: {manifest_path}")
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("actual_additional_timesteps") != 425984:
                    problems.append(
                        f"{arch} seed_{seed} {generation}: actual_additional_timesteps="
                        f"{manifest.get('actual_additional_timesteps')} != 425984"
                    )
                if manifest.get("architecture") != arch:
                    problems.append(f"{arch} seed_{seed} {generation}: architecture field mismatch")
                if manifest.get("seed") != seed:
                    problems.append(f"{arch} seed_{seed} {generation}: seed field mismatch")
                if manifest.get("generation") != generation:
                    problems.append(f"{arch} seed_{seed} {generation}: generation field mismatch")
                if manifest.get("git_commit") != TRAINING_CODE_COMMIT:
                    problems.append(
                        f"{arch} seed_{seed} {generation}: git_commit={manifest.get('git_commit')} "
                        f"!= expected {TRAINING_CODE_COMMIT}"
                    )
                if generation == "d1":
                    pool = manifest.get("opponent_pool_spec") or {}
                    weights = pool.get("weights", {})
                    if weights != {"A0": 0.5, "A1-seen-201": 0.25, "A1-seen-202": 0.25}:
                        problems.append(f"{arch} seed_{seed} d1: unexpected opponent mixture {weights}")
                resume_manifest_path = out_dir / "checkpoints" / "resume_manifest.json"
                if not resume_manifest_path.exists():
                    problems.append(f"MISSING resume_manifest.json: {resume_manifest_path}")
                else:
                    rm = json.loads(resume_manifest_path.read_text())
                    if rm.get("completed") is not True:
                        problems.append(f"{arch} seed_{seed} {generation}: resume_manifest.completed != True")
                for fname in ("final_model.zip", "best_model.zip"):
                    if not (out_dir / fname).exists():
                        problems.append(f"MISSING {fname}: {out_dir / fname}")

            # Starting D0 checkpoint consistency between the matched D0+/D1 pair.
            d0_plus_manifest = json.loads((ADAPT_ROOT / "d0_plus_scripted" / arch / f"seed_{seed}" / "training_manifest.json").read_text())
            d1_manifest = json.loads((ADAPT_ROOT / "d1_adaptive_mixture" / arch / f"seed_{seed}" / "training_manifest.json").read_text())
            if d0_plus_manifest["starting_checkpoint_sha256"] != d1_manifest["starting_checkpoint_sha256"]:
                problems.append(f"{arch} seed_{seed}: D0+ and D1 started from DIFFERENT D0 checkpoints")

    # D0 checkpoint hashes still match the original recorded baseline.
    d0_hashes = json.loads((D0_ROOT / "model_hashes.json").read_text())["hashes"]
    for path, expected in d0_hashes.items():
        actual = _sha256_file(D0_ROOT / path)
        if actual != expected:
            problems.append(f"D0 CHECKPOINT HASH MISMATCH: {path}")

    # A1 checkpoint hashes still match the original recorded baseline.
    a1_hashes = json.loads((A1_ROOT / "model_hashes.json").read_text())
    for entry in a1_hashes:
        path = A1_ROOT / f"seed_{entry['seed']}" / "best_model.zip"
        if not path.exists():
            path = A1_ROOT / f"seed_{entry['seed']}" / "final_model.zip"
        if _sha256_file(path) != entry["sha256"]:
            problems.append(f"A1 CHECKPOINT HASH MISMATCH: seed_{entry['seed']}")

    return problems


def build_model_registry() -> dict:
    """Builds (and writes) the frozen `defender_adaptation_model_registry.json`
    -- the ONE authoritative source of model paths/hashes for all downstream
    evaluation code (Phase 1: 'do not infer paths dynamically during
    evaluation after this registry is frozen')."""
    problems = audit_campaign_outputs()
    if problems:
        raise RuntimeError("Campaign audit found problems, refusing to build model registry:\n" + "\n".join(problems))

    d0_hashes = json.loads((D0_ROOT / "model_hashes.json").read_text())["hashes"]
    a1_hashes = {e["seed"]: e["sha256"] for e in json.loads((A1_ROOT / "model_hashes.json").read_text())}

    registry: dict[str, dict] = {}
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            d0_path = D0_ROOT / arch / f"seed_{seed}" / "best_model.zip"
            registry[f"d0|{arch}|{seed}"] = {
                "architecture": arch, "defender_seed": seed, "generation": "d0",
                "checkpoint_role": "d0_start", "path": str(d0_path),
                "sha256": d0_hashes[f"{arch}/seed_{seed}/best_model.zip"],
            }
            for generation, subdir in (("d0_plus", "d0_plus_scripted"), ("d1", "d1_adaptive_mixture")):
                out_dir = ADAPT_ROOT / subdir / arch / f"seed_{seed}"
                final_path = out_dir / "final_model.zip"
                best_path = out_dir / "best_model.zip"
                manifest = json.loads((out_dir / "training_manifest.json").read_text())
                registry[f"{generation}|{arch}|{seed}"] = {
                    "architecture": arch, "defender_seed": seed, "generation": generation,
                    "checkpoint_role": "final", "path": str(final_path), "sha256": _sha256_file(final_path),
                    "best_model_path": str(best_path), "best_model_sha256": _sha256_file(best_path),
                    "starting_checkpoint_sha256": manifest["starting_checkpoint_sha256"],
                    "actual_additional_timesteps": manifest["actual_additional_timesteps"],
                    "training_code_commit": manifest["git_commit"],
                    "opponent_pool_spec": manifest.get("opponent_pool_spec"),
                }

    attacker_registry = {}
    for seed in ATTACKER_TRAINING_SEEDS:
        path = _attacker_checkpoint_path(seed)
        attacker_registry[str(seed)] = {"path": str(path), "sha256": a1_hashes[seed]}
        if a1_hashes[seed] != _sha256_file(path):
            raise RuntimeError(f"A1 seed {seed} hash mismatch when building registry")

    full_registry = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_code_commit": TRAINING_CODE_COMMIT,
        "defenders": registry,
        "attackers": attacker_registry,
    }
    MODEL_REGISTRY_PATH.write_text(json.dumps(full_registry, indent=2, default=str))
    return full_registry


def load_model_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"model registry not found at {MODEL_REGISTRY_PATH}; call build_model_registry() first")
    return json.loads(MODEL_REGISTRY_PATH.read_text())


def verify_checkpoint_integrity(registry: dict) -> list[str]:
    """Phase 8: re-hash every registry-referenced checkpoint (9 D0 + 18
    D0+/D1 final + 18 D0+/D1 best + 3 A1) against the registry's own
    recorded hashes. Call before the first episode AND after final
    completion; any mismatch is a hard failure (Phase 8: 'any unexpected
    model mutation is a hard failure')."""
    problems = []
    for key, entry in registry["defenders"].items():
        actual = _sha256_file(Path(entry["path"]))
        if actual != entry["sha256"]:
            problems.append(f"{key}: path hash mismatch ({entry['path']})")
        if "best_model_path" in entry:
            actual_best = _sha256_file(Path(entry["best_model_path"]))
            if actual_best != entry["best_model_sha256"]:
                problems.append(f"{key}: best_model_path hash mismatch ({entry['best_model_path']})")
    for seed, entry in registry["attackers"].items():
        actual = _sha256_file(Path(entry["path"]))
        if actual != entry["sha256"]:
            problems.append(f"attacker seed_{seed}: hash mismatch ({entry['path']})")
    return problems


# =============================================================================
# Defender instances (built from the FROZEN registry -- Phase 1: no dynamic
# path inference after the registry exists).
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DefenderInstance:
    architecture: str
    defender_seed: int
    generation: str           # "d0" | "d0_plus" | "d1"
    checkpoint_role: str      # "d0_start" | "final"
    estimator: str
    needs_sector_features: bool
    is_lr_ppo: bool
    checkpoint_path: str
    checkpoint_sha256: str
    build: Callable[[EnvConfig], object]


def build_defender_instances(registry: dict | None = None) -> list[DefenderInstance]:
    registry = registry if registry is not None else load_model_registry()
    instances = []
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            for generation in GENERATIONS:
                entry = registry["defenders"][f"{generation}|{arch}|{seed}"]
                path = entry["path"]
                if arch == "lrppo_clutter_v1":
                    build = lambda cfg, p=path: LeadResidualPPOPolicyWrapper.load(
                        p, residual_max_angle_deg=DEFAULT_RESIDUAL_MAX_ANGLE_DEG, config=cfg, deterministic=True,
                    )
                elif arch == "ppo_kalman_clutter_v1":
                    build = lambda cfg, p=path: get_policy("ppo_kalman", model_path=str(p), deterministic=True)
                else:
                    build = lambda cfg, p=path: get_policy("ppo", model_path=str(p), deterministic=True)
                instances.append(DefenderInstance(
                    architecture=arch, defender_seed=seed, generation=generation,
                    checkpoint_role=entry["checkpoint_role"], estimator=spec["estimator"],
                    needs_sector_features=spec["needs_sector_features"], is_lr_ppo=spec["is_lr_ppo"],
                    checkpoint_path=path, checkpoint_sha256=entry["sha256"], build=build,
                ))
    return instances


@dataclasses.dataclass(frozen=True)
class AttackerCondition:
    name: str              # "A0" | "A1-seen-201" | "A1-seen-202" | "A1-unseen-203"
    seed: int | None
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    model: object | None


def build_attacker_conditions(registry: dict | None = None) -> list[AttackerCondition]:
    registry = registry if registry is not None else load_model_registry()
    conditions = [AttackerCondition("A0", None, None, None, None)]
    for name, seed in ATTACKER_CONDITIONS[1:]:
        entry = registry["attackers"][str(seed)]
        model = PPO.load(str(entry["path"]))
        conditions.append(AttackerCondition(name, seed, entry["path"], entry["sha256"], model))
    return conditions


def _episode_key(d: DefenderInstance, attacker: AttackerCondition, level: str, eval_seed: int) -> str:
    return "|".join(str(x) for x in (
        d.architecture, d.defender_seed, d.generation, d.checkpoint_role,
        attacker.name, attacker.seed, level, eval_seed,
    ))


def _row_key(row: dict) -> str:
    return "|".join(str(row[f]) for f in EPISODE_KEY_FIELDS)


def _config_hash(config: EnvConfig) -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True, default=str).encode()).hexdigest()[:16]


# =============================================================================
# PHASE 7: immutable evaluation manifest (32,400 expected keys).
# =============================================================================

def build_evaluation_manifest(registry: dict | None = None) -> dict:
    """Every one of the 27 x 4 x 3 x 100 = 32,400 expected (defender,
    attacker, level, eval_seed) combinations, with full provenance. Pure
    function of frozen inputs (registry + fixed architectures/seeds/
    attacker conditions/clutter levels/eval seeds) -- deterministic,
    regenerating it always reproduces the identical manifest (Phase 7:
    'do not regenerate it differently during resume')."""
    registry = registry if registry is not None else load_model_registry()
    defenders = build_defender_instances(registry)
    attacker_conditions_meta = [
        {"name": name, "seed": seed, **(
            {"checkpoint_path": registry["attackers"][str(seed)]["path"],
             "checkpoint_sha256": registry["attackers"][str(seed)]["sha256"]}
            if seed is not None else {"checkpoint_path": None, "checkpoint_sha256": None}
        )}
        for name, seed in ATTACKER_CONDITIONS
    ]
    config_hash_cache: dict[tuple[str, str], str] = {}

    def _get_config_hash(level: str, estimator: str) -> str:
        key = (level, estimator)
        if key not in config_hash_cache:
            config_hash_cache[key] = _config_hash(build_canonical_env_config(level, estimator))
        return config_hash_cache[key]

    keys = []
    for d in defenders:
        for attacker_meta in attacker_conditions_meta:
            for level in EVAL_LEVELS:
                for eval_seed in EVAL_SEEDS:
                    keys.append({
                        "key": "|".join(str(x) for x in (
                            d.architecture, d.defender_seed, d.generation, d.checkpoint_role,
                            attacker_meta["name"], attacker_meta["seed"], level, eval_seed,
                        )),
                        "architecture": d.architecture, "defender_seed": d.defender_seed,
                        "generation": d.generation, "checkpoint_role": d.checkpoint_role,
                        "defender_checkpoint_path": d.checkpoint_path, "defender_checkpoint_sha256": d.checkpoint_sha256,
                        "attacker_type": attacker_meta["name"], "attacker_seed": attacker_meta["seed"],
                        "attacker_checkpoint_path": attacker_meta["checkpoint_path"],
                        "attacker_checkpoint_sha256": attacker_meta["checkpoint_sha256"],
                        "level": level, "eval_seed": eval_seed,
                        "clutter_config_hash": _get_config_hash(level, d.estimator),
                        "code_commit": TRAINING_CODE_COMMIT,
                        "physical_collision_config": "journal-physical-collision-semantics-v1 (unchanged)",
                        "awareness_mode": "map_local",
                        "observation_version": "36-D (PPO/PPO-Kalman)" if not d.is_lr_ppo else "39-D (LR-PPO)",
                    })
    manifest = {
        "total_expected_keys": len(keys),
        "n_defenders": len(defenders),
        "n_attacker_conditions": len(attacker_conditions_meta),
        "n_levels": len(EVAL_LEVELS),
        "n_eval_seeds": len(EVAL_SEEDS),
        "eval_seed_range": [EVAL_SEED_START, EVAL_SEED_START + N_EVAL_SEEDS - 1],
        "entries": keys,
    }
    return manifest


def write_evaluation_manifest(path: Path | None = None) -> dict:
    path = path if path is not None else (OUT_DIR / "evaluation_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_evaluation_manifest()
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


# =============================================================================
# Episode execution (Phase 4 common-random-numbers + Phase 10/11 outcomes/metrics).
# =============================================================================

NEAR_30_DEG_MARGIN = 2.0  # degrees; "near the 30 deg bound" defined BEFORE inspecting results (Phase 31)


def _percentile_or_nan(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def _run_matched_episode(
    env: SoldierEnv,
    defender_instance: DefenderInstance,
    defender_policy,
    lead_policy_for_residual,
    attacker_model,
    eval_seed: int,
) -> dict:
    """Runs ONE held-out episode. `attacker_model=None` => scripted hostile
    (enemy_tactical_override left unset, bit-for-bit legacy behavior --
    Phase 2 audit); otherwise the given FROZEN (inference-only) learned
    attacker model drives the hostile via the override hook, exactly as in
    experiments/adaptive_attacker_evaluation.py. For a GIVEN (level,
    eval_seed), `env.reset(seed=eval_seed)` reproduces an IDENTICAL spawn
    state / protected-asset walk / static-obstacle layout / dynamic-
    obstacle trajectories / measurement-noise stream regardless of which
    defender/attacker condition is run (Phase 4 common-random-numbers
    protocol) -- ONLY the attacker's own tactical decisions (and, as a
    downstream consequence, the defender's reactive trajectory) can differ
    between conditions sharing the same (level, eval_seed)."""
    pending = {"action": np.zeros(3, dtype=np.float32)}
    env.enemy_tactical_override = (lambda: pending["action"]) if attacker_model is not None else None

    obs, info = env.reset(seed=eval_seed)
    defender_policy.reset()
    if lead_policy_for_residual is not None:
        lead_policy_for_residual.reset()

    start_asset_dist = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
    min_defender_attacker_dist = float(np.linalg.norm(env._defender_pos - env._enemy_pos))
    min_attacker_asset_dist = start_asset_dist
    defender_path_length = 0.0
    attacker_path_length = 0.0
    prev_defender_pos = env._defender_pos.copy()
    prev_attacker_pos = env._enemy_pos.copy()
    detection_time: int | None = None
    defender_avoidance_steps = 0
    override_angles: list[float] = []
    static_clearances: list[float] = []
    dynamic_clearances: list[float] = []
    residual_angles: list[float] = []

    steps = 0
    done = False
    while not done:
        if attacker_model is not None:
            attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
            action, _ = attacker_model.predict(attacker_obs, deterministic=True)
            pending["action"] = np.asarray(action, dtype=np.float32)

        defender_obs = env._get_obs()
        defender_info = build_policy_info(env._get_info(), defender_instance.estimator)
        sector = None
        if defender_instance.needs_sector_features:
            sector = compute_sector_features(
                defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
            )
        if defender_instance.is_lr_ppo:
            defender_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
            if lead_policy_for_residual is not None:
                lead_direction = np.asarray(lead_policy_for_residual.act(defender_obs, defender_info), dtype=np.float64)
                final_direction = np.asarray(defender_action, dtype=np.float64)
                ln, fn = np.linalg.norm(lead_direction), np.linalg.norm(final_direction)
                if ln > 1e-8 and fn > 1e-8:
                    cos_angle = float(np.clip(np.dot(lead_direction, final_direction) / (ln * fn), -1.0, 1.0))
                    residual_angles.append(float(np.degrees(np.arccos(cos_angle))))
        elif defender_instance.needs_sector_features:
            defender_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)
        else:
            defender_action = defender_policy.act(defender_obs, defender_info)

        obs, reward, terminated, truncated, info = env.step(np.asarray(defender_action, dtype=np.float32))
        steps += 1
        done = terminated or truncated

        if detection_time is None and info.get("enemy_detected"):
            detection_time = steps
        cur_defender_pos = env._defender_pos.copy()
        cur_attacker_pos = env._enemy_pos.copy()
        defender_path_length += float(np.linalg.norm(cur_defender_pos - prev_defender_pos))
        attacker_path_length += float(np.linalg.norm(cur_attacker_pos - prev_attacker_pos))
        prev_defender_pos, prev_attacker_pos = cur_defender_pos, cur_attacker_pos
        min_defender_attacker_dist = min(min_defender_attacker_dist, float(np.linalg.norm(cur_defender_pos - cur_attacker_pos)))
        min_attacker_asset_dist = min(min_attacker_asset_dist, float(np.linalg.norm(env._soldier_pos - cur_attacker_pos)))
        if info.get("defender_avoidance_active"):
            defender_avoidance_steps += 1
            override_angles.append(float(info.get("defender_avoidance_angle_deg", 0.0)))
        static_clearances.append(float(info.get("defender_min_obstacle_clearance", float("inf"))))
        dynamic_clearances.append(float(info.get("defender_min_dynamic_obstacle_clearance", float("inf"))))

    outcome = info["outcome"]
    row = {
        "outcome": outcome,
        "defender_safe_intercept": 1 if outcome == "intercepted" else 0,
        "attacker_asset_reach": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        "defender_static_obstacle_collision": 1 if outcome == "defender_static_obstacle_collision" else 0,
        "defender_dynamic_obstacle_collision": 1 if outcome == "defender_dynamic_obstacle_collision" else 0,
        "enemy_static_obstacle_collision": 1 if outcome == "enemy_static_obstacle_collision" else 0,
        "enemy_dynamic_obstacle_collision": 1 if outcome == "enemy_dynamic_obstacle_collision" else 0,
        "episode_length": steps,
        "detection_time": detection_time,
        "interception_time": steps if outcome == "intercepted" else None,
        "defender_path_length": defender_path_length,
        "attacker_path_length": attacker_path_length,
        "min_defender_attacker_distance": min_defender_attacker_dist,
        "min_attacker_asset_distance": min_attacker_asset_dist,
        "defender_safety_intervention_fraction": defender_avoidance_steps / steps if steps else 0.0,
        "mean_override_angle_deg": float(np.mean(override_angles)) if override_angles else 0.0,
        "max_override_angle_deg": float(np.max(override_angles)) if override_angles else 0.0,
        "p95_override_angle_deg": _percentile_or_nan(override_angles, 95) if override_angles else 0.0,
        "static_min_clearance": float(np.min(static_clearances)) if static_clearances else float("inf"),
        "dynamic_min_clearance": float(np.min(dynamic_clearances)) if dynamic_clearances else float("inf"),
    }
    if defender_instance.is_lr_ppo:
        row["mean_residual_angle_deg"] = float(np.mean(residual_angles)) if residual_angles else 0.0
        row["max_residual_angle_deg"] = float(np.max(residual_angles)) if residual_angles else 0.0
        row["p95_residual_angle_deg"] = _percentile_or_nan(residual_angles, 95) if residual_angles else 0.0
        row["fraction_near_30deg"] = (
            float(np.mean([1.0 if (DEFAULT_RESIDUAL_MAX_ANGLE_DEG - a) <= NEAR_30_DEG_MARGIN else 0.0 for a in residual_angles]))
            if residual_angles else 0.0
        )
    return row


# =============================================================================
# PHASE 12: resumable JSONL persistence (same validated PATTERN as
# experiments/adaptive_attacker_evaluation.py, parameterized for THIS
# module's own (richer) episode key schema).
# =============================================================================

def is_valid_episode_row(row: dict) -> bool:
    for field in EPISODE_KEY_FIELDS + ("outcome",):
        if field not in row:
            return False
    if row["outcome"] not in VALID_OUTCOMES:
        return False
    for field in MANDATORY_NUMERIC_FIELDS:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            return False
    return True


def load_completed_episodes(jsonl_path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not jsonl_path.exists():
        return completed
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            is_last = (i == len(lines) - 1)
            if not is_last:
                raise ValueError(
                    f"malformed JSONL line {i + 1}/{len(lines)} in {jsonl_path} is NOT the final line -- "
                    f"indicates corruption beyond a simple truncated-final-write, must be investigated manually."
                )
            continue
        if not is_valid_episode_row(row):
            continue
        key = _row_key(row)
        if key in completed and completed[key] != row:
            raise ValueError(f"duplicate episode key {key!r} in {jsonl_path} with DIFFERING content")
        completed[key] = row
    return completed


def append_episode_result(jsonl_path: Path, row: dict, completed: dict[str, dict]) -> None:
    key = _row_key(row)
    if key in completed:
        if completed[key] != row:
            raise ValueError(f"refusing to append duplicate episode key {key!r} with different content")
        return
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    completed[key] = row


def build_recovery_manifest(jsonl_path: Path | None = None, evaluation_manifest: dict | None = None) -> dict:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    evaluation_manifest = evaluation_manifest if evaluation_manifest is not None else build_evaluation_manifest()
    completed = load_completed_episodes(jsonl_path)
    expected_keys = [e["key"] for e in evaluation_manifest["entries"]]
    total_expected = len(expected_keys)
    n_valid = sum(1 for k in expected_keys if k in completed)
    return {
        "total_expected": total_expected,
        "complete_valid": n_valid,
        "missing": total_expected - n_valid,
        "percent_complete": (100.0 * n_valid / total_expected) if total_expected else 100.0,
    }


# =============================================================================
# PHASE 16: the main resumable evaluation loop.
# =============================================================================

def evaluate_all(jsonl_path: Path | None = None, registry: dict | None = None) -> pd.DataFrame:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else load_model_registry()
    completed = load_completed_episodes(jsonl_path)

    defenders = build_defender_instances(registry)
    attacker_conditions = build_attacker_conditions(registry)
    lead_policy_cache: dict[str, object] = {}
    eval_code_commit = _eval_git_commit()

    rows = list(completed.values())
    n_skipped = 0
    n_executed = 0
    for d in defenders:
        for level in EVAL_LEVELS:
            base_config = build_canonical_env_config(level, d.estimator)
            defender_policy = d.build(base_config)
            lead_policy_for_residual = None
            if d.is_lr_ppo:
                cache_key = f"{level}:{d.estimator}"
                if cache_key not in lead_policy_cache:
                    lead_policy_cache[cache_key] = get_policy("lead", config=base_config)
                lead_policy_for_residual = lead_policy_cache[cache_key]
            env = SoldierEnv(base_config)
            for attacker in attacker_conditions:
                t0 = time.time()
                n_new = 0
                for eval_seed in EVAL_SEEDS:
                    key = _episode_key(d, attacker, level, eval_seed)
                    if key in completed:
                        n_skipped += 1
                        continue
                    metrics = _run_matched_episode(env, d, defender_policy, lead_policy_for_residual, attacker.model, eval_seed)
                    metrics.update(
                        architecture=d.architecture, defender_seed=d.defender_seed, generation=d.generation,
                        checkpoint_role=d.checkpoint_role, attacker_type=attacker.name, attacker_seed=attacker.seed,
                        level=level, eval_seed=eval_seed, eval_code_commit=eval_code_commit,
                    )
                    append_episode_result(jsonl_path, metrics, completed)
                    rows.append(metrics)
                    n_executed += 1
                    n_new += 1
                if n_new:
                    print(f"{d.architecture:>22s} seed={d.defender_seed} {d.generation:<8s} level={level} "
                          f"attacker={attacker.name:<16s} (+{n_new} new, {time.time()-t0:.1f}s)", flush=True)
    print(f"=== evaluation loop done: {n_executed} executed, {n_skipped} skipped (already complete) ===", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-registry", action="store_true")
    args = parser.parse_args()

    if args.rebuild_registry or not MODEL_REGISTRY_PATH.exists():
        print("Building model registry (Phase 1 audit)...", flush=True)
        build_model_registry()

    registry = load_model_registry()
    print("Checkpoint integrity check (before)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED before evaluation:\n" + "\n".join(problems))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "evaluation_manifest.json"
    if not manifest_path.exists():
        write_evaluation_manifest(manifest_path)
    evaluation_manifest = json.loads(manifest_path.read_text())

    recovery = build_recovery_manifest(OUT_DIR / "episodes.jsonl", evaluation_manifest)
    print("=== recovery manifest (before) ===", flush=True)
    print(json.dumps(recovery, indent=2), flush=True)

    df = evaluate_all(OUT_DIR / "episodes.jsonl", registry)

    final_recovery = build_recovery_manifest(OUT_DIR / "episodes.jsonl", evaluation_manifest)
    print("=== recovery manifest (after) ===", flush=True)
    print(json.dumps(final_recovery, indent=2), flush=True)
    if final_recovery["missing"] != 0:
        raise RuntimeError(f"evaluation incomplete: {final_recovery['missing']} missing -- re-run to resume")

    print("Checkpoint integrity check (after)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED after evaluation:\n" + "\n".join(problems))

    df.to_csv(OUT_DIR / "episodes.csv", index=False)
    print(f"wrote {len(df)} episode rows to {OUT_DIR / 'episodes.csv'}", flush=True)
