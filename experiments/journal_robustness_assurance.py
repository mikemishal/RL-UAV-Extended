"""Final journal robustness/assurance evaluation (closes the remaining
robustness evidence for the clutter/adaptive-attacker paper). Pure
evaluation: reuses the FROZEN D0+/D1 checkpoints and the A1-unseen-203
attacker checkpoint UNCHANGED. No training, no architecture change, no
environment-semantics change -- only legitimate EnvConfig field overrides
(measurement_var, detection_radius, v_d, v_e) already supported by
`build_canonical_env_config(..., extra_overrides=...)`.

================================================================================
PHASE 1: provenance of original LR-PPO robustness protocol values
================================================================================
Audited (read-only) from the ALREADY-EXISTING, locked robustness-sweep
protocol modules in this repository (never modified by this task):

| Variable | Original sweep values | Nominal | Journal R1/R2/R3 picks |
|---|---|---|---|
| measurement_var (R1) | (0.05, 0.25, 0.50, 1.00, 2.00, 4.00) [`robustness_measurement_noise_protocol.MEASUREMENT_VAR_VALUES`] | 0.50 | low=0.05, nominal=0.50 (reused), high=4.00 |
| detection_radius (R2) | (5.0, 10.0, 15.0, 20.0, 25.0, 30.0) [`robustness_detection_protocol.DETECTION_RADIUS_VALUES`] | 15.0 | short=5.0, nominal=15.0 (reused), long=30.0 |
| defender speed v_d (R3) | (12.0, 15.0, 18.0, 21.0, 24.0) [`robustness_mobility_protocol.DEFENDER_SPEED_VALUES`] | 18.0 | see mobility table below |
| hostile speed v_e (R3) | (8.0, 10.0, 12.0, 14.0, 16.0) [`robustness_mobility_protocol.HOSTILE_SPEED_VALUES`] | 12.0 | see mobility table below |

R3 mobility operating points (actual tested (v_d, v_e) pairs from the
original two 1-D sweeps -- NOT a new Cartesian grid point):
    defender-disadvantaged : (v_d=12.0, v_e=12.0), ratio=1.00 (lowest ratio
        actually tested in either 1-D sweep; the original protocol never
        tested v_e > v_d, so this is the closest available "defender has
        least kinematic advantage" point)
    nominal                : (v_d=18.0, v_e=12.0), ratio=1.50 (canonical)
    defender-advantaged    : (v_d=18.0, v_e=8.0),  ratio=2.25 (highest ratio
        actually tested in either 1-D sweep)

No new numerical values were invented; every R1/R2/R3 setting below is an
exact member of an existing, pre-specified original-paper sweep grid.

================================================================================
PHASE 3/4: nominal-cell reuse + new seed namespace
================================================================================
Nominal-point cells (noise=0.50, detection=15.0, v_d=18/v_e=12) at C0 and
C2 against A1-unseen-203 for D0+/D1 are REUSED verbatim from:
    - results/defender_adaptation_canonical_c0_evaluation/episodes.csv (C0)
    - results/defender_adaptation_evaluation/episodes.csv (C2)
No new episodes are run for the nominal point.

New seed namespace audit: the highest previously-reserved 10,000-aligned
seed block found anywhere in experiments/*.py is
`attacker_heldout_eval` = 190,000-199,999
(experiments/adaptive_attacker_env.py::ATTACKER_HELDOUT_EVAL_SEED_START).
No seed constant of any kind exists in 130,100-189,999 or >=200,000
(grep-audited). This module reserves the next free 10,000-aligned block:
JOURNAL_ROBUSTNESS_SEED_START = 200_000, N=100 (200,000-200,099) -- the
SAME 100 seeds are reused (common random numbers) across every
(dimension, setting, clutter level, architecture, generation) cell,
exactly matching the established EVAL_SEEDS-reuse-across-levels pattern
in experiments/defender_adaptation_evaluation.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.adaptive_attacker_env import build_attacker_observation
from experiments.defender_adaptation_evaluation import (
    PROJECT_ROOT, ARCHITECTURES, VALID_OUTCOMES,
    load_model_registry, build_defender_instances, build_attacker_conditions,
    verify_checkpoint_integrity, _config_hash, _eval_git_commit,
)
from experiments.defender_adaptation_canonical_c0_evaluation import CANONICAL_EXTRA_OVERRIDES
from experiments.robustness_measurement_noise_protocol import MEASUREMENT_VAR_VALUES, NOMINAL_MEASUREMENT_VAR
from experiments.robustness_detection_protocol import DETECTION_RADIUS_VALUES, NOMINAL_DETECTION_RADIUS
from experiments.robustness_mobility_protocol import (
    DEFENDER_SPEED_VALUES, HOSTILE_SPEED_VALUES, NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED,
)

OUT_DIR = PROJECT_ROOT / "results" / "journal_robustness_assurance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATIONS = ("d0_plus", "d1")
CLUTTER_LEVELS = ("C0", "C2")
PRIMARY_ATTACKER = "A1-unseen-203"

JOURNAL_ROBUSTNESS_SEED_START = 200_000
N_EVAL_SEEDS = 100
EVAL_SEEDS: tuple[int, ...] = tuple(range(JOURNAL_ROBUSTNESS_SEED_START, JOURNAL_ROBUSTNESS_SEED_START + N_EVAL_SEEDS))

assert NOMINAL_MEASUREMENT_VAR in MEASUREMENT_VAR_VALUES
assert NOMINAL_DETECTION_RADIUS in DETECTION_RADIUS_VALUES
assert NOMINAL_DEFENDER_SPEED in DEFENDER_SPEED_VALUES and NOMINAL_HOSTILE_SPEED in HOSTILE_SPEED_VALUES

# =============================================================================
# R1/R2/R3 off-nominal operating points (every value is a MEMBER of an
# existing, pre-specified original-paper sweep grid -- see module docstring).
# =============================================================================
R1_SETTINGS: dict[str, float] = {"low": min(MEASUREMENT_VAR_VALUES), "high": max(MEASUREMENT_VAR_VALUES)}
R2_SETTINGS: dict[str, float] = {"short": min(DETECTION_RADIUS_VALUES), "long": max(DETECTION_RADIUS_VALUES)}
R3_SETTINGS: dict[str, tuple[float, float]] = {
    "defender_disadvantaged": (12.0, 12.0),   # ratio 1.00 -- lowest tested ratio
    "defender_advantaged": (18.0, 8.0),       # ratio 2.25 -- highest tested ratio
}
for _vd, _ve in R3_SETTINGS.values():
    assert _vd in DEFENDER_SPEED_VALUES and _ve in HOSTILE_SPEED_VALUES

NOMINAL_POINT = {
    "measurement_var": NOMINAL_MEASUREMENT_VAR, "detection_radius": NOMINAL_DETECTION_RADIUS,
    "v_d": NOMINAL_DEFENDER_SPEED, "v_e": NOMINAL_HOSTILE_SPEED,
}


def _overrides_for(dimension: str, setting_label: str) -> dict:
    if dimension == "R1_measurement_noise":
        return {"measurement_var": R1_SETTINGS[setting_label]}
    if dimension == "R2_detection_radius":
        return {"detection_radius": R2_SETTINGS[setting_label]}
    if dimension == "R3_relative_mobility":
        v_d, v_e = R3_SETTINGS[setting_label]
        return {"v_d": v_d, "v_e": v_e}
    raise ValueError(f"unknown dimension {dimension!r}")


# Exactly 6 off-nominal (dimension, setting_label) cells, per Phase 2/5.
OFF_NOMINAL_CELLS: tuple[tuple[str, str], ...] = (
    ("R1_measurement_noise", "low"), ("R1_measurement_noise", "high"),
    ("R2_detection_radius", "short"), ("R2_detection_radius", "long"),
    ("R3_relative_mobility", "defender_disadvantaged"), ("R3_relative_mobility", "defender_advantaged"),
)


def _env_config_for(level: str, estimator: str, dimension: str, setting_label: str) -> EnvConfig:
    overrides = dict(CANONICAL_EXTRA_OVERRIDES) if level == "C0" else {}
    overrides.update(_overrides_for(dimension, setting_label))
    return build_canonical_env_config(level, estimator, extra_overrides=overrides)


# =============================================================================
# PHASE 6: pre-run configuration assertions -- fail loudly on mismatch.
# =============================================================================

def assert_robustness_config(cfg: EnvConfig, level: str, dimension: str, setting_label: str) -> dict:
    expected = dict(NOMINAL_POINT)
    expected.update(_overrides_for(dimension, setting_label))
    checks = {
        "defender_standby_until_detection": (cfg.defender_standby_until_detection, True),
        "measurement_var": (cfg.measurement_var, expected["measurement_var"]),
        "detection_radius": (cfg.detection_radius, expected["detection_radius"]),
        "v_d": (cfg.v_d, expected["v_d"]),
        "v_e": (cfg.v_e, expected["v_e"]),
    }
    if level == "C0":
        checks["obstacles_enabled"] = (cfg.obstacles_enabled, False)
        checks["defender_obstacle_avoidance_enabled"] = (cfg.defender_obstacle_avoidance_enabled, False)
        checks["enemy_obstacle_avoidance_enabled"] = (cfg.enemy_obstacle_avoidance_enabled, False)
    elif level == "C2":
        checks["obstacles_enabled"] = (cfg.obstacles_enabled, True)
        checks["defender_obstacle_avoidance_enabled"] = (cfg.defender_obstacle_avoidance_enabled, True)
        checks["enemy_obstacle_avoidance_enabled"] = (cfg.enemy_obstacle_avoidance_enabled, True)
        checks["defender_obstacle_awareness_mode"] = (cfg.defender_obstacle_awareness_mode, "map_local")
        checks["enemy_obstacle_awareness_mode"] = (cfg.enemy_obstacle_awareness_mode, "map_local")
    else:
        raise ValueError(f"unexpected level {level!r}")
    failures = [f"{k}: expected {expected!r}, got {actual!r}" for k, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise AssertionError(
            f"Robustness config assertion FAILED for level={level} dimension={dimension} setting={setting_label}:\n"
            + "\n".join(failures)
        )
    return dataclasses.asdict(cfg)


# =============================================================================
# PHASE 7: episode runner -- extends the validated defender_adaptation_
# evaluation._run_matched_episode pattern with Kalman tracking-error metrics
# (ground-truth comparison for diagnostic measurement ONLY, never fed to
# any policy).
# =============================================================================

NEAR_BOUND_MARGIN_DEG = 2.0


def _percentile_or_nan(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def run_robustness_episode(env: SoldierEnv, defender_instance, defender_policy, lead_policy_for_residual,
                            attacker_model, eval_seed: int) -> dict:
    pending = {"action": np.zeros(3, dtype=np.float32)}
    env.enemy_tactical_override = lambda: pending["action"]

    obs, info = env.reset(seed=eval_seed)
    defender_policy.reset()
    if lead_policy_for_residual is not None:
        lead_policy_for_residual.reset()

    min_defender_attacker_dist = float(np.linalg.norm(env._defender_pos - env._enemy_pos))
    min_attacker_asset_dist = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
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
    kalman_pos_errors: list[float] = []
    kalman_vel_errors: list[float] = []

    steps = 0
    done = False
    while not done:
        attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
        action, _ = attacker_model.predict(attacker_obs, deterministic=True)
        pending["action"] = np.asarray(action, dtype=np.float32)

        raw_info = env._get_info()
        defender_obs = env._get_obs()
        defender_info = build_policy_info(raw_info, defender_instance.estimator)
        sector = compute_sector_features(
            defender_pos=env._defender_pos, defender_vel=env._defender_vel,
            perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
        )

        if defender_instance.is_lr_ppo:
            defender_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
            lead_direction = np.asarray(lead_policy_for_residual.act(defender_obs, defender_info), dtype=np.float64)
            final_direction = np.asarray(defender_action, dtype=np.float64)
            ln, fn = np.linalg.norm(lead_direction), np.linalg.norm(final_direction)
            if ln > 1e-8 and fn > 1e-8:
                cos_angle = float(np.clip(np.dot(lead_direction, final_direction) / (ln * fn), -1.0, 1.0))
                residual_angles.append(float(np.degrees(np.arccos(cos_angle))))
        else:
            defender_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)

        if defender_instance.estimator == "kalman" and raw_info.get("e_hat") is not None:
            kalman_pos_errors.append(float(np.linalg.norm(np.asarray(raw_info["e_hat"]) - env._enemy_pos)))
            if raw_info.get("v_hat") is not None:
                kalman_vel_errors.append(float(np.linalg.norm(np.asarray(raw_info["v_hat"]) - env._enemy_vel)))

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
        "static_min_clearance": float(np.min(static_clearances)) if static_clearances else float("inf"),
        "dynamic_min_clearance": float(np.min(dynamic_clearances)) if dynamic_clearances else float("inf"),
    }
    if defender_instance.is_lr_ppo:
        row["mean_residual_angle_deg"] = float(np.mean(residual_angles)) if residual_angles else 0.0
        row["p95_residual_angle_deg"] = _percentile_or_nan(residual_angles, 95)
        row["fraction_near_30deg"] = (
            float(np.mean([1.0 if (30.0 - a) <= NEAR_BOUND_MARGIN_DEG else 0.0 for a in residual_angles]))
            if residual_angles else 0.0
        )
    if defender_instance.estimator == "kalman":
        row["mean_kalman_position_error"] = float(np.mean(kalman_pos_errors)) if kalman_pos_errors else None
        row["mean_kalman_velocity_error"] = float(np.mean(kalman_vel_errors)) if kalman_vel_errors else None
    return row


# =============================================================================
# PHASE 21: resumable JSONL persistence (own key schema: dimension/setting
# distinguish cells that defender_adaptation_evaluation's generic
# EPISODE_KEY_FIELDS cannot).
# =============================================================================

EPISODE_KEY_FIELDS = (
    "architecture", "defender_seed", "generation", "dimension", "setting_label", "level", "eval_seed",
)
MANDATORY_NUMERIC_FIELDS = (
    "episode_length", "min_defender_attacker_distance", "min_attacker_asset_distance",
    "defender_path_length", "attacker_path_length", "defender_safety_intervention_fraction",
)


def _row_key(row: dict) -> str:
    return "|".join(str(row[f]) for f in EPISODE_KEY_FIELDS)


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
            if i != len(lines) - 1:
                raise ValueError(f"malformed JSONL line {i + 1}/{len(lines)} in {jsonl_path} is NOT the final line.")
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


# =============================================================================
# Manifest + main resumable evaluation loop.
# =============================================================================

def build_evaluation_manifest(registry: dict | None = None) -> dict:
    registry = registry if registry is not None else load_model_registry()
    defenders = [d for d in build_defender_instances(registry) if d.generation in GENERATIONS]
    entries = []
    for d in defenders:
        for dimension, setting_label in OFF_NOMINAL_CELLS:
            for level in CLUTTER_LEVELS:
                for eval_seed in EVAL_SEEDS:
                    key = "|".join(str(x) for x in (
                        d.architecture, d.defender_seed, d.generation, dimension, setting_label, level, eval_seed,
                    ))
                    entries.append({
                        "key": key, "architecture": d.architecture, "defender_seed": d.defender_seed,
                        "generation": d.generation, "dimension": dimension, "setting_label": setting_label,
                        "level": level, "eval_seed": eval_seed,
                    })
    return {
        "total_expected_keys": len(entries), "n_defenders": len(defenders),
        "n_off_nominal_cells": len(OFF_NOMINAL_CELLS), "n_levels": len(CLUTTER_LEVELS), "n_eval_seeds": len(EVAL_SEEDS),
        "eval_seed_range": [JOURNAL_ROBUSTNESS_SEED_START, JOURNAL_ROBUSTNESS_SEED_START + N_EVAL_SEEDS - 1],
        "entries": entries,
    }


def build_recovery_manifest(jsonl_path: Path | None = None, evaluation_manifest: dict | None = None) -> dict:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    evaluation_manifest = evaluation_manifest if evaluation_manifest is not None else build_evaluation_manifest()
    completed = load_completed_episodes(jsonl_path)
    expected_keys = [e["key"] for e in evaluation_manifest["entries"]]
    total_expected = len(expected_keys)
    n_valid = sum(1 for k in expected_keys if k in completed)
    return {"total_expected": total_expected, "complete_valid": n_valid, "missing": total_expected - n_valid,
            "percent_complete": (100.0 * n_valid / total_expected) if total_expected else 100.0}


def evaluate_all(jsonl_path: Path | None = None, registry: dict | None = None) -> pd.DataFrame:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "episodes.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else load_model_registry()
    completed = load_completed_episodes(jsonl_path)

    defenders = [d for d in build_defender_instances(registry) if d.generation in GENERATIONS]
    attacker = next(a for a in build_attacker_conditions(registry) if a.name == PRIMARY_ATTACKER)
    lead_policy_cache: dict[str, object] = {}
    eval_code_commit = _eval_git_commit()

    rows = list(completed.values())
    n_skipped = n_executed = 0
    for d in defenders:
        for dimension, setting_label in OFF_NOMINAL_CELLS:
            for level in CLUTTER_LEVELS:
                base_config = _env_config_for(level, d.estimator, dimension, setting_label)
                assert_robustness_config(base_config, level, dimension, setting_label)
                defender_policy = d.build(base_config)
                lead_policy_for_residual = None
                if d.is_lr_ppo:
                    cache_key = f"{level}:{d.estimator}:{dimension}:{setting_label}"
                    if cache_key not in lead_policy_cache:
                        lead_policy_cache[cache_key] = get_policy("lead", config=base_config)
                    lead_policy_for_residual = lead_policy_cache[cache_key]
                env = SoldierEnv(base_config)
                t0 = time.time()
                n_new = 0
                for eval_seed in EVAL_SEEDS:
                    key = "|".join(str(x) for x in (
                        d.architecture, d.defender_seed, d.generation, dimension, setting_label, level, eval_seed,
                    ))
                    if key in completed:
                        n_skipped += 1
                        continue
                    metrics = run_robustness_episode(env, d, defender_policy, lead_policy_for_residual, attacker.model, eval_seed)
                    metrics.update(
                        architecture=d.architecture, defender_seed=d.defender_seed, generation=d.generation,
                        dimension=dimension, setting_label=setting_label, level=level, eval_seed=eval_seed,
                        attacker_type=PRIMARY_ATTACKER, attacker_seed=attacker.seed,
                        config_hash=_config_hash(base_config), eval_code_commit=eval_code_commit,
                    )
                    append_episode_result(jsonl_path, metrics, completed)
                    rows.append(metrics)
                    n_executed += 1
                    n_new += 1
                if n_new:
                    print(f"{d.architecture:>22s} seed={d.defender_seed} {d.generation:<8s} {dimension:<22s} "
                          f"{setting_label:<22s} level={level} (+{n_new} new, {time.time()-t0:.1f}s)", flush=True)
    print(f"=== robustness evaluation loop done: {n_executed} executed, {n_skipped} skipped (already complete) ===", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    registry = load_model_registry()
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity check FAILED before any episode runs:\n" + "\n".join(problems))
    print("Checkpoint integrity check (before): OK", flush=True)

    manifest = build_evaluation_manifest(registry)
    (OUT_DIR / "evaluation_manifest.json").write_text(json.dumps(manifest, default=str))
    print(f"=== manifest: {manifest['total_expected_keys']} expected episodes "
          f"({manifest['n_defenders']} defenders x {manifest['n_off_nominal_cells']} off-nominal cells x "
          f"{manifest['n_levels']} levels x {manifest['n_eval_seeds']} seeds) ===", flush=True)
    print("=== recovery manifest (before) ===", json.dumps(build_recovery_manifest(evaluation_manifest=manifest)), flush=True)

    evaluate_all(registry=registry)

    print("=== recovery manifest (after) ===", json.dumps(build_recovery_manifest(evaluation_manifest=manifest)), flush=True)
    problems_after = verify_checkpoint_integrity(registry)
    if problems_after:
        raise RuntimeError("Checkpoint integrity check FAILED after episode runs:\n" + "\n".join(problems_after))
    print("Checkpoint integrity check (after): OK", flush=True)

    df = pd.read_json(OUT_DIR / "episodes.jsonl", lines=True)
    df.to_csv(OUT_DIR / "episodes.csv", index=False)
    print(f"wrote {len(df)} episode rows to {OUT_DIR / 'episodes.csv'}", flush=True)
