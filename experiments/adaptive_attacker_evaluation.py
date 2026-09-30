"""
Phase 19-25 evaluation for Adaptive Attacker v1 (journal extension).
Compares, under IDENTICAL matched scenarios (Phase 21: same clutter level,
initial states, obstacle layouts, dynamic-obstacle trajectories, defender
checkpoint, and seed):

    - the EXISTING hand-designed ("scripted") hostile (enemy_tactical_override
      left unset) -- baseline;
    - EACH of the 3 learned Adaptive Attacker v1 seeds (201/202/203)
      individually (Phase 25: never report only the best seed);

against:

    - SEEN defenders (Phase 19): the exact frozen defender-pool families
      used during attacker training -- Lead + clutter-trained
      PPO-C/PPO-Kalman-C/LR-PPO-C (all 3 seeds each);
    - UNSEEN defenders (Phase 20): families NEVER used during attacker
      training -- Greedy, PN, RMPC (scripted/optimization-based) + the
      FROZEN CONFERENCE PPO/PPO-Kalman/LR-PPO checkpoints (16-D/19-D
      observation, structurally different from the 36-D/39-D
      clutter-trained variants used in training).

SAMPLE SIZE NOTE (honest, reduced-from-an-ideal-N, consistent with the
prior clutter-training-scaleup sub-task's precedent): a full high-N sweep
across 22 defender instances x 4 attacker conditions x 3 clutter levels
would take many additional hours (RMPC alone is ~150-400x slower per
episode than every other controller). This module defaults to
N_EPISODES_DEFAULT=15 (non-RMPC) and RMPC_N_EPISODES=5 with RMPC
restricted to 2 attacker conditions (scripted + the single best-performing
learned seed) -- smaller than the N=100 used for the defender-side
held-out evaluation, but still directly informative for the primary
questions this phase asks (does the learned attacker degrade frozen
defenders, and is that degradation general across seen/unseen
defenders/clutter levels). Flagged explicitly in every output artifact;
the harness scales to any N via function arguments with no code changes.

Usage:
    python -m experiments.adaptive_attacker_evaluation
"""

from __future__ import annotations

import argparse
import dataclasses
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
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.lead_residual_diagnostic_protocol import RESIDUAL_MAX_ANGLE_DEG
from experiments.clutter_benchmark_protocol import (
    FROZEN_MODELS, LEARNED_ROOTS, RMPCPolicy, RMPCConfig, _frozen_rmpc_config,
)
from experiments.adaptive_attacker_env import (
    ATTACKER_HELDOUT_EVAL_SEED_START,
    ATTACKER_TRAINING_SEEDS,
    build_attacker_observation,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "adaptive_attacker_evaluation"
ATTACKER_CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "adaptive_attacker_400k"

EVAL_LEVELS = ("C1", "C2", "C3")
N_EPISODES_DEFAULT = 15
RMPC_N_EPISODES_DEFAULT = 5
RMPC_BEST_ATTACKER_SEED = 202  # highest diagnostic-scale validation success_rate (0.55) -- see session notes

# Recovery infrastructure (added after an unexpected machine restart mid-run
# -- infrastructure only, does NOT change N/seeds/protocol/models). Each
# completed episode is appended to this JSONL file IMMEDIATELY, so a future
# interruption loses at most one in-flight episode instead of the whole run.
EPISODES_JSONL_NAME = "episodes.jsonl"
EPISODE_KEY_FIELDS = ("level", "defender_instance", "attacker_condition", "seed")
VALID_OUTCOMES = (
    "intercepted", "soldier_caught", "unsafe_intercept", "timeout",
    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
)
MANDATORY_NUMERIC_FIELDS = (
    "episode_length", "time_to_outcome_s", "asset_progress_total", "min_defender_separation",
    "defender_avoidance_fraction", "attacker_avoidance_fraction",
)


def held_out_seeds(n_episodes: int) -> tuple[int, ...]:
    return tuple(range(ATTACKER_HELDOUT_EVAL_SEED_START, ATTACKER_HELDOUT_EVAL_SEED_START + n_episodes))


@dataclasses.dataclass(frozen=True)
class EvalDefender:
    family: str
    instance_id: str
    estimator: str
    needs_sector_features: bool
    is_lr_ppo: bool
    seen_in_training: bool
    build: Callable[[EnvConfig], object]


@dataclasses.dataclass(frozen=True)
class AttackerCondition:
    name: str          # "scripted" | f"learned_seed_{seed}"
    seed: int | None    # None for scripted
    model: object | None


def _attacker_checkpoint_path(seed: int) -> Path:
    run_dir = ATTACKER_CAMPAIGN_ROOT / f"seed_{seed}"
    path = run_dir / "best_model.zip"
    if not path.exists():
        path = run_dir / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"no adaptive-attacker checkpoint found for seed_{seed} under {run_dir}")
    return path


def build_attacker_conditions() -> list[AttackerCondition]:
    conditions = [AttackerCondition("scripted", None, None)]
    for seed in ATTACKER_TRAINING_SEEDS:
        model = PPO.load(str(_attacker_checkpoint_path(seed)))
        conditions.append(AttackerCondition(f"learned_seed_{seed}", seed, model))
    return conditions


def _clutter_checkpoint_path(arch: str, seed: int) -> Path:
    run_dir = PROJECT_ROOT / "results" / "clutter_training_400k" / arch / f"seed_{seed}"
    path = run_dir / "best_model.zip"
    if not path.exists():
        path = run_dir / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint found for {arch} seed_{seed} under {run_dir}")
    return path


def build_seen_defender_instances() -> list[EvalDefender]:
    """Phase 19: EXACT training-pool families (Lead + all 9 clutter-trained
    checkpoints -- 3 seeds each of PPO-C/PPO-Kalman-C/LR-PPO-C)."""
    instances = [
        EvalDefender("Lead", "Lead", "measurement", False, False, True,
                     build=lambda cfg: get_policy("lead", config=cfg)),
    ]
    for seed in (71, 72, 73):
        path = _clutter_checkpoint_path("ppo_clutter_v1", seed)
        instances.append(EvalDefender(
            "PPO-C", f"PPO-C-{seed}", "measurement", True, False, True,
            build=lambda cfg, p=path: get_policy("ppo", model_path=str(p), deterministic=True),
        ))
    for seed in (74, 75, 76):
        path = _clutter_checkpoint_path("ppo_kalman_clutter_v1", seed)
        instances.append(EvalDefender(
            "PPO-Kalman-C", f"PPOK-C-{seed}", "kalman", True, False, True,
            build=lambda cfg, p=path: get_policy("ppo_kalman", model_path=str(p), deterministic=True),
        ))
    for seed in (77, 78, 79):
        path = _clutter_checkpoint_path("lrppo_clutter_v1", seed)
        instances.append(EvalDefender(
            "LR-PPO-C", f"LRPPO-C-{seed}", "measurement", True, True, True,
            build=lambda cfg, p=path: LeadResidualPPOPolicyWrapper.load(
                p, residual_max_angle_deg=30.0, config=cfg, deterministic=True,
            ),
        ))
    return instances


def build_unseen_defender_instances() -> list[EvalDefender]:
    """Phase 20: families NEVER present in the attacker's training pool --
    Greedy, PN, RMPC (scripted/optimization-based), and the FROZEN
    CONFERENCE PPO/PPO-Kalman/LR-PPO checkpoints (16-D/19-D observation,
    structurally different from the 36-D/39-D clutter-trained variants
    the attacker actually trained against)."""
    instances = [
        EvalDefender("Greedy", "Greedy", "measurement", False, False, False,
                     build=lambda cfg: get_policy("greedy")),
        EvalDefender("PN", "PN", "measurement", False, False, False,
                     build=lambda cfg: get_policy("pn", config=cfg)),
        EvalDefender("RMPC", "RMPC", "measurement", False, False, False,
                     build=lambda cfg: RMPCPolicy(config=cfg, rmpc_config=_frozen_rmpc_config())),
    ]
    for seed in LEARNED_ROOTS["PPO"]:
        spec = FROZEN_MODELS[("direct", seed)]
        instances.append(EvalDefender(
            "PPO (frozen conference)", f"PPO-{seed}", "measurement", False, False, False,
            build=lambda cfg, p=spec["path"]: get_policy("ppo", model_path=str(p), deterministic=True),
        ))
    for seed in LEARNED_ROOTS["PPO-Kalman"]:
        spec = FROZEN_MODELS[("kalman", seed)]
        instances.append(EvalDefender(
            "PPO-Kalman (frozen conference)", f"PPOK-{seed}", "kalman", False, False, False,
            build=lambda cfg, p=spec["path"]: get_policy("ppo_kalman", model_path=str(p), deterministic=True),
        ))
    for seed in LEARNED_ROOTS["LR-PPO"]:
        spec = FROZEN_MODELS[("lr_ppo", seed)]
        instances.append(EvalDefender(
            "LR-PPO (frozen conference)", f"LRPPO-{seed}", "measurement", False, False, False,
            build=lambda cfg, p=spec["path"]: LeadResidualPPOPolicyWrapper.load(
                p, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=cfg, deterministic=True,
            ),
        ))
    return instances


def _run_matched_episode(
    env: SoldierEnv,
    defender_policy,
    defender: EvalDefender,
    attacker_model,
    seed: int,
) -> dict:
    """Runs ONE episode. `attacker_model=None` => scripted hostile
    (enemy_tactical_override left unset); otherwise the given learned
    attacker model drives the hostile via the override hook. Both
    conditions, for the SAME seed, get IDENTICAL initial state/obstacle
    layout/dynamic-obstacle trajectories -- only the enemy's own tactical
    decisions differ (Phase 21 matched-scenario requirement)."""
    pending = {"action": np.zeros(3, dtype=np.float32)}
    if attacker_model is not None:
        env.enemy_tactical_override = lambda: pending["action"]
    else:
        env.enemy_tactical_override = None

    obs, info = env.reset(seed=seed)
    defender_policy.reset()
    start_dist_asset = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
    min_defender_separation = float(np.linalg.norm(env._defender_pos - env._enemy_pos))
    defender_avoidance_steps = 0
    attacker_avoidance_steps = 0
    steps = 0
    done = False
    while not done:
        if attacker_model is not None:
            attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
            action, _ = attacker_model.predict(attacker_obs, deterministic=True)
            pending["action"] = np.asarray(action, dtype=np.float32)

        defender_obs = env._get_obs()
        defender_info = build_policy_info(env._get_info(), defender.estimator)
        if defender.needs_sector_features:
            sector = compute_sector_features(
                defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
            )
            if defender.is_lr_ppo:
                defender_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
            else:
                defender_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)
        else:
            defender_action = defender_policy.act(defender_obs, defender_info)

        obs, reward, terminated, truncated, info = env.step(np.asarray(defender_action, dtype=np.float32))
        steps += 1
        done = terminated or truncated
        min_defender_separation = min(min_defender_separation, float(np.linalg.norm(env._defender_pos - env._enemy_pos)))
        if info.get("defender_avoidance_active"):
            defender_avoidance_steps += 1
        if info.get("enemy_obstacle_avoidance_active"):
            attacker_avoidance_steps += 1

    end_dist_asset = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
    outcome = info["outcome"]
    return {
        "seed": seed, "outcome": outcome,
        "defender_safe_intercept": 1 if outcome == "intercepted" else 0,
        "attacker_soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "attacker_environmental_failure": 1 if outcome in (
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision") else 0,
        "defender_environmental_failure": 1 if outcome in (
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision") else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        "episode_length": steps,
        "time_to_outcome_s": steps * env.config.dt,
        "asset_progress_total": start_dist_asset - end_dist_asset,
        "min_defender_separation": min_defender_separation,
        "defender_avoidance_fraction": defender_avoidance_steps / steps if steps else 0.0,
        "attacker_avoidance_fraction": attacker_avoidance_steps / steps if steps else 0.0,
    }


def _episode_key(level: str, defender_instance: str, attacker_condition: str, seed: int) -> str:
    return f"{level}|{defender_instance}|{attacker_condition}|{seed}"


def _row_key(row: dict) -> str:
    return _episode_key(row["level"], row["defender_instance"], row["attacker_condition"], row["seed"])


def is_valid_episode_row(row: dict) -> bool:
    """Phase R5 integrity check: correct identity fields present, outcome
    is one of the environment's known mutually-exclusive terminal
    outcomes, and no NaN/inf in any mandatory numeric field."""
    for field in EPISODE_KEY_FIELDS + ("outcome",):
        if field not in row or row[field] is None:
            return False
    if row["outcome"] not in VALID_OUTCOMES:
        return False
    for field in MANDATORY_NUMERIC_FIELDS:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            return False
    return True


def load_completed_episodes(jsonl_path: Path) -> dict[str, dict]:
    """Phase R3/R5/R6: reads the episode-level JSONL (if present),
    returning ONLY valid, parseable rows keyed by their unique
    (level, defender_instance, attacker_condition, seed) key. A
    malformed/truncated FINAL line (machine restart mid-write) is
    silently dropped; all preceding valid lines are kept -- never an
    all-or-nothing discard of the whole file."""
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
                    f"this indicates corruption beyond a simple truncated-final-write and must be "
                    f"investigated manually, not silently discarded."
                )
            continue  # malformed FINAL line: mid-write truncation, safe to drop
        if not is_valid_episode_row(row):
            continue
        key = _row_key(row)
        if key in completed and completed[key] != row:
            raise ValueError(
                f"duplicate episode key {key!r} in {jsonl_path} with DIFFERING content -- "
                f"stopping rather than silently choosing one (Phase R9)."
            )
        completed[key] = row
    return completed


def append_episode_result(jsonl_path: Path, row: dict, completed: dict[str, dict]) -> None:
    """Phase R8/R9: atomic-ish append (write + flush + fsync) of ONE
    completed episode. Refuses to write a duplicate key whose content
    differs from what is already recorded; a duplicate with IDENTICAL
    content is a harmless no-op (not re-appended)."""
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


def build_recovery_manifest(n_episodes: int = N_EPISODES_DEFAULT, rmpc_n_episodes: int = RMPC_N_EPISODES_DEFAULT,
                             jsonl_path: Path | None = None) -> dict:
    """Phase R4: every EXPECTED evaluation unit, derived from the actual
    evaluation configuration (not guessed), cross-referenced against what
    is already present/valid in the episode JSONL."""
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / EPISODES_JSONL_NAME)
    completed = load_completed_episodes(jsonl_path)

    attacker_conditions = build_attacker_conditions()
    rmpc_conditions = [c for c in attacker_conditions if c.seed in (None, RMPC_BEST_ATTACKER_SEED)]
    defenders = build_seen_defender_instances() + build_unseen_defender_instances()

    expected_keys = []
    by_family: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    by_level: dict[str, int] = {}
    rmpc_remaining = 0
    for level in EVAL_LEVELS:
        for defender in defenders:
            is_rmpc = defender.family == "RMPC"
            conditions = rmpc_conditions if is_rmpc else attacker_conditions
            n = rmpc_n_episodes if is_rmpc else n_episodes
            for condition in conditions:
                for seed in held_out_seeds(n):
                    key = _episode_key(level, defender.instance_id, condition.name, seed)
                    expected_keys.append(key)
                    if key not in completed:
                        by_family[defender.family] = by_family.get(defender.family, 0) + 1
                        by_condition[condition.name] = by_condition.get(condition.name, 0) + 1
                        by_level[level] = by_level.get(level, 0) + 1
                        if is_rmpc:
                            rmpc_remaining += 1

    total_expected = len(expected_keys)
    n_valid = sum(1 for k in expected_keys if k in completed)
    n_missing = total_expected - n_valid
    return {
        "total_expected": total_expected,
        "complete_valid": n_valid,
        "missing": n_missing,
        "percent_complete": (100.0 * n_valid / total_expected) if total_expected else 100.0,
        "remaining_by_defender_family": by_family,
        "remaining_by_attacker_condition": by_condition,
        "remaining_by_level": by_level,
        "rmpc_remaining": rmpc_remaining,
    }


def evaluate_all(n_episodes: int = N_EPISODES_DEFAULT, rmpc_n_episodes: int = RMPC_N_EPISODES_DEFAULT,
                  jsonl_path: Path | None = None) -> pd.DataFrame:
    """Resumable evaluation loop (recovery infrastructure). Persists every
    completed episode to `jsonl_path` (append-only, flushed+fsync'd)
    IMMEDIATELY -- an interruption loses at most one in-flight episode,
    not the whole run. On (re)start, any (level, defender_instance,
    attacker_condition, seed) key already present and VALID in the JSONL
    is skipped (never re-executed), so this function is naturally
    idempotent/resumable across restarts. Scientific protocol (N, seeds,
    models, controllers, clutter config) is UNCHANGED by this --
    persistence/skip logic only."""
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / EPISODES_JSONL_NAME)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_episodes(jsonl_path)

    attacker_conditions = build_attacker_conditions()
    rmpc_conditions = [c for c in attacker_conditions if c.seed in (None, RMPC_BEST_ATTACKER_SEED)]
    defenders = build_seen_defender_instances() + build_unseen_defender_instances()

    rows = list(completed.values())
    n_skipped = 0
    n_executed = 0
    for level in EVAL_LEVELS:
        for defender in defenders:
            is_rmpc = defender.family == "RMPC"
            conditions = rmpc_conditions if is_rmpc else attacker_conditions
            n = rmpc_n_episodes if is_rmpc else n_episodes
            base_config = None
            defender_policy = None
            env = None
            for condition in conditions:
                seeds = held_out_seeds(n)
                t0 = time.time()
                n_new_this_group = 0
                for seed in seeds:
                    key = _episode_key(level, defender.instance_id, condition.name, seed)
                    if key in completed:
                        n_skipped += 1
                        continue
                    if base_config is None:
                        base_config = build_canonical_env_config(level, defender.estimator)
                        defender_policy = defender.build(base_config)
                        env = SoldierEnv(base_config)
                    metrics = _run_matched_episode(env, defender_policy, defender, condition.model, seed)
                    metrics.update(
                        level=level, defender_family=defender.family, defender_instance=defender.instance_id,
                        seen_in_training=defender.seen_in_training, attacker_condition=condition.name,
                    )
                    append_episode_result(jsonl_path, metrics, completed)
                    rows.append(metrics)
                    n_executed += 1
                    n_new_this_group += 1
                if n_new_this_group:
                    print(f"level={level} defender={defender.instance_id:>22s} attacker={condition.name:>16s} "
                          f"n={n} (+{n_new_this_group} new, {time.time()-t0:.1f}s)", flush=True)
                else:
                    print(f"level={level} defender={defender.instance_id:>22s} attacker={condition.name:>16s} "
                          f"n={n} (all {n} already complete, skipped)", flush=True)
    print(f"=== evaluation loop done: {n_executed} episodes executed, {n_skipped} already-complete "
          f"episodes skipped ===", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["level", "defender_family", "seen_in_training", "attacker_condition"]).agg(
        n=("seed", "count"),
        defender_safe_intercept_rate=("defender_safe_intercept", "mean"),
        attacker_soldier_caught_rate=("attacker_soldier_caught", "mean"),
        unsafe_intercept_rate=("unsafe_intercept", "mean"),
        attacker_environmental_failure_rate=("attacker_environmental_failure", "mean"),
        defender_environmental_failure_rate=("defender_environmental_failure", "mean"),
        timeout_rate=("timeout", "mean"),
        mean_episode_length=("episode_length", "mean"),
        mean_time_to_outcome_s=("time_to_outcome_s", "mean"),
        mean_asset_progress=("asset_progress_total", "mean"),
        mean_min_defender_separation=("min_defender_separation", "mean"),
        mean_defender_avoidance_fraction=("defender_avoidance_fraction", "mean"),
        mean_attacker_avoidance_fraction=("attacker_avoidance_fraction", "mean"),
    ).reset_index()


def compute_degradation_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Phase 21/24: Delta defender_success = success_vs_learned -
    success_vs_scripted, aggregated ACROSS clutter levels AND (for
    multi-seed families) across defender seeds, per defender FAMILY, for
    EACH learned-attacker seed AND pooled across all 3."""
    rows = []
    for family in sorted(summary["defender_family"].unique()):
        fam_df = summary[summary["defender_family"] == family]
        scripted = fam_df[fam_df["attacker_condition"] == "scripted"]
        scripted_rate = float((scripted["defender_safe_intercept_rate"] * scripted["n"]).sum() / scripted["n"].sum())
        scripted_env_fail = float((scripted["attacker_environmental_failure_rate"] * scripted["n"]).sum() / scripted["n"].sum())
        for condition in sorted(c for c in fam_df["attacker_condition"].unique() if c != "scripted"):
            cond_df = fam_df[fam_df["attacker_condition"] == condition]
            if cond_df["n"].sum() == 0:
                continue
            learned_rate = float((cond_df["defender_safe_intercept_rate"] * cond_df["n"]).sum() / cond_df["n"].sum())
            learned_env_fail = float((cond_df["attacker_environmental_failure_rate"] * cond_df["n"]).sum() / cond_df["n"].sum())
            rows.append({
                "defender_family": family, "attacker_condition": condition,
                "scripted_safe_intercept_rate": scripted_rate,
                "learned_safe_intercept_rate": learned_rate,
                "delta_defender_success": learned_rate - scripted_rate,
                "scripted_attacker_environmental_failure_rate": scripted_env_fail,
                "learned_attacker_environmental_failure_rate": learned_env_fail,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES_DEFAULT)
    parser.add_argument("--rmpc-n-episodes", type=int, default=RMPC_N_EPISODES_DEFAULT)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / EPISODES_JSONL_NAME

    # Phase R14: report the recovery manifest BEFORE resuming.
    manifest = build_recovery_manifest(args.n_episodes, args.rmpc_n_episodes, jsonl_path)
    print("=== recovery manifest (before resuming) ===", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)

    df = evaluate_all(n_episodes=args.n_episodes, rmpc_n_episodes=args.rmpc_n_episodes, jsonl_path=jsonl_path)

    # Phase R15: rebuild the manifest from scratch and verify completeness.
    final_manifest = build_recovery_manifest(args.n_episodes, args.rmpc_n_episodes, jsonl_path)
    print("=== recovery manifest (after run) ===", flush=True)
    print(json.dumps(final_manifest, indent=2), flush=True)
    if final_manifest["missing"] != 0:
        raise RuntimeError(
            f"evaluation incomplete after run: {final_manifest['missing']} units still missing "
            f"(expected 0) -- re-run this script to resume."
        )

    # Phase R16: regenerate ALL aggregates from scratch from the canonical
    # (now-complete) episode dataset -- never trust/append to pre-restart
    # aggregate files.
    df.to_csv(OUT_DIR / "episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 240)
    print(summary.to_string(index=False))

    degradation = compute_degradation_matrix(summary)
    degradation.to_csv(OUT_DIR / "degradation_matrix.csv", index=False)
    print(degradation.to_string(index=False))

    print(f"\nNOTE: N_EPISODES={args.n_episodes} (RMPC={args.rmpc_n_episodes}); "
          f"see module docstring for the documented reduced-sample decision.", flush=True)
