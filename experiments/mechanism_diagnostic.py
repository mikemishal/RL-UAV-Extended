"""Phase 1-3/6/7 mechanism diagnostic: instrumented (step-level) rollouts for
the architecture-mechanism diagnostic task. Pure additive diagnostic code --
NO training, NO architecture change, NO checkpoint mutation. Reuses the
FROZEN model registry / defender instances / attacker conditions / canonical
env-config builders from experiments/defender_adaptation_evaluation.py and
experiments/defender_adaptation_canonical_c0_evaluation.py UNCHANGED.

This module adds step-level instrumentation (defender/attacker kinematics,
Lead-guidance internals for LR-PPO, residual angle, outcome) that was never
persisted by the original (episode-level-only) evaluation harnesses. Running
this module executes NEW, SMALL, targeted diagnostic rollouts -- reusing the
SAME matched eval-seed values already used by the frozen canonical-C0 (C0)
and clutter (C1-C3) evaluations (a disjoint-seed-namespace SUBSET, not a new
namespace), per the task's "use matched seeds wherever possible" and "small
targeted diagnostic rollouts are allowed" instructions.

Scope (modest, NOT a full campaign):
    Phase 3 (LR-PPO Lead diagnostic):   lrppo_clutter_v1, d0/d0_plus/d1,
        A0 + A1-unseen-203, C0 + C3, N=30 matched seeds/cell.
    Phase 6 (PPO-Kalman mechanism):     ppo_kalman_clutter_v1, d0_plus/d1,
        A1-unseen-203, C0 + C3, N=30 matched seeds/cell.
    Phase 7 (PPO-C C0 regression):      ppo_clutter_v1, d0_plus/d1,
        A1-unseen-203, C0 + C3, N=30 matched seeds/cell.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_composition import angle_between_deg

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.adaptive_attacker_env import build_attacker_observation
from experiments.defender_adaptation_evaluation import (
    PROJECT_ROOT, load_model_registry, build_defender_instances, build_attacker_conditions,
    verify_checkpoint_integrity, EVAL_SEEDS as C1C3_EVAL_SEEDS,
)
from experiments.defender_adaptation_canonical_c0_evaluation import (
    EVAL_SEEDS as CANONICAL_C0_EVAL_SEEDS, CANONICAL_EXTRA_OVERRIDES, CANONICAL_C0_LEVEL,
)

OUT_DIR = PROJECT_ROOT / "results" / "mechanism_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DIAGNOSTIC_SEEDS = 20  # modest, step-level-logging subset of the EXISTING matched eval-seed ranges
DIAG_SEEDS_C0 = tuple(CANONICAL_C0_EVAL_SEEDS[:N_DIAGNOSTIC_SEEDS])      # subset of 130000-130099
DIAG_SEEDS_C3 = tuple(C1C3_EVAL_SEEDS[:N_DIAGNOSTIC_SEEDS])              # subset of 90010000-90010099
DIAG_LEVELS = ("C0", "C3")


def _env_config_for(level: str, estimator: str):
    if level == "C0":
        return build_canonical_env_config(CANONICAL_C0_LEVEL, estimator, extra_overrides=CANONICAL_EXTRA_OVERRIDES)
    return build_canonical_env_config(level, estimator)


def _diag_seeds_for(level: str) -> tuple[int, ...]:
    return DIAG_SEEDS_C0 if level == "C0" else DIAG_SEEDS_C3


def _maneuver_intensity(vel_history: list[np.ndarray], idx: int, dt: float) -> float:
    """Finite-difference acceleration magnitude of the GROUND-TRUTH attacker
    velocity (diagnostic-only; never fed to any policy)."""
    if idx == 0 or idx >= len(vel_history):
        return 0.0
    return float(np.linalg.norm(vel_history[idx] - vel_history[idx - 1]) / dt)


@dataclasses.dataclass
class StepRecord:
    step: int
    defender_pos: tuple
    defender_vel: tuple
    attacker_pos_truth: tuple
    attacker_vel_truth: tuple
    final_action: tuple
    enemy_detected: bool
    guidance_mode: str | None
    lead_direction: tuple | None
    residual_angle_deg: float | None
    intercept_point: tuple | None
    intercept_time: float | None
    target_pos_est: tuple | None
    target_vel_est: tuple | None
    maneuver_intensity: float = 0.0
    positional_lead_error: float | None = None
    angular_lead_error_deg: float | None = None


def run_instrumented_episode(
    env: SoldierEnv, defender_instance, defender_policy, lead_diag_policy, attacker_model, eval_seed: int,
) -> tuple[dict, list[StepRecord]]:
    """Step-level-instrumented single-episode rollout. Ground-truth
    `attacker_pos_truth`/`attacker_vel_truth` are recorded for DIAGNOSTIC
    analysis only (Lead-prediction-error computation) -- never passed to
    any policy's act()/predict(), matching every other baseline's
    ground-truth-independence contract in this repository."""
    pending = {"action": np.zeros(3, dtype=np.float32)}
    env.enemy_tactical_override = (lambda: pending["action"]) if attacker_model is not None else None

    obs, info = env.reset(seed=eval_seed)
    defender_policy.reset()
    if lead_diag_policy is not None:
        lead_diag_policy.reset()

    steps_records: list[StepRecord] = []
    vel_truth_history: list[np.ndarray] = []
    defender_path_length = 0.0
    prev_defender_pos = env._defender_pos.copy()
    detection_time: int | None = None
    residual_angles: list[float] = []

    steps = 0
    done = False
    while not done:
        if attacker_model is not None:
            attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
            action, _ = attacker_model.predict(attacker_obs, deterministic=True)
            pending["action"] = np.asarray(action, dtype=np.float32)

        raw_info = env._get_info()
        defender_obs = env._get_obs()
        defender_info = build_policy_info(raw_info, defender_instance.estimator)
        sector = None
        if defender_instance.needs_sector_features:
            sector = compute_sector_features(
                defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
            )

        lead_direction = None
        residual_angle = None
        intercept_point = intercept_time = target_pos_est = target_vel_est = guidance_mode = None
        if defender_instance.is_lr_ppo:
            final_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
            lead_direction = np.asarray(lead_diag_policy.act(defender_obs, defender_info), dtype=np.float64)
            final_arr = np.asarray(final_action, dtype=np.float64)
            if np.linalg.norm(lead_direction) > 1e-8 and np.linalg.norm(final_arr) > 1e-8:
                residual_angle = angle_between_deg(lead_direction, final_arr)
                residual_angles.append(residual_angle)
            intercept_point = lead_diag_policy.last_intercept_point
            intercept_time = lead_diag_policy.last_intercept_time
            target_pos_est = lead_diag_policy.last_target_position
            target_vel_est = lead_diag_policy.last_target_velocity_estimate
            guidance_mode = lead_diag_policy.last_guidance_mode
        elif defender_instance.needs_sector_features:
            final_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)
        else:
            final_action = defender_policy.act(defender_obs, defender_info)

        vel_truth_history.append(env._enemy_vel.copy())
        steps_records.append(StepRecord(
            step=steps,
            defender_pos=tuple(env._defender_pos.tolist()),
            defender_vel=tuple(env._defender_vel.tolist()),
            attacker_pos_truth=tuple(env._enemy_pos.tolist()),
            attacker_vel_truth=tuple(env._enemy_vel.tolist()),
            final_action=tuple(np.asarray(final_action, dtype=np.float64).tolist()),
            enemy_detected=bool(raw_info.get("enemy_detected", False)),
            guidance_mode=guidance_mode,
            lead_direction=tuple(lead_direction.tolist()) if lead_direction is not None else None,
            residual_angle_deg=residual_angle,
            intercept_point=tuple(np.asarray(intercept_point).tolist()) if intercept_point is not None else None,
            intercept_time=float(intercept_time) if intercept_time is not None else None,
            target_pos_est=tuple(np.asarray(target_pos_est).tolist()) if target_pos_est is not None else None,
            target_vel_est=tuple(np.asarray(target_vel_est).tolist()) if target_vel_est is not None else None,
        ))

        obs, reward, terminated, truncated, info = env.step(np.asarray(final_action, dtype=np.float32))
        steps += 1
        done = terminated or truncated

        if detection_time is None and info.get("enemy_detected"):
            detection_time = steps
        cur_defender_pos = env._defender_pos.copy()
        defender_path_length += float(np.linalg.norm(cur_defender_pos - prev_defender_pos))
        prev_defender_pos = cur_defender_pos

    outcome = info["outcome"]
    dt = env.config.dt
    v_d = env.config.v_d

    # Post-hoc long-horizon Lead-prediction error (ground truth used ONLY
    # for this offline diagnostic metric, never for any policy decision).
    attacker_truth_traj = [r.attacker_pos_truth for r in steps_records]
    for i, rec in enumerate(steps_records):
        rec.maneuver_intensity = _maneuver_intensity(vel_truth_history, i, dt)
        rec.positional_lead_error = None
        rec.angular_lead_error_deg = None
        if rec.intercept_time is not None and rec.intercept_point is not None and rec.guidance_mode == "lead":
            future_idx = i + int(round(rec.intercept_time / dt))
            if future_idx < len(attacker_truth_traj):
                actual_future_pos = np.asarray(attacker_truth_traj[future_idx])
                predicted_point = np.asarray(rec.intercept_point)
                rec.positional_lead_error = float(np.linalg.norm(actual_future_pos - predicted_point))
                def_pos = np.asarray(rec.defender_pos)
                v_pred = predicted_point - def_pos
                v_true = actual_future_pos - def_pos
                if np.linalg.norm(v_pred) > 1e-6 and np.linalg.norm(v_true) > 1e-6:
                    rec.angular_lead_error_deg = angle_between_deg(v_pred, v_true)

    episode_summary = {
        "outcome": outcome,
        "defender_safe_intercept": 1 if outcome == "intercepted" else 0,
        "attacker_asset_reach": 1 if outcome == "soldier_caught" else 0,
        "episode_length": steps,
        "detection_time": detection_time,
        "defender_path_length": defender_path_length,
        "mean_residual_angle_deg": float(np.mean(residual_angles)) if residual_angles else 0.0,
        "n_lead_steps": sum(1 for r in steps_records if r.guidance_mode == "lead"),
    }
    pos_errs = [r.positional_lead_error for r in steps_records if r.positional_lead_error is not None]
    ang_errs = [r.angular_lead_error_deg for r in steps_records if r.angular_lead_error_deg is not None]
    episode_summary["mean_positional_lead_error"] = float(np.mean(pos_errs)) if pos_errs else None
    episode_summary["median_positional_lead_error"] = float(np.median(pos_errs)) if pos_errs else None
    episode_summary["mean_angular_lead_error_deg"] = float(np.mean(ang_errs)) if ang_errs else None
    episode_summary["median_angular_lead_error_deg"] = float(np.median(ang_errs)) if ang_errs else None
    episode_summary["p95_angular_lead_error_deg"] = float(np.percentile(ang_errs, 95)) if ang_errs else None
    return episode_summary, steps_records


# =============================================================================
# Driver: collect diagnostic episodes for one architecture across a chosen
# set of (generation, attacker_type, level) cells. Step-level records are
# persisted only for the `store_steps_generations` subset (keeps on-disk
# volume modest while still enabling Phase 5/9/10 analyses/figures).
# =============================================================================

def collect_diagnostics(
    architecture: str, generations: tuple[str, ...], attacker_names: tuple[str, ...], levels: tuple[str, ...],
    store_steps_generations: tuple[str, ...], tag: str,
) -> pd.DataFrame:
    registry = load_model_registry()
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity check failed before diagnostic rollouts:\n" + "\n".join(problems))

    all_defenders = [d for d in build_defender_instances(registry)
                     if d.architecture == architecture and d.generation in generations]
    all_attackers = [a for a in build_attacker_conditions(registry) if a.name in attacker_names]

    episode_rows = []
    step_rows = []
    lead_policy_cache: dict[str, object] = {}
    for level in levels:
        seeds = _diag_seeds_for(level)
        for d in all_defenders:
            base_config = _env_config_for(level, d.estimator)
            defender_policy = d.build(base_config)
            lead_diag_policy = None
            if d.is_lr_ppo:
                cache_key = f"{level}:{d.estimator}"
                if cache_key not in lead_policy_cache:
                    lead_policy_cache[cache_key] = get_policy("lead", config=base_config)
                lead_diag_policy = lead_policy_cache[cache_key]
            env = SoldierEnv(base_config)
            for attacker in all_attackers:
                for eval_seed in seeds:
                    summary, steps = run_instrumented_episode(
                        env, d, defender_policy, lead_diag_policy, attacker.model, eval_seed)
                    summary.update(architecture=d.architecture, defender_seed=d.defender_seed,
                                   generation=d.generation, attacker_type=attacker.name, level=level, eval_seed=eval_seed)
                    episode_rows.append(summary)
                    if d.generation in store_steps_generations:
                        for r in steps:
                            row = dataclasses.asdict(r)
                            row.update(architecture=d.architecture, defender_seed=d.defender_seed,
                                       generation=d.generation, attacker_type=attacker.name, level=level, eval_seed=eval_seed)
                            step_rows.append(row)
            print(f"{tag}: {d.architecture} seed={d.defender_seed} {d.generation} level={level} done", flush=True)

    ep_df = pd.DataFrame(episode_rows)
    ep_df.to_csv(OUT_DIR / f"{tag}_episodes.csv", index=False)
    if step_rows:
        step_df = pd.DataFrame(step_rows)
        step_df.to_csv(OUT_DIR / f"{tag}_steps.csv.gz", index=False, compression="gzip")
    return ep_df


def run_phase3_lrppo_lead() -> pd.DataFrame:
    return collect_diagnostics(
        architecture="lrppo_clutter_v1", generations=("d0", "d0_plus", "d1"),
        attacker_names=("A0", "A1-unseen-203"), levels=DIAG_LEVELS,
        store_steps_generations=("d0_plus", "d1"), tag="lrppo_lead",
    )


def run_phase6_ppo_kalman() -> pd.DataFrame:
    return collect_diagnostics(
        architecture="ppo_kalman_clutter_v1", generations=("d0_plus", "d1"),
        attacker_names=("A1-unseen-203",), levels=DIAG_LEVELS,
        store_steps_generations=("d0_plus", "d1"), tag="ppo_kalman_mechanism",
    )


def run_phase7_ppo_c() -> pd.DataFrame:
    return collect_diagnostics(
        architecture="ppo_clutter_v1", generations=("d0_plus", "d1"),
        attacker_names=("A1-unseen-203",), levels=DIAG_LEVELS,
        store_steps_generations=("d0_plus", "d1"), tag="ppo_c_regression",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["3", "6", "7", "all"], default="all")
    args = parser.parse_args()

    manifest = {"n_diagnostic_seeds": N_DIAGNOSTIC_SEEDS, "levels": DIAG_LEVELS,
                "seeds_c0": list(DIAG_SEEDS_C0), "seeds_c3": list(DIAG_SEEDS_C3)}
    (OUT_DIR / "diagnostic_seed_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.phase in ("3", "all"):
        run_phase3_lrppo_lead()
    if args.phase in ("6", "all"):
        run_phase6_ppo_kalman()
    if args.phase in ("7", "all"):
        run_phase7_ppo_c()
    print("=== mechanism diagnostic rollouts complete ===", flush=True)

