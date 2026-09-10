"""Phase 7: per-step diagnostic logging.

`run_episode_with_diagnostics()` steps a policy through one episode and
returns a list of per-step dict rows (one per environment step) containing
every diagnostic field listed in the Phase-7 specification that is
actually available for the given controller/environment configuration
(unavailable fields are `None`, never fabricated).

CRITICAL: this module never mutates `obs`/`info`, never changes the
action returned by `policy.act(...)`, and never changes reward/termination
logic -- it only reads already-computed values (either from `info` or from
a policy's own public diagnostic attributes) after the real step has
already happened. Logging is a pure side observer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.constant_acceleration_lead_policy import ConstantAccelerationLeadPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info

# Canonical column order (superset across all controllers/configs; unknown
# fields are written as empty/None). Grouped per the Phase-7 specification.
STEP_LOG_COLUMNS = (
    # Episode identifiers
    "seed", "controller", "scenario", "step_index", "sim_time",
    # Lead diagnostics
    "lead_command_x", "lead_command_y", "lead_command_z",
    "lead_intercept_time", "lead_intercept_point_x", "lead_intercept_point_y", "lead_intercept_point_z",
    "lead_fallback_used", "lead_solution_valid", "lead_guidance_mode",
    # CA-Lead diagnostics
    "ca_used_acceleration", "ca_intercept_time", "ca_guidance_mode",
    # LR-PPO diagnostics
    "final_command_x", "final_command_y", "final_command_z",
    "residual_angle_deg",
    # Hostile state/behavior
    "e_hat_x", "e_hat_y", "e_hat_z", "v_hat_x", "v_hat_y", "v_hat_z",
    "true_enemy_pos_x", "true_enemy_pos_y", "true_enemy_pos_z",
    "true_enemy_vel_x", "true_enemy_vel_y", "true_enemy_vel_z",
    "estimated_acceleration_x", "estimated_acceleration_y", "estimated_acceleration_z",
    "enemy_turn_rate", "enemy_evasion_active", "enemy_obstacle_avoidance_active",
    # Defender
    "defender_pos_x", "defender_pos_y", "defender_pos_z",
    "defender_vel_x", "defender_vel_y", "defender_vel_z",
    "defender_accel", "defender_accel_saturated", "defender_turn_rate", "defender_turn_saturated",
    "defender_climb_saturated",
    # Sensing
    "enemy_measurement_x", "enemy_measurement_y", "enemy_measurement_z",
    "tracking_error",
    # Geometry
    "defender_enemy_dist", "enemy_soldier_dist", "obstacle_min_clearance",
    # Outcome (final step only; blank otherwise)
    "outcome", "success", "episode_length",
)


def _xyz(prefix: str, vec) -> dict:
    if vec is None:
        return {f"{prefix}_x": None, f"{prefix}_y": None, f"{prefix}_z": None}
    vec = np.asarray(vec, dtype=np.float64)
    return {f"{prefix}_x": float(vec[0]), f"{prefix}_y": float(vec[1]), f"{prefix}_z": float(vec[2])}


def angle_between_deg(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float | None:
    """Angle in degrees between two vectors, using the SAME clip-then-arccos
    pattern used throughout this codebase. Returns None if either vector is
    (near-)zero (angle undefined)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    un = float(np.linalg.norm(u))
    vn = float(np.linalg.norm(v))
    if un <= eps or vn <= eps:
        return None
    cos_angle = float(np.clip(np.dot(u, v) / (un * vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _lead_diagnostics(lead_policy) -> dict:
    if lead_policy is None:
        return {
            "lead_intercept_time": None, **_xyz("lead_intercept_point", None),
            "lead_fallback_used": None, "lead_solution_valid": None, "lead_guidance_mode": None,
        }
    mode = getattr(lead_policy, "last_guidance_mode", None)
    return {
        "lead_intercept_time": getattr(lead_policy, "last_intercept_time", None),
        **_xyz("lead_intercept_point", getattr(lead_policy, "last_intercept_point", None)),
        "lead_fallback_used": None if mode is None else (mode != "lead"),
        "lead_solution_valid": None if mode is None else (mode == "lead"),
        "lead_guidance_mode": mode,
    }


def run_episode_with_diagnostics(
    env: SoldierEnv,
    policy: Any,
    seed: int,
    controller: str,
    scenario: str,
    estimator_mode: str = "measurement",
    lead_reference_policy=None,
    ca_reference_policy=None,
    max_steps: int | None = None,
) -> list[dict]:
    """
    Run ONE episode with `policy`, logging per-step diagnostics.

    `lead_reference_policy`/`ca_reference_policy` are OPTIONAL companion
    diagnostic controllers (e.g. a `LeadInterceptPolicy`/
    `ConstantAccelerationLeadPolicy`) stepped IN PARALLEL purely to record
    what Lead/CA would have commanded at each step -- their outputs are
    NEVER fed back into the environment or the evaluated `policy`.

    If `policy` itself exposes `_lead_policy` (e.g. `LeadResidualPPOPolicyWrapper`),
    its Lead diagnostics are used automatically when `lead_reference_policy`
    is not separately supplied, and `residual_angle_deg` is computed
    between the final action and that internal Lead direction.

    Returns a list of per-step dict rows (see `STEP_LOG_COLUMNS`).
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    if lead_reference_policy is not None:
        lead_reference_policy.reset()
    if ca_reference_policy is not None:
        ca_reference_policy.reset()

    internal_lead_policy = getattr(policy, "_lead_policy", None)
    # If `policy` IS ITSELF a LeadInterceptPolicy, use it directly as the
    # Lead-diagnostic source when no separate reference policy was
    # supplied (e.g. logging the Standard Lead controller's own episode).
    if internal_lead_policy is None and lead_reference_policy is None and isinstance(policy, LeadInterceptPolicy):
        internal_lead_policy = policy
    # Same self-detection for a CA-Lead controller logging its own episode
    # (avoids calling .act() a second time, which would corrupt its
    # internal velocity-estimate history).
    internal_ca_policy = None
    if ca_reference_policy is None and isinstance(policy, ConstantAccelerationLeadPolicy):
        internal_ca_policy = policy
    rows: list[dict] = []
    step_index = 0
    outcome = "ongoing"
    terminated = False
    truncated = False

    while not (terminated or truncated):
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)

        if lead_reference_policy is not None:
            lead_action = lead_reference_policy.act(obs, policy_info)
        else:
            lead_action = None
        if ca_reference_policy is not None:
            ca_reference_policy.act(obs, policy_info)

        row: dict[str, Any] = {col: None for col in STEP_LOG_COLUMNS}
        row.update({
            "seed": seed, "controller": controller, "scenario": scenario,
            "step_index": step_index, "sim_time": step_index * float(env.config.dt),
        })
        row.update(_xyz("lead_command", lead_action if lead_action is not None else action))

        effective_lead_policy = lead_reference_policy if lead_reference_policy is not None else internal_lead_policy
        row.update(_lead_diagnostics(effective_lead_policy))

        effective_ca_policy = ca_reference_policy if ca_reference_policy is not None else internal_ca_policy
        if effective_ca_policy is not None:
            ca_solution = getattr(effective_ca_policy, "last_solution", None)
            row["ca_used_acceleration"] = None if ca_solution is None else ca_solution.used_acceleration
            row["ca_intercept_time"] = None if ca_solution is None else ca_solution.intercept_time
            row["ca_guidance_mode"] = getattr(effective_ca_policy, "last_guidance_mode", None)

        row.update(_xyz("final_command", action))
        if effective_lead_policy is not None and getattr(effective_lead_policy, "last_action", None) is not None:
            row["residual_angle_deg"] = angle_between_deg(action, effective_lead_policy.last_action)

        row.update(_xyz("e_hat", info.get("e_hat")))
        row.update(_xyz("v_hat", info.get("v_hat")))
        row.update(_xyz("true_enemy_pos", info.get("enemy_pos")))
        row.update(_xyz("true_enemy_vel", info.get("enemy_vel")))
        row["enemy_turn_rate"] = info.get("enemy_turn_rate")
        row["enemy_evasion_active"] = info.get("enemy_evasion_active")
        row["enemy_obstacle_avoidance_active"] = info.get("enemy_obstacle_avoidance_active")

        row.update(_xyz("defender_pos", info.get("defender_pos")))
        row.update(_xyz("defender_vel", info.get("defender_vel")))
        row["defender_accel"] = info.get("defender_accel")
        row["defender_accel_saturated"] = info.get("defender_accel_saturated")
        row["defender_turn_rate"] = info.get("defender_turn_rate")
        row["defender_turn_saturated"] = info.get("defender_turn_saturated")
        row["defender_climb_saturated"] = info.get("defender_climb_saturated")

        row.update(_xyz("enemy_measurement", info.get("enemy_measurement")))
        row["tracking_error"] = info.get("tracking_error")

        row["defender_enemy_dist"] = info.get("defender_enemy_dist")
        row["enemy_soldier_dist"] = info.get("enemy_soldier_dist")
        row["obstacle_min_clearance"] = info.get("enemy_min_obstacle_clearance")

        rows.append(row)

        obs, reward, terminated, truncated, info = env.step(action)
        step_index += 1
        if max_steps is not None and step_index >= max_steps:
            truncated = True

    outcome = info.get("outcome", outcome)
    if rows:
        rows[-1]["outcome"] = outcome
        rows[-1]["success"] = 1 if outcome == "intercepted" else 0
        rows[-1]["episode_length"] = step_index
    return rows


def write_step_logs_csv(rows: list[dict], path: str | Path) -> None:
    """Deterministic CSV writer: fixed column order (STEP_LOG_COLUMNS),
    one row per logged step, across possibly many episodes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(STEP_LOG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
