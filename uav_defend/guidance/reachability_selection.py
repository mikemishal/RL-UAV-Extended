"""
Shared reachability-selection machinery (Phase 76-78) used identically by
`RALAPGPolicy` (learned prediction) and `ReachabilityAwareLeadPolicy` (CV
prediction) -- per Phase 76, the ONLY difference between the two
controllers must be the target predictor, never the candidate-time grid,
defender dynamics model, reachability metric, intercept-radius criterion,
or selection rule.

Two reachability rollout modes (Phase 78):
  FIXED_DIRECTION -- freeze one desired direction (toward the tau-horizon
    predicted point) for the whole candidate horizon (Phase 54-58
    original, simplest/most interpretable).
  RECEDING_TARGET -- at each rollout step k, re-aim toward the predicted
    hostile position AT THAT SAME elapsed simulated time (a predicted
    "moving" aim point, i.e. simulated pure pursuit against the predicted
    trajectory), which may be less conservative in turn-limited regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.policies.baseline.lead_math import pure_pursuit_direction

PredictFn = Callable[[float], np.ndarray]  # h (seconds) -> predicted absolute target position


class ReachabilityMode(Enum):
    FIXED_DIRECTION = "fixed_direction"
    RECEDING_TARGET = "receding_target"


@dataclass(frozen=True)
class CandidateEvaluation:
    tau: float
    predicted_target_point: np.ndarray
    E_reach: float
    theta_req_deg: float | None
    accel_saturated_fraction: float
    turn_saturated_fraction: float
    climb_saturated_fraction: float
    feasible: bool


@dataclass(frozen=True)
class ReachabilitySolution:
    action: np.ndarray
    selected_tau: float | None
    candidates: tuple[CandidateEvaluation, ...]
    fallback_used: bool
    guidance_mode: str


def _fixed_direction_rollout(defender_position, defender_velocity, direction, config, tau, max_steps_cap):
    from uav_defend.guidance.dynamic_feasibility import simulate_feasibility_rollout
    return simulate_feasibility_rollout(defender_position, defender_velocity, direction, config, tau, max_steps_cap=max_steps_cap)


def _receding_target_rollout(defender_position, defender_velocity, predict_fn: PredictFn, config, tau, max_steps_cap):
    """Phase 78 RECEDING_TARGET: re-aim toward predict_fn(k*dt) at every
    simulated step, rather than one frozen direction. Uses the same
    `advance_velocity`/`apply_boundary` dynamics as `SoldierEnv`; does not
    mutate any environment."""
    dt = float(config.dt)
    eps = float(config.eps)
    n_steps = 0 if tau <= 0 else min(int(round(tau / dt)), max_steps_cap)

    pos = np.asarray(defender_position, dtype=np.float64).copy()
    vel = np.asarray(defender_velocity, dtype=np.float64).copy()

    accel_sat = turn_sat = climb_sat = 0
    for k in range(n_steps):
        aim_point = predict_fn(k * dt)
        direction = pure_pursuit_direction(aim_point, pos, eps)
        desired_velocity = config.v_d * direction
        next_vel, diag = advance_velocity(
            current_velocity=vel, desired_velocity=desired_velocity,
            max_speed=config.v_d, max_accel=config.defender_max_accel,
            max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
            max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
            dt=dt, eps=eps,
        )
        accel_sat += int(diag["accel_saturated"])
        turn_sat += int(diag["turn_saturated"])
        climb_sat += int(diag["climb_saturated"])
        new_pos = pos + next_vel * dt
        new_pos, next_vel = apply_boundary(new_pos, next_vel, L=config.L, max_altitude=config.max_altitude)
        pos, vel = new_pos, next_vel

    denom = max(n_steps, 1)

    class _Result:
        pass
    result = _Result()
    result.terminal_position = pos
    result.accel_saturated_fraction = accel_sat / denom
    result.turn_saturated_fraction = turn_sat / denom
    result.climb_saturated_fraction = climb_sat / denom
    result.n_steps = n_steps
    return result


def evaluate_candidate(
    defender_position: np.ndarray, defender_velocity: np.ndarray, predict_fn: PredictFn,
    tau: float, config, intercept_radius: float, max_steps_cap: int,
    mode: ReachabilityMode = ReachabilityMode.FIXED_DIRECTION,
) -> CandidateEvaluation:
    p_target = predict_fn(tau)
    if not np.all(np.isfinite(p_target)):
        p_target = predict_fn(0.0)  # guard: never evaluate a non-finite candidate point

    direction = pure_pursuit_direction(p_target, defender_position, config.eps)

    if mode is ReachabilityMode.FIXED_DIRECTION:
        rollout = _fixed_direction_rollout(defender_position, defender_velocity, direction, config, tau, max_steps_cap)
    else:
        rollout = _receding_target_rollout(defender_position, defender_velocity, predict_fn, config, tau, max_steps_cap)

    from uav_defend.dynamics.constrained_point_mass import advance_velocity as _av  # noqa: F401  (import kept local to avoid cycle at module import time)
    speed = float(np.linalg.norm(defender_velocity))
    if speed > config.eps:
        cos_theta = float(np.clip(np.dot(np.asarray(defender_velocity) / speed, direction), -1.0, 1.0))
        theta_req_deg = float(np.degrees(np.arccos(cos_theta)))
    else:
        theta_req_deg = None

    e_reach = float(np.linalg.norm(rollout.terminal_position - p_target)) if rollout.n_steps > 0 else float("inf")

    return CandidateEvaluation(
        tau=tau, predicted_target_point=p_target, E_reach=e_reach, theta_req_deg=theta_req_deg,
        accel_saturated_fraction=rollout.accel_saturated_fraction,
        turn_saturated_fraction=rollout.turn_saturated_fraction,
        climb_saturated_fraction=rollout.climb_saturated_fraction,
        feasible=e_reach <= intercept_radius,
    )


def select_candidate(candidates: tuple[CandidateEvaluation, ...]) -> tuple[CandidateEvaluation, bool]:
    """Phase 57 lexicographic selection rule -- IDENTICAL for RA-Lead and
    RA-LAPG. Returns (selected_candidate, fallback_used)."""
    def _heading_key(c: CandidateEvaluation) -> float:
        return c.theta_req_deg if c.theta_req_deg is not None else 0.0

    feasible_candidates = [c for c in candidates if c.feasible]
    if feasible_candidates:
        return min(feasible_candidates, key=lambda c: (c.tau, c.E_reach, _heading_key(c))), False
    return min(candidates, key=lambda c: (c.E_reach, c.tau, _heading_key(c))), True
