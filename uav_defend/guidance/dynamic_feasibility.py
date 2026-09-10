"""
Policy-independent dynamic-feasibility diagnostics (Phase 35-37).

Answers: is a nominal Lead command dynamically achievable by the
defender in the time available before the predicted intercept?

DIAGNOSTIC ONLY. This module is never called from `SoldierEnv.step()` or
any deployable policy's `act()`. It reuses the EXACT SAME pure dynamics
functions the environment uses (`uav_defend.dynamics.constrained_point_mass
.advance_velocity` / `.apply_boundary`) so that "how far the defender could
actually get in tau seconds" is computed with the identical constrained
point-mass model the real simulation enforces -- never a duplicated or
approximate reimplementation.

Individual interpretable features are kept separate (Phase 36 explicitly
asks NOT to collapse them into one arbitrary weighted score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary


@dataclass(frozen=True)
class FeasibilityRolloutResult:
    """Result of forward-simulating a FIXED Lead command direction for a
    short horizon, using the real constrained defender dynamics."""

    terminal_position: np.ndarray
    accel_saturated_fraction: float
    turn_saturated_fraction: float
    climb_saturated_fraction: float
    obstacle_path_conflict: bool | None  # None if no obstacle layout supplied
    n_steps: int


@dataclass(frozen=True)
class FeasibilityFeatures:
    """Individually-interpretable feasibility diagnostics for one
    (defender state, Lead command, intercept solution) snapshot."""

    theta_req_deg: float | None  # required heading change to align with Lead's direction
    theta_available_deg: float | None  # turn authority available before tau
    D_turn: float | None  # normalized turn demand (theta_req / theta_available)
    required_speed: float | None  # ||intercept_point - defender_position|| / tau
    closing_speed_ratio: float | None  # required_speed / v_d
    predicted_accel_saturated_fraction: float
    predicted_turn_saturated_fraction: float
    predicted_climb_saturated_fraction: float
    E_reach: float | None  # miss distance of the dynamically-simulated defender vs. the nominal intercept point
    obstacle_path_conflict: bool | None
    n_rollout_steps: int


def simulate_feasibility_rollout(
    defender_position: np.ndarray,
    defender_velocity: np.ndarray,
    lead_direction: np.ndarray,
    config,
    tau: float | None,
    obstacle_layout=None,
    max_steps_cap: int = 40,
) -> FeasibilityRolloutResult:
    """
    Forward-simulate repeated application of the CURRENT Lead command
    direction, held fixed, for `min(tau, max_steps_cap * dt)` seconds,
    using the same `advance_velocity` / `apply_boundary` functions
    `SoldierEnv` uses. Does not mutate any environment or RNG state.

    This is a "frozen-command" feasibility check: it asks "if the
    defender kept applying today's Lead direction unchanged, how far
    could its own constrained dynamics actually carry it before tau?" --
    not a full-fidelity re-simulation of the reactive Lead law (which
    would update direction every step as new hostile estimates arrive).
    """
    dt = float(config.dt)
    eps = float(config.eps)

    n_steps = 0 if (tau is None or tau <= 0) else min(int(round(tau / dt)), max_steps_cap)

    pos = np.asarray(defender_position, dtype=np.float64).copy()
    vel = np.asarray(defender_velocity, dtype=np.float64).copy()
    direction = np.asarray(lead_direction, dtype=np.float64)
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm > eps:
        direction = direction / dir_norm
    desired_velocity = config.v_d * direction

    accel_sat_count = turn_sat_count = climb_sat_count = 0
    obstacle_conflict = False if obstacle_layout is not None else None

    for _ in range(n_steps):
        next_vel, diag = advance_velocity(
            current_velocity=vel, desired_velocity=desired_velocity,
            max_speed=config.v_d, max_accel=config.defender_max_accel,
            max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
            max_climb_rate=config.defender_max_climb_rate,
            max_descent_rate=config.defender_max_descent_rate,
            dt=dt, eps=eps,
        )
        accel_sat_count += int(diag["accel_saturated"])
        turn_sat_count += int(diag["turn_saturated"])
        climb_sat_count += int(diag["climb_saturated"])

        new_pos = pos + next_vel * dt
        new_pos, next_vel = apply_boundary(new_pos, next_vel, L=config.L, max_altitude=config.max_altitude)

        if obstacle_layout is not None and obstacle_layout.segment_intersects(pos, new_pos):
            obstacle_conflict = True

        pos, vel = new_pos, next_vel

    denom = max(n_steps, 1)
    return FeasibilityRolloutResult(
        terminal_position=pos,
        accel_saturated_fraction=accel_sat_count / denom,
        turn_saturated_fraction=turn_sat_count / denom,
        climb_saturated_fraction=climb_sat_count / denom,
        obstacle_path_conflict=obstacle_conflict,
        n_steps=n_steps,
    )


def compute_feasibility_features(
    defender_position: np.ndarray,
    defender_velocity: np.ndarray,
    lead_direction: np.ndarray,
    intercept_point: np.ndarray | None,
    intercept_time: float | None,
    config,
    obstacle_layout=None,
    max_steps_cap: int = 40,
) -> FeasibilityFeatures:
    """Compute the individual, interpretable feasibility diagnostics
    (Phase 36) for one snapshot of (defender state, Lead command,
    intercept solution)."""
    eps = float(config.eps)
    defender_velocity = np.asarray(defender_velocity, dtype=np.float64)
    lead_direction = np.asarray(lead_direction, dtype=np.float64)
    dir_norm = float(np.linalg.norm(lead_direction))
    unit_direction = lead_direction / dir_norm if dir_norm > eps else lead_direction

    speed = float(np.linalg.norm(defender_velocity))
    if speed > eps and dir_norm > eps:
        cos_theta = float(np.clip(np.dot(defender_velocity / speed, unit_direction), -1.0, 1.0))
        theta_req_deg = float(np.degrees(np.arccos(cos_theta)))
    else:
        theta_req_deg = None  # heading undefined at rest, or no commanded direction

    tau = None if intercept_time is None else max(float(intercept_time), 0.0)
    omega_max_rad = np.radians(config.defender_max_turn_rate_deg)
    theta_available_deg = float(np.degrees(omega_max_rad * tau)) if tau is not None else None

    D_turn = None
    if theta_req_deg is not None and theta_available_deg is not None:
        D_turn = theta_req_deg / max(theta_available_deg, eps)

    required_speed = closing_speed_ratio = None
    if intercept_point is not None and tau is not None and tau > eps:
        required_speed = float(np.linalg.norm(np.asarray(intercept_point) - np.asarray(defender_position)) / tau)
        closing_speed_ratio = required_speed / config.v_d

    rollout = simulate_feasibility_rollout(
        defender_position, defender_velocity, unit_direction, config, tau, obstacle_layout, max_steps_cap,
    )

    E_reach = None
    if intercept_point is not None and rollout.n_steps > 0:
        E_reach = float(np.linalg.norm(rollout.terminal_position - np.asarray(intercept_point)))

    return FeasibilityFeatures(
        theta_req_deg=theta_req_deg,
        theta_available_deg=theta_available_deg,
        D_turn=D_turn,
        required_speed=required_speed,
        closing_speed_ratio=closing_speed_ratio,
        predicted_accel_saturated_fraction=rollout.accel_saturated_fraction,
        predicted_turn_saturated_fraction=rollout.turn_saturated_fraction,
        predicted_climb_saturated_fraction=rollout.climb_saturated_fraction,
        E_reach=E_reach,
        obstacle_path_conflict=rollout.obstacle_path_conflict,
        n_rollout_steps=rollout.n_steps,
    )
