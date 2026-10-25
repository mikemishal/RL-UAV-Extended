"""Diagnostic-only: reactive terminal reacquisition geometry for the V2 attacker (A0). Pure
functions only (closing-velocity/fly-by detection, curvature-feasible speed laws, reacquisition
exit condition); NOT wired into production. See
docs/journal_v2_terminal_reacquisition.md for the full investigation and why this design was
ultimately rejected (both tried variants caused a new collision on the one seed they targeted).

Fly-by DETECTION is purely causal (no future information): the range's closing-to-opening
transition is observable the instant it happens because it is computed from the CURRENT (already
post-turn) velocity, not from a future one. A fly-by is only geometrically relevant when the miss
distance is comparable to the vehicle's own current turning radius,
`R_turn = speed / max_turn_rate_rad`:

    relevant  <=>  range <= capture_radius + R_turn

Two curvature-feasible SPEED laws were tried (direction is always left as pure current-position
pursuit, never a lead point or waypoint):

  * Variant A (instantaneous): `v_cmd = min(max_speed, max_turn_rate_rad * max(range -
    capture_radius, 0))` -- the highest speed whose own turning radius still fits the remaining
    range.
  * Variant B (deceleration-aware): accounts for the extra distance consumed while the vehicle
    decelerates from its current speed to the target, by solving a quadratic for the highest
    target speed that remains feasible once that distance is subtracted from the available range.
"""

from __future__ import annotations

import numpy as np

from uav_defend.obstacles.avoidance import DynamicsLimits


def closing_velocity(position: np.ndarray, velocity: np.ndarray, asset_position: np.ndarray, eps: float) -> float:
    """Current range rate: positive while approaching, non-positive once level or receding."""
    r = np.asarray(asset_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(r))
    if norm <= eps:
        return 0.0
    return float(np.dot(np.asarray(velocity, dtype=np.float64), r / norm))


def heading_error_degrees(velocity: np.ndarray, position: np.ndarray, asset_position: np.ndarray, eps: float) -> float:
    v = np.asarray(velocity, dtype=np.float64)
    los = np.asarray(asset_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    nv, nl = float(np.linalg.norm(v)), float(np.linalg.norm(los))
    if nv <= eps or nl <= eps:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v, los)) / (nv * nl), -1.0, 1.0))))


def turning_radius(speed: float, max_turn_rate_rad: float, eps: float) -> float:
    return speed / max(max_turn_rate_rad, eps)


def flyby_detected(
    previous_closing: float | None, closing_now: float, range_now: float, speed_now: float,
    capture_radius: float, limits: DynamicsLimits,
) -> bool:
    """Causal fly-by detector: range just stopped decreasing, no capture, and the miss is
    geometrically relevant to the vehicle's own current turning radius."""
    if previous_closing is None or previous_closing <= 0.0 or closing_now > 0.0:
        return False
    r_turn = turning_radius(speed_now, limits.max_turn_rate_rad, limits.eps)
    return range_now <= capture_radius + r_turn


def curvature_speed(range_now: float, capture_radius: float, max_speed: float, limits: DynamicsLimits) -> float:
    """Variant A: the highest speed whose own turning radius does not exceed the remaining
    distance to the capture sphere."""
    return float(np.clip(limits.max_turn_rate_rad * max(range_now - capture_radius, 0.0), 0.0, max_speed))


def curvature_speed_decel_aware(
    range_now: float, capture_radius: float, current_speed: float, max_speed: float, limits: DynamicsLimits,
) -> float:
    """Variant B: accounts for the distance consumed while decelerating from `current_speed`
    before the turn even begins (`(v0^2 - v_target^2) / (2*accel)`, standard kinematics)."""
    accel = limits.max_accel
    omega = max(limits.max_turn_rate_rad, limits.eps)
    k = max(range_now - capture_radius, 0.0)
    c = k - (current_speed * current_speed) / (2.0 * accel)
    b = 2.0 * accel / omega
    d = 2.0 * accel * c
    discriminant = b * b - 4.0 * d
    if discriminant < 0.0:
        return curvature_speed(range_now, capture_radius, max_speed, limits)
    root = (b - np.sqrt(discriminant)) / 2.0
    return float(np.clip(min(root, current_speed), 0.0, max_speed))


def reacquisition_achieved(closing_now: float, heading_error_deg: float) -> bool:
    """Exit condition from CURRENT geometry only: closing again, broadly facing the asset."""
    return closing_now > 0.0 and heading_error_deg <= 90.0
