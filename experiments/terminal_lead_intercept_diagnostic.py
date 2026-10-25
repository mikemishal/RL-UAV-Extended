"""Diagnostic-only (Phases 3-13): capture-assured terminal lead-intercept candidate search for the
V2 attacker (A0). NOT wired into production. Uses only the already-accepted, already-committed
constrained point-mass dynamics (`uav_defend.dynamics.constrained_point_mass`) -- no new production
code is touched by this module.

Asset-state source (Phase 4): the asset's current position is directly observable (A0's existing
mission command already uses it: `r_hat = (soldier_pos - enemy_pos)/|...|`). Its velocity is NOT
separately exposed to A0 anywhere in the existing observation/info contract, so it is estimated the
same way any other finite-difference quantity available to mission logic would be: from the most
recently observed asset position and the one immediately before it, divided by dt. This is not
privileged information -- it is exactly the two most recent REAL positions a receding-horizon
controller already has. No future or ground-truth-only asset velocity field is read.

Lead-intercept equation (Phase 5): for commanded speed s and relative position r = p_asset - p_me
(both at the CURRENT decision point), assuming the asset continues at constant velocity v_asset,
solve for the smallest positive t_i with ||r + v_asset*t_i|| = s*t_i (standard constant-velocity
intercept quadratic a*t^2+b*t+c=0, a=|v_asset|^2-s^2, b=2*r.v_asset, c=|r|^2). The predicted
intercept point is p_i = p_asset + v_asset*t_i; initial desired direction = normalize(p_i - p_me).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.obstacles.avoidance import DynamicsLimits
from uav_defend.obstacles.layout import ObstacleLayout
from uav_defend.obstacles.swept_contact import first_contact

GRID_SPEEDS = tuple(range(0, 13))  # 0..12 m/s, 1 m/s resolution (v_e = 12)


def estimate_asset_velocity(asset_now: np.ndarray, asset_prev: np.ndarray | None, dt: float) -> np.ndarray:
    """Finite-difference estimate from the two most recently OBSERVED asset positions -- the same
    kind of quantity mission logic already has access to. Zero if there is no previous observation
    (first step) or no observable motion."""
    if asset_prev is None:
        return np.zeros(3, dtype=np.float64)
    return (np.asarray(asset_now, dtype=np.float64) - np.asarray(asset_prev, dtype=np.float64)) / dt


def solve_lead_intercept_time(r: np.ndarray, v_asset: np.ndarray, speed: float, eps: float) -> float | None:
    """Smallest positive t with ||r + v_asset*t|| = speed*t, or None if no positive real root."""
    a = float(np.dot(v_asset, v_asset)) - speed * speed
    b = 2.0 * float(np.dot(r, v_asset))
    c = float(np.dot(r, r))
    if abs(a) < eps:
        if abs(b) < eps:
            return None
        t = -c / b
        return t if t > eps else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    roots = [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]
    positive = [t for t in roots if t > eps]
    return min(positive) if positive else None


def lead_direction(position: np.ndarray, v_asset: np.ndarray, asset_position: np.ndarray, speed: float,
                    fallback: np.ndarray, eps: float) -> np.ndarray:
    """Lead-intercept unit direction from the current state, or `fallback` (pure pursuit) if no
    positive-time solution exists at this speed (e.g. the asset is unreachable at that speed)."""
    r = np.asarray(asset_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    if speed < eps:
        return np.zeros(3, dtype=np.float64)
    t = solve_lead_intercept_time(r, v_asset, speed, eps)
    if t is None:
        return np.asarray(fallback, dtype=np.float64)
    intercept_point = np.asarray(asset_position, dtype=np.float64) + v_asset * t
    direction = intercept_point - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > eps else np.asarray(fallback, dtype=np.float64)


def _step_at_speed(position32, velocity32, direction, speed, limits: DynamicsLimits):
    """Exactly `SoldierEnv._move_enemy`'s update, generalized with an explicit commanded speed
    (the real env always uses speed == limits.max_speed; this diagnostic needs other values).
    Uses only the already-accepted `advance_velocity`/`apply_boundary` primitives -- no equations
    are duplicated or reinvented."""
    next_v, _ = advance_velocity(
        current_velocity=velocity32, desired_velocity=speed * direction, max_speed=limits.max_speed,
        max_accel=limits.max_accel, max_turn_rate_rad=limits.max_turn_rate_rad,
        max_climb_rate=limits.max_climb_rate, max_descent_rate=limits.max_descent_rate,
        dt=limits.dt, eps=limits.eps,
    )
    new_p, new_v = apply_boundary(position32 + next_v * limits.dt, next_v, L=limits.L, max_altitude=limits.max_altitude)
    return new_p.astype(np.float32), new_v.astype(np.float32)


@dataclass(frozen=True)
class LeadRollout:
    speed: float
    captured: bool
    capture_step: int | None
    capture_time_s: float | None
    obstacle_conflict_step: int | None        # first step a SEGMENT intersects a perceived obstacle
    avoidance_would_activate_step: int | None  # first step the FULL-SPEED pure-pursuit rollout from
                                                 # that predicted state would itself collide (i.e. the
                                                 # accepted avoidance trigger would fire there)
    path_length_m: float
    min_clearance_m: float
    heading_change_deg: float
    positions: list[np.ndarray]

    @property
    def capture_assured(self) -> bool:
        """Phase 7: capture occurs, strictly before any obstacle conflict, and strictly before any
        step at which the existing avoidance controller's own trigger would have activated."""
        if not self.captured:
            return False
        if self.obstacle_conflict_step is not None and self.obstacle_conflict_step <= self.capture_step:
            return False
        if self.avoidance_would_activate_step is not None and self.avoidance_would_activate_step < self.capture_step:
            return False
        return True


def _mission_direction(position, asset_position, fallback, eps):
    r = np.asarray(asset_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(r))
    if norm <= eps:
        return np.asarray(fallback, dtype=np.float64)
    return r / norm


def _pure_pursuit_rollout_collides(position32, velocity32, asset_position, limits: DynamicsLimits, horizon: int,
                                    layout: ObstacleLayout, fallback) -> bool:
    """Would the EXISTING accepted avoidance trigger (full-speed pure pursuit, static-asset
    assumption, same as `direct_rollout` in uav_defend/obstacles/stateful_avoidance.py) see a
    collision from this predicted state? Reused here only to know whether the LEAD candidate would
    be flying somewhere the real avoidance controller would itself have reacted to -- not to
    duplicate or alter that controller."""
    p, v = position32, velocity32
    positions = [p.astype(np.float64)]
    for _ in range(horizon):
        direction = _mission_direction(p, asset_position, fallback, limits.eps)
        p, v = _step_at_speed(p, v, direction, limits.max_speed, limits)
        positions.append(p.astype(np.float64))
    return any(layout.first_segment_intersection(a, b).intersects for a, b in zip(positions, positions[1:]))


def lead_intercept_rollout(
    position32: np.ndarray, velocity32: np.ndarray, speed: float, asset_position: np.ndarray,
    v_asset: np.ndarray, limits: DynamicsLimits, horizon: int, stop_radius: float, layout: ObstacleLayout,
    fallback: np.ndarray,
) -> LeadRollout:
    """Execution-faithful rollout of commanded speed `speed` under receding-horizon lead-intercept
    guidance (direction re-solved from the predicted current state every step, per Phase 5), with
    the asset predicted at constant velocity `v_asset` from its CURRENT observed position (Phase
    4) -- never a privileged future ground-truth position."""
    p, v = position32, velocity32
    positions = [p.astype(np.float64)]
    velocities = [v.astype(np.float64)]
    asset0 = np.asarray(asset_position, dtype=np.float64)
    captured, capture_step, capture_time = False, None, None
    obstacle_conflict_step = None
    avoidance_trigger_step = None
    for k in range(horizon):
        predicted_asset = asset0 + v_asset * (k * limits.dt)
        direction = lead_direction(p, v_asset, predicted_asset, speed, fallback, limits.eps)
        p, v = _step_at_speed(p, v, direction, speed, limits)
        positions.append(p.astype(np.float64))
        velocities.append(v.astype(np.float64))
        predicted_asset_next = asset0 + v_asset * ((k + 1) * limits.dt)
        hit = first_contact(positions[-2], positions[-1], predicted_asset, predicted_asset_next, stop_radius)
        if hit is not None and not captured:
            captured, capture_step, capture_time = True, k, (k + 1) * limits.dt
            break
        if obstacle_conflict_step is None and layout.first_segment_intersection(positions[-2], positions[-1]).intersects:
            obstacle_conflict_step = k
        if avoidance_trigger_step is None and _pure_pursuit_rollout_collides(
            p, v, predicted_asset_next, limits, horizon - k - 1, layout, fallback):
            avoidance_trigger_step = k

    path_length = float(sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:])))
    min_clearance = min((layout.min_clearance(point) for point in positions), default=float("inf"))
    heading_change = 0.0
    for a, b in zip(velocities, velocities[1:]):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na > limits.eps and nb > limits.eps:
            heading_change += float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))
    return LeadRollout(
        speed=speed, captured=captured, capture_step=capture_step, capture_time_s=capture_time,
        obstacle_conflict_step=obstacle_conflict_step, avoidance_would_activate_step=avoidance_trigger_step,
        path_length_m=path_length, min_clearance_m=min_clearance, heading_change_deg=heading_change,
        positions=positions,
    )


def best_capture_assured_candidate(
    position32, velocity32, asset_position, v_asset, limits: DynamicsLimits, horizon: int, stop_radius: float,
    layout: ObstacleLayout, fallback,
) -> tuple[LeadRollout | None, list[LeadRollout]]:
    """Phase 8 lexicographic ranking among CAPTURE-ASSURED candidates only: earliest capture, then
    shortest path to capture, then largest minimum clearance before capture, then smallest
    integrated heading change, then highest speed. Returns (best or None, all evaluated rollouts)."""
    evaluations = [
        lead_intercept_rollout(position32, velocity32, float(s), asset_position, v_asset, limits, horizon,
                               stop_radius, layout, fallback)
        for s in GRID_SPEEDS
    ]
    assured = [r for r in evaluations if r.capture_assured]
    if not assured:
        return None, evaluations
    best = min(assured, key=lambda r: (r.capture_step, r.path_length_m, -r.min_clearance_m, r.heading_change_deg, -r.speed))
    return best, evaluations
