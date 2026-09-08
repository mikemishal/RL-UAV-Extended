"""
Pure, dependency-free constrained 3-D point-mass dynamics helper.

Factored out of `uav_defend.envs.soldier_env.SoldierEnv._advance_velocity`
(the single canonical constrained-velocity update previously shared, via a
bound method, by both the defender and the hostile UAV). This module
contains EXACTLY the same equations, made into a pure function so that it
can be reused by planning/prediction code (e.g. the RMPC baseline) WITHOUT
constructing or mutating a `SoldierEnv` instance.

This is a 3-D constrained point-mass dynamics model -- NOT six-DOF or
aerodynamic flight dynamics.

`SoldierEnv._advance_velocity` is now a thin wrapper that delegates here
with `dt=self.config.dt, eps=self.config.eps` -- see
`test_constrained_point_mass_regression.py` for the regression proof that
this factoring preserves bit-for-bit-equivalent (within floating-point
tolerance) behavior versus the prior in-place implementation.
"""

from __future__ import annotations

import numpy as np


def advance_velocity(
    current_velocity: np.ndarray,
    desired_velocity: np.ndarray,
    max_speed: float,
    max_accel: float,
    max_turn_rate_rad: float,
    max_climb_rate: float,
    max_descent_rate: float,
    dt: float,
    eps: float,
) -> tuple[np.ndarray, dict]:
    """
    Advance a persistent 3D point-mass velocity toward a desired velocity,
    subject to a maximum horizontal turn rate, maximum acceleration,
    maximum total speed, and maximum climb/descent rate.

    PURE function: does not read/mutate any environment, RNG, or global
    state; does not mutate its input arrays.

    Order of operations:
      1. Vertical-rate command limiting: clip desired_velocity[2] into
         [-max_descent_rate, max_climb_rate].
      2. Horizontal turn-rate limiting: rotate the current horizontal
         heading toward the desired horizontal heading by at most
         max_turn_rate_rad * dt (shortest signed angular path, wrapped
         to [-pi, pi]), adopting the desired horizontal speed magnitude.
         Special cases (documented explicitly, not merely defaulted):
           - If current horizontal speed <= eps, the previous heading is
             undefined; heading initializes directly toward the desired
             heading with NO rate limit applied (there is nothing to
             rate-limit against). Acceleration limiting (step 3) still
             governs how quickly speed itself can rise.
           - If desired horizontal speed <= eps (e.g. a "stop" command),
             heading has no defined target; the current heading is held
             while horizontal speed is commanded toward zero.
         The result is the "dynamically permissible target velocity".
      3. Acceleration limiting: clip the change from current_velocity to
         the target velocity to at most max_accel * dt in magnitude,
         preserving the direction of the change. Because this is
         equivalent to next = current + t*(target - current) for some
         t in [0, 1], the result is a convex combination of current and
         target.
      4. Total-speed / vertical-rate safety enforcement: since both
         current_velocity and the target velocity already satisfy the
         max_speed ball and the climb/descent bounds (both are convex
         constraints), and step 3 produces a convex combination of the
         two, the result satisfies both bounds automatically. A
         defensive clip is applied regardless, to guard against
         floating-point overshoot (never used to silently paper over a
         design error).

    Args:
        current_velocity: Current persistent velocity [vx, vy, vz], shape (3,).
        desired_velocity: Controller/guidance-commanded desired velocity
            [vx, vy, vz], shape (3,); ||desired_velocity|| is expected to
            be <= max_speed (e.g. max_speed * unit_direction).
        max_speed: Maximum total speed (m/s).
        max_accel: Maximum acceleration magnitude (m/s^2).
        max_turn_rate_rad: Maximum horizontal heading-rate magnitude (rad/s).
        max_climb_rate: Maximum upward vertical speed (m/s), positive.
        max_descent_rate: Maximum downward vertical speed magnitude (m/s), positive.
        dt: Time step duration (s).
        eps: Numerical-stability threshold.

    Returns:
        (next_velocity, diagnostics):
            next_velocity: Constrained velocity, shape (3,), float32.
            diagnostics: dict with keys "accel_used" (m/s^2, actually
                applied this step), "turn_rate_used_deg" (deg/s, actually
                applied this step), "accel_saturated" (bool),
                "turn_saturated" (bool), "climb_saturated" (bool, True if
                the vertical-rate command limit in step 1 was active).
    """
    current_velocity = np.asarray(current_velocity, dtype=np.float64)
    desired_velocity = np.asarray(desired_velocity, dtype=np.float64)

    # 1. Vertical-rate command limiting
    desired_vz = desired_velocity[2]
    desired_vz_clipped = float(np.clip(desired_vz, -max_descent_rate, max_climb_rate))
    climb_saturated = abs(desired_vz_clipped - desired_vz) > 1e-9

    # 2. Horizontal turn-rate limiting
    cur_h = current_velocity[0:2]
    des_h = desired_velocity[0:2]
    ch = float(np.linalg.norm(cur_h))
    dh = float(np.linalg.norm(des_h))

    turn_rate_used_rad = 0.0
    turn_saturated = False

    if dh <= eps:
        # Desired horizontal command is (near) zero: hold current
        # heading, command zero horizontal speed (bounded braking).
        target_heading = np.arctan2(cur_h[1], cur_h[0]) if ch > eps else 0.0
        target_h_mag = 0.0
    elif ch <= eps:
        # Current horizontal velocity undefined (at/near rest):
        # initialize heading directly toward the desired heading; no
        # rate limit applies to an undefined previous heading.
        target_heading = np.arctan2(des_h[1], des_h[0])
        target_h_mag = dh
    else:
        psi_current = np.arctan2(cur_h[1], cur_h[0])
        psi_desired = np.arctan2(des_h[1], des_h[0])
        delta_psi = psi_desired - psi_current
        delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        max_delta = max_turn_rate_rad * dt
        if abs(delta_psi) > max_delta:
            delta_psi = np.sign(delta_psi) * max_delta
            turn_saturated = True
        target_heading = psi_current + delta_psi
        target_h_mag = dh
        turn_rate_used_rad = abs(delta_psi) / dt if dt > 0 else 0.0

    target_velocity = np.array([
        target_h_mag * np.cos(target_heading),
        target_h_mag * np.sin(target_heading),
        desired_vz_clipped,
    ])

    # 3. Acceleration limiting (bounded convex-combination step toward target)
    delta_v = target_velocity - current_velocity
    delta_v_norm = float(np.linalg.norm(delta_v))
    max_delta_v = max_accel * dt
    accel_saturated = delta_v_norm > max_delta_v
    if accel_saturated and delta_v_norm > eps:
        delta_v = delta_v / delta_v_norm * max_delta_v
    next_velocity = current_velocity + delta_v
    accel_used = float(np.linalg.norm(delta_v)) / dt if dt > 0 else 0.0

    # 4. Total-speed / vertical-rate safety enforcement (defensive clip)
    speed = float(np.linalg.norm(next_velocity))
    if speed > max_speed and speed > eps:
        next_velocity = next_velocity / speed * max_speed
    next_velocity[2] = np.clip(next_velocity[2], -max_descent_rate, max_climb_rate)

    diagnostics = {
        "accel_used": accel_used,
        "turn_rate_used_deg": float(np.degrees(turn_rate_used_rad)),
        "accel_saturated": bool(accel_saturated),
        "turn_saturated": bool(turn_saturated),
        "climb_saturated": bool(climb_saturated),
    }

    return next_velocity.astype(np.float32), diagnostics


def apply_boundary(
    pos: np.ndarray, vel: np.ndarray, L: float, max_altitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip position to the 3D engagement volume (x, y in [-L, L], z in
    [0, max_altitude]) and zero ONLY the outward-normal velocity component
    at any boundary the position has reached. Tangential velocity
    components are preserved unchanged (no bounce/reflection for
    persistent-velocity vehicles).

    PURE function: does not mutate its input arrays. Factored out of
    `SoldierEnv._apply_boundary` (the single canonical domain-boundary
    update) so short-horizon prediction code (e.g.
    `uav_defend/obstacles/avoidance.py`) can apply the SAME ground/ceiling/
    wall clipping the real simulation uses -- without this, unclipped
    prediction can let a mover's z "escape" below the ground plane (or
    beyond a wall) in a way that never actually happens in the real
    simulation, causing a predicted trajectory to spuriously miss an
    obstacle it would truly collide with at the real (clipped) altitude.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    vel = np.array(vel, dtype=np.float64, copy=True)

    if pos[0] > L:
        pos[0] = L
        if vel[0] > 0:
            vel[0] = 0.0
    elif pos[0] < -L:
        pos[0] = -L
        if vel[0] < 0:
            vel[0] = 0.0

    if pos[1] > L:
        pos[1] = L
        if vel[1] > 0:
            vel[1] = 0.0
    elif pos[1] < -L:
        pos[1] = -L
        if vel[1] < 0:
            vel[1] = 0.0

    if pos[2] > max_altitude:
        pos[2] = max_altitude
        if vel[2] > 0:
            vel[2] = 0.0
    elif pos[2] < 0.0:
        pos[2] = 0.0
        if vel[2] < 0:
            vel[2] = 0.0

    return pos, vel
