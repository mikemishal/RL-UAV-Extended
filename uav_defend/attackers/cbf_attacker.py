"""A1-CBF: a deterministic, literature-motivated attacker for a velocity-controlled quadrotor abstraction.

Architecture (frozen, see docs/journal_cbf_attacker/foundation/README.md):

    current relative target state -> nominal velocity guidance -> CBF/QP safety filter -> quadrotor velocity executor

This module implements the pursuit/dynamics FOUNDATION only:
  * `CBFAttackerController.nominal_velocity`  -- analytical moving-target pursuit (Prompt 1: proportional law on the
    CURRENT relative position; no target-velocity feedforward, no learned prediction -- see that method's docstring).
  * `CBFAttackerController.filter_velocity`   -- the CBF/QP safety-filter INTERFACE. Prompt 1 is the identity filter
    (v_safe = v_nom, zero constraints); a later prompt replaces only this method's body with an actual QP solve over
    the (currently always empty) `hazards` sequence. No other part of this module or of `SoldierEnv` needs to change
    when that happens.
  * `integrate_quadrotor_substeps`            -- the velocity EXECUTOR: an acceleration-limited (NOT heading/turn-rate
    limited) 3-D point-mass integrator appropriate for a multirotor, substepped at `CBF_DT_SUB` under one external
    `SoldierEnv` step (see that function's docstring for the exact substep contract and why it does not double-count
    swept mission contacts).

Nothing here imports `SoldierEnv`; everything is a pure function / stateless class, independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_defend.dynamics.constrained_point_mass import apply_boundary

# Internal integration substep (s) for the quadrotor velocity executor. The external SoldierEnv step remains
# dt = 0.5 s (unchanged, for compatibility with every existing experiment); this is strictly an internal refinement
# of how ONE entity's one-step motion is integrated, not a change to the external step rate or to any other mover.
CBF_DT_SUB = 0.1

# Frozen nominal-guidance gain (s^-1): v_nom = min(max_speed, kp * ||r||) * r_hat. Not tunable after the foundation
# acceptance experiment is run (see the stage README); do not change this value based on acceptance results.
CBF_KP = 1.0

# Default physical limits, matching the canonical V2 hostile-UAV config values (uav_defend/config/env_config.py:
# v_e=12.0, enemy_max_accel=6.0, enemy_max_climb_rate=5.0, enemy_max_descent_rate=5.0) -- see the module README for
# the rationale for keeping these numbers even though the legacy heading/turn-rate limit is intentionally dropped.
DEFAULT_MAX_HORIZONTAL_SPEED = 12.0
DEFAULT_MAX_ACCEL = 6.0
DEFAULT_MAX_CLIMB_RATE = 5.0
DEFAULT_MAX_DESCENT_RATE = 5.0


def substep_count(dt: float, dt_sub: float = CBF_DT_SUB) -> int:
    """Number of internal substeps covering one external step of duration `dt` (5 for the canonical dt=0.5, dt_sub=0.1).
    Raises if `dt` is not (to floating-point tolerance) an integer multiple of `dt_sub`, so a misconfigured dt can
    never silently produce an incomplete or over-long substep sequence."""
    n = dt / dt_sub
    rounded = round(n)
    if rounded < 1 or abs(n - rounded) > 1e-6:
        raise ValueError(f"dt={dt} is not an integer multiple of dt_sub={dt_sub} (n={n})")
    return int(rounded)


@dataclass(frozen=True)
class CBFAttackerLimits:
    """Physical limits for the quadrotor velocity executor. Defaults match the canonical V2 hostile-UAV config."""

    max_horizontal_speed: float = DEFAULT_MAX_HORIZONTAL_SPEED
    max_accel: float = DEFAULT_MAX_ACCEL
    max_climb_rate: float = DEFAULT_MAX_CLIMB_RATE
    max_descent_rate: float = DEFAULT_MAX_DESCENT_RATE


def nominal_velocity_command(relative_position: np.ndarray, *, kp: float = CBF_KP, max_speed: float = DEFAULT_MAX_HORIZONTAL_SPEED,
                              eps: float = 1e-8) -> np.ndarray:
    """Analytical moving-target pursuit (Prompt 1, frozen):

        v_nom = min(max_speed, kp * ||r||) * r / ||r||    for ||r|| > eps,   else [0, 0, 0] (hover; no random fallback)

    where `r` is the CURRENT 3-D relative position (asset - attacker) at the time this command is computed -- never
    the initial/obsolete asset location, and never a learned or predicted future position. No target-velocity
    feedforward is added: because the canonical asset motion is an i.i.d. Gaussian random walk with no persistent
    velocity (see docs/journal_v2_a1_curriculum/stage4/README.md), there is no well-defined target velocity to feed
    forward, and doing so would not improve pursuit of this specific target model -- this is a deliberate, documented
    omission, not an oversight. With kp=1 s^-1 and max_speed=12 m/s this naturally slows the commanded speed as range
    shrinks (12 m/s at >=12 m away, 6 m/s at 6 m, 3 m/s at 3 m, 2 m/s at 2 m, hover at r=0) with NO minimum forward
    speed -- hover is always a reachable command.
    """
    r = np.asarray(relative_position, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(r))
    if distance <= eps:
        return np.zeros(3, dtype=np.float64)
    speed = min(max_speed, kp * distance)
    return (speed / distance) * r


@dataclass
class CBFAttackerController:
    """A1-CBF attacker: analytical pursuit guidance + a safety-filter interface. Stateless across steps (no
    persistent planner/RNG state); one instance may be shared freely across episodes/workers."""

    kp: float = CBF_KP
    limits: CBFAttackerLimits = field(default_factory=CBFAttackerLimits)
    eps: float = 1e-8

    def nominal_velocity(self, relative_position: np.ndarray) -> np.ndarray:
        """Target-relative nominal guidance component (see `nominal_velocity_command`)."""
        return nominal_velocity_command(relative_position, kp=self.kp, max_speed=self.limits.max_horizontal_speed, eps=self.eps)

    def filter_velocity(self, v_nom: np.ndarray, *, state: dict | None = None, hazards: tuple = (), defender_hazard=None) -> tuple[np.ndarray, dict]:
        """CBF/QP safety-filter component. Identity when `hazards` is empty AND `defender_hazard` is None
        (`v_safe = v_nom` exactly, bit-for-bit -- this is the ENTIRE behavior for Prompt 1 / C0, and remains
        exactly this for any hazard-free, defender-unobservable step in any clutter/defender level, preserving the
        foundation stage's acceptance result unperturbed). When `hazards` is non-empty and/or `defender_hazard` is
        provided (Prompt 2/4: static-obstacle and/or defender-avoidance CBF constraints -- a
        `uav_defend.attackers.cbf_obstacle.StaticHazard`/`uav_defend.attackers.cbf_defender.DefenderHazard`),
        solves the JOINT `min_v 0.5*||v - v_nom||^2` subject to every active constraint via
        `uav_defend.attackers.cbf_obstacle.solve_velocity_cbf_qp` (hard -> soft-slack -> emergency-brake fallback
        chain, shared between obstacle and defender constraint classes; see that function's docstring). `state` is
        accepted (own position/velocity) for forward compatibility; the QP itself does not read it (each hazard
        already carries its own precomputed normal/clearance/stopping-margin/estimated-velocity).
        """
        v_nom = np.asarray(v_nom, dtype=np.float64).reshape(3)
        constraint_count = len(hazards)
        if constraint_count == 0 and defender_hazard is None:
            v_safe = v_nom.copy()
            diagnostics = {
                "cbf_enabled": True, "cbf_constraint_count": 0, "cbf_qp_status": "identity_no_constraints",
                "cbf_intervention_norm": 0.0, "qp_success": True, "qp_iterations": 0, "qp_objective": 0.0,
                "qp_max_constraint_residual": 0.0, "soft_slack": 0.0, "obstacle_constraint_residual": 0.0,
                "defender_constraint_residual": 0.0, "intervention_source": "none", "infeasibility_source": None,
            }
            return v_safe, diagnostics

        from uav_defend.attackers.cbf_obstacle import CBF_ALPHA, solve_velocity_cbf_qp
        from uav_defend.attackers.cbf_defender import CBF_ALPHA_DEFENDER

        v_safe, qp_diag = solve_velocity_cbf_qp(
            v_nom, list(hazards), defender_hazard=defender_hazard, alpha=CBF_ALPHA, alpha_defender=CBF_ALPHA_DEFENDER,
            max_horizontal_speed=self.limits.max_horizontal_speed,
            max_climb_rate=self.limits.max_climb_rate, max_descent_rate=self.limits.max_descent_rate,
        )
        intervention_norm = float(np.linalg.norm(v_safe - v_nom))
        diagnostics = {
            "cbf_enabled": True, "cbf_constraint_count": int(constraint_count), "cbf_intervention_norm": intervention_norm,
            **qp_diag,
        }
        return v_safe, diagnostics


def apply_one_substep(
    position: np.ndarray, velocity: np.ndarray, v_cmd: np.ndarray, *, limits: CBFAttackerLimits, dt_sub: float,
    L: float, max_altitude: float, eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """ONE substep of the quadrotor velocity executor (the per-substep body factored out of
    `integrate_quadrotor_substeps` so a safety filter that must be re-evaluated every substep -- see
    `uav_defend.attackers.cbf_obstacle.integrate_obstacle_cbf_substeps` -- can reuse the EXACT SAME physical update,
    never a second/duplicated implementation): (1) acceleration-limited step toward `v_cmd`; (2) hard horizontal-
    speed cap; (3) hard climb/descent cap; (4) position update + `apply_boundary`. Returns
    `(position, velocity, substep_diagnostics)` -- NOT yet aggregated across substeps (the caller's loop does that).
    """
    pos = np.asarray(position, dtype=np.float64).copy()
    vel = np.asarray(velocity, dtype=np.float64).copy()
    v_cmd = np.asarray(v_cmd, dtype=np.float64).reshape(3)

    delta_v = v_cmd - vel
    delta_v_norm = float(np.linalg.norm(delta_v))
    max_delta_v = limits.max_accel * dt_sub
    accel_saturated = delta_v_norm > max_delta_v
    if accel_saturated and delta_v_norm > eps:
        delta_v = delta_v / delta_v_norm * max_delta_v
    vel = vel + delta_v
    accel_used = float(np.linalg.norm(delta_v)) / dt_sub if dt_sub > 0 else 0.0

    horiz = vel[:2]
    horiz_speed = float(np.linalg.norm(horiz))
    horizontal_saturated = horiz_speed > limits.max_horizontal_speed
    if horizontal_saturated and horiz_speed > eps:
        vel[:2] = horiz / horiz_speed * limits.max_horizontal_speed

    vertical_saturated = False
    if vel[2] > limits.max_climb_rate:
        vertical_saturated = True
        vel[2] = limits.max_climb_rate
    elif vel[2] < -limits.max_descent_rate:
        vertical_saturated = True
        vel[2] = -limits.max_descent_rate

    pos = pos + vel * dt_sub
    pos, vel = apply_boundary(pos, vel, L=L, max_altitude=max_altitude)

    diagnostics = {
        "accel_used": accel_used, "accel_saturated": bool(accel_saturated),
        "horizontal_speed_saturated": bool(horizontal_saturated), "vertical_rate_saturated": bool(vertical_saturated),
    }
    return pos, vel, diagnostics


def integrate_quadrotor_substeps(
    position: np.ndarray, velocity: np.ndarray, v_cmd: np.ndarray, *, limits: CBFAttackerLimits, dt: float,
    dt_sub: float = CBF_DT_SUB, L: float, max_altitude: float, eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Velocity-executor component: integrates ONE external `dt` (default 0.5 s) as `substep_count(dt, dt_sub)`
    (default 5) internal substeps of duration `dt_sub` (default 0.1 s), with `v_cmd` HELD CONSTANT across all of
    them (computed once per external step by the caller, exactly like the legacy guidance -- see
    `CBFAttackerController.nominal_velocity`). Appropriate when `v_cmd` is NOT safety-critical (Prompt 1's identity
    safety filter, or any hazard-free step) -- for a safety-critical command that must react faster than once per
    external step, see `uav_defend.attackers.cbf_obstacle.integrate_obstacle_cbf_substeps`, which re-solves the
    safety filter every substep using this SAME per-substep physical update (`apply_one_substep`).

    Per substep, in order (frozen; matches the brief's `Delta v = v_cmd - v; ||Delta v|| <= a_max * dt_sub` contract
    exactly, with NO legacy heading/turn-rate limiting -- a quadrotor can accelerate laterally, decelerate, hover and
    reverse without maintaining forward airspeed, unlike the legacy fixed-wing-style model):
      1. Acceleration-limited step toward `v_cmd`: delta_v = v_cmd - v; clip ||delta_v|| <= max_accel * dt_sub
         (preserving direction), in the FULL 3-D velocity (not decomposed into horizontal heading + magnitude).
      2. Hard horizontal-speed cap: clip ||v[:2]|| to `limits.max_horizontal_speed` (direction-preserving).
      3. Hard climb/descent cap: clip v[2] to [-max_descent_rate, max_climb_rate].
      4. Position update: p += v * dt_sub, followed by the EXISTING `apply_boundary` (bit-identical arena-boundary
         handling used everywhere else in `SoldierEnv`).

    Swept mission-contact detection (capture / interception) is NOT performed here and is UNCHANGED in `SoldierEnv`:
    it still tests exactly ONE linear segment from the pre-step to the post-step (here: post-ALL-5-substeps)
    position over the full external `dt` -- see `uav_defend/envs/soldier_env.py:step()`. This function only changes
    HOW the attacker's one-step motion is produced (acceleration-limited substepping instead of a single
    turn-rate-limited update); it introduces no new/parallel contact test, so no event can be double-counted.

    Returns (new_position (float32, shape (3,)), new_velocity (float32, shape (3,)), diagnostics) where diagnostics
    has the keys "accel_used" (max over substeps, m/s^2), "turn_rate_used_deg" (always 0.0 -- no heading concept),
    "accel_saturated", "horizontal_speed_saturated", "vertical_rate_saturated" (bool, True if ANY substep saturated
    that limit) and "climb_saturated" (alias of vertical_rate_saturated, for `SoldierEnv._get_info()` schema
    compatibility with the legacy dynamics-diagnostics dict).
    """
    pos = np.asarray(position, dtype=np.float64).copy()
    vel = np.asarray(velocity, dtype=np.float64).copy()
    v_cmd = np.asarray(v_cmd, dtype=np.float64).reshape(3)
    n = substep_count(dt, dt_sub)
    max_accel_used = 0.0
    accel_saturated = horizontal_saturated = vertical_saturated = False
    for _ in range(n):
        pos, vel, sub_diag = apply_one_substep(pos, vel, v_cmd, limits=limits, dt_sub=dt_sub, L=L, max_altitude=max_altitude, eps=eps)
        max_accel_used = max(max_accel_used, sub_diag["accel_used"])
        accel_saturated = accel_saturated or sub_diag["accel_saturated"]
        horizontal_saturated = horizontal_saturated or sub_diag["horizontal_speed_saturated"]
        vertical_saturated = vertical_saturated or sub_diag["vertical_rate_saturated"]

    diagnostics = {
        "accel_used": max_accel_used,
        "turn_rate_used_deg": 0.0,
        "accel_saturated": bool(accel_saturated),
        "turn_saturated": False,
        "horizontal_speed_saturated": bool(horizontal_saturated),
        "vertical_rate_saturated": bool(vertical_saturated),
        "climb_saturated": bool(vertical_saturated),
    }
    return pos.astype(np.float32), vel.astype(np.float32), diagnostics
