"""Pure constant-velocity Lead-intercept mathematics, factored out of
`uav_defend.policies.baseline.lead_intercept_policy.LeadInterceptPolicy`
(Phase 5 of the learning-benefit diagnostic study; see
docs/learning_benefit_analysis.md if present) so it can be reused by
diagnostic/analysis controllers -- in particular the True-State
constant-velocity Lead controller (`uav_defend.policies.analysis.
true_state_lead_policy`) -- WITHOUT duplicating the analytical geometry.

This module is intentionally policy-independent: it takes a target
position/velocity (estimated OR true) and a defender position, and returns
the SAME constant-velocity intercept geometry `LeadInterceptPolicy` has
always used. It does not know or care where the target state came from.

CRITICAL: `LeadInterceptPolicy.act()` was refactored to call
`solve_cv_lead()` below, but its escort/no-target/no-velocity fallback
handling (which happens BEFORE a target velocity is known) is unchanged.
See tests/test_lead_math_regression.py for the regression proof that this
refactor preserves byte-for-bit-identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeadSolution:
    """Result of `solve_cv_lead()`.

    `valid` is True only for guidance_mode == "lead" (a full analytical
    solution was found); every other guidance_mode is a fallback
    (`fallback_used=True`), and `action` is the pure-pursuit direction
    toward `target_position` in that case.
    """

    action: np.ndarray
    guidance_mode: str
    valid: bool
    fallback_used: bool
    relative_position: np.ndarray | None
    quadratic_a: float | None
    quadratic_b: float | None
    quadratic_c: float | None
    discriminant: float | None
    intercept_time: float | None
    intercept_point: np.ndarray | None


def solve_quadratic_intercept_time(
    a: float, b: float, c: float, eps: float
) -> tuple[float | None, float | None, str]:
    """
    Solve a t^2 + b t + c = 0 for the earliest positive real root,
    robustly handling the normal-quadratic, near-linear, and
    floating-point-roundoff-near-zero-discriminant cases.

    Returns:
        (t_intercept, discriminant, reason) where:
            t_intercept: earliest positive real root, or None if none exists.
            discriminant: b^2-4ac (normal-quadratic branch only), else None.
            reason: "ok" | "no_real_solution" | "no_positive_root"
    """
    if abs(a) > eps:
        # CASE A: normal quadratic.
        discriminant = b * b - 4.0 * a * c
        if discriminant < -eps:
            return None, discriminant, "no_real_solution"
        if discriminant < 0.0:
            discriminant = 0.0  # floating-point roundoff clamp

        sqrt_D = float(np.sqrt(discriminant))
        sign_b = 1.0 if b >= 0.0 else -1.0
        q = -0.5 * (b + sign_b * sqrt_D)

        roots: list[float] = []
        if abs(q) > eps:
            # Numerically stable formulation (avoids catastrophic
            # cancellation in (-b +/- sqrt(D))/(2a) when b and sqrt_D
            # are close in magnitude).
            roots.append(q / a)
            roots.append(c / q)
        else:
            # q ~ 0: fall back to the direct quadratic formula, which is
            # safe here precisely because q's cancellation risk is what
            # the stable form exists to avoid, and q~0 means that risk
            # is not present in this branch.
            roots.append((-b + sqrt_D) / (2.0 * a))
            roots.append((-b - sqrt_D) / (2.0 * a))

        positive_roots = [t for t in roots if np.isfinite(t) and t > eps]
        if not positive_roots:
            return None, discriminant, "no_positive_root"
        return min(positive_roots), discriminant, "ok"

    # CASE B: near-linear geometry (a ~ 0).
    if abs(b) > eps:
        t = -c / b
        if np.isfinite(t) and t > eps:
            return t, None, "ok"
        return None, None, "no_positive_root"

    # abs(a) <= eps and abs(b) <= eps: no useful predicted solution.
    return None, None, "no_real_solution"


def pure_pursuit_direction(target: np.ndarray | None, defender_position: np.ndarray, eps: float) -> np.ndarray:
    """Pure-pursuit fallback direction toward `target` (zero vector if
    `target` is None or coincides with `defender_position`)."""
    if target is None:
        return np.zeros(3, dtype=np.float32)
    target = np.asarray(target, dtype=np.float64)
    direction = target - np.asarray(defender_position, dtype=np.float64)
    dist = float(np.linalg.norm(direction))
    if dist < eps:
        return np.zeros(3, dtype=np.float32)
    return (direction / dist).astype(np.float32)


def get_target_state_estimate(
    info: dict, state_source: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Return (target_position, target_velocity) using ONLY legitimate
    environment info fields for the given `state_source` ("measurement" or
    "kalman"). NEVER reads info["enemy_pos"] / info["enemy_vel"] (ground
    truth). Shared by every ESTIMATED-state Lead-family controller
    (`LeadInterceptPolicy`, the Constant-Acceleration Lead diagnostic
    controller) so this lookup is not duplicated.
    """
    if state_source == "kalman":
        e_hat = info.get("e_hat")
        v_hat = info.get("v_hat")
        if e_hat is None:
            return None, None
        target_pos = np.asarray(e_hat, dtype=np.float64)
        target_vel = np.asarray(v_hat, dtype=np.float64) if v_hat is not None else None
        return target_pos, target_vel

    # state_source == "measurement": consume the environment's
    # standardized finite-difference measurement velocity -- the same
    # estimator used by the Direct observation and shared across all
    # measurement-mode controllers. No independent per-policy finite
    # differencing.
    meas = info.get("enemy_measurement")
    if meas is None:
        return None, None
    target_pos = np.asarray(meas, dtype=np.float64)
    if info.get("enemy_measurement_velocity_valid", False):
        target_vel = np.asarray(info.get("enemy_measurement_velocity"), dtype=np.float64)
    else:
        target_vel = None
    return target_pos, target_vel


def solve_cv_lead(
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    defender_position: np.ndarray,
    v_d: float,
    eps: float,
) -> LeadSolution:
    """
    Pure 3-D constant-velocity Lead-intercept solve, given a KNOWN target
    position/velocity (may be an estimate OR ground truth -- this function
    does not care) and defender position.

    Intercept geometry (see `LeadInterceptPolicy` for the full derivation):
        r = target_position - defender_position
        Assumed constant target motion: p(t) = target_position + target_velocity * t
        Ideal interceptor at speed s = v_d: ||r + target_velocity * t|| = s * t
        => quadratic  a t^2 + b t + c = 0  where
               a = target_velocity . target_velocity - s^2
               b = 2 * (r . target_velocity)
               c = r . r
        Earliest positive real root t_intercept is selected.

    Callers are expected to have ALREADY handled the "no target known yet"
    / "no velocity estimate yet" cases (those precede having both a target
    position AND velocity, and are policy-specific escort/fallback
    behavior -- see `LeadInterceptPolicy.act()`).
    """
    defender_position = np.asarray(defender_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)

    r = target_position - defender_position
    R = float(np.linalg.norm(r))

    if R <= eps:
        return LeadSolution(
            action=pure_pursuit_direction(target_position, defender_position, eps),
            guidance_mode="degenerate_fallback", valid=False, fallback_used=True,
            relative_position=r.astype(np.float32),
            quadratic_a=None, quadratic_b=None, quadratic_c=None, discriminant=None,
            intercept_time=None, intercept_point=None,
        )

    s = v_d
    a = float(np.dot(target_velocity, target_velocity) - s * s)
    b = float(2.0 * np.dot(r, target_velocity))
    c = float(np.dot(r, r))

    t_intercept, discriminant, reason = solve_quadratic_intercept_time(a, b, c, eps)

    if reason != "ok":
        mode = "no_real_solution_fallback" if reason == "no_real_solution" else "no_positive_root_fallback"
        return LeadSolution(
            action=pure_pursuit_direction(target_position, defender_position, eps),
            guidance_mode=mode, valid=False, fallback_used=True,
            relative_position=r.astype(np.float32),
            quadratic_a=a, quadratic_b=b, quadratic_c=c, discriminant=discriminant,
            intercept_time=None, intercept_point=None,
        )

    # Predicted intercept point. NOT clipped to the engagement volume --
    # this is a mathematical aim point; only its direction is used.
    p_intercept = target_position + target_velocity * t_intercept
    if not np.all(np.isfinite(p_intercept)):
        return LeadSolution(
            action=pure_pursuit_direction(target_position, defender_position, eps),
            guidance_mode="degenerate_fallback", valid=False, fallback_used=True,
            relative_position=r.astype(np.float32),
            quadratic_a=a, quadratic_b=b, quadratic_c=c, discriminant=discriminant,
            intercept_time=t_intercept, intercept_point=None,
        )

    lead_vector = p_intercept - defender_position
    lead_norm = float(np.linalg.norm(lead_vector))
    if lead_norm <= eps or not np.isfinite(lead_norm):
        return LeadSolution(
            action=pure_pursuit_direction(target_position, defender_position, eps),
            guidance_mode="degenerate_fallback", valid=False, fallback_used=True,
            relative_position=r.astype(np.float32),
            quadratic_a=a, quadratic_b=b, quadratic_c=c, discriminant=discriminant,
            intercept_time=t_intercept, intercept_point=p_intercept.astype(np.float32),
        )

    action = (lead_vector / lead_norm).astype(np.float32)
    return LeadSolution(
        action=action, guidance_mode="lead", valid=True, fallback_used=False,
        relative_position=r.astype(np.float32),
        quadratic_a=a, quadratic_b=b, quadratic_c=c, discriminant=discriminant,
        intercept_time=t_intercept, intercept_point=p_intercept.astype(np.float32),
    )
