"""Pure constant-acceleration (CA) Lead-intercept quartic mathematics.

Interception condition with constant target acceleration `a`:

    || r + v*tau + 0.5*a*tau^2 || = v_d * tau

Squaring and expanding (r, v, a are the target's relative position,
velocity, and acceleration relative to the defender) yields the quartic:

    C4*tau^4 + C3*tau^3 + C2*tau^2 + C1*tau + C0 = 0

    C4 = 0.25 * (a . a)
    C3 = (v . a)
    C2 = (v . v) + (r . a) - v_d^2
    C1 = 2 * (r . v)
    C0 = (r . r)

When a = 0 exactly, C4 = C3 = 0 and (C2, C1, C0) reduce to EXACTLY the
constant-velocity quadratic coefficients (a_cv, b_cv, c_cv) used by
`uav_defend.policies.baseline.lead_math.solve_cv_lead` -- this module
special-cases that regime and delegates directly to `solve_cv_lead` for
bit-identical behavior rather than relying on quartic root-finding to
numerically reproduce the quadratic case.

Root selection: real roots (numerical tolerance `imag_tolerance` on the
imaginary part), positive roots (`> eps`), finite roots; the EARLIEST
(minimum) valid positive root is chosen. If no valid root exists, the
caller falls back to the existing Standard (CV) Lead behavior -- this
module never fabricates a solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.policies.baseline.lead_math import LeadSolution, pure_pursuit_direction, solve_cv_lead

DEFAULT_IMAG_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CALeadSolution:
    """Result of `solve_ca_lead()`."""

    action: np.ndarray
    guidance_mode: str
    used_acceleration: bool
    valid: bool
    fallback_used: bool
    relative_position: np.ndarray | None
    acceleration: np.ndarray | None
    quartic_coefficients: tuple[float, float, float, float, float] | None
    intercept_time: float | None
    intercept_point: np.ndarray | None


def solve_quartic_intercept_time(
    r: np.ndarray, v: np.ndarray, a: np.ndarray, v_d: float, eps: float,
    imag_tolerance: float = DEFAULT_IMAG_TOLERANCE,
) -> tuple[float | None, tuple[float, float, float, float, float]]:
    """Solve the CA-Lead quartic for the earliest positive real root.

    Returns (t_intercept_or_None, (C4, C3, C2, C1, C0)).
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    c4 = 0.25 * float(np.dot(a, a))
    c3 = float(np.dot(v, a))
    c2 = float(np.dot(v, v) + np.dot(r, a) - v_d * v_d)
    c1 = float(2.0 * np.dot(r, v))
    c0 = float(np.dot(r, r))
    coefficients = (c4, c3, c2, c1, c0)

    roots = np.roots([c4, c3, c2, c1, c0])
    positive_real_roots = []
    for root in roots:
        if not np.isfinite(root.real) or not np.isfinite(root.imag):
            continue  # complex/degenerate infinite root: not a valid intercept time
        tol = imag_tolerance * max(1.0, abs(root.real))
        if abs(root.imag) > tol:
            continue  # meaningfully complex root: ignore
        if root.real > eps:
            positive_real_roots.append(float(root.real))

    if not positive_real_roots:
        return None, coefficients
    return min(positive_real_roots), coefficients


def solve_ca_lead(
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_acceleration: np.ndarray,
    defender_position: np.ndarray,
    v_d: float,
    eps: float,
    imag_tolerance: float = DEFAULT_IMAG_TOLERANCE,
) -> CALeadSolution:
    """
    Constant-acceleration Lead-intercept solve. Falls back to the existing
    Standard (constant-velocity) Lead behavior -- via `solve_cv_lead`, NOT
    a duplicated implementation -- whenever:
      - the target position coincides with the defender (degenerate), or
      - the acceleration is (numerically) exactly zero (delegates for
        bit-identical CV behavior), or
      - the quartic has no valid (real, positive, finite) root.
    """
    defender_position = np.asarray(defender_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    target_acceleration = np.asarray(target_acceleration, dtype=np.float64)

    r = target_position - defender_position
    R = float(np.linalg.norm(r))

    if R <= eps:
        return CALeadSolution(
            action=pure_pursuit_direction(target_position, defender_position, eps),
            guidance_mode="degenerate_fallback", used_acceleration=False, valid=False, fallback_used=True,
            relative_position=r.astype(np.float32), acceleration=None, quartic_coefficients=None,
            intercept_time=None, intercept_point=None,
        )

    accel_norm_sq = float(np.dot(target_acceleration, target_acceleration))
    if accel_norm_sq <= eps * eps:
        # Exactly (numerically) zero acceleration: delegate to the SAME
        # constant-velocity solver Standard Lead uses, for bit-identical
        # behavior in this regime (see module docstring).
        cv_solution = solve_cv_lead(target_position, target_velocity, defender_position, v_d, eps)
        return CALeadSolution(
            action=cv_solution.action, guidance_mode=cv_solution.guidance_mode,
            used_acceleration=False, valid=cv_solution.valid, fallback_used=True,
            relative_position=cv_solution.relative_position, acceleration=None, quartic_coefficients=None,
            intercept_time=cv_solution.intercept_time, intercept_point=cv_solution.intercept_point,
        )

    t_intercept, coefficients = solve_quartic_intercept_time(
        r, target_velocity, target_acceleration, v_d, eps, imag_tolerance,
    )

    if t_intercept is None:
        # No valid CA root: fall back to Standard (CV) Lead behavior.
        cv_solution = solve_cv_lead(target_position, target_velocity, defender_position, v_d, eps)
        return CALeadSolution(
            action=cv_solution.action, guidance_mode=cv_solution.guidance_mode,
            used_acceleration=False, valid=cv_solution.valid, fallback_used=True,
            relative_position=cv_solution.relative_position, acceleration=target_acceleration.astype(np.float32),
            quartic_coefficients=coefficients,
            intercept_time=cv_solution.intercept_time, intercept_point=cv_solution.intercept_point,
        )

    p_intercept = target_position + target_velocity * t_intercept + 0.5 * target_acceleration * t_intercept ** 2
    if not np.all(np.isfinite(p_intercept)):
        cv_solution = solve_cv_lead(target_position, target_velocity, defender_position, v_d, eps)
        return CALeadSolution(
            action=cv_solution.action, guidance_mode=cv_solution.guidance_mode,
            used_acceleration=False, valid=cv_solution.valid, fallback_used=True,
            relative_position=cv_solution.relative_position, acceleration=target_acceleration.astype(np.float32),
            quartic_coefficients=coefficients,
            intercept_time=cv_solution.intercept_time, intercept_point=cv_solution.intercept_point,
        )

    lead_vector = p_intercept - defender_position
    lead_norm = float(np.linalg.norm(lead_vector))
    if lead_norm <= eps or not np.isfinite(lead_norm):
        cv_solution = solve_cv_lead(target_position, target_velocity, defender_position, v_d, eps)
        return CALeadSolution(
            action=cv_solution.action, guidance_mode=cv_solution.guidance_mode,
            used_acceleration=False, valid=cv_solution.valid, fallback_used=True,
            relative_position=cv_solution.relative_position, acceleration=target_acceleration.astype(np.float32),
            quartic_coefficients=coefficients,
            intercept_time=cv_solution.intercept_time, intercept_point=cv_solution.intercept_point,
        )

    action = (lead_vector / lead_norm).astype(np.float32)
    return CALeadSolution(
        action=action, guidance_mode="ca_lead", used_acceleration=True, valid=True, fallback_used=False,
        relative_position=r.astype(np.float32), acceleration=target_acceleration.astype(np.float32),
        quartic_coefficients=coefficients,
        intercept_time=t_intercept, intercept_point=p_intercept.astype(np.float32),
    )
