"""Continuous (swept) moving-sphere contact detection for one simulation step.

Two bodies A and B move linearly over a step, tau in [0, 1]:

    pA(tau) = pA0 + tau * (pA1 - pA0)
    pB(tau) = pB0 + tau * (pB1 - pB0)

Their relative position is r(tau) = r0 + tau * dr with
r0 = pA0 - pB0 and dr = (pA1 - pA0) - (pB1 - pB0). Contact is the FIRST
tau in [0, 1] with ||r(tau)|| <= R.

The solution is expressed through the closest point of approach (CPA),
tau_cpa = -(r0 . dr) / (dr . dr) and d_cpa = ||r(tau_cpa)||, instead of the
textbook quadratic formula. The entry time is then

    tau_in = tau_cpa - sqrt(max(R^2 - d_cpa^2, 0) / (dr . dr)),

which has no catastrophic cancellation for fast, nearly tangent passes.

Used by SoldierEnv for attacker->asset capture and defender->attacker
interception under mission_contact_mode == "swept". Pure, RNG-free, and
independent of SoldierEnv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Absolute distance tolerance (m) for grazing/tangent contact; far below any
# physically meaningful radius but above float64 round-off at ~100 m scale.
CONTACT_DISTANCE_TOLERANCE = 1e-9
# dr . dr below this is treated as zero relative displacement.
_MIN_RELATIVE_DISPLACEMENT_SQ = 1e-18


@dataclass(frozen=True)
class SweptContact:
    """First contact within one step. `fraction` is tau in [0, 1]."""

    fraction: float
    distance: float  # ||r(fraction)|| (<= radius + tolerance)
    position_a: np.ndarray
    position_b: np.ndarray

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.position_a + self.position_b)


def first_contact(
    p0_a: np.ndarray,
    p1_a: np.ndarray,
    p0_b: np.ndarray,
    p1_b: np.ndarray,
    radius: float,
) -> SweptContact | None:
    """First contact of two linearly moving points within `radius`, or None.

    Handles: already inside at tau=0 (fraction 0), zero relative
    displacement, tangent contact (within CONTACT_DISTANCE_TOLERANCE),
    pass-through (both endpoints outside), and both bodies moving.
    """
    if radius < 0.0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    p0_a = np.asarray(p0_a, dtype=np.float64)
    p1_a = np.asarray(p1_a, dtype=np.float64)
    p0_b = np.asarray(p0_b, dtype=np.float64)
    p1_b = np.asarray(p1_b, dtype=np.float64)

    r0 = p0_a - p0_b
    dr = (p1_a - p0_a) - (p1_b - p0_b)
    limit = radius + CONTACT_DISTANCE_TOLERANCE

    d0 = float(np.linalg.norm(r0))
    if d0 <= limit:
        return _make(0.0, p0_a, p1_a, p0_b, p1_b, d0)

    a = float(np.dot(dr, dr))
    if a < _MIN_RELATIVE_DISPLACEMENT_SQ:
        return None

    tau_cpa = -float(np.dot(r0, dr)) / a
    d_cpa = float(np.linalg.norm(r0 + tau_cpa * dr))
    if d_cpa > limit:
        return None

    half_chord = float(np.sqrt(max(radius * radius - d_cpa * d_cpa, 0.0) / a))
    tau_in = tau_cpa - half_chord
    tau_out = tau_cpa + half_chord
    if tau_out < 0.0 or tau_in > 1.0:
        return None

    tau = float(np.clip(tau_in, 0.0, 1.0))
    distance = float(np.linalg.norm(r0 + tau * dr))
    return _make(tau, p0_a, p1_a, p0_b, p1_b, distance)


def _make(tau: float, p0_a, p1_a, p0_b, p1_b, distance: float) -> SweptContact:
    return SweptContact(
        fraction=tau,
        distance=distance,
        position_a=p0_a + tau * (p1_a - p0_a),
        position_b=p0_b + tau * (p1_b - p0_b),
    )
