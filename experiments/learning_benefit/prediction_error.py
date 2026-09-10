"""Phase 8: retrospective (non-causal, offline) Lead future-prediction-error
analysis.

For a REALIZED episode (the true hostile trajectory already happened,
recorded step-by-step), and for every step `t` where Standard Lead (or
CA-Lead) reported a valid intercept time `tau`, compute:

    E_CV(t, tau) = || p_true(t + tau) - p_CV(t + tau) ||
    p_CV(t + tau) = p_hat(t) + v_hat(t) * tau

using LINEAR INTERPOLATION between recorded true-position samples when
`t + tau` falls between two integer step indices, and marking the sample
UNAVAILABLE (None) if `t + tau` lies beyond the end of the realized
trajectory. This is a pure, non-causal ANALYSIS computation -- it never
feeds back into the policy or environment, and it is computed strictly
AFTER an episode has fully run (using only the stored trajectory), so it
cannot perturb RNG state or the realized hostile process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    """One recorded true-hostile-position sample at (0-indexed) step `t`,
    at simulation time `t * dt`."""

    step_index: int
    sim_time: float
    position: np.ndarray


def interpolate_true_position(samples: list[TrajectorySample], query_time: float) -> np.ndarray | None:
    """Linear interpolation of the true hostile position at `query_time`
    (seconds) from a time-ordered list of `samples`. Returns None if
    `query_time` lies before the first sample or after the last (the
    episode ended before that time -- truth is not fabricated/extrapolated)."""
    if not samples:
        return None
    if query_time < samples[0].sim_time - 1e-9 or query_time > samples[-1].sim_time + 1e-9:
        return None
    if len(samples) == 1:
        return samples[0].position.copy()

    for i in range(len(samples) - 1):
        t0, t1 = samples[i].sim_time, samples[i + 1].sim_time
        if t0 - 1e-9 <= query_time <= t1 + 1e-9:
            if abs(t1 - t0) < 1e-12:
                return samples[i].position.copy()
            frac = (query_time - t0) / (t1 - t0)
            return samples[i].position + frac * (samples[i + 1].position - samples[i].position)
    return None  # unreachable for a sorted, gap-free sample list


def compute_cv_prediction_error(
    samples: list[TrajectorySample],
    step_index: int,
    dt: float,
    estimated_position_at_t: np.ndarray,
    estimated_velocity_at_t: np.ndarray,
    intercept_time: float,
) -> float | None:
    """E_CV(t, tau) = || p_true(t+tau) - p_CV(t+tau) ||, or None if
    t + tau lies beyond the realized episode's recorded trajectory."""
    query_time = step_index * dt + intercept_time
    p_true = interpolate_true_position(samples, query_time)
    if p_true is None:
        return None
    p_cv = np.asarray(estimated_position_at_t, dtype=np.float64) + np.asarray(estimated_velocity_at_t, dtype=np.float64) * intercept_time
    return float(np.linalg.norm(p_true - p_cv))


def compute_ca_prediction_error(
    samples: list[TrajectorySample],
    step_index: int,
    dt: float,
    estimated_position_at_t: np.ndarray,
    estimated_velocity_at_t: np.ndarray,
    estimated_acceleration_at_t: np.ndarray,
    intercept_time: float,
) -> float | None:
    """E_CA(t, tau) using the constant-acceleration prediction
    p_hat(t) + v_hat(t)*tau + 0.5*a_hat(t)*tau^2, or None if t + tau lies
    beyond the realized episode's recorded trajectory."""
    query_time = step_index * dt + intercept_time
    p_true = interpolate_true_position(samples, query_time)
    if p_true is None:
        return None
    p_ca = (
        np.asarray(estimated_position_at_t, dtype=np.float64)
        + np.asarray(estimated_velocity_at_t, dtype=np.float64) * intercept_time
        + 0.5 * np.asarray(estimated_acceleration_at_t, dtype=np.float64) * intercept_time ** 2
    )
    return float(np.linalg.norm(p_true - p_ca))
