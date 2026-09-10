"""
Baseline temporal predictors (Phase 29 A/B): Constant-Velocity and
Constant-Acceleration prediction, evaluated against the causal history
windows produced by `trajectory_dataset.py`. These are NOT guidance
controllers -- they only predict a future hostile position from recent
estimated state, for comparison against the learned GRU-residual
predictor (Phase 30).
"""

from __future__ import annotations

import numpy as np


def cv_predict(sample: dict) -> dict[float, np.ndarray]:
    """Constant-velocity prediction: p_CV(t+h) = p_hat_t + v_hat_t * h.
    Uses ONLY the current step's estimated position/velocity (already
    computed and stored on the sample by `trajectory_dataset.build_samples`
    to guarantee no future leakage)."""
    return {h: pred.copy() for h, pred in sample["cv_pred"].items()}


def ca_predict(sample: dict, dt: float) -> dict[float, np.ndarray]:
    """Constant-acceleration prediction using the SAME simple
    finite-difference acceleration methodology as the diagnostic study's
    `ConstantAccelerationLeadPolicy`: acceleration estimated from the two
    most recent velocity estimates in the causal history window.

    p_CA(t+h) = p_hat_t + v_hat_t*h + 0.5*a_hat*h^2

    Falls back to a zero acceleration estimate (== CV) when fewer than two
    detected steps of history are available, mirroring the diagnostic
    study's "no_acceleration_estimate_fallback" behavior.
    """
    history = sample["history"]
    mask = sample["history_mask"]
    est_pos = sample["est_pos_now"]
    est_vel = sample["est_vel_now"]

    if len(mask) >= 2 and mask[-1] == 1 and mask[-2] == 1:
        v_now = history[-1][3:6]
        v_prev = history[-2][3:6]
        accel = (v_now - v_prev) / dt
    else:
        accel = np.zeros(3)

    return {h: est_pos + est_vel * h + 0.5 * accel * h * h for h in sample["cv_pred"]}
