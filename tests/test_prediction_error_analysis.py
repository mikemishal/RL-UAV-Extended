"""Tests for retrospective Lead prediction-error analysis (Phase 8 / 19).

Covers:
 - exact interpolation at an integer sample time
 - fractional-time interpolation between two samples
 - unavailable-beyond-episode-end handling
 - a known synthetic trajectory error computation
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.learning_benefit.prediction_error import (
    TrajectorySample,
    compute_ca_prediction_error,
    compute_cv_prediction_error,
    interpolate_true_position,
)


def _samples():
    # Straight-line motion at 2 m/s along +x, dt=0.5s.
    dt = 0.5
    return [
        TrajectorySample(step_index=i, sim_time=i * dt, position=np.array([2.0 * i * dt, 0.0, 0.0]))
        for i in range(6)
    ]


def test_interpolation_at_integer_sample_time():
    samples = _samples()
    p = interpolate_true_position(samples, query_time=1.0)  # exactly sample index 2
    assert np.allclose(p, [2.0, 0.0, 0.0])


def test_interpolation_at_fractional_time():
    samples = _samples()
    p = interpolate_true_position(samples, query_time=0.75)  # between t=0.5 (idx1) and t=1.0 (idx2)
    assert np.allclose(p, [1.5, 0.0, 0.0])


def test_interpolation_unavailable_beyond_episode_end():
    samples = _samples()  # last sim_time = 2.5
    assert interpolate_true_position(samples, query_time=10.0) is None


def test_interpolation_unavailable_before_episode_start():
    samples = _samples()
    assert interpolate_true_position(samples, query_time=-1.0) is None


def test_cv_prediction_error_known_synthetic_trajectory():
    samples = _samples()
    dt = 0.5
    # At step 0 (t=0), estimated position/velocity match truth exactly:
    # perfect CV prediction => zero error.
    err = compute_cv_prediction_error(
        samples, step_index=0, dt=dt,
        estimated_position_at_t=np.array([0.0, 0.0, 0.0]),
        estimated_velocity_at_t=np.array([2.0, 0.0, 0.0]),
        intercept_time=1.0,
    )
    assert err is not None
    assert err < 1e-9


def test_cv_prediction_error_with_known_offset():
    samples = _samples()
    dt = 0.5
    # Estimated velocity is wrong (1.0 instead of true 2.0 m/s): at tau=1.0,
    # predicted x = 0 + 1.0*1.0 = 1.0; true x at t=1.0 is 2.0*1.0=2.0 => error=1.0
    err = compute_cv_prediction_error(
        samples, step_index=0, dt=dt,
        estimated_position_at_t=np.array([0.0, 0.0, 0.0]),
        estimated_velocity_at_t=np.array([1.0, 0.0, 0.0]),
        intercept_time=1.0,
    )
    assert err is not None
    assert abs(err - 1.0) < 1e-9


def test_cv_prediction_error_unavailable_when_beyond_end():
    samples = _samples()
    err = compute_cv_prediction_error(
        samples, step_index=0, dt=0.5,
        estimated_position_at_t=np.zeros(3), estimated_velocity_at_t=np.array([2.0, 0.0, 0.0]),
        intercept_time=100.0,
    )
    assert err is None


def test_ca_prediction_error_matches_true_quadratic_motion():
    # Build a trajectory with constant acceleration a=1 m/s^2 along x,
    # starting at rest: x(t) = 0.5*1*t^2.
    dt = 0.5
    samples = [
        TrajectorySample(step_index=i, sim_time=i * dt, position=np.array([0.5 * (i * dt) ** 2, 0.0, 0.0]))
        for i in range(6)
    ]
    err = compute_ca_prediction_error(
        samples, step_index=0, dt=dt,
        estimated_position_at_t=np.array([0.0, 0.0, 0.0]),
        estimated_velocity_at_t=np.array([0.0, 0.0, 0.0]),
        estimated_acceleration_at_t=np.array([1.0, 0.0, 0.0]),
        intercept_time=1.0,
    )
    assert err is not None
    assert err < 1e-9


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
