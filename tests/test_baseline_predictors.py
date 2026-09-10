"""Tests for CV/CA baseline temporal predictors (Phase 29 A/B) using
synthetic trajectories with known closed-form answers."""

import numpy as np

from experiments.fa_lapg_foundation.baseline_predictors import ca_predict, cv_predict
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, EpisodeRecord, EpisodeStep, build_samples

DT = 0.5


def _episode_with_kinematics(n_steps: int, v0: np.ndarray, a: np.ndarray) -> EpisodeRecord:
    """Synthetic hostile motion following p(t) = p0 + v0*t + 0.5*a*t^2,
    with the "estimated" state exactly equal to the analytic state at
    each step (noiseless, so CV/CA predictions have known closed forms)."""
    steps = []
    p0 = np.array([0.0, 0.0, 0.0])
    for i in range(n_steps):
        t = i * DT
        pos = p0 + v0 * t + 0.5 * a * t * t
        vel = v0 + a * t
        features = np.zeros(FEATURE_DIM)
        features[0:3] = pos
        features[3:6] = vel
        steps.append(EpisodeStep(i, t, True, features, pos.copy()))
    return EpisodeRecord(seed=1, scenario="synthetic", dt=DT, steps=steps)


def test_cv_predict_matches_constant_velocity_closed_form():
    v0 = np.array([2.0, -1.0, 0.0])
    episode = _episode_with_kinematics(20, v0=v0, a=np.zeros(3))
    samples = build_samples(episode, horizons=(0.5, 1.0, 2.0))
    sample = samples[10]
    for h in (0.5, 1.0, 2.0):
        expected = sample["est_pos_now"] + sample["est_vel_now"] * h
        assert np.allclose(cv_predict(sample)[h], expected)


def test_cv_predict_underestimates_under_true_acceleration():
    v0 = np.array([2.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0])
    episode = _episode_with_kinematics(20, v0=v0, a=a)
    samples = build_samples(episode, horizons=(2.0,))
    sample = samples[10]
    target = sample["targets"][2.0]
    cv_pred = cv_predict(sample)[2.0]
    assert target is not None
    # CV ignores acceleration -> must be measurably wrong when a != 0
    assert np.linalg.norm(target - cv_pred) > 0.1


def test_ca_predict_recovers_constant_acceleration_exactly():
    v0 = np.array([2.0, 0.0, 0.0])
    a = np.array([1.0, -0.5, 0.0])
    episode = _episode_with_kinematics(20, v0=v0, a=a)
    samples = build_samples(episode, horizons=(0.5, 1.0, 2.0, 3.0))
    sample = samples[10]  # enough history for a 2-point finite-difference accel estimate
    for h in (0.5, 1.0, 2.0, 3.0):
        target = sample["targets"][h]
        assert target is not None
        ca_pred = ca_predict(sample, dt=DT)[h]
        # exact under true constant acceleration + noiseless estimates
        assert np.allclose(target, ca_pred, atol=1e-6)


def test_ca_predict_falls_back_to_cv_with_less_than_two_history_steps():
    v0 = np.array([2.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0])
    episode = _episode_with_kinematics(20, v0=v0, a=a)
    samples = build_samples(episode, horizons=(1.0,))
    first_sample = samples[0]  # only one detected step so far -> no accel estimate
    assert np.allclose(ca_predict(first_sample, dt=DT)[1.0], cv_predict(first_sample)[1.0])
