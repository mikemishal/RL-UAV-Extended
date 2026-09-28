"""
Tests for the fixed-size directional obstacle-sector observation (journal
extension, Phase 13 of the obstacle-aware-retraining sub-task). Covers the
14 numbered scenarios from the task specification.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.obstacles.sector_features import (
    N_SECTORS,
    SECTOR_FEATURE_DIM,
    SECTOR_NAMES,
    compute_sector_features,
)
from uav_defend.obstacles.perception import (
    PerceivedDynamicObstacle,
    PerceivedObstacleSet,
    PerceivedStaticObstacle,
)
from experiments.journal_clutter_protocol import build_canonical_env_config
from uav_defend.envs import SoldierEnv
from uav_defend.envs.obstacle_observation_wrapper import ObstacleSectorObservationWrapper

R = 15.0


def _static(min_c, max_c, obstacle_id=0) -> PerceivedStaticObstacle:
    return PerceivedStaticObstacle(
        id=obstacle_id, min_corner=np.asarray(min_c, dtype=np.float64),
        max_corner=np.asarray(max_c, dtype=np.float64), source="local_sensor",
    )


def _dynamic(position, velocity=(0.0, 0.0, 0.0), radius=1.0, obstacle_id=0) -> PerceivedDynamicObstacle:
    return PerceivedDynamicObstacle(
        id=obstacle_id, position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64), radius=radius, obstacle_type="bird",
    )


# 1. fixed observation dimension regardless of obstacle count -----------------

def test_fixed_dimension_regardless_of_obstacle_count():
    empty = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), PerceivedObstacleSet(), R)
    many = PerceivedObstacleSet(static=tuple(
        _static((5 + i, -1, -1), (7 + i, 1, 1), obstacle_id=i) for i in range(12)
    ))
    full = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), many, R)
    assert empty.shape == (SECTOR_FEATURE_DIM,) == full.shape
    assert SECTOR_FEATURE_DIM == N_SECTORS * 2 == 20


# 2. no obstacle gives deterministic empty-sector values ----------------------

def test_no_obstacle_gives_all_ones():
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), PerceivedObstacleSet(), R)
    assert np.allclose(feats, 1.0)


# 3. obstacle directly ahead appears in front sector --------------------------

def test_obstacle_directly_ahead_in_front_sector():
    obs = _static((5.0, -1, -1), (7.0, 1, 1))
    pset = PerceivedObstacleSet(static=(obs,))
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    front = SECTOR_NAMES.index("front")
    assert feats[front, 0] < 1.0
    for i in range(N_SECTORS):
        if i != front:
            assert feats[i, 0] == 1.0


# 4. left/right rotation works correctly --------------------------------------

def test_left_right_rotation():
    obs = _static((-1, 5.0, -1), (1, 7.0, 1))  # world +y from origin
    pset = PerceivedObstacleSet(static=(obs,))
    # Defender heading +x: world +y is a LEFT turn -> "left" sector.
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    assert feats[SECTOR_NAMES.index("left"), 0] < 1.0
    # Defender heading +y (rotated 90 deg): the SAME world obstacle is now
    # directly ahead ("front").
    feats2 = compute_sector_features(np.zeros(3), np.array([0.0, 1.0, 0]), pset, R).reshape(N_SECTORS, 2)
    assert feats2[SECTOR_NAMES.index("front"), 0] < 1.0
    # Defender heading -x: world +y is now a RIGHT turn -> "right" sector.
    feats3 = compute_sector_features(np.zeros(3), np.array([-1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    assert feats3[SECTOR_NAMES.index("right"), 0] < 1.0


# 5. vertical obstacle relation maps correctly --------------------------------

def test_vertical_obstacle_mapping():
    above = _static((-1, -1, 10.0), (1, 1, 12.0))
    below = _static((-1, -1, -12.0), (1, 1, -10.0), obstacle_id=1)
    pset = PerceivedObstacleSet(static=(above, below))
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    assert feats[SECTOR_NAMES.index("up"), 0] < 1.0
    assert feats[SECTOR_NAMES.index("down"), 0] < 1.0
    for name in ("front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right"):
        assert feats[SECTOR_NAMES.index(name), 0] == 1.0


# 6. nearest relevant obstacle selected consistently --------------------------

def test_nearest_obstacle_in_sector_wins():
    near = _static((5.0, -1, -1), (6.0, 1, 1), obstacle_id=0)
    far = _static((9.0, -1, -1), (10.0, 1, 1), obstacle_id=1)
    pset = PerceivedObstacleSet(static=(far, near))  # far listed first, must not win
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    front = SECTOR_NAMES.index("front")
    expected = near.min_corner[0] / R  # nearer obstacle's distance normalized
    assert np.isclose(feats[front, 0], expected, atol=1e-6)


# 7. static and dynamic channels remain separate ------------------------------

def test_static_and_dynamic_channels_independent():
    static_obs = _static((5.0, -1, -1), (7.0, 1, 1))
    dynamic_obs = _dynamic((0.0, 5.0, 0.0), radius=1.0)  # left sector, dynamic only
    pset = PerceivedObstacleSet(static=(static_obs,), dynamic=(dynamic_obs,))
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    front = SECTOR_NAMES.index("front")
    left = SECTOR_NAMES.index("left")
    assert feats[front, 0] < 1.0 and feats[front, 1] == 1.0  # static only in front
    assert feats[left, 1] < 1.0 and feats[left, 0] == 1.0    # dynamic only in left


# 8. dynamic closing rate -- N/A (2-channel design omits it, documented) -----

def test_dynamic_channel_uses_center_minus_radius_clearance():
    dynamic_obs = _dynamic((5.0, 0.0, 0.0), radius=2.0)
    pset = PerceivedObstacleSet(dynamic=(dynamic_obs,))
    feats = compute_sector_features(np.zeros(3), np.array([1.0, 0, 0]), pset, R).reshape(N_SECTORS, 2)
    front = SECTOR_NAMES.index("front")
    assert np.isclose(feats[front, 1], (5.0 - 2.0) / R, atol=1e-6)


# 9/10/11. Level B vs Level C perception-vs-physics separation (env-level) --

def test_level_b_unsensed_obstacle_absent_from_sector_features():
    config = build_canonical_env_config(
        "C1", "measurement", awareness_mode="local", extra_overrides={"defender_obstacle_sensor_range": 1.0},
    )
    env = SoldierEnv(config=config)
    wrapped = ObstacleSectorObservationWrapper(env)
    obs, info = wrapped.reset(seed=5)
    # Far outside the tiny sensor range -> Level B genuinely knows nothing yet.
    assert len(env._defender_last_perceived.static) == 0
    sector_part = obs[16:]
    assert np.allclose(sector_part, 1.0)


def test_level_c_mapped_obstacle_present_in_sector_features():
    """A map-known static obstacle is in the perceived set from the start,
    regardless of distance -- but sector features still normalize/clip by
    `sensor_range` (documented behavior: a mapped-but-far obstacle reads as
    1.0, identical to "no obstacle"). Place it well within range so this
    test isolates "is Level C perception wired into the features", not
    the (separately correct) far-away clipping behavior."""
    config = build_canonical_env_config(
        "C0", "measurement", awareness_mode="map_local",
        extra_overrides={
            "obstacles_enabled": True, "obstacle_layout_mode": "fixed",
            "obstacle_fixed_spec": ((5.0, 0.0, 5.0, 2.0, 2.0, 5.0),),
            "defender_standby_until_detection": False,
        },
    )
    env = SoldierEnv(config=config)
    wrapped = ObstacleSectorObservationWrapper(env)
    obs, info = wrapped.reset(seed=5)
    obs, _, term, trunc, info = wrapped.step(np.zeros(3, dtype=np.float32))
    assert len(env._defender_last_perceived.static) >= 1  # mapped, known immediately
    sector_part = obs[16:]
    assert not np.allclose(sector_part, 1.0)  # at least one sector reflects it


def test_no_ground_truth_leakage_beyond_sensor_range():
    """An obstacle placed far outside sensor range but WITHIN the ground-
    truth layout must not appear in Level B sector features."""
    config = build_canonical_env_config(
        "C0", "measurement", awareness_mode="local",
        extra_overrides={
            "obstacles_enabled": True, "obstacle_layout_mode": "fixed",
            "obstacle_fixed_spec": ((40.0, 40.0, 5.0, 2.0, 2.0, 5.0),),
            "defender_obstacle_sensor_range": 2.0,
        },
    )
    env = SoldierEnv(config=config)
    wrapped = ObstacleSectorObservationWrapper(env)
    obs, info = wrapped.reset(seed=1)
    assert np.allclose(obs[16:], 1.0)  # obstacle exists in ground truth, but unperceived


# 12. identical seed produces identical observation sequence -----------------

def test_seed_reproducibility():
    config = build_canonical_env_config("C2", "measurement")
    results = []
    for _ in range(2):
        env = SoldierEnv(config=config)
        wrapped = ObstacleSectorObservationWrapper(env)
        obs, info = wrapped.reset(seed=42)
        seq = [obs.copy()]
        for _ in range(20):
            obs, _, term, trunc, info = wrapped.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            seq.append(obs.copy())
            if term or trunc:
                break
        results.append(seq)
    assert len(results[0]) == len(results[1])
    for a, b in zip(results[0], results[1]):
        np.testing.assert_array_equal(a, b)


# 13. no NaN/inf ---------------------------------------------------------------

def test_no_nan_or_inf():
    config = build_canonical_env_config("C3", "measurement")
    env = SoldierEnv(config=config)
    wrapped = ObstacleSectorObservationWrapper(env)
    obs, info = wrapped.reset(seed=7)
    assert np.all(np.isfinite(obs))
    for _ in range(30):
        obs, _, term, trunc, info = wrapped.step(np.array([1.0, 0.0, 1.0], dtype=np.float32))
        assert np.all(np.isfinite(obs))
        if term or trunc:
            break


# 14. conference observation path remains unchanged for old models -----------

def test_unwrapped_env_observation_unaffected():
    """The wrapper is purely additive: the SAME SoldierEnv, used WITHOUT
    the wrapper, still returns the original 16-D observation."""
    config = build_canonical_env_config("C2", "measurement")
    env_plain = SoldierEnv(config=config)
    obs_plain, _ = env_plain.reset(seed=3)
    assert obs_plain.shape == (16,)

    env_wrapped = SoldierEnv(config=config)
    wrapped = ObstacleSectorObservationWrapper(env_wrapped)
    obs_wrapped, _ = wrapped.reset(seed=3)
    assert obs_wrapped.shape == (36,)
    np.testing.assert_array_equal(obs_plain, obs_wrapped[:16])


def test_wrapper_requires_defender_avoidance_enabled():
    config = build_canonical_env_config("C1", "measurement")
    import dataclasses
    config = dataclasses.replace(config, defender_obstacle_avoidance_enabled=False)
    env = SoldierEnv(config=config)
    with pytest.raises(ValueError):
        ObstacleSectorObservationWrapper(env)
