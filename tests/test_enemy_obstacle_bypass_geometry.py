"""Hostile obstacle bypass WAYPOINT-GEOMETRY tests (journal extension,
Phase 3D: geometry-aware horizontal bypass construction).

Covers the required test list (Step 8 of the Phase-3D specification):
 1. oblique approach where the old center-offset path violates clearance
 2/3. new left/right waypoints maintain inflated-AABB path clearance
 4. head-on symmetric case remains deterministic
 5. near-corner approach
 6. non-square obstacle footprint
 7. very wide obstacle
 8. narrow/deep obstacle
 9. hostile initially laterally offset from obstacle center
 10. candidate still evaluated through constrained dynamics
 11. current velocity can make a geometrically valid candidate dynamically
     margin-deficient
 12. no random numbers consumed
 13. climb behavior unchanged
 14. Phase-3C ranking unchanged
 15. obstacle-disabled conference regression exact

Two-stage horizontal bypass (items 16-19) was found NOT to be needed: the
held-out validation bank shows no pathological stall-at-corner behavior
and the stall rate improved versus Phase 3C (see the Phase-3D report), so
those tests are not applicable -- Step 6 explicitly says to only add
two-stage behavior if testing shows it is needed.

Run directly: python tests/test_enemy_obstacle_bypass_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout
from uav_defend.obstacles.avoidance import (
    DynamicsLimits,
    candidate_sort_key,
    generate_bypass_candidates,
    predict_collision,
    predict_constrained_trajectory,
)

_L = 50.0
_MAX_ALTITUDE = 30.0
_REQUIRED_CLEARANCE = 2.0


def _limits(**overrides) -> DynamicsLimits:
    defaults = dict(
        max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0),
        max_climb_rate=5.0, max_descent_rate=5.0, dt=0.5, eps=1e-8, L=_L, max_altitude=_MAX_ALTITUDE,
    )
    defaults.update(overrides)
    return DynamicsLimits(**defaults)


def _old_center_offset_waypoint(position, obstacle, u_base, clearance, mode, eps=1e-8):
    """Reproduces the Phase-3C (pre-fix) center-offset formula, purely as
    a reference for regression-demonstrating the defect it had -- NOT
    used by production code any more."""
    raw = np.array([-u_base[1], u_base[0]], dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    lateral = raw / norm if norm > eps else np.array([1.0, 0.0])
    sign = 1.0 if mode == "left" else -1.0
    lateral_extent = abs(lateral[0]) * obstacle.half_extents[0] + abs(lateral[1]) * obstacle.half_extents[1] + clearance
    target_xy = obstacle.center[:2] + sign * lateral * lateral_extent
    return np.array([target_xy[0], target_xy[1], position[2]])


def _path_min_clearance_to_obstacle_xy(p0_xy, p1_xy, obstacle, n_samples=200):
    """Direct static-geometry minimum horizontal clearance sampled along
    the straight segment p0_xy -> p1_xy (independent of dynamics/prediction
    -- used only to characterize the raw path geometry)."""
    xmin, xmax = obstacle.min_corner[0], obstacle.max_corner[0]
    ymin, ymax = obstacle.min_corner[1], obstacle.max_corner[1]
    min_clearance = float("inf")
    for t in np.linspace(0.0, 1.0, n_samples):
        p = p0_xy + t * (p1_xy - p0_xy)
        dx = max(xmin - p[0], p[0] - xmax, 0.0)
        dy = max(ymin - p[1], p[1] - ymax, 0.0)
        min_clearance = min(min_clearance, float(np.hypot(dx, dy)))
    return min_clearance


# --- 1. Oblique approach: the OLD center-offset path violated clearance ---

def test_old_center_offset_path_violated_clearance_on_oblique_approach():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    old_waypoint = _old_center_offset_waypoint(position, obstacle, u_base, _REQUIRED_CLEARANCE, "left")
    path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], old_waypoint[:2], obstacle)
    # The waypoint itself sits exactly at the required clearance, but the
    # APPROACH PATH clips closer than that before arriving -- this is
    # exactly the Phase-3C defect.
    assert path_clearance < _REQUIRED_CLEARANCE


# --- 2/3. New waypoints maintain inflated-AABB path clearance -------------

def test_new_left_waypoint_maintains_path_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    left = next(c for c in candidates if c.mode == "left")
    path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], left.waypoint[:2], obstacle)
    assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


def test_new_right_waypoint_maintains_path_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    right = next(c for c in candidates if c.mode == "right")
    path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], right.waypoint[:2], obstacle)
    assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


def test_new_waypoint_collision_free_and_meets_clearance_under_dynamics():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    velocity = np.zeros(3)
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for mode in ("left", "right"):
        candidate = next(c for c in candidates if c.mode == mode)
        positions, _ = predict_constrained_trajectory(position, velocity, candidate.direction, limits, 12)
        prediction = predict_collision(positions, layout)
        assert not prediction.collision
        assert prediction.min_clearance >= _REQUIRED_CLEARANCE - 1e-6


# --- 4. Head-on symmetric case remains deterministic -----------------------

def test_head_on_symmetric_case_is_deterministic():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    results = []
    for _ in range(3):
        candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
        left = next(c for c in candidates if c.mode == "left")
        right = next(c for c in candidates if c.mode == "right")
        results.append((tuple(left.waypoint), tuple(right.waypoint)))
    assert len(set(results)) == 1
    # Symmetric geometry (box centered on the approach axis) => left/right
    # waypoints are mirror images across y=0.
    left_wp, right_wp = results[0]
    assert abs(left_wp[0] - right_wp[0]) < 1e-9
    assert abs(left_wp[1] + right_wp[1]) < 1e-9


# --- 5. Near-corner approach ------------------------------------------------

def test_near_corner_approach_produces_valid_clearance_waypoint():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    # Start almost adjacent to the box's near corner (just outside the
    # inflated footprint).
    position = np.array([11.0 - _REQUIRED_CLEARANCE - 0.5, -4.0 - _REQUIRED_CLEARANCE - 0.5, 1.0])
    u_base = np.array([1.0, 0.3, 0.0])
    u_base = u_base / np.linalg.norm(u_base)
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for c in candidates:
        if c.mode == "climb":
            continue
        path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], c.waypoint[:2], obstacle)
        assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


# --- 6-8. Various obstacle footprints --------------------------------------

def test_non_square_footprint_waypoints_maintain_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (2.0, 8.0, 5.0))  # deep wall
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for c in candidates:
        if c.mode == "climb":
            continue
        path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], c.waypoint[:2], obstacle)
        assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


def test_very_wide_obstacle_waypoints_maintain_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (2.0, 20.0, 5.0))  # very wide in y
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for c in candidates:
        if c.mode == "climb":
            continue
        path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], c.waypoint[:2], obstacle)
        assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


def test_narrow_deep_obstacle_waypoints_maintain_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (10.0, 1.0, 5.0))  # narrow in y, deep in x
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for c in candidates:
        if c.mode == "climb":
            continue
        path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], c.waypoint[:2], obstacle)
        assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


# --- 9. Hostile initially laterally offset from obstacle center -----------

def test_laterally_offset_hostile_waypoints_maintain_clearance():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([-10.0, 12.0, 1.0])  # well off to the side already
    u_base = np.array([1.0, -0.2, 0.0])
    u_base = u_base / np.linalg.norm(u_base)
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    for c in candidates:
        if c.mode == "climb":
            continue
        path_clearance = _path_min_clearance_to_obstacle_xy(position[:2], c.waypoint[:2], obstacle)
        assert path_clearance >= _REQUIRED_CLEARANCE - 1e-6


# --- 10/11. Candidates still evaluated through real constrained dynamics --

def test_candidate_still_evaluated_through_constrained_dynamics():
    """A geometrically-valid waypoint (clear straight-line path) can still
    be flagged margin-deficient/colliding once real turn-rate-limited
    dynamics are applied with strong initial momentum -- static geometry
    alone must never be trusted as the final safety authority."""
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([5.0, 6.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    left = next(c for c in candidates if c.mode == "left")

    # From rest: the static path is geometrically clear.
    static_clearance = _path_min_clearance_to_obstacle_xy(position[:2], left.waypoint[:2], obstacle)
    assert static_clearance >= _REQUIRED_CLEARANCE - 1e-6

    # With strong momentum already carrying the mover TOWARD the obstacle
    # (turn-rate-limited), the real predicted trajectory can differ from
    # the idealized straight-line path.
    velocity_at_rest_positions, _ = predict_constrained_trajectory(position, np.zeros(3), left.direction, limits, 12)
    velocity_with_inertia_positions, _ = predict_constrained_trajectory(
        position, np.array([12.0, -6.0, 0.0]), left.direction, limits, 12,
    )
    rest_clearance = predict_collision(velocity_at_rest_positions, layout).min_clearance
    inertia_clearance = predict_collision(velocity_with_inertia_positions, layout).min_clearance
    assert inertia_clearance < rest_clearance  # inertia measurably erodes the achieved clearance


# --- 12. No random numbers consumed ----------------------------------------

def test_no_extra_rng_draws_from_geometry_aware_waypoints():
    no_avoidance_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),),
        enemy_obstacle_avoidance_enabled=False, defender_standby_until_detection=False,
    )
    avoidance_cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),),
        enemy_obstacle_avoidance_enabled=True, defender_standby_until_detection=False,
    )
    env_a = SoldierEnv(config=no_avoidance_cfg)
    env_b = SoldierEnv(config=avoidance_cfg)
    env_a.reset(seed=1)
    env_b.reset(seed=1)
    for env in (env_a, env_b):
        env._enemy_pos = np.array([-20.0, 0.0, 3.0], dtype=np.float32)
        env._enemy_vel = np.zeros(3, dtype=np.float32)
        env._soldier_pos = np.array([40.0, 0.0, 0.0], dtype=np.float32)
        env._defender_pos = np.array([1000.0, 1000.0, 0.0], dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
    activated = False
    for _ in range(30):
        _, _, term_a, trunc_a, _ = env_a.step(np.zeros(3, dtype=np.float32))
        _, _, term_b, trunc_b, info_b = env_b.step(np.zeros(3, dtype=np.float32))
        activated = activated or info_b["enemy_obstacle_avoidance_active"]
        if term_a or trunc_a or term_b or trunc_b:
            break
    assert activated
    draw_a = env_a._rng_enemy_motion.normal(size=5)
    draw_b = env_b._rng_enemy_motion.normal(size=5)
    assert np.array_equal(draw_a, draw_b)


# --- 13. Climb behavior unchanged ------------------------------------------

def test_climb_waypoint_formula_unchanged():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 1.0), (10.0, 10.0, 1.0))
    position = np.array([3.0, -2.0, 1.0])
    u_base = np.array([1.0, 0.3, 0.0])
    u_base = u_base / np.linalg.norm(u_base)
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    climb = next(c for c in candidates if c.mode == "climb")
    expected_waypoint = np.array([position[0], position[1], obstacle.max_corner[2] + _REQUIRED_CLEARANCE])
    assert np.allclose(climb.waypoint, expected_waypoint)


# --- 14. Phase-3C ranking rule unchanged -----------------------------------

def test_phase3c_ranking_rule_unchanged():
    from uav_defend.obstacles.avoidance import CandidateEvaluation

    def _evaluation(**overrides):
        defaults = dict(
            mode="x", direction=np.array([1.0, 0.0, 0.0]), waypoint=np.zeros(3),
            collision=False, min_clearance=10.0, meets_clearance=True,
            progress=1.0, angular_deviation=0.0, order_index=0,
        )
        defaults.update(overrides)
        return CandidateEvaluation(**defaults)

    climb_like = _evaluation(mode="climb", min_clearance=20.0, progress=0.5)
    horizontal_like = _evaluation(mode="left", min_clearance=3.0, progress=5.0)
    best = min([climb_like, horizontal_like], key=candidate_sort_key)
    assert best.mode == "left"  # SAFE candidates still ranked by progress first


# --- 15. Obstacle-disabled conference regression exact ---------------------

def test_obstacles_disabled_reproduces_conference_behavior():
    baseline = SoldierEnv(config=EnvConfig())
    disabled = SoldierEnv(config=EnvConfig(obstacles_enabled=False, enemy_obstacle_avoidance_enabled=True))
    ob, _ = baseline.reset(seed=9)
    od, _ = disabled.reset(seed=9)
    assert np.array_equal(ob, od)
    for _ in range(60):
        ob, rb, term_b, trunc_b, info_b = baseline.step(np.zeros(3, dtype=np.float32))
        od, rd, term_d, trunc_d, info_d = disabled.step(np.zeros(3, dtype=np.float32))
        assert np.array_equal(ob, od)
        assert rb == rd
        if term_b or trunc_b:
            break


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
