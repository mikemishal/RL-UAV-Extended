"""Hostile obstacle bypass CANDIDATE-SELECTION tests (journal extension,
Phase 3C: mission-efficiency refinement).

Covers the required test list (Step 5 of the Phase-3C specification):
 1. greater-progress candidate wins among SAFE candidates even if another
    SAFE candidate (e.g. climb) has larger raw clearance
 2. a margin-deficient candidate never beats a SAFE candidate, regardless
    of progress
 3. among collision-free-but-margin-deficient candidates, larger clearance
    wins
 4. if every candidate collides, the maximum-clearance one is used
    (fallback_used=True)
 5. selection is deterministic
 6. existing left/right/climb feasibility tests still pass (see
    test_enemy_obstacle_avoidance.py -- re-run together with this file)
 7. persistence/release/resume-pursuit behavior is unchanged (see
    test_enemy_obstacle_avoidance.py's persistence test -- unaffected by
    this change, re-verified there)
 8. avoidance-disabled hostile trajectory remains exact (see
    test_enemy_obstacle_avoidance.py -- unaffected, re-verified there)
 9. obstacles-disabled conference regression remains exact (see
    test_enemy_obstacle_avoidance.py -- unaffected, re-verified there)

Run directly: python tests/test_enemy_obstacle_bypass_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout
from uav_defend.obstacles.avoidance import (
    CandidateEvaluation,
    DynamicsLimits,
    candidate_category,
    candidate_sort_key,
    evaluate_and_select_candidate,
    generate_bypass_candidates,
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


def _evaluation(**overrides) -> CandidateEvaluation:
    defaults = dict(
        mode="x", direction=np.array([1.0, 0.0, 0.0]), waypoint=np.zeros(3),
        collision=False, min_clearance=10.0, meets_clearance=True,
        progress=1.0, angular_deviation=0.0, order_index=0,
    )
    defaults.update(overrides)
    return CandidateEvaluation(**defaults)


# --- 1. Greater progress wins among SAFE candidates, even over larger raw
#        clearance (this is THE Phase-3C fix, exercised at the rule level).

def test_safe_candidate_with_more_progress_wins_over_larger_clearance():
    climb_like = _evaluation(mode="climb", collision=False, meets_clearance=True, min_clearance=20.0, progress=0.5)
    horizontal_like = _evaluation(mode="left", collision=False, meets_clearance=True, min_clearance=3.0, progress=5.0)
    best = min([climb_like, horizontal_like], key=candidate_sort_key)
    assert best.mode == "left"


# --- 1b. Same rule exercised end-to-end through real geometry/dynamics.

def test_progress_efficient_bypass_preferred_over_conservative_climb_end_to_end():
    # A short obstacle (climb is trivially very safe) with the mover
    # starting already aligned with the "left" bypass waypoint's lateral
    # offset (so the straight-line approach maintains clearance the whole
    # way, rather than clipping closer to the box's near corner first --
    # see the Phase-3C report for this geometric nuance of the unchanged
    # Phase-3B waypoint formula), and the protected asset positioned so
    # left makes much more progress than climbing in place.
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([-20.0, 6.0, 1.0])
    velocity = np.zeros(3)
    u_base = np.array([1.0, 0.0, 0.0])
    target_position = np.array([30.0, 20.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    best, fallback_used = evaluate_and_select_candidate(
        position, velocity, u_base, candidates, layout, target_position, _limits(), horizon_steps=12,
        required_clearance=_REQUIRED_CLEARANCE,
    )
    assert not fallback_used
    assert best.meets_clearance
    assert best.mode != "climb"


# --- 2. A margin-deficient candidate never beats a SAFE candidate, no
#        matter how much progress it makes.

def test_margin_deficient_candidate_never_beats_safe_candidate():
    safe_low_progress = _evaluation(mode="right", collision=False, meets_clearance=True, min_clearance=2.5, progress=0.1)
    deficient_high_progress = _evaluation(mode="left", collision=False, meets_clearance=False, min_clearance=0.5, progress=100.0)
    best = min([safe_low_progress, deficient_high_progress], key=candidate_sort_key)
    assert best.mode == "right"
    assert candidate_category(best) == candidate_category(safe_low_progress)


# --- 3. Among margin-deficient (but collision-free) candidates, larger
#        clearance wins.

def test_among_margin_deficient_candidates_larger_clearance_wins():
    lower_clearance = _evaluation(mode="left", collision=False, meets_clearance=False, min_clearance=0.5, progress=10.0)
    higher_clearance = _evaluation(mode="right", collision=False, meets_clearance=False, min_clearance=1.5, progress=1.0)
    best = min([lower_clearance, higher_clearance], key=candidate_sort_key)
    assert best.mode == "right"


# --- 4. If every candidate collides, maximum-clearance fallback is used.

def test_all_colliding_uses_maximum_clearance_fallback():
    worse = _evaluation(mode="left", collision=True, meets_clearance=False, min_clearance=0.2, progress=5.0)
    better_clearance = _evaluation(mode="right", collision=True, meets_clearance=False, min_clearance=1.0, progress=0.0)
    best = min([worse, better_clearance], key=candidate_sort_key)
    assert best.mode == "right"
    assert candidate_category(best) == 2  # CATEGORY_COLLIDING


def test_all_colliding_end_to_end_marks_fallback_used():
    # A single obstacle so large no bypass direction clears it within the
    # short horizon: every candidate collides.
    obstacle = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (60.0, 60.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([-59.0, 0.0, 1.0])
    velocity = np.zeros(3)
    u_base = np.array([1.0, 0.0, 0.0])
    target_position = np.array([200.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
    best, fallback_used = evaluate_and_select_candidate(
        position, velocity, u_base, candidates, layout, target_position, _limits(), horizon_steps=12,
        required_clearance=_REQUIRED_CLEARANCE,
    )
    assert fallback_used is True
    assert best.collision is True


# --- 5. Selection is deterministic.

def test_selection_is_deterministic():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    velocity = np.zeros(3)
    u_base = np.array([1.0, 0.0, 0.0])
    target_position = np.array([30.0, 20.0, 0.0])
    limits = _limits()

    results = []
    for _ in range(3):
        candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=_REQUIRED_CLEARANCE, max_altitude=_MAX_ALTITUDE)
        best, fallback_used = evaluate_and_select_candidate(
            position, velocity, u_base, candidates, layout, target_position, limits, horizon_steps=12,
            required_clearance=_REQUIRED_CLEARANCE,
        )
        results.append((best.mode, fallback_used, tuple(best.direction)))
    assert len(set(results)) == 1


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
