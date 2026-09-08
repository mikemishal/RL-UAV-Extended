"""Unit tests for `uav_defend.obstacles.geometry.AABBObstacle`.

Run directly: python tests/test_obstacle_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle


def _box() -> AABBObstacle:
    return AABBObstacle.from_center_half_extents(center=(0.0, 0.0, 5.0), half_extents=(2.0, 3.0, 5.0))


# --- containment ---------------------------------------------------------

def test_center_point_is_contained():
    b = _box()
    assert b.contains_point((0.0, 0.0, 5.0))


def test_external_point_is_not_contained():
    b = _box()
    assert not b.contains_point((100.0, 0.0, 5.0))


def test_boundary_point_counts_as_contained():
    b = _box()
    assert b.contains_point((2.0, 0.0, 5.0))  # exactly on +x face
    assert b.contains_point((-2.0, -3.0, 0.0))  # exactly on a corner


# --- distance --------------------------------------------------------------

def test_zero_distance_inside():
    b = _box()
    assert b.distance_to_point((0.0, 0.0, 5.0)) == 0.0
    assert b.distance_to_point((2.0, 0.0, 5.0)) == 0.0  # boundary


def test_distance_outside_each_face():
    b = _box()  # min=(-2,-3,0), max=(2,3,10)
    assert np.isclose(b.distance_to_point((5.0, 0.0, 5.0)), 3.0)   # +x face
    assert np.isclose(b.distance_to_point((-5.0, 0.0, 5.0)), 3.0)  # -x face
    assert np.isclose(b.distance_to_point((0.0, 6.0, 5.0)), 3.0)   # +y face
    assert np.isclose(b.distance_to_point((0.0, -6.0, 5.0)), 3.0)  # -y face
    assert np.isclose(b.distance_to_point((0.0, 0.0, 13.0)), 3.0)  # +z face
    assert np.isclose(b.distance_to_point((0.0, 0.0, -3.0)), 3.0)  # -z face


def test_distance_outside_edge_and_corner():
    b = _box()  # min=(-2,-3,0), max=(2,3,10)
    # Just outside an edge (x and y both beyond the box, z inside).
    assert np.isclose(b.distance_to_point((5.0, 6.0, 5.0)), np.hypot(3.0, 3.0))
    # Just outside a corner (all three axes beyond the box).
    assert np.isclose(b.distance_to_point((5.0, 6.0, 13.0)), np.sqrt(3.0**2 + 3.0**2 + 3.0**2))


def test_clearance_to_point_subtracts_radius():
    b = _box()
    assert np.isclose(b.clearance_to_point((5.0, 0.0, 5.0), radius=1.0), 2.0)
    assert b.clearance_to_point((0.0, 0.0, 5.0), radius=1.0) < 0  # inside minus radius is negative


# --- sphere intersection ----------------------------------------------------

def test_sphere_collision_true_when_overlapping():
    b = _box()
    assert b.intersects_sphere((5.0, 0.0, 5.0), radius=3.5)


def test_sphere_collision_false_when_far():
    b = _box()
    assert not b.intersects_sphere((5.0, 0.0, 5.0), radius=1.0)


def test_sphere_collision_boundary_counts():
    b = _box()
    assert b.intersects_sphere((5.0, 0.0, 5.0), radius=3.0)  # exactly touching


# --- segment intersection ----------------------------------------------------

def test_segment_crosses_box():
    b = _box()
    assert b.segment_intersects((-10.0, 0.0, 5.0), (10.0, 0.0, 5.0))


def test_segment_misses_box():
    b = _box()
    assert not b.segment_intersects((-10.0, 10.0, 5.0), (10.0, 10.0, 5.0))


def test_segment_touches_boundary():
    b = _box()
    # Segment runs exactly along the +x face plane (x=2).
    assert b.segment_intersects((2.0, -10.0, 5.0), (2.0, 10.0, 5.0))


def test_vertical_segment_behavior():
    b = _box()
    assert b.segment_intersects((0.0, 0.0, -10.0), (0.0, 0.0, 20.0))
    assert not b.segment_intersects((100.0, 0.0, -10.0), (100.0, 0.0, 20.0))


def test_degenerate_zero_length_segment_reduces_to_point_test():
    b = _box()
    assert b.segment_intersects((0.0, 0.0, 5.0), (0.0, 0.0, 5.0))
    assert not b.segment_intersects((100.0, 0.0, 5.0), (100.0, 0.0, 5.0))


# --- invalid construction ----------------------------------------------------

def test_invalid_box_rejected_when_max_below_min():
    try:
        AABBObstacle(min_corner=(1.0, 1.0, 1.0), max_corner=(0.0, 1.0, 1.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_box_rejected_when_degenerate_zero_extent():
    try:
        AABBObstacle(min_corner=(0.0, 0.0, 0.0), max_corner=(0.0, 1.0, 1.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_half_extents_rejected():
    try:
        AABBObstacle.from_center_half_extents(center=(0.0, 0.0, 0.0), half_extents=(1.0, -1.0, 1.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_shape_rejected():
    try:
        AABBObstacle(min_corner=(0.0, 0.0), max_corner=(1.0, 1.0, 1.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_overlaps_detects_overlap_and_margin():
    a = AABBObstacle.from_center_half_extents((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b_touch = AABBObstacle.from_center_half_extents((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b_far = AABBObstacle.from_center_half_extents((10.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert a.overlaps(b_touch)  # touching counts as overlap
    assert not a.overlaps(b_far)
    assert a.overlaps(b_far, margin=100.0)  # margin can force overlap detection


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
