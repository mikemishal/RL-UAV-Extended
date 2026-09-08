"""Swept-motion (segment) collision-detection tests for
`AABBObstacle.segment_intersection` / `ObstacleLayout.first_segment_intersection`.

Covers the 12 critical cases required for physically accurate collision
detection against fast-moving entities (tunneling, boundary contact,
multi-obstacle first-contact selection, degenerate segments, etc.).

Run directly: python tests/test_obstacle_collision_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout


def _thin_wall() -> AABBObstacle:
    """A thin wall spanning x in [-0.5, 0.5], y in [-20, 20], z in [0, 10]."""
    return AABBObstacle(min_corner=(-0.5, -20.0, 0.0), max_corner=(0.5, 20.0, 10.0))


# 1. Tunneling: start and end both outside a thin obstacle, but the segment
# between them crosses straight through it -- this is the essential case a
# fast-moving entity (endpoint-only checking would miss).
def test_tunneling_through_thin_obstacle():
    wall = _thin_wall()
    start = (-9.0, 0.0, 5.0)  # far outside, -x side
    end = (9.0, 0.0, 5.0)     # far outside, +x side
    assert not wall.contains_point(start)
    assert not wall.contains_point(end)
    result = wall.segment_intersection(start, end)
    assert result.intersects
    assert 0.0 < result.t_enter < 1.0


# 2. Movement ending inside the obstacle.
def test_movement_ending_inside_obstacle():
    wall = _thin_wall()
    result = wall.segment_intersection((-9.0, 0.0, 5.0), (0.0, 0.0, 5.0))
    assert result.intersects
    assert wall.contains_point(result.point)


# 3. Movement beginning exactly on the boundary.
def test_movement_beginning_on_boundary():
    wall = _thin_wall()
    result = wall.segment_intersection((-0.5, 0.0, 5.0), (-9.0, 0.0, 5.0))
    assert result.intersects
    assert np.isclose(result.t_enter, 0.0)


# 4. Tangent/boundary contact counts as collision.
def test_tangent_boundary_contact_counts_as_collision():
    wall = _thin_wall()
    # Segment runs exactly along the -x face plane (x = -0.5).
    result = wall.segment_intersection((-0.5, -30.0, 5.0), (-0.5, 30.0, 5.0))
    assert result.intersects


# 5. Vertical segment crosses the building top.
def test_vertical_segment_crosses_building_top():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (5.0, 5.0, 5.0))
    result = box.segment_intersection((0.0, 0.0, 20.0), (0.0, 0.0, -5.0))
    assert result.intersects
    assert np.isclose(result.point[2], 10.0)  # enters through the top face (z=10)


# 6. Segment passes above a building without collision.
def test_segment_passes_above_building_without_collision():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (5.0, 5.0, 5.0))  # top at z=10
    result = box.segment_intersection((-20.0, 0.0, 15.0), (20.0, 0.0, 15.0))
    assert not result.intersects


# 7. Segment passes beside a building without collision.
def test_segment_passes_beside_building_without_collision():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (5.0, 5.0, 5.0))
    result = box.segment_intersection((-20.0, 20.0, 5.0), (20.0, 20.0, 5.0))
    assert not result.intersects


# 8/9. Zero-length segment inside/outside an obstacle.
def test_zero_length_segment_inside_obstacle():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (5.0, 5.0, 5.0))
    result = box.segment_intersection((0.0, 0.0, 5.0), (0.0, 0.0, 5.0))
    assert result.intersects
    assert np.isclose(result.t_enter, 0.0)


def test_zero_length_segment_outside_obstacle():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 5.0), (5.0, 5.0, 5.0))
    result = box.segment_intersection((100.0, 0.0, 5.0), (100.0, 0.0, 5.0))
    assert not result.intersects


# 10. Multiple obstacles along one trajectory: first intersection correctly identified.
def test_multiple_obstacles_first_intersection_identified():
    near = AABBObstacle.from_center_half_extents((10.0, 0.0, 5.0), (2.0, 2.0, 5.0))
    far = AABBObstacle.from_center_half_extents((30.0, 0.0, 5.0), (2.0, 2.0, 5.0))
    layout = ObstacleLayout(obstacles=(far, near))  # deliberately out of spatial order
    result = layout.first_segment_intersection((0.0, 0.0, 5.0), (50.0, 0.0, 5.0))
    assert result.intersects
    assert result.obstacle_index == 1  # "near" is at index 1 in this layout
    assert result.t_enter < 0.5  # enters the nearer obstacle first


# 11. Collision point / t_enter is mathematically correct.
def test_collision_point_and_t_enter_mathematically_correct():
    box = AABBObstacle.from_center_half_extents((10.0, 0.0, 5.0), (2.0, 2.0, 5.0))  # x in [8,12]
    start = np.array([0.0, 0.0, 5.0])
    end = np.array([20.0, 0.0, 5.0])
    result = box.segment_intersection(start, end)
    assert result.intersects
    expected_t_enter = 8.0 / 20.0  # enters the box at x=8
    assert np.isclose(result.t_enter, expected_t_enter)
    expected_point = start + expected_t_enter * (end - start)
    assert np.allclose(result.point, expected_point)


# 12. No-obstacle layout returns no collision.
def test_no_obstacle_layout_returns_no_collision():
    layout = ObstacleLayout()
    result = layout.first_segment_intersection((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
    assert not result.intersects
    assert result.obstacle_index is None
    assert result.t_enter is None
    assert result.point is None


# --- Additional t_enter/t_exit consistency checks ---------------------------

def test_t_exit_greater_than_or_equal_t_enter():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    result = box.segment_intersection((-5.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    assert result.intersects
    assert result.t_exit >= result.t_enter


def test_segment_intersects_matches_segment_intersection_boolean():
    box = AABBObstacle.from_center_half_extents((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    for start, end in [((-5, 0, 0), (5, 0, 0)), ((-5, 10, 0), (5, 10, 0))]:
        assert box.segment_intersects(start, end) == box.segment_intersection(start, end).intersects


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
