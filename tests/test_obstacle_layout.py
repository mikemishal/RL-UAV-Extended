"""Unit tests for `uav_defend.obstacles.layout` (`ObstacleLayout`,
`generate_layout`).

Run directly: python tests/test_obstacle_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout, generate_layout


def _rng(seed=0):
    return np.random.default_rng(seed)


# --- ObstacleLayout queries ----------------------------------------------

def test_empty_layout_queries():
    layout = ObstacleLayout()
    assert len(layout) == 0
    assert not layout.contains_point((0.0, 0.0, 0.0))
    assert layout.containing_obstacle_index((0.0, 0.0, 0.0)) is None
    assert layout.min_clearance((0.0, 0.0, 0.0)) == float("inf")
    assert layout.nearest_obstacle_index((0.0, 0.0, 0.0)) is None
    assert not layout.segment_intersects((-1, 0, 0), (1, 0, 0))
    assert layout.first_intersecting_obstacle((-1, 0, 0), (1, 0, 0)) is None
    assert layout.to_dict() == {"count": 0, "obstacles": []}


def test_multiple_obstacle_queries():
    a = AABBObstacle.from_center_half_extents((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    b = AABBObstacle.from_center_half_extents((10.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    layout = ObstacleLayout(obstacles=(a, b))
    assert len(layout) == 2
    assert layout.contains_point((0.0, 0.0, 1.0))
    assert layout.contains_point((10.0, 0.0, 1.0))
    assert not layout.contains_point((5.0, 0.0, 1.0))
    assert layout.containing_obstacle_index((10.0, 0.0, 1.0)) == 1


def test_minimum_clearance_selects_correct_obstacle():
    near = AABBObstacle.from_center_half_extents((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    far = AABBObstacle.from_center_half_extents((20.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    layout = ObstacleLayout(obstacles=(near, far))
    point = (3.0, 0.0, 1.0)
    assert np.isclose(layout.min_clearance(point), near.clearance_to_point(point))
    assert layout.nearest_obstacle_index(point) == 0


def test_segment_query_selects_intersected_obstacle():
    a = AABBObstacle.from_center_half_extents((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    b = AABBObstacle.from_center_half_extents((10.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    layout = ObstacleLayout(obstacles=(a, b))
    assert layout.first_intersecting_obstacle((10.0, -5.0, 1.0), (10.0, 5.0, 1.0)) == 1
    assert layout.first_intersecting_obstacle((100.0, -5.0, 1.0), (100.0, 5.0, 1.0)) is None


def test_serialization_deterministic():
    a = AABBObstacle.from_center_half_extents((1.0, 2.0, 3.0), (0.5, 0.5, 0.5))
    layout = ObstacleLayout(obstacles=(a,))
    d1 = layout.to_dict()
    d2 = layout.to_dict()
    assert d1 == d2
    assert d1["count"] == 1
    assert d1["obstacles"][0]["center"] == [1.0, 2.0, 3.0]


# --- generate_layout: disabled/fixed ----------------------------------------

def test_disabled_layout_is_empty_and_uses_no_rng():
    cfg = EnvConfig(obstacles_enabled=False)
    layout = generate_layout(cfg, _rng(), (0, 0, 0), (0, 0, 0))
    assert len(layout) == 0


def test_none_mode_layout_is_empty():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="none")
    layout = generate_layout(cfg, _rng(), (0, 0, 0), (0, 0, 0))
    assert len(layout) == 0


def test_fixed_mode_uses_exact_spec():
    cfg = EnvConfig(
        obstacles_enabled=True,
        obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((10.0, 10.0, 2.0, 1.0, 1.0, 2.0), (-10.0, -10.0, 3.0, 2.0, 2.0, 3.0)),
    )
    layout = generate_layout(cfg, _rng(), (0, 0, 0), (0, 0, 0))
    assert len(layout) == 2
    assert layout[0].center.tolist() == [10.0, 10.0, 2.0]
    assert layout[1].half_extents.tolist() == [2.0, 2.0, 3.0]


def test_fixed_mode_rejects_out_of_domain_obstacle():
    cfg = EnvConfig(
        obstacles_enabled=True,
        obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((49.9, 0.0, 1.0, 1.0, 1.0, 1.0),),  # extends past L=50
    )
    try:
        generate_layout(cfg, _rng(), (0, 0, 0), (0, 0, 0))
        raise AssertionError("expected ValueError for out-of-domain fixed obstacle")
    except ValueError:
        pass


# --- generate_layout: random -------------------------------------------------

def test_same_seed_identical_obstacles():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=6,
                     obstacle_clearance_from_asset=3.0)
    layout_a = generate_layout(cfg, _rng(42), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(40, 40, 15))
    layout_b = generate_layout(cfg, _rng(42), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(40, 40, 15))
    assert layout_a.to_dict() == layout_b.to_dict()


def test_different_seeds_produce_different_layout():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=6,
                     obstacle_clearance_from_asset=3.0)
    layout_a = generate_layout(cfg, _rng(1), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(40, 40, 15))
    layout_b = generate_layout(cfg, _rng(2), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(40, 40, 15))
    assert layout_a.to_dict() != layout_b.to_dict()


def test_random_layout_no_overlap():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=10,
                     obstacle_min_size=3.0, obstacle_max_size=6.0, obstacle_clearance_from_asset=3.0)
    layout = generate_layout(cfg, _rng(7), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(45, 10, 15))
    obstacles = list(layout)
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            assert not obstacles[i].overlaps(obstacles[j]), f"obstacles {i} and {j} overlap"


def test_random_layout_domain_bounds_respected():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=15,
                     obstacle_clearance_from_asset=3.0)
    layout = generate_layout(cfg, _rng(3), (0, 0, 0), (0, 0, 0), enemy_spawn_position=(45, 10, 15))
    for o in layout:
        assert np.all(o.min_corner[:2] >= -cfg.L - 1e-9)
        assert np.all(o.max_corner[:2] <= cfg.L + 1e-9)
        assert o.min_corner[2] >= -1e-9
        assert o.max_corner[2] <= cfg.max_altitude + 1e-9


def test_random_layout_asset_clearance_respected():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=10,
                     obstacle_clearance_from_asset=4.0)
    asset_pos = (0.0, 0.0, 0.0)
    layout = generate_layout(cfg, _rng(11), asset_pos, asset_pos, enemy_spawn_position=(45, -30, 12))
    for o in layout:
        assert o.clearance_to_point(asset_pos) >= cfg.obstacle_clearance_from_asset - 1e-9


def test_impossible_configuration_raises_runtime_error():
    # Footprint too large to ever fit inside the domain.
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=1,
                     obstacle_min_size=500.0, obstacle_max_size=500.0)
    try:
        generate_layout(cfg, _rng(0), (0, 0, 0), (0, 0, 0))
        raise AssertionError("expected RuntimeError for an impossible layout request")
    except RuntimeError:
        pass


def test_impossible_configuration_raises_runtime_error_via_clearance_saturation():
    # Small domain relative to count/clearance forces exhaustion of the
    # per-obstacle placement-attempt budget rather than an infinite loop.
    cfg = EnvConfig(
        L=6.0, obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=20,
        obstacle_min_size=2.0, obstacle_max_size=3.0, obstacle_clearance_from_asset=5.5,
        obstacle_max_placement_attempts=50,
    )
    try:
        generate_layout(cfg, _rng(0), (0, 0, 0), (0, 0, 0))
        raise AssertionError("expected RuntimeError for an impossible layout request")
    except RuntimeError:
        pass


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
