"""
Deterministic tests for the hostile-UAS perception/awareness layer
(uav_defend/obstacles/perception.py) -- static-obstacle sensing, map
knowledge, and memory (Section 17).
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.obstacles import AABBObstacle, ObstacleLayout
from uav_defend.obstacles.perception import EnemyObstaclePerception


def _far_box(map_known=True):
    return AABBObstacle.from_center_half_extents((40.0, 0.0, 5.0), (2.0, 2.0, 5.0), map_known=map_known)


def test_level_b_does_not_perceive_distant_obstacle():
    perception = EnemyObstaclePerception(
        mode="local", static_sensor_range=10.0, dynamic_sensor_range=10.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(),))
    perceived = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())
    assert len(perceived.static) == 0


def test_level_b_perceives_static_obstacle_after_entering_range():
    perception = EnemyObstaclePerception(
        mode="local", static_sensor_range=10.0, dynamic_sensor_range=10.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(),))
    far = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())
    near = perception.update(np.array([32.0, 0.0, 5.0]), layout, ())  # clearance to box = 32-38=-6 (inside range)
    assert len(far.static) == 0
    assert len(near.static) == 1
    assert near.static[0].source == "local_sensor"


def test_avoidance_activates_only_after_obstacle_perceived():
    from uav_defend.config.env_config import EnvConfig
    from uav_defend.envs import SoldierEnv

    base_config = EnvConfig()
    env = SoldierEnv(config=base_config)
    _, info = env.reset(seed=10)
    enemy_start = info["enemy_pos"].copy()
    soldier_pos = info["soldier_pos"].copy()
    midpoint = (enemy_start + soldier_pos) / 2.0
    cx = float(np.clip(midpoint[0], -base_config.L + 8, base_config.L - 8))
    cy = float(np.clip(midpoint[1], -base_config.L + 8, base_config.L - 8))
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((cx, cy, 9.0, 6.0, 6.0, 8.0),),
        enemy_obstacle_avoidance_enabled=True,
        enemy_obstacle_awareness_mode="local", enemy_obstacle_sensor_range=8.0,
    )
    env2 = SoldierEnv(config=config)
    _, info = env2.reset(seed=10)
    activated_before_perceived = False
    was_ever_perceived = False
    for _ in range(300):
        obs, reward, done, truncated, info = env2.step(np.zeros(3, dtype=np.float32))
        if info["enemy_obstacle_avoidance_active"] and info["enemy_num_locally_sensed_static"] == 0:
            activated_before_perceived = True
        was_ever_perceived = was_ever_perceived or info["enemy_num_locally_sensed_static"] > 0
        if done or truncated:
            break
    assert not activated_before_perceived
    assert was_ever_perceived


def test_mapped_obstacle_known_before_entering_sensing_range():
    perception = EnemyObstaclePerception(
        mode="map_local", static_sensor_range=5.0, dynamic_sensor_range=5.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(map_known=True),))
    perceived = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())  # far away, small sensor range
    assert len(perceived.static) == 1
    assert perceived.static[0].source == "map"


def test_unmapped_obstacle_not_known_before_sensing_range_under_level_c():
    perception = EnemyObstaclePerception(
        mode="map_local", static_sensor_range=5.0, dynamic_sensor_range=5.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(map_known=False),))
    perceived = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())
    assert len(perceived.static) == 0


def test_unmapped_obstacle_appears_after_local_detection():
    perception = EnemyObstaclePerception(
        mode="map_local", static_sensor_range=10.0, dynamic_sensor_range=10.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(map_known=False),))
    far = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())
    near = perception.update(np.array([32.0, 0.0, 5.0]), layout, ())
    assert len(far.static) == 0
    assert len(near.static) == 1
    assert near.static[0].source == "local_sensor"


def test_locally_detected_static_obstacle_remembered_when_configured():
    perception = EnemyObstaclePerception(
        mode="local", static_sensor_range=10.0, dynamic_sensor_range=10.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    layout = ObstacleLayout(obstacles=(_far_box(),))
    perception.update(np.array([32.0, 0.0, 5.0]), layout, ())  # in range -> detected
    later = perception.update(np.array([-40.0, 0.0, 5.0]), layout, ())  # far away again
    assert len(later.static) == 1  # remembered despite leaving range


def test_same_seed_reproduces_identical_perception_history():
    from uav_defend.config.env_config import EnvConfig
    from uav_defend.envs import SoldierEnv

    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=4,
        obstacle_unmapped_fraction=0.5,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="map_local",
    )
    env_a, env_b = SoldierEnv(config=config), SoldierEnv(config=config)
    _, info_a = env_a.reset(seed=321)
    _, info_b = env_b.reset(seed=321)
    for _ in range(60):
        _, _, done_a, trunc_a, info_a = env_a.step(np.zeros(3, dtype=np.float32))
        _, _, done_b, trunc_b, info_b = env_b.step(np.zeros(3, dtype=np.float32))
        assert info_a["enemy_perceived_static_ids"] == info_b["enemy_perceived_static_ids"]
        assert info_a["enemy_total_perceived_obstacles"] == info_b["enemy_total_perceived_obstacles"]
        if done_a or trunc_a:
            break


def test_obstacle_exactly_at_sensing_boundary_handled_consistently():
    perception = EnemyObstaclePerception(
        mode="local", static_sensor_range=10.0, dynamic_sensor_range=10.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    box = AABBObstacle.from_center_half_extents((20.0, 0.0, 5.0), (2.0, 2.0, 5.0))
    layout = ObstacleLayout(obstacles=(box,))
    # Clearance to box surface exactly equals sensor range (20-2-8=10).
    perceived = perception.update(np.array([8.0, 0.0, 5.0]), layout, ())
    assert len(perceived.static) == 1  # boundary contact counts as perceived (<=)


def test_obstacle_extent_distance_not_just_center_distance():
    perception = EnemyObstaclePerception(
        mode="local", static_sensor_range=5.0, dynamic_sensor_range=5.0,
        static_memory_enabled=True, dynamic_memory_steps=5,
    )
    # A LARGE obstacle whose CENTER is far but whose near FACE is close.
    box = AABBObstacle.from_center_half_extents((30.0, 0.0, 5.0), (20.0, 2.0, 5.0))
    layout = ObstacleLayout(obstacles=(box,))
    # Center distance = 30m (way outside range), but surface distance = 30-20=10m (still outside 5m range)...
    perceived_far = perception.update(np.array([0.0, 0.0, 5.0]), layout, ())
    assert len(perceived_far.static) == 0
    # Move closer so surface distance <= 5m (center distance would still suggest "far" if used naively).
    perceived_near = perception.update(np.array([4.0, 0.0, 5.0]), layout, ())  # surface dist = 10-4=6... adjust
    perceived_near2 = perception.update(np.array([6.0, 0.0, 5.0]), layout, ())  # surface dist = 10-6=4 <= 5
    assert len(perceived_near2.static) == 1
