"""
Deterministic tests for hostile-UAS dynamic-obstacle perception (Section
18): finite sensing range, track memory/expiry, and reacquisition.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.obstacles import ObstacleLayout
from uav_defend.obstacles.dynamic import DynamicObstacle
from uav_defend.obstacles.perception import EnemyObstaclePerception


def _bird(id_=0, position=(20.0, 0.0, 10.0), velocity=(0.0, 2.0, 0.0), radius=0.5):
    return DynamicObstacle(id=id_, obstacle_type="bird",
                            position=np.array(position, dtype=np.float64),
                            velocity=np.array(velocity, dtype=np.float64), radius=radius)


def _perception(mode="local", dynamic_range=10.0, memory_steps=3):
    return EnemyObstaclePerception(
        mode=mode, static_sensor_range=10.0, dynamic_sensor_range=dynamic_range,
        static_memory_enabled=True, dynamic_memory_steps=memory_steps,
    )


def test_dynamic_obstacle_outside_sensor_range_absent():
    perception = _perception(dynamic_range=10.0)
    bird = _bird(position=(30.0, 0.0, 10.0))
    perceived = perception.update(np.array([0.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    assert len(perceived.dynamic) == 0


def test_dynamic_obstacle_visible_after_entering_range():
    perception = _perception(dynamic_range=10.0)
    bird = _bird(position=(30.0, 0.0, 10.0))
    far = perception.update(np.array([0.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    near = perception.update(np.array([25.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    assert len(far.dynamic) == 0
    assert len(near.dynamic) == 1
    assert near.dynamic[0].source == "local_sensor"


def test_level_c_does_not_know_unsensed_bird_location():
    perception = _perception(mode="map_local", dynamic_range=5.0)
    bird = _bird(position=(30.0, 0.0, 10.0))
    perceived = perception.update(np.array([0.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    assert len(perceived.dynamic) == 0  # map knowledge never applies to dynamic obstacles


def test_velocity_available_only_once_sensed():
    perception = _perception(dynamic_range=10.0)
    bird = _bird(position=(30.0, 0.0, 10.0), velocity=(1.0, 2.0, 0.0))
    near = perception.update(np.array([25.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    assert len(near.dynamic) == 1
    assert np.array_equal(near.dynamic[0].velocity, bird.velocity)


def test_track_expires_after_memory_timeout():
    perception = _perception(dynamic_range=10.0, memory_steps=2)
    bird = _bird(position=(25.0, 0.0, 10.0))
    perception.update(np.array([20.0, 0.0, 10.0]), ObstacleLayout(), (bird,))  # observed (dist=5)
    far_pos = np.array([-40.0, 0.0, 10.0])
    step1 = perception.update(far_pos, ObstacleLayout(), (bird,))  # age 1 (memory_steps=2 -> still remembered)
    step2 = perception.update(far_pos, ObstacleLayout(), (bird,))  # age 2 -> still remembered (<=2)
    step3 = perception.update(far_pos, ObstacleLayout(), (bird,))  # age 3 -> expired
    assert len(step1.dynamic) == 1
    assert len(step2.dynamic) == 1
    assert len(step3.dynamic) == 0


def test_reacquired_dynamic_obstacle_updates_correctly():
    perception = _perception(dynamic_range=10.0, memory_steps=2)
    bird = _bird(position=(25.0, 0.0, 10.0), velocity=(0.0, 1.0, 0.0))
    perception.update(np.array([20.0, 0.0, 10.0]), ObstacleLayout(), (bird,))
    moved_bird = _bird(position=(25.0, 5.0, 10.0), velocity=(0.0, 1.0, 0.0))
    reacquired = perception.update(np.array([20.0, 5.0, 10.0]), ObstacleLayout(), (moved_bird,))
    assert len(reacquired.dynamic) == 1
    assert np.array_equal(reacquired.dynamic[0].position, moved_bird.position)


def test_moving_obstacle_crossing_sensing_boundary_deterministic():
    perception_a = _perception(dynamic_range=10.0, memory_steps=1)
    perception_b = _perception(dynamic_range=10.0, memory_steps=1)
    positions = [np.array([float(x), 0.0, 10.0]) for x in (0, 5, 10, 15, 20)]
    bird = _bird(position=(15.0, 0.0, 10.0))
    history_a = [len(perception_a.update(p, ObstacleLayout(), (bird,)).dynamic) for p in positions]
    history_b = [len(perception_b.update(p, ObstacleLayout(), (bird,)).dynamic) for p in positions]
    assert history_a == history_b


def test_dynamic_avoidance_only_when_perceived():
    from uav_defend.config.env_config import EnvConfig
    from uav_defend.envs import SoldierEnv

    config = EnvConfig(
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=3,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        enemy_dynamic_obstacle_sensor_range=8.0,
    )
    env = SoldierEnv(config=config)
    _, info = env.reset(seed=41)
    violation = False
    for _ in range(300):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_avoidance_obstacle_type"] in ("bird", "airborne_clutter") and info["enemy_num_locally_sensed_dynamic"] == 0:
            violation = True
        if done or truncated:
            break
    assert not violation


def test_no_stale_dynamic_obstacle_causes_indefinite_avoidance():
    from uav_defend.config.env_config import EnvConfig
    from uav_defend.envs import SoldierEnv

    config = EnvConfig(
        dynamic_obstacles_enabled=True, dynamic_obstacle_count=2,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        enemy_dynamic_obstacle_memory_steps=3,
    )
    env = SoldierEnv(config=config)
    _, info = env.reset(seed=42)
    active_flags = []
    for _ in range(400):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        active_flags.append(info["enemy_avoidance_active"])
        if done or truncated:
            break
    if any(active_flags):
        assert not all(active_flags)


def test_multiple_locally_perceived_dynamic_obstacles_returned_correctly():
    perception = _perception(dynamic_range=15.0)
    birds = (_bird(id_=0, position=(10.0, 0.0, 10.0)), _bird(id_=1, position=(-10.0, 0.0, 10.0)),
             _bird(id_=2, position=(0.0, 10.0, 10.0)))
    perceived = perception.update(np.array([0.0, 0.0, 10.0]), ObstacleLayout(), birds)
    assert len(perceived.dynamic) == 3
    assert {o.id for o in perceived.dynamic} == {0, 1, 2}
