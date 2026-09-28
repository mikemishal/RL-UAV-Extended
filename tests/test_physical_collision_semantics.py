"""
Deterministic tests for physical obstacle-collision semantics (journal
extension, physical-collision-semantics phase): a ground-truth static or
dynamic obstacle collision now terminates the episode immediately with a
dedicated outcome, clamps the colliding entity's position to its first
contact point (never letting it pass through), and is never misclassified
as any of the four original mission outcomes. The shared reactive
obstacle-safety layer (Level B/C perception, avoidance) is unchanged --
this phase only changes what happens when that layer FAILS (or is
disabled) and a real collision occurs.

Covers the numbered scenario list for this phase:
    1.  defender static collision terminates
    2.  defender dynamic collision terminates
    3.  attacker static collision terminates
    4.  attacker dynamic collision terminates
    5.  collision is not misclassified as interception
    6.  attacker collision is not counted as defender interception
    7.  swept collision prevents tunneling (thin wall / crossing bird / high speed)
    8.  safe near-miss does not terminate
    9.  first event wins when collision/interception happen in the same step
    10. Level B unseen obstacle can still physically collide
    11. Level C mapped obstacle avoids collision when navigation succeeds
    12. safety-layer failure results in actual physical failure
    13. C0 exact regression preserved
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.obstacles.dynamic import DynamicObstacle
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info


def _safe_fixed_spec(center: np.ndarray, half_xy: float, half_z: float, config: EnvConfig) -> tuple:
    margin = 0.5
    cx = float(np.clip(center[0], -config.L + half_xy + margin, config.L - half_xy - margin))
    cy = float(np.clip(center[1], -config.L + half_xy + margin, config.L - half_xy - margin))
    cz = float(np.clip(center[2], half_z + margin, config.max_altitude - half_z - margin))
    return (cx, cy, cz, half_xy, half_xy, half_z)


# --- 1. defender static collision terminates --------------------------------

def test_defender_static_collision_terminates():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, reward, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["defender_obstacle_collision"]:
            assert term is True
            assert trunc is False
            assert info["outcome"] == "defender_static_obstacle_collision"
            assert reward == config.reward_defender_static_obstacle_collision
            return
    pytest.fail("expected a defender static collision within 10 steps")


# --- 2. defender dynamic collision terminates -------------------------------

def test_defender_dynamic_collision_terminates():
    config = EnvConfig(dynamic_obstacles_enabled=True, defender_standby_until_detection=False)
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    env._dynamic_obstacles = (
        DynamicObstacle(id=0, obstacle_type="bird", position=np.array([5.0, 0.0, 5.0]),
                         velocity=np.zeros(3), radius=1.0),
    )
    for _ in range(10):
        _, reward, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["defender_dynamic_obstacle_collision"]:
            assert term is True
            assert info["outcome"] == "defender_dynamic_obstacle_collision"
            assert reward == config.reward_defender_dynamic_obstacle_collision
            return
    pytest.fail("expected a defender dynamic collision within 10 steps")


# --- 3. attacker static collision terminates ---------------------------------

def test_attacker_static_collision_terminates():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        enemy_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._enemy_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._enemy_vel = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(10):
        _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_obstacle_collision"]:
            assert term is True
            assert info["outcome"] == "enemy_static_obstacle_collision"
            assert reward == config.reward_enemy_static_obstacle_collision
            return
    pytest.fail("expected an enemy static collision within 10 steps")


# --- 4. attacker dynamic collision terminates --------------------------------

def test_attacker_dynamic_collision_terminates():
    config = EnvConfig(dynamic_obstacles_enabled=True, defender_standby_until_detection=False)
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._soldier_pos = np.array([20.0, 0.0, 0.0], dtype=np.float32)  # enemy pursuit heads +x, toward the bird
    env._enemy_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    env._enemy_vel = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    # Close and large enough to catch the enemy's very next displacement
    # regardless of exact weave/pursuit heading this step.
    env._dynamic_obstacles = (
        DynamicObstacle(id=0, obstacle_type="bird", position=np.array([2.0, 0.0, 5.0]),
                         velocity=np.zeros(3), radius=3.0),
    )
    for _ in range(10):
        _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_dynamic_obstacle_collision"]:
            assert term is True
            assert info["outcome"] == "enemy_dynamic_obstacle_collision"
            assert reward == config.reward_enemy_dynamic_obstacle_collision
            return
    pytest.fail("expected an enemy dynamic collision within 10 steps")


# --- 5. collision is not misclassified as interception -----------------------

def test_collision_not_misclassified_as_mission_outcome():
    """A defender collision that ALSO happens to land within intercept
    range of the enemy must be reported as a collision, never as
    intercepted/soldier_caught/unsafe_intercept/timeout."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-30.0, 0.0, 5.0, 4.0, 4.0, 5.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
        intercept_radius=50.0,  # generously wide, so intercept geometry is ALSO satisfied
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-40.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    env._enemy_pos = np.array([-30.0, 0.0, 5.0], dtype=np.float32)  # inside the obstacle's own footprint region
    env._enemy_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["defender_obstacle_collision"]:
            assert info["outcome"] == "defender_static_obstacle_collision"
            assert info["outcome"] not in ("intercepted", "soldier_caught", "unsafe_intercept", "timeout")
            return
    pytest.fail("expected a defender static collision within 10 steps")


# --- 6. attacker collision is not counted as defender interception ----------

def test_attacker_collision_not_counted_as_interception():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        enemy_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
        intercept_radius=100.0,  # defender is "close enough" by geometry alone
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._enemy_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._enemy_vel = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    env._defender_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # clear of the wall's [-15,-5] footprint
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_obstacle_collision"]:
            assert info["outcome"] == "enemy_static_obstacle_collision"
            assert info["outcome"] != "intercepted"
            return
    pytest.fail("expected an enemy static collision within 10 steps")


# --- 7. swept collision prevents tunneling ----------------------------------

def test_tunneling_prevented_thin_wall_high_speed():
    """A thin wall crossed in a single dt at high relative speed must
    still be detected via swept intersection, not skipped because only
    the (pre/post) endpoints are checked."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        # Thin wall: only 0.5 half-extent in x (1 unit thick).
        obstacle_fixed_spec=((-30.0, 0.0, 5.0, 0.5, 20.0, 5.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
        v_d=40.0,  # fast enough to cross in well under one dt otherwise
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-35.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.array([40.0, 0.0, 0.0], dtype=np.float32)
    _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["defender_obstacle_collision"] is True
    assert term is True
    assert info["outcome"] == "defender_static_obstacle_collision"


def test_tunneling_prevented_crossing_bird():
    """A fast-crossing dynamic obstacle (bird) that only briefly overlaps
    the defender's path must still register as a first-contact collision
    via the swept sphere-vs-sphere check, not merely a same-step endpoint
    check."""
    config = EnvConfig(dynamic_obstacles_enabled=True, defender_standby_until_detection=False,
                        v_d=20.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([0.0, -10.0, 5.0], dtype=np.float32)
    env._defender_vel = np.array([0.0, 20.0, 0.0], dtype=np.float32)
    # Bird crossing left-to-right exactly through the defender's path at
    # y=0, timed so the two swept segments intersect mid-step (not at
    # either segment's endpoint).
    env._dynamic_obstacles = (
        DynamicObstacle(id=0, obstacle_type="bird", position=np.array([-5.0, 0.0, 5.0]),
                         velocity=np.array([10.0, 0.0, 0.0]), radius=1.0),
    )
    _, _, term, trunc, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert info["defender_dynamic_obstacle_collision"] is True
    assert term is True
    assert info["outcome"] == "defender_dynamic_obstacle_collision"


def test_tunneling_prevented_high_relative_speed_dynamic():
    """Both the mover and the dynamic obstacle moving quickly toward each
    other (high closing speed) must still be caught by the swept check,
    not just the min-distance-at-endpoints heuristic."""
    config = EnvConfig(dynamic_obstacles_enabled=True, defender_standby_until_detection=False,
                        v_d=30.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-8.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.array([30.0, 0.0, 0.0], dtype=np.float32)
    env._dynamic_obstacles = (
        DynamicObstacle(id=0, obstacle_type="bird", position=np.array([8.0, 0.0, 5.0]),
                         velocity=np.array([-30.0, 0.0, 0.0]), radius=1.0),
    )
    _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["defender_dynamic_obstacle_collision"] is True
    assert term is True


# --- 8. safe near-miss does not terminate -----------------------------------

def test_safe_near_miss_does_not_terminate():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-30.0, 0.0, 5.0, 2.0, 2.0, 5.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    # Passes well clear of the obstacle's y in [-2, 2] footprint.
    env._defender_pos = np.array([-40.0, 10.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert info["defender_obstacle_collision"] is False
        assert term is False
        assert info["outcome"] == "ongoing"


# --- 9. first event wins: collision vs interception in the same step -------

def test_collision_wins_over_interception_in_same_step():
    """Phase 6 precedence: if a collision AND intercept-range geometry are
    both satisfied in the same step, the collision outcome wins."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
        intercept_radius=100.0,  # trivially satisfied every step
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    env._enemy_pos = np.array([-29.0, 0.0, 2.0], dtype=np.float32)  # well within intercept_radius
    env._enemy_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["defender_obstacle_collision"]:
            assert info["outcome"] == "defender_static_obstacle_collision"
            assert term is True
            return
    pytest.fail("expected a defender static collision within 10 steps")


# --- 10. Level B unseen obstacle can still physically collide --------------

def test_level_b_unseen_obstacle_still_physically_collides():
    """Perception is separate from physics: an obstacle outside the
    defender's Level B local sensor range is invisible to the safety layer
    but still causes a REAL ground-truth collision if flown into."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="local",
        defender_obstacle_sensor_range=1.0,  # far too short to see the wall in advance
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(15):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["defender_obstacle_collision"]:
            assert info["outcome"] == "defender_static_obstacle_collision"
            assert term is True
            return
    pytest.fail("expected the unseen wall to still cause a physical collision")


# --- 11. Level C mapped obstacle avoids collision when navigation succeeds --

def test_level_c_mapped_obstacle_avoids_collision():
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        obstacle_fixed_map_known=(True,),
        defender_obstacle_avoidance_enabled=True, defender_obstacle_awareness_mode="map_local",
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(60):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert info["defender_obstacle_collision"] is False
        if term or trunc:
            break


# --- 12. safety-layer failure results in actual physical failure -----------

def test_safety_layer_disabled_results_in_physical_collision():
    """With the safety layer explicitly disabled, a controller flying
    straight at an obstacle physically collides -- confirming collision
    consequences are real, not merely diagnostic, when the safety layer
    does not intervene."""
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 40.0, 2.0),),
        defender_obstacle_avoidance_enabled=False,
        defender_standby_until_detection=False,
    )
    env = SoldierEnv(config=config)
    env.reset(seed=1)
    env._defender_pos = np.array([-30.0, 0.0, 2.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if term or trunc:
            assert info["outcome"] == "defender_static_obstacle_collision"
            return
    pytest.fail("expected termination via physical collision within 10 steps")


# --- 13. C0 exact regression preserved --------------------------------------

def test_c0_exact_regression_preserved():
    """With obstacles/dynamic-obstacles disabled, behavior is bit-for-bit
    identical to the plain conference-baseline EnvConfig() -- the new
    physical-collision code paths are simply never reached."""
    baseline_config = EnvConfig()
    c0_config = EnvConfig(obstacles_enabled=False, dynamic_obstacles_enabled=False)
    for seed in (0, 1, 42):
        env_a = SoldierEnv(config=baseline_config)
        env_b = SoldierEnv(config=c0_config)
        obs_a, info_a = env_a.reset(seed=seed)
        obs_b, info_b = env_b.reset(seed=seed)
        policy_a, policy_b = GreedyInterceptPolicy(), GreedyInterceptPolicy()
        policy_a.reset()
        policy_b.reset()
        for _ in range(200):
            action_a = np.asarray(policy_a.act(obs_a, build_policy_info(info_a, "measurement")), dtype=np.float32)
            action_b = np.asarray(policy_b.act(obs_b, build_policy_info(info_b, "measurement")), dtype=np.float32)
            obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(action_a)
            obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(action_b)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert reward_a == reward_b
            assert term_a == term_b
            assert trunc_a == trunc_b
            assert info_a["outcome"] == info_b["outcome"]
            if term_a or trunc_a:
                break
