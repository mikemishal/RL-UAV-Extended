"""Hostile obstacle-navigation tests (journal extension, Phase 3B).

Covers the required test list (Step 12 of the Phase-3B specification):
 1. prediction follows u_base (not only the raw pursuit ray)
 2. current persistent velocity/inertia matters
 3-6. left / right / climb bypass, and climb rejected when infeasible
 7-9. candidate motion respects accel / turn-rate / climb-rate limits
 10. deterministic selection
 11. persistence prevents left/right oscillation
 12. avoidance releases after obstacle clearance
 13. normal pursuit resumes after release
 14. avoidance disabled reproduces the pre-obstacle hostile trajectory
 15. obstacles disabled reproduces conference behavior exactly

Plus the RNG-purity invariant (Step 13): the obstacle navigator consumes
NO randomness, and enabling/using it never perturbs the hostile RNG stream.

Run directly: python tests/test_enemy_obstacle_avoidance.py
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
    ObstacleAvoidancePlanner,
    generate_bypass_candidates,
    predict_collision,
    predict_constrained_trajectory,
)

_L = 50.0
_MAX_ALTITUDE = 30.0


def _limits(**overrides) -> DynamicsLimits:
    defaults = dict(
        max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0),
        max_climb_rate=5.0, max_descent_rate=5.0, dt=0.5, eps=1e-8, L=_L, max_altitude=_MAX_ALTITUDE,
    )
    defaults.update(overrides)
    return DynamicsLimits(**defaults)


def _building_config(**overrides) -> EnvConfig:
    defaults = dict(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        # Far from the origin so reset-time containment validation at the
        # true (0,0,0) reset position passes; scenarios reposition entities
        # via direct state overrides.
        obstacle_fixed_spec=((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),),
        enemy_obstacle_avoidance_enabled=True,
        defender_standby_until_detection=False,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _run_hostile_only(cfg: EnvConfig, seed: int, enemy_start, soldier_pos, n_steps: int = 45):
    """Run with the defender held far away (no interference) and the
    hostile's start position/velocity controlled directly."""
    env = SoldierEnv(config=cfg)
    env.reset(seed=seed)
    env._enemy_pos = np.asarray(enemy_start, dtype=np.float32)
    env._enemy_vel = np.zeros(3, dtype=np.float32)
    env._soldier_pos = np.asarray(soldier_pos, dtype=np.float32)
    env._defender_pos = np.array([1000.0, 1000.0, 0.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    positions = [env._enemy_pos.copy()]
    infos = []
    collided = False
    for _ in range(n_steps):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        positions.append(info["enemy_pos"].copy())
        infos.append(info)
        collided = collided or info["enemy_obstacle_collision"]
        if term or trunc:
            break
    return positions, infos, collided


# --- 1. Prediction follows u_base, not just the raw pursuit ray ------------

def test_prediction_follows_u_base_not_pursuit_ray_alone():
    """Pursuit ray (straight line hostile->soldier) is clear of the
    obstacle, but a strong reactive-evasion push (defender very close)
    bends u_base into the obstacle -- avoidance must still activate."""
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    soldier_pos = np.array([0.0, -100.0, 0.0])  # pursuit ray: straight down -y, clear of the obstacle (x~15)
    pursuit_ray_direction = np.array([0.0, -1.0, 0.0])
    assert not predict_collision(
        predict_constrained_trajectory(position, np.zeros(3), pursuit_ray_direction, _limits(), 12)[0], layout,
    ).collision  # sanity: the raw pursuit ray alone is clear

    # u_base bent toward the obstacle by a strong evasion/weave contribution.
    u_base = np.array([1.0, 0.0, 0.0])
    planner = ObstacleAvoidancePlanner(clearance=2.0, horizon_steps=12, release_steps=4, max_altitude=_MAX_ALTITUDE)
    result = planner.plan(position, np.zeros(3), u_base, layout, soldier_pos, _limits())
    assert result["active"] is True


# --- 2. Current persistent velocity/inertia matters -------------------------

def test_current_velocity_inertia_causes_predicted_collision():
    """u_base itself points safely away from the obstacle, but strong
    current velocity pointed AT the obstacle, combined with turn-rate
    limits, still causes a predicted collision -- avoidance must detect
    this (a naive "check only the new direction" scheme would miss it)."""
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([5.0, 0.0, 1.0])
    velocity = np.array([12.0, 0.0, 0.0])  # already at max speed, pointed straight at the obstacle
    u_base = np.array([0.0, 1.0, 0.0])  # desired direction is safely sideways

    # A naive scheme that only checks whether flying u_base FROM REST would
    # collide (ignoring current velocity) would say "clear":
    naive_positions, _ = predict_constrained_trajectory(position, np.zeros(3), u_base, _limits(), 12)
    assert not predict_collision(naive_positions, layout).collision

    # The REAL prediction, starting from the actual (nonzero) velocity,
    # must detect the collision caused by turn-rate-limited inertia.
    real_positions, _ = predict_constrained_trajectory(position, velocity, u_base, _limits(), 12)
    assert predict_collision(real_positions, layout).collision


# --- 3-6. Bypass candidates -------------------------------------------------

def test_left_bypass_candidate_is_collision_free():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    left = next(c for c in candidates if c.mode == "left")
    positions, _ = predict_constrained_trajectory(position, np.zeros(3), left.direction, _limits(), 12)
    assert not predict_collision(positions, layout).collision


def test_right_bypass_candidate_is_collision_free():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    right = next(c for c in candidates if c.mode == "right")
    positions, _ = predict_constrained_trajectory(position, np.zeros(3), right.direction, _limits(), 12)
    assert not predict_collision(positions, layout).collision


def test_climb_bypass_candidate_is_collision_free_for_short_obstacle():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 1.0), (10.0, 10.0, 1.0))  # height 2, wide footprint
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    climb = next(c for c in candidates if c.mode == "climb")
    positions, _ = predict_constrained_trajectory(position, np.zeros(3), climb.direction, _limits(), 12)
    assert not predict_collision(positions, layout).collision


def test_impossible_climb_due_to_altitude_ceiling_is_rejected():
    # Obstacle top + clearance exceeds max_altitude: climb candidate must
    # be entirely omitted, not merely a "bad" candidate.
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 20.0), (4.0, 4.0, 20.0))  # top at z=40 > max_altitude=30
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    assert all(c.mode != "climb" for c in candidates)
    assert {c.mode for c in candidates} == {"left", "right"}


# --- 7-9. Candidate motion respects dynamic limits --------------------------

def test_candidate_motion_respects_acceleration_limit():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    for candidate in candidates:
        # Short horizon: stays well within the domain, isolating the
        # accel-limiting behavior from unrelated domain-boundary velocity
        # resets (a separate, already-tested physical event).
        _, velocities = predict_constrained_trajectory(position, np.zeros(3), candidate.direction, limits, 4)
        for k in range(len(velocities) - 1):
            delta_v = np.linalg.norm(velocities[k + 1] - velocities[k])
            assert delta_v <= limits.max_accel * limits.dt + 1e-6


def test_candidate_motion_respects_turn_rate_limit():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    velocity = np.array([12.0, 0.0, 0.0])  # nonzero initial heading so turn-rate limiting is exercised
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    max_delta_heading = limits.max_turn_rate_rad * limits.dt + 1e-6
    for candidate in candidates:
        # Short horizon: avoids incidental domain-boundary velocity resets.
        _, velocities = predict_constrained_trajectory(position, velocity, candidate.direction, limits, 4)
        for k in range(len(velocities) - 1):
            h0 = velocities[k][:2]
            h1 = velocities[k + 1][:2]
            if np.linalg.norm(h0) <= limits.eps or np.linalg.norm(h1) <= limits.eps:
                continue
            cos_angle = np.clip(np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1)), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            assert angle <= max_delta_heading


def test_candidate_motion_respects_climb_rate_limit():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 1.0), (10.0, 10.0, 1.0))
    position = np.array([0.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    candidates = generate_bypass_candidates(position, obstacle, u_base, clearance=2.0, max_altitude=_MAX_ALTITUDE)
    for candidate in candidates:
        _, velocities = predict_constrained_trajectory(position, np.zeros(3), candidate.direction, limits, 12)
        for v in velocities:
            assert -limits.max_descent_rate - 1e-6 <= v[2] <= limits.max_climb_rate + 1e-6


# --- 10. Deterministic selection --------------------------------------------

def test_avoidance_choice_is_deterministic():
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([-10.0, 0.0, 1.0])
    velocity = np.zeros(3)
    u_base = np.array([1.0, 0.0, 0.0])
    soldier_pos = np.array([40.0, 0.0, 0.0])
    limits = _limits()

    results = []
    for _ in range(3):
        planner = ObstacleAvoidancePlanner(clearance=2.0, horizon_steps=12, release_steps=4, max_altitude=_MAX_ALTITUDE)
        results.append(planner.plan(position, velocity, u_base, layout, soldier_pos, limits))
    assert all(r["mode"] == results[0]["mode"] for r in results)
    assert all(np.array_equal(r["direction"], results[0]["direction"]) for r in results)


# --- 11-13. Persistence, release, resumed pursuit --------------------------

def test_persistence_prevents_oscillation_and_releases_then_resumes_pursuit():
    positions, infos, collided = _run_hostile_only(
        _building_config(), seed=1, enemy_start=(-20.0, 0.0, 3.0), soldier_pos=(40.0, 0.0, 0.0),
    )
    assert not collided
    modes = [i["enemy_obstacle_avoidance_mode"] for i in infos if i["enemy_obstacle_avoidance_active"]]
    assert len(modes) > 0
    # 11. persistence: the SAME mode is held for consecutive active steps,
    # not re-decided (potentially oscillating) every single step.
    run_lengths = []
    current_run = 1
    for a, b in zip(modes, modes[1:]):
        if a == b:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    assert max(run_lengths) >= 2

    # 12. avoidance releases at some point (active becomes False again
    # after having been True).
    active_flags = [i["enemy_obstacle_avoidance_active"] for i in infos]
    first_active = active_flags.index(True)
    assert False in active_flags[first_active:]
    release_index = active_flags.index(False, first_active)

    # 13. after release, normal pursuit resumes: mode is cleared.
    assert infos[release_index]["enemy_obstacle_avoidance_mode"] is None


# --- 14. Avoidance disabled reproduces the pre-obstacle hostile trajectory --

def test_avoidance_disabled_matches_no_obstacle_hostile_trajectory():
    no_obstacle_cfg = EnvConfig(defender_standby_until_detection=False)
    obstacles_present_avoidance_off_cfg = _building_config(enemy_obstacle_avoidance_enabled=False)
    for seed in (0, 1, 5):
        env_a = SoldierEnv(config=no_obstacle_cfg)
        env_b = SoldierEnv(config=obstacles_present_avoidance_off_cfg)
        env_a.reset(seed=seed)
        env_b.reset(seed=seed)
        for env in (env_a, env_b):
            env._enemy_pos = np.array([-20.0, 0.0, 3.0], dtype=np.float32)
            env._enemy_vel = np.zeros(3, dtype=np.float32)
            env._soldier_pos = np.array([40.0, 0.0, 0.0], dtype=np.float32)
            env._defender_pos = np.array([1000.0, 1000.0, 0.0], dtype=np.float32)
            env._defender_vel = np.zeros(3, dtype=np.float32)
        for _ in range(30):
            _, _, term_a, trunc_a, info_a = env_a.step(np.zeros(3, dtype=np.float32))
            _, _, term_b, trunc_b, info_b = env_b.step(np.zeros(3, dtype=np.float32))
            assert info_b["enemy_obstacle_avoidance_active"] is False
            assert np.array_equal(info_a["enemy_pos"], info_b["enemy_pos"])
            assert np.array_equal(info_a["enemy_vel"], info_b["enemy_vel"])
            if term_a or trunc_a:
                break


# --- 15. Obstacles disabled reproduces conference behavior exactly --------

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
        assert info_d["enemy_obstacle_avoidance_active"] is False
        if term_b or trunc_b:
            break


# --- Step 13: RNG purity -----------------------------------------------------

def test_obstacle_navigation_consumes_no_extra_rng_draws():
    """Enabling avoidance (which actively engages, redirecting the
    hostile) must not change the NUMBER OF DRAWS consumed from
    `_rng_enemy_motion` -- the planner is entirely deterministic/RNG-free."""
    no_avoidance_cfg = _building_config(enemy_obstacle_avoidance_enabled=False)
    avoidance_cfg = _building_config(enemy_obstacle_avoidance_enabled=True)
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
    assert activated  # the scenario must actually exercise avoidance

    draw_a = env_a._rng_enemy_motion.normal(size=5)
    draw_b = env_b._rng_enemy_motion.normal(size=5)
    assert np.array_equal(draw_a, draw_b)


def test_obstacle_navigation_consumes_no_randomness_directly():
    """The planner/prediction primitives take no RNG argument at all and
    are pure functions of their inputs -- calling them repeatedly with
    identical inputs must always return identical output (already also
    covered behaviorally by test_avoidance_choice_is_deterministic)."""
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, 5.0), (4.0, 4.0, 5.0))
    layout = ObstacleLayout(obstacles=(obstacle,))
    position = np.array([-10.0, 0.0, 1.0])
    u_base = np.array([1.0, 0.0, 0.0])
    limits = _limits()
    positions_a, _ = predict_constrained_trajectory(position, np.zeros(3), u_base, limits, 12)
    positions_b, _ = predict_constrained_trajectory(position, np.zeros(3), u_base, limits, 12)
    assert all(np.array_equal(a, b) for a, b in zip(positions_a, positions_b))
    assert predict_collision(positions_a, layout) == predict_collision(positions_b, layout)


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
