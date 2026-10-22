"""Predictor-vs-execution fidelity of the V2 attacker's obstacle planner.

The planner scores each bypass candidate with a predicted trajectory. These tests
fly the real `SoldierEnv` and require that the prediction made when a candidate is
selected equals what the environment then actually executes, under constrained
vehicle dynamics (turn-rate, acceleration, speed, climb-rate limits) and with only
the documented external changes (moving asset, newly perceived obstacles, moving
obstacles, a later reselection) excluded: here the asset is stationary and the
whole static layout is perceived.

No training and no learned policy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import uav_defend.obstacles.avoidance as avoidance
from uav_defend.config.env_config import EnvConfig
from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.avoidance import (
    DynamicsLimits,
    ObstacleAvoidancePlanner,
    _executed_step,
    predict_constrained_trajectory,
    predict_executed_trajectory,
    predict_waypoint_following_trajectory,
    truncate_prediction_at_target,
)
from uav_defend.obstacles.dynamic import DynamicObstacle
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout

from experiments.journal_v2_protocol import V2_SEMANTIC_FIELDS, build_v1_env_config, build_v2_env_config

FAR_DEFENDER = np.array([1000.0, 1000.0, 0.0], dtype=np.float32)
TOLERANCE_M = 1e-5  # state is float32 end to end; the replay is bit-exact, this only guards platform noise


def _config(spec, **overrides) -> EnvConfig:
    params = dict(
        V2_SEMANTIC_FIELDS, obstacles_enabled=True, obstacle_layout_mode="fixed", obstacle_fixed_spec=spec,
        enemy_obstacle_avoidance_enabled=True, defender_standby_until_detection=False,
    )
    params.update(overrides)
    return EnvConfig(**params)


class Flight:
    """One recorded flight with the planner's selections and pre-step states."""

    def __init__(self):
        self.pre: list[tuple[np.ndarray, np.ndarray]] = []
        self.pos: list[np.ndarray] = []
        self.vel: list[np.ndarray] = []
        self.selected_at: list[int] = []  # 0-based step indices at which a candidate was (re)selected
        self.waypoints: list[np.ndarray | None] = []
        self.active: list[bool] = []
        self.modes: list[str | None] = []
        self.layouts: list[ObstacleLayout] = []
        self.saturation: list[dict] = []
        self.limits: DynamicsLimits | None = None
        self.target: np.ndarray | None = None
        self.stop_radius: float | None = None
        self.u_base: list[np.ndarray] = []


def fly(config, enemy_pos, enemy_vel, soldier_pos, steps, monkeypatch, dynamic_obstacles=()) -> Flight:
    env = SoldierEnv(config)
    env.reset(seed=0)
    if dynamic_obstacles:
        env._dynamic_obstacles = tuple(dynamic_obstacles)  # stationary: the swept box then never changes
    env._enemy_pos = np.asarray(enemy_pos, dtype=np.float32)
    env._enemy_vel = np.asarray(enemy_vel, dtype=np.float32)
    env._soldier_pos = np.asarray(soldier_pos, dtype=np.float32)
    env._defender_pos = FAR_DEFENDER.copy()
    env._defender_vel = np.zeros(3, dtype=np.float32)
    env._move_soldier = lambda: None  # stationary asset: removes an external change
    planner = env._enemy_obstacle_planner

    flight = Flight()
    flight.limits = env._enemy_dynamics_limits()
    flight.target = env._soldier_pos.copy()
    flight.stop_radius = planner.mission_stop_radius
    selections = []
    original = avoidance.evaluate_and_select_candidate

    def counting(*args, **kwargs):
        selections.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(avoidance, "evaluate_and_select_candidate", counting)
    original_plan = planner.plan

    def observing_plan(position, velocity, u_base, layout, target_position, limits):
        flight.u_base.append(np.array(u_base, dtype=np.float64))
        flight.layouts.append(layout)
        return original_plan(position, velocity, u_base, layout, target_position, limits)

    planner.plan = observing_plan
    for step in range(steps):
        flight.pre.append((env._enemy_pos.copy(), env._enemy_vel.copy()))
        before = len(selections)
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        if len(selections) > before:
            flight.selected_at.append(step)
        flight.pos.append(env._enemy_pos.copy())
        flight.vel.append(env._enemy_vel.copy())
        flight.waypoints.append(None if planner.waypoint is None else planner.waypoint.copy())
        flight.active.append(bool(planner.active))
        flight.modes.append(planner.mode)
        flight.saturation.append(dict(env._enemy_dynamics_info))
        if terminated or truncated:
            break
    return flight


def first_interval(flight: Flight, horizon: int):
    """(step, steps) of the first selection and how long it stays the selection."""
    assert flight.selected_at, "the scenario never triggered an avoidance selection"
    start = flight.selected_at[0]
    later = [s for s in flight.selected_at if s > start]
    stop = later[0] if later else len(flight.pos)
    return start, min(stop - start, horizon, len(flight.pos) - start)


def predicted(flight: Flight, start: int, steps: int, horizon: int):
    p0, v0 = flight.pre[start]
    waypoint = flight.waypoints[start]
    positions, velocities, release = predict_executed_trajectory(
        p0, v0, waypoint, flight.u_base[start], flight.target, flight.layouts[start], flight.limits, horizon,
        flight.stop_radius,
    )
    # An episode ends at the first swept contact with the asset, and the planner cuts its own
    # prediction there too (the final environment position is the contact point).
    positions = truncate_prediction_at_target(positions, flight.target, flight.stop_radius)
    return positions[: steps + 1], velocities[: steps + 1], release


def _same_layout(a: ObstacleLayout, b: ObstacleLayout) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(x.min_corner, y.min_corner) and np.array_equal(x.max_corner, y.max_corner)
        for x, y in zip(a.obstacles, b.obstacles)
    )


def stable_steps(flight: Flight, start: int, steps: int) -> int:
    """Steps over which the planner's obstacle layout stays what it was at selection.

    The prediction treats the layout as static (the planner's own assumption), so a
    change in the perceived/threatening obstacle set is a documented external change
    that ends the comparison interval. Static scenarios never change."""
    stable = 1
    while stable < steps and _same_layout(flight.layouts[start + stable], flight.layouts[start]):
        stable += 1
    return stable


def assert_prediction_matches_execution(flight: Flight, horizon: int = 12, min_steps: int = 3):
    start, steps = first_interval(flight, horizon)
    steps = stable_steps(flight, start, steps)
    assert steps >= min_steps, f"interval of {steps} step(s) is too short to be a meaningful comparison"
    positions, velocities, release = predicted(flight, start, steps, horizon)
    actual = [flight.pre[start][0]] + flight.pos[start : start + steps]
    errors = [float(np.linalg.norm(a - b)) for a, b in zip(positions, actual)]
    assert max(errors) < TOLERANCE_M, f"max prediction error {max(errors):.3e} m"
    actual_release = next((k + 1 for k in range(steps) if not flight.active[start + k]), None)
    assert (release if release is not None and release <= steps else None) == actual_release
    return start, steps, release, errors


# ----------------------------------------------------------------- scenarios
WALL = ((15.0, 0.0, 5.0, 4.0, 50.0, 5.0),)          # full-width wall, top 10 m: only a climb can clear it
TALL_WALL = ((15.0, 0.0, 10.0, 4.0, 50.0, 10.0),)   # top 20 m: a long, climb-rate-limited ascent
SMALL_BUILDING = ((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),)
ASSET = (40.0, 0.0, 0.0)


def _fixed_direction_error(flight: Flight, start: int, steps: int) -> float:
    """Worst position error of the ORIGINAL predictor (initial direction held for the horizon)."""
    p0, v0 = flight.pre[start]
    direction = flight.waypoints[start] - p0.astype(np.float64)
    direction = direction / np.linalg.norm(direction)
    positions, _ = predict_constrained_trajectory(p0, v0, direction, flight.limits, steps)
    actual = [p0] + flight.pos[start : start + steps]
    return max(float(np.linalg.norm(a - b)) for a, b in zip(positions, actual))


def test_a_high_speed_persisted_climb_matches_execution(monkeypatch):
    flight = fly(_config(WALL), (-25, 0, 1), (12, 0, 0), ASSET, 40, monkeypatch)
    start, steps, release, _ = assert_prediction_matches_execution(flight)
    assert flight.modes[start] == "climb" and steps == 12
    assert np.linalg.norm(flight.pre[start][1]) == pytest.approx(12.0, abs=1e-4)  # high speed
    # The waypoint is the point above the activation position, which the vehicle leaves at once.
    assert _fixed_direction_error(flight, start, steps) > 5.0  # the original predictor was far off


def test_b_lateral_waypoint_matches_execution_including_release_to_direct(monkeypatch):
    flight = fly(_config(SMALL_BUILDING), (0, 0, 1), (12, 0, 0), ASSET, 40, monkeypatch)
    start, steps, release, _ = assert_prediction_matches_execution(flight)
    assert flight.modes[start] in ("left", "right") and release is not None
    assert not flight.active[start + release - 1]  # execution really returned to DIRECT then
    assert _fixed_direction_error(flight, start, steps) > 1.0


def test_c_turn_rate_limited_reversal_matches_execution(monkeypatch):
    """Heading opposite the asset bearing: the turn direction is decided by rounding."""
    flight = fly(_config(((0.0, 6.0, 5.0, 3.0, 3.0, 5.0),)), (6, 0, 1), (-12, 0, 0), ASSET, 40, monkeypatch)
    start, steps, _, _ = assert_prediction_matches_execution(flight)
    heading = np.degrees(np.arctan2(flight.pre[start][1][1], flight.pre[start][1][0]))
    assert abs(abs(heading) - 180.0) < 1e-3
    assert any(info["turn_saturated"] for info in flight.saturation[start : start + steps])


def test_c2_turn_rate_limited_quarter_turn_matches_execution(monkeypatch):
    flight = fly(_config(((12.0, 0.0, 5.0, 3.0, 8.0, 5.0),)), (8, 2, 1), (0, 12, 0), ASSET, 40, monkeypatch)
    start, steps, _, _ = assert_prediction_matches_execution(flight)
    assert any(info["turn_saturated"] for info in flight.saturation[start : start + steps])


def test_d_climb_rate_limited_ascent_matches_execution(monkeypatch):
    flight = fly(_config(TALL_WALL), (-25, 0, 1), (12, 0, 0), ASSET, 40, monkeypatch)
    start, steps, _, _ = assert_prediction_matches_execution(flight)
    assert flight.modes[start] == "climb"
    assert any(info["climb_saturated"] for info in flight.saturation[start : start + steps])
    assert max(p[2] for p in flight.pos[start : start + steps]) > 15.0  # a long climb at 5 m/s


def test_e_acceleration_limited_start_from_rest_matches_execution(monkeypatch):
    flight = fly(_config(SMALL_BUILDING), (8, 0, 1), (0, 0, 0), ASSET, 40, monkeypatch)
    start, steps, _, _ = assert_prediction_matches_execution(flight)
    assert any(info["accel_saturated"] for info in flight.saturation[start : start + steps])


def test_f_straight_persisted_flight_agrees_with_the_fixed_direction_model():
    """No turn needed: holding the direction and re-deriving it coincide, so the models must agree."""
    limits = DynamicsLimits(
        max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0), max_climb_rate=5.0, max_descent_rate=5.0,
        dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0,
    )
    p0 = np.array([-30.0, 0.0, 5.0], dtype=np.float32)
    v0 = np.array([8.0, 0.0, 0.0], dtype=np.float32)
    waypoint = np.array([500.0, 0.0, 5.0])
    persisted, _ = predict_waypoint_following_trajectory(p0, v0, waypoint, np.array([1.0, 0.0, 0.0]), limits, 12)
    fixed, _ = predict_constrained_trajectory(p0, v0, np.array([1.0, 0.0, 0.0]), limits, 12)
    assert max(np.linalg.norm(a - b) for a, b in zip(persisted, fixed)) < 1e-3


def test_g_dynamic_obstacle_swept_box_matches_execution_while_the_layout_is_unchanged(monkeypatch):
    bird = DynamicObstacle(id=0, obstacle_type="bird", position=np.array([20.0, 0.0, 10.0]),
                           velocity=np.zeros(3), radius=3.0)
    flight = fly(_config(()), (0, 0, 10), (12, 0, 0), (45, 0, 10), 40, monkeypatch, dynamic_obstacles=(bird,))
    start, steps, _, _ = assert_prediction_matches_execution(flight, min_steps=2)
    assert len(flight.layouts[start]) == 1  # no static obstacles: the avoided box is the swept dynamic one
    assert flight.modes[start] in ("left", "right", "climb")


# ------------------------------------------------------------------ real episode
def _real_episode_flight(level, seed, steps):
    cfg = dataclasses.replace(build_v2_env_config(level, "measurement"), defender_standby_until_detection=False)
    env = SoldierEnv(cfg)
    env.enemy_tactical_override = None
    env.reset(seed=seed)
    env._defender_pos = FAR_DEFENDER.copy()
    env._defender_vel = np.zeros(3, dtype=np.float32)
    planner = env._enemy_obstacle_planner
    rows = []
    for _ in range(steps):
        pre = (env._enemy_pos.copy(), env._enemy_vel.copy())
        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        rows.append((pre, env._enemy_pos.copy(), None if planner.waypoint is None else planner.waypoint.copy(), planner.mode))
        if terminated or truncated:
            break
    return env, rows


def test_climb_waypoint_over_the_activation_point_needs_a_float32_exact_replay():
    """C3 seed 130006, step 67: a climb waypoint directly above the activation point.

    After a step or two the desired heading is almost exactly opposite the current
    heading, so which way the turn-rate-limited rotation goes is decided by
    float32 rounding. The planner's replay must follow the environment; a float64
    replica flies the other way (a 34.7 m error here)."""
    env, rows = _real_episode_flight("C3", 130006, 90)
    start, persisted_steps = 66, 7
    (p0, v0), _, waypoint, mode = rows[start]
    assert mode == "climb" and all(np.array_equal(r[2], waypoint) for r in rows[start : start + persisted_steps])
    limits = env._enemy_dynamics_limits()
    persisted, _ = predict_waypoint_following_trajectory(p0, v0, waypoint, np.array([1.0, 0.0, 0.0]), limits, persisted_steps)
    actual = [p0] + [r[1] for r in rows[start : start + persisted_steps]]
    assert max(np.linalg.norm(a - b) for a, b in zip(persisted, actual)) < TOLERANCE_M

    # float64 reference of the same control law
    p = p0.astype(np.float64)
    v = v0.astype(np.float64)
    reference = [p.copy()]
    for _ in range(persisted_steps):
        direction = waypoint - p
        direction = direction / np.linalg.norm(direction)
        v, _ = advance_velocity(
            current_velocity=v, desired_velocity=limits.max_speed * direction, max_speed=limits.max_speed,
            max_accel=limits.max_accel, max_turn_rate_rad=limits.max_turn_rate_rad, max_climb_rate=limits.max_climb_rate,
            max_descent_rate=limits.max_descent_rate, dt=limits.dt, eps=limits.eps,
        )
        p, v = apply_boundary(p + np.asarray(v, dtype=np.float64) * limits.dt, np.asarray(v, dtype=np.float64),
                              L=limits.L, max_altitude=limits.max_altitude)
        reference.append(p.copy())
    assert max(np.linalg.norm(a - b) for a, b in zip(reference, actual)) > 5.0


# --------------------------------------------------- shared dynamics / wiring
def test_original_fixed_direction_predictor_is_unchanged():
    limits = DynamicsLimits(
        max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0), max_climb_rate=5.0, max_descent_rate=5.0,
        dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0,
    )
    p0, v0 = np.array([3.0, -4.0, 6.0]), np.array([1.0, 7.0, -2.0])
    direction = np.array([-0.6, 0.3, 0.74])
    direction = direction / np.linalg.norm(direction)
    positions, velocities = predict_constrained_trajectory(p0, v0, direction, limits, 12)
    p, v = p0.copy(), v0.copy()
    for k in range(12):
        v, _ = advance_velocity(
            current_velocity=v, desired_velocity=limits.max_speed * direction, max_speed=limits.max_speed,
            max_accel=limits.max_accel, max_turn_rate_rad=limits.max_turn_rate_rad, max_climb_rate=limits.max_climb_rate,
            max_descent_rate=limits.max_descent_rate, dt=limits.dt, eps=limits.eps,
        )
        v = np.asarray(v, dtype=np.float64)
        p, v = apply_boundary(p + v * limits.dt, v, L=limits.L, max_altitude=limits.max_altitude)
        assert np.array_equal(positions[k + 1], p) and np.array_equal(velocities[k + 1], v)


def test_executed_step_is_the_environment_update(monkeypatch):
    env = SoldierEnv(_config(()))
    env.reset(seed=0)
    env._enemy_pos = np.array([3.0, -4.0, 6.0], dtype=np.float32)
    env._enemy_vel = np.array([1.0, 7.0, -2.0], dtype=np.float32)
    env._soldier_pos = np.array(ASSET, dtype=np.float32)
    env._defender_pos = FAR_DEFENDER.copy()
    env._move_soldier = lambda: None
    p0, v0 = env._enemy_pos.copy(), env._enemy_vel.copy()
    direction = np.asarray(env._soldier_pos - env._enemy_pos, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    expected_p, expected_v = _executed_step(p0, v0, direction, env._enemy_dynamics_limits())
    env.step(np.zeros(3, dtype=np.float32))
    assert np.array_equal(env._enemy_pos, expected_p) and np.array_equal(env._enemy_vel, expected_v)


def test_only_the_v2_attacker_planner_uses_executed_prediction():
    v2 = SoldierEnv(build_v2_env_config("C2", "measurement"))
    v1 = SoldierEnv(build_v1_env_config("C2", "measurement"))
    assert v2._enemy_obstacle_planner.executed_prediction is True
    assert v1._enemy_obstacle_planner.executed_prediction is False
    assert v2._defender_obstacle_planner.executed_prediction is False
    assert v1._defender_obstacle_planner.executed_prediction is False
    assert ObstacleAvoidancePlanner(clearance=2.0, horizon_steps=12, release_steps=4, max_altitude=30.0).executed_prediction is False


def _digest(builder, level, seed, steps=140):
    import hashlib

    from uav_defend.policies.registry import get_policy
    from uav_defend.policies.sanitize import build_policy_info

    env = SoldierEnv(builder(level, "measurement"))
    env.enemy_tactical_override = None
    policy = get_policy("greedy")
    env.reset(seed=seed)
    digest = hashlib.sha256()
    for _ in range(steps):
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), "measurement")), dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        for key in ("enemy_pos", "enemy_vel", "defender_pos", "defender_vel", "soldier_pos"):
            digest.update(np.asarray(info[key], dtype=np.float32).tobytes())
        if terminated or truncated:
            break
    return digest.hexdigest()


@pytest.mark.parametrize("level,seed,expected", [
    ("C2", 130001, "1d1e7f7476e94431c4740a601483fc355ffa42f926fba5f04dd1c703f4563f9f"),
    ("C3", 130002, "90b65cabbff5519738749e42afca0774c3ecfdc218707646f16289b4a663934a"),
])
def test_v1_trajectories_including_the_defender_planner_are_bit_identical_to_revision_181f466(level, seed, expected):
    """Digests captured from the unmodified 181f466 sources (greedy defender, canonical V1)."""
    assert _digest(build_v1_env_config, level, seed) == expected
