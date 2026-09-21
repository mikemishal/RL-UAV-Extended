"""Simple, deterministic dynamic (moving) obstacles -- journal-level aerial
clutter (e.g. small birds / other non-combat airborne objects), NOT a
detailed biological flight model.

Motion model: approximately constant horizontal speed and altitude, with an
optional small bounded stochastic per-step heading change (all randomness
drawn from a dedicated, caller-supplied `numpy.random.Generator`, so motion
is fully deterministic given a seed). Domain boundaries are handled by a
simple horizontal reflection (mirrors the reflecting-boundary convention
used elsewhere in this repository), altitude is held fixed.

This module is intentionally independent of `SoldierEnv` and of
`uav_defend/obstacles/avoidance.py`'s static-obstacle machinery: it knows
nothing about pursuit/evasion/collision-avoidance policy. See
`dynamic_obstacle_swept_aabb()` for the bridge that lets
`ObstacleAvoidancePlanner` (built for STATIC `AABBObstacle`s) reuse its
existing candidate-generation/selection logic for dynamic obstacles too,
by approximating a dynamic obstacle's short-horizon swept volume as an
inflated axis-aligned box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout

DYNAMIC_OBSTACLE_TYPES = ("bird", "airborne_clutter")


@dataclass(frozen=True)
class DynamicObstacle:
    """A single moving, spherical-collision-geometry obstacle."""

    id: int
    obstacle_type: str
    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    radius: float

    def __post_init__(self) -> None:
        if self.obstacle_type not in DYNAMIC_OBSTACLE_TYPES:
            raise ValueError(f"obstacle_type must be one of {DYNAMIC_OBSTACLE_TYPES}, got {self.obstacle_type!r}")
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64))
        object.__setattr__(self, "velocity", np.asarray(self.velocity, dtype=np.float64))


def _reflect_horizontal(position: np.ndarray, velocity: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Reflect (x, y) off the domain walls [-L, L]^2; z/vz untouched
    (dynamic obstacles fly at a fixed altitude)."""
    position = position.copy()
    velocity = velocity.copy()
    for axis in (0, 1):
        while position[axis] < -L or position[axis] > L:
            if position[axis] < -L:
                position[axis] = -2 * L - position[axis]
                velocity[axis] = -velocity[axis]
            elif position[axis] > L:
                position[axis] = 2 * L - position[axis]
                velocity[axis] = -velocity[axis]
    return position, velocity


def generate_dynamic_obstacles(
    config,
    rng: np.random.Generator,
    static_layout: ObstacleLayout,
    asset_position: np.ndarray,
    defender_position: np.ndarray,
    enemy_position: np.ndarray,
) -> tuple[DynamicObstacle, ...]:
    """Deterministic (given `rng`'s state) generation of
    `config.dynamic_obstacle_count` dynamic obstacles. Rejection-samples
    placement to avoid spawning inside a static obstacle or too close to
    the asset/defender/hostile initial positions; falls back to accepting
    the last-tried candidate after `obstacle_max_placement_attempts`
    attempts (dynamic clutter is not mission-placement-critical the way
    static obstacles are, so this never raises)."""
    L = config.L
    clearance = config.dynamic_obstacle_clearance
    obstacles: list[DynamicObstacle] = []
    for i in range(config.dynamic_obstacle_count):
        candidate = None
        for _attempt in range(config.obstacle_max_placement_attempts):
            radius = rng.uniform(config.dynamic_obstacle_min_radius, config.dynamic_obstacle_max_radius)
            speed = rng.uniform(config.dynamic_obstacle_min_speed, config.dynamic_obstacle_max_speed)
            altitude = rng.uniform(config.dynamic_obstacle_min_altitude, config.dynamic_obstacle_max_altitude)
            x = rng.uniform(-L, L)
            y = rng.uniform(-L, L)
            heading = rng.uniform(-np.pi, np.pi)
            position = np.array([x, y, altitude], dtype=np.float64)
            velocity = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0], dtype=np.float64)
            candidate = DynamicObstacle(id=i, obstacle_type=DYNAMIC_OBSTACLE_TYPES[i % 2],
                                        position=position, velocity=velocity, radius=radius)
            required = radius + clearance
            if static_layout.min_clearance(position) < required:
                continue
            if any(np.linalg.norm(position - p) < required for p in (asset_position, defender_position, enemy_position)):
                continue
            break
        obstacles.append(candidate)
    return tuple(obstacles)


def step_dynamic_obstacles(
    obstacles: tuple[DynamicObstacle, ...],
    rng: np.random.Generator,
    config,
) -> tuple[DynamicObstacle, ...]:
    """Advance every obstacle by one `config.dt`: unchanged speed/altitude,
    heading rotated by a small `Normal(0, dynamic_obstacle_heading_noise)`
    increment (radians), then horizontal reflection off the domain walls."""
    updated = []
    for obs in obstacles:
        speed = float(np.linalg.norm(obs.velocity[:2]))
        heading = float(np.arctan2(obs.velocity[1], obs.velocity[0])) if speed > 0 else 0.0
        heading += rng.normal(0.0, config.dynamic_obstacle_heading_noise)
        new_velocity = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0], dtype=np.float64)
        new_position = obs.position + new_velocity * config.dt
        new_position, new_velocity = _reflect_horizontal(new_position, new_velocity, config.L)
        updated.append(DynamicObstacle(id=obs.id, obstacle_type=obs.obstacle_type,
                                        position=new_position, velocity=new_velocity, radius=obs.radius))
    return tuple(updated)


def dynamic_obstacle_swept_aabb(obstacle: DynamicObstacle, horizon_steps: int, dt: float, clearance: float) -> AABBObstacle:
    """Conservative axis-aligned bounding box enclosing `obstacle`'s
    current position and its constant-velocity-extrapolated position
    `horizon_steps * dt` seconds ahead, inflated by `obstacle.radius +
    clearance` on every axis. This lets the EXISTING static-obstacle
    avoidance machinery (`generate_bypass_candidates`,
    `evaluate_and_select_candidate`, swept-segment collision queries) be
    reused verbatim for a moving obstacle, by treating it as one more
    (temporary, per-step) `AABBObstacle` in the layout passed to
    `ObstacleAvoidancePlanner.plan()`. NOTE: this box spans the obstacle's
    ENTIRE path over the horizon, so it is only a meaningful avoidance
    target for obstacles already confirmed to threaten a conflict -- see
    `dynamic_obstacle_threatens()`, which must gate inclusion of this box
    in a layout (an unconditionally-included box would spuriously flag
    obstacles that merely happen to lie along the same corridor while
    actually receding)."""
    future = obstacle.position + obstacle.velocity * (horizon_steps * dt)
    margin = obstacle.radius + clearance
    lo = np.minimum(obstacle.position, future) - margin
    hi = np.maximum(obstacle.position, future) + margin
    return AABBObstacle(min_corner=lo, max_corner=hi)


def dynamic_obstacle_threatens(
    mover_position: np.ndarray, mover_velocity: np.ndarray, obstacle: DynamicObstacle,
    horizon_steps: int, dt: float, clearance: float,
) -> bool:
    """Closest-point-of-approach pre-check: True iff, extrapolating BOTH
    the mover (at its current velocity -- a constant-velocity proxy; the
    authoritative check is the full constrained-dynamics candidate
    evaluation against the box this gates) and `obstacle` linearly over
    `horizon_steps * dt`, their minimum predicted separation is <=
    `obstacle.radius + clearance`. A receding/merely-nearby obstacle whose
    CPA distance is large returns False -- this is what prevents
    `dynamic_obstacle_swept_aabb`'s full-path box from ever being built
    for (and triggering avoidance against) an obstacle that poses no
    actual predicted conflict."""
    horizon_s = horizon_steps * dt
    mover_future = np.asarray(mover_position, dtype=np.float64) + np.asarray(mover_velocity, dtype=np.float64) * horizon_s
    obstacle_future = obstacle.position + obstacle.velocity * horizon_s
    cpa_distance = swept_sphere_vs_sphere_min_distance(mover_position, mover_future, obstacle.position, obstacle_future)
    return cpa_distance <= obstacle.radius + clearance


def swept_sphere_vs_sphere_min_distance(
    p0_a: np.ndarray, p1_a: np.ndarray, p0_b: np.ndarray, p1_b: np.ndarray,
) -> float:
    """Minimum center-to-center distance between two points each moving
    linearly over the SAME unit time interval (`a`: p0_a -> p1_a, `b`:
    p0_b -> p1_b), i.e. the closest point of approach of their relative
    motion, clamped to the segment (t in [0, 1]). Used for ACTUAL
    (per-step, not predictive) swept-collision diagnostics: a "collision"
    is `min_distance <= radius_a + radius_b`."""
    p0_a, p1_a = np.asarray(p0_a, dtype=np.float64), np.asarray(p1_a, dtype=np.float64)
    p0_b, p1_b = np.asarray(p0_b, dtype=np.float64), np.asarray(p1_b, dtype=np.float64)
    rel_pos0 = p0_a - p0_b
    rel_disp = (p1_a - p0_a) - (p1_b - p0_b)
    denom = float(np.dot(rel_disp, rel_disp))
    if denom < 1e-12:
        t_star = 0.0
    else:
        t_star = float(np.clip(-np.dot(rel_pos0, rel_disp) / denom, 0.0, 1.0))
    closest = rel_pos0 + t_star * rel_disp
    return float(np.linalg.norm(closest))
