"""Obstacle-layout container and deterministic layout generation.

`ObstacleLayout` is a simple immutable collection of `AABBObstacle`
instances with aggregate query helpers (containment, clearance, segment
intersection). It has no dependency on `SoldierEnv` or any policy code.

`generate_layout()` is the single entry point `SoldierEnv.reset()` uses to
build a layout for a given episode. It is deterministic given the same
`EnvConfig` and the same `numpy.random.Generator` state -- callers are
responsible for supplying a dedicated obstacle RNG stream (see
`SoldierEnv.reset()`), independent of the four pre-existing exogenous RNG
streams (spawn, soldier, enemy motion, sensor), so that enabling/disabling
or reconfiguring obstacles never perturbs the conference-baseline
trajectories those streams produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.obstacles.geometry import AABBObstacle


@dataclass(frozen=True)
class ObstacleLayout:
    """An immutable collection of `AABBObstacle` instances."""

    obstacles: tuple[AABBObstacle, ...] = ()

    def __len__(self) -> int:
        return len(self.obstacles)

    def __iter__(self):
        return iter(self.obstacles)

    def __getitem__(self, index: int) -> AABBObstacle:
        return self.obstacles[index]

    def contains_point(self, point) -> bool:
        return any(o.contains_point(point) for o in self.obstacles)

    def containing_obstacle_index(self, point) -> int | None:
        for i, o in enumerate(self.obstacles):
            if o.contains_point(point):
                return i
        return None

    def min_clearance(self, point, radius: float = 0.0) -> float:
        """Minimum clearance from `point` (a sphere of `radius`) to any
        obstacle; `+inf` if the layout is empty."""
        if not self.obstacles:
            return float("inf")
        return min(o.clearance_to_point(point, radius) for o in self.obstacles)

    def nearest_obstacle_index(self, point, radius: float = 0.0) -> int | None:
        """Index of the obstacle achieving `min_clearance`, or None if the
        layout is empty."""
        if not self.obstacles:
            return None
        clearances = [o.clearance_to_point(point, radius) for o in self.obstacles]
        return int(np.argmin(clearances))

    def segment_intersects(self, start, end) -> bool:
        return any(o.segment_intersects(start, end) for o in self.obstacles)

    def first_intersecting_obstacle(self, start, end) -> int | None:
        """Index of the first obstacle (in layout order) intersected by the
        segment `start` -> `end`, or None if none intersect."""
        for i, o in enumerate(self.obstacles):
            if o.segment_intersects(start, end):
                return i
        return None

    def to_dict(self) -> dict:
        """Deterministic, JSON/CSV-friendly serialization for logging."""
        return {
            "count": len(self.obstacles),
            "obstacles": [o.to_dict() for o in self.obstacles],
        }


def _validate_within_domain(obstacle: AABBObstacle, config: EnvConfig) -> None:
    L = config.L
    if (
        obstacle.min_corner[0] < -L or obstacle.max_corner[0] > L
        or obstacle.min_corner[1] < -L or obstacle.max_corner[1] > L
        or obstacle.min_corner[2] < 0.0 or obstacle.max_corner[2] > config.max_altitude
    ):
        raise ValueError(
            f"obstacle {obstacle.to_dict()} is not fully inside the engagement volume "
            f"[-{L}, {L}]^2 x [0, {config.max_altitude}]"
        )


def _fixed_layout(config: EnvConfig) -> ObstacleLayout:
    obstacles = []
    for spec in config.obstacle_fixed_spec:
        cx, cy, cz, hx, hy, hz = spec
        obstacle = AABBObstacle.from_center_half_extents((cx, cy, cz), (hx, hy, hz))
        _validate_within_domain(obstacle, config)
        obstacles.append(obstacle)
    return ObstacleLayout(obstacles=tuple(obstacles))


def _random_layout(
    config: EnvConfig,
    rng: np.random.Generator,
    asset_position: np.ndarray,
    defender_position: np.ndarray,
    enemy_spawn_position: np.ndarray | None,
) -> ObstacleLayout:
    L = config.L
    placed: list[AABBObstacle] = []

    for i in range(config.obstacle_count):
        accepted: AABBObstacle | None = None
        for _attempt in range(config.obstacle_max_placement_attempts):
            size_x = rng.uniform(config.obstacle_min_size, config.obstacle_max_size)
            size_y = rng.uniform(config.obstacle_min_size, config.obstacle_max_size)
            height = rng.uniform(config.obstacle_min_height, min(config.obstacle_max_height, config.max_altitude))
            half_extents = np.array([size_x / 2.0, size_y / 2.0, height / 2.0])

            if L - half_extents[0] <= 0 or L - half_extents[1] <= 0:
                # Requested footprint cannot fit inside the domain at all;
                # further attempts with the same size range would also fail.
                raise RuntimeError(
                    f"obstacle_max_size={config.obstacle_max_size} is too large to fit within "
                    f"the domain half-size L={L}"
                )
            center_x = rng.uniform(-L + half_extents[0], L - half_extents[0])
            center_y = rng.uniform(-L + half_extents[1], L - half_extents[1])
            center_z = half_extents[2]  # ground-based: min_corner_z == 0

            candidate = AABBObstacle.from_center_half_extents(
                (center_x, center_y, center_z), half_extents
            )

            if candidate.clearance_to_point(asset_position) < config.obstacle_clearance_from_asset:
                continue
            if candidate.clearance_to_point(defender_position) < config.obstacle_clearance_from_asset:
                continue
            if enemy_spawn_position is not None:
                if candidate.clearance_to_point(enemy_spawn_position) < config.obstacle_clearance_from_spawn:
                    continue
            if any(candidate.overlaps(other, margin=config.obstacle_min_separation) for other in placed):
                continue

            accepted = candidate
            break

        if accepted is None:
            raise RuntimeError(
                f"Could not place obstacle {i + 1}/{config.obstacle_count} within "
                f"{config.obstacle_max_placement_attempts} attempts under the configured "
                "size/clearance/separation constraints. Relax obstacle_count, obstacle sizes, "
                "or clearance settings."
            )
        placed.append(accepted)

    return ObstacleLayout(obstacles=tuple(placed))


def generate_layout(
    config: EnvConfig,
    rng: np.random.Generator,
    asset_position,
    defender_position,
    enemy_spawn_position=None,
) -> ObstacleLayout:
    """Build the obstacle layout for one episode.

    Args:
        config: Environment configuration (see `EnvConfig` obstacle fields).
        rng: Dedicated obstacle RNG stream (never shared with the four
            pre-existing exogenous streams). Unused (but still accepted,
            for a uniform call signature) when `obstacle_layout_mode` is
            "none" or "fixed", so those modes never consume any random
            draws.
        asset_position: Protected-asset (soldier) initial position, shape (3,).
        defender_position: Defender initial position, shape (3,).
        enemy_spawn_position: Hostile spawn position, shape (3,), if already
            known at layout-generation time; otherwise None.

    Returns:
        An `ObstacleLayout` (empty when obstacles are disabled or
        `obstacle_layout_mode == "none"`).
    """
    if not config.obstacles_enabled or config.obstacle_layout_mode == "none":
        return ObstacleLayout(obstacles=())
    if config.obstacle_layout_mode == "fixed":
        return _fixed_layout(config)
    if config.obstacle_layout_mode == "random":
        return _random_layout(config, rng, asset_position, defender_position, enemy_spawn_position)
    raise ValueError(f"unknown obstacle_layout_mode: {config.obstacle_layout_mode!r}")
