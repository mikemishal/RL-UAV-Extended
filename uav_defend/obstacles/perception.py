"""
Hostile-UAS environmental-awareness/perception layer (journal extension).

Explicitly separates the avoidance planner's input from ground-truth world
state:

    World (ground truth: static layout + dynamic obstacles)
        -> EnemyObstaclePerception.update(...)
        -> PerceivedObstacleSet  (only what this awareness mode may see)
        -> ObstacleAvoidancePlanner (unchanged; sees ONLY the perceived set)

CRITICAL INVARIANT (see uav_defend/envs/soldier_env.py's collision
diagnostics): perception state != physics state. Ground-truth collision
checking always uses the FULL world layout/dynamic-obstacle list,
regardless of `enemy_obstacle_awareness_mode` -- an attacker can and does
collide with an obstacle it never perceived.

Awareness modes:
    "disabled"  -- backward-compatible: perceives the ENTIRE ground-truth
        static layout and every current dynamic obstacle (pre-perception-
        layer behavior, unchanged). Not a realistic journal attacker mode.
    "local"     -- Level B: no prior map knowledge. A static obstacle is
        perceived once the attacker comes within `enemy_obstacle_sensor_range`
        of its AABB surface (not just its center); a dynamic obstacle is
        perceived once within `enemy_dynamic_obstacle_sensor_range` of its
        center. Sensing is a simple omnidirectional volume (no field of
        view/occlusion/noise/missed detections -- an explicitly documented
        abstraction for this first implementation; see task spec Section 5).
    "map_local" -- Level C: `O_perceived = O_map union O_local`, where
        O_map is every static obstacle with `AABBObstacle.map_known=True`
        (known from episode start, regardless of range) and O_local is
        local sensing of everything else (unmapped static + all dynamic).

Memory:
    - Static: once locally detected (mode "local"), or always (map-known
      obstacles under "map_local"), an obstacle is remembered for the rest
      of the episode if `enemy_static_obstacle_memory_enabled` (default
      True) -- it does not "disappear" if the attacker moves back out of
      range. Map knowledge itself is fixed at episode start and never
      silently updated using ground truth for unmapped obstacles (Section
      11): an obstacle with `map_known=False` is absent from O_map and can
      ONLY enter the perceived set via actual local detection.
    - Dynamic: NEVER globally known in advance (Section 8). A track
      expires (is dropped) after `enemy_dynamic_obstacle_memory_steps`
      consecutive steps without re-observation -- stale tracks are never
      used indefinitely.

At this phase, sensed values equal ground truth once detected (no
measurement noise) -- see `PerceivedStaticObstacle`/`PerceivedDynamicObstacle`
docstrings for how this can later be replaced by an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_defend.obstacles.dynamic import DynamicObstacle
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout

VALID_AWARENESS_MODES = ("disabled", "local", "map_local")


@dataclass(frozen=True)
class PerceivedStaticObstacle:
    """A static obstacle as known to the avoidance planner. Geometry is
    ground truth once perceived (no noise modeled yet); `source` records
    WHY it is known, for diagnostics."""

    id: int  # stable index into the ground-truth ObstacleLayout
    min_corner: np.ndarray
    max_corner: np.ndarray
    source: str  # "map" | "local_sensor"


@dataclass(frozen=True)
class PerceivedDynamicObstacle:
    """A dynamic obstacle as known to the avoidance planner (always via
    local sensing -- see module docstring, Section 8). `position`/
    `velocity` are ground truth once detected (no noise modeled yet)."""

    id: int  # DynamicObstacle.id
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    obstacle_type: str
    source: str = "local_sensor"


@dataclass(frozen=True)
class PerceivedObstacleSet:
    """Everything the avoidance planner may currently use -- NEVER the
    ground-truth world state directly."""

    static: tuple[PerceivedStaticObstacle, ...] = ()
    dynamic: tuple[PerceivedDynamicObstacle, ...] = ()

    def to_static_layout(self) -> ObstacleLayout:
        """Bridge to the existing (unmodified) avoidance planner's expected
        `ObstacleLayout` of `AABBObstacle`s."""
        return ObstacleLayout(
            obstacles=tuple(
                AABBObstacle(min_corner=o.min_corner, max_corner=o.max_corner) for o in self.static
            )
        )


def _static_obstacle_perceived(
    obstacle: AABBObstacle, position: np.ndarray, sensor_range: float,
) -> bool:
    """Distance to the obstacle's AABB SURFACE (not just its center), per
    Section 4/17-item-10: `obstacle.clearance_to_point(position) <=
    sensor_range` -- zero/negative clearance (already inside/touching) is
    always perceived."""
    return obstacle.clearance_to_point(position) <= sensor_range


@dataclass
class EnemyObstaclePerception:
    """Stateful perception update -- owns static/dynamic obstacle memory
    for ONE hostile UAV. Consumes ZERO randomness (purely a function of
    ground-truth state + configuration): perceiving/forgetting an obstacle
    never perturbs any RNG stream, so identical seeds reproduce identical
    perception histories.
    """

    mode: str
    static_sensor_range: float
    dynamic_sensor_range: float
    static_memory_enabled: bool
    dynamic_memory_steps: int

    _remembered_static_ids: set = field(default_factory=set, init=False)
    # id -> (last-known PerceivedDynamicObstacle, steps since last observed)
    _dynamic_tracks: dict = field(default_factory=dict, init=False)
    new_static_detections: int = field(default=0, init=False)  # this-step count, for diagnostics

    def __post_init__(self) -> None:
        if self.mode not in VALID_AWARENESS_MODES:
            raise ValueError(f"mode must be one of {VALID_AWARENESS_MODES}, got {self.mode!r}")

    def reset(self) -> None:
        self._remembered_static_ids = set()
        self._dynamic_tracks = {}
        self.new_static_detections = 0

    def update(
        self,
        position: np.ndarray,
        static_layout: ObstacleLayout,
        dynamic_obstacles: tuple[DynamicObstacle, ...],
    ) -> PerceivedObstacleSet:
        """Compute this step's `PerceivedObstacleSet` and advance memory
        state (call exactly once per simulation step)."""
        position = np.asarray(position, dtype=np.float64)
        self.new_static_detections = 0

        if self.mode == "disabled":
            static = tuple(
                PerceivedStaticObstacle(id=i, min_corner=o.min_corner, max_corner=o.max_corner, source="map")
                for i, o in enumerate(static_layout)
            )
            dynamic = tuple(
                PerceivedDynamicObstacle(id=o.id, position=o.position, velocity=o.velocity,
                                          radius=o.radius, obstacle_type=o.obstacle_type)
                for o in dynamic_obstacles
            )
            return PerceivedObstacleSet(static=static, dynamic=dynamic)

        # --- Static: map knowledge (Level C only) + local sensing + memory ---
        newly_detected_ids = set()
        for i, obstacle in enumerate(static_layout):
            if i in self._remembered_static_ids:
                continue
            if self.mode == "map_local" and obstacle.map_known:
                newly_detected_ids.add(i)
                continue
            if _static_obstacle_perceived(obstacle, position, self.static_sensor_range):
                newly_detected_ids.add(i)

        self.new_static_detections = len(newly_detected_ids - self._remembered_static_ids)
        if self.static_memory_enabled:
            self._remembered_static_ids |= newly_detected_ids
            visible_ids = self._remembered_static_ids
        else:
            # No memory: an obstacle counts only while its perception
            # condition (map-known, or currently within range) holds THIS step.
            visible_ids = newly_detected_ids

        static = tuple(
            PerceivedStaticObstacle(
                id=i, min_corner=static_layout[i].min_corner, max_corner=static_layout[i].max_corner,
                source="map" if (self.mode == "map_local" and static_layout[i].map_known) else "local_sensor",
            )
            for i in sorted(visible_ids)
        )

        # --- Dynamic: local sensing only, with a short-timeout track memory ---
        currently_sensed = {}
        for obs in dynamic_obstacles:
            if float(np.linalg.norm(obs.position - position)) <= self.dynamic_sensor_range:
                currently_sensed[obs.id] = PerceivedDynamicObstacle(
                    id=obs.id, position=obs.position, velocity=obs.velocity,
                    radius=obs.radius, obstacle_type=obs.obstacle_type,
                )

        updated_tracks = {}
        for obs_id, perceived in currently_sensed.items():
            updated_tracks[obs_id] = (perceived, 0)
        for obs_id, (perceived, age) in self._dynamic_tracks.items():
            if obs_id in updated_tracks:
                continue  # freshly re-observed this step, already added above
            new_age = age + 1
            if new_age <= self.dynamic_memory_steps:
                updated_tracks[obs_id] = (perceived, new_age)
            # else: track expired -- dropped, never used indefinitely.
        self._dynamic_tracks = updated_tracks

        dynamic = tuple(perceived for perceived, _age in self._dynamic_tracks.values())
        return PerceivedObstacleSet(static=static, dynamic=dynamic)
