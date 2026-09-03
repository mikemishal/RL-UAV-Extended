"""3-D obstacle geometry and layout primitives for the journal extension.

Canonical representation: `AABBObstacle` stores the box as `min_corner` /
`max_corner` (opposite corners). This representation was chosen because
point-containment, clearance, and segment-intersection (slab method) tests
are all most directly and numerically-stably expressed in terms of a box's
min/max corners; `center` / `half_extents` are exposed as derived
convenience properties.

This module is intentionally independent of `SoldierEnv` and any policy
code: it is pure NumPy-based 3-D geometry, with no external geometry
package dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SegmentIntersection:
    """Result of `AABBObstacle.segment_intersection()`.

    `t_enter`/`t_exit` are the segment parameters (in `[0, 1]`, where 0 is
    `start` and 1 is `end`) at which the segment enters/exits the box;
    `point = start + t_enter * (end - start)` is the first-contact point.
    All three are `None` when `intersects` is False.
    """

    intersects: bool
    t_enter: float | None = None
    t_exit: float | None = None
    point: np.ndarray | None = None


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class AABBObstacle:
    """An axis-aligned 3-D rectangular-prism (box) obstacle.

    Boundary contact counts as containment/intersection (inclusive tests,
    within `tolerance`) -- this matches the "touching a wall is a
    collision" convention used throughout this repository's geometry.
    """

    min_corner: np.ndarray
    max_corner: np.ndarray
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        min_c = _as_vec3(self.min_corner, "min_corner")
        max_c = _as_vec3(self.max_corner, "max_corner")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {self.tolerance}")
        if np.any(max_c - min_c <= 0):
            raise ValueError(
                "AABBObstacle requires strictly positive extents on every axis "
                f"(max_corner - min_corner > 0); got min_corner={min_c}, max_corner={max_c}"
            )
        object.__setattr__(self, "min_corner", min_c)
        object.__setattr__(self, "max_corner", max_c)

    @classmethod
    def from_center_half_extents(
        cls, center, half_extents, tolerance: float = DEFAULT_TOLERANCE
    ) -> "AABBObstacle":
        """Construct from `center` (3,) and `half_extents` (3,), both > 0."""
        center = _as_vec3(center, "center")
        half_extents = _as_vec3(half_extents, "half_extents")
        if np.any(half_extents <= 0):
            raise ValueError(f"half_extents must be > 0 on every axis, got {half_extents}")
        return cls(min_corner=center - half_extents, max_corner=center + half_extents, tolerance=tolerance)

    @property
    def center(self) -> np.ndarray:
        return (self.min_corner + self.max_corner) / 2.0

    @property
    def half_extents(self) -> np.ndarray:
        return (self.max_corner - self.min_corner) / 2.0

    def contains_point(self, point) -> bool:
        """True iff `point` is inside the box or exactly on its boundary."""
        p = _as_vec3(point, "point")
        tol = self.tolerance
        return bool(np.all(p >= self.min_corner - tol) and np.all(p <= self.max_corner + tol))

    def distance_to_point(self, point) -> float:
        """Euclidean distance from `point` to the nearest point of the box
        (0.0 if `point` is inside or on the boundary). Correct for the
        nearest-face, nearest-edge, and nearest-corner cases alike."""
        p = _as_vec3(point, "point")
        outside = np.maximum(np.maximum(self.min_corner - p, p - self.max_corner), 0.0)
        return float(np.linalg.norm(outside))

    def clearance_to_point(self, point, radius: float = 0.0) -> float:
        """`distance_to_point(point) - radius` -- negative means the given
        sphere of `radius` around `point` overlaps/collides with the box."""
        return self.distance_to_point(point) - radius

    def intersects_sphere(self, center, radius: float) -> bool:
        """True iff a sphere of `radius` centered at `center` touches or
        overlaps the box."""
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        return self.distance_to_point(center) <= radius + self.tolerance

    def segment_intersects(self, start, end) -> bool:
        """True iff the line segment `start` -> `end` touches or crosses
        the box (slab method). A degenerate zero-length segment reduces to
        a point-containment test. See `segment_intersection()` for
        first-contact parameter/point information."""
        return self.segment_intersection(start, end).intersects

    def segment_intersection(self, start, end) -> SegmentIntersection:
        """Full swept-motion intersection test (slab method), returning
        first-contact information. Required for collision detection against
        fast-moving entities: checking only the endpoint positions can miss
        an obstacle the segment tunnels through between simulation steps."""
        p0 = _as_vec3(start, "start")
        p1 = _as_vec3(end, "end")
        direction = p1 - p0
        t_min, t_max = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < self.tolerance:
                # Segment parallel to this axis's pair of slab planes: it can
                # only intersect if it already lies within the slab.
                if p0[axis] < self.min_corner[axis] - self.tolerance or p0[axis] > self.max_corner[axis] + self.tolerance:
                    return SegmentIntersection(intersects=False)
                continue
            t1 = (self.min_corner[axis] - p0[axis]) / direction[axis]
            t2 = (self.max_corner[axis] - p0[axis]) / direction[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max + self.tolerance:
                return SegmentIntersection(intersects=False)
        point = p0 + t_min * direction
        return SegmentIntersection(intersects=True, t_enter=t_min, t_exit=t_max, point=point)

    def overlaps(self, other: "AABBObstacle", margin: float = 0.0) -> bool:
        """True iff this box and `other` overlap/touch, after expanding both
        by `margin` (used to enforce minimum obstacle-to-obstacle separation
        during layout generation)."""
        for axis in range(3):
            if self.max_corner[axis] + margin < other.min_corner[axis] - self.tolerance:
                return False
            if other.max_corner[axis] + margin < self.min_corner[axis] - self.tolerance:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "min_corner": self.min_corner.tolist(),
            "max_corner": self.max_corner.tolist(),
            "center": self.center.tolist(),
            "half_extents": self.half_extents.tolist(),
        }
