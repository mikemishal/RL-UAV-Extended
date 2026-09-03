"""Deterministic 3-D obstacle geometry and layout for the journal extension.

Public API:
    AABBObstacle              -- axis-aligned rectangular-prism obstacle.
    SegmentIntersection       -- single-obstacle swept-motion query result.
    ObstacleLayout            -- immutable collection of obstacles with query helpers.
    LayoutSegmentIntersection -- layout-wide first-contact swept-motion query result.
    generate_layout           -- deterministic per-episode layout builder.
"""

from uav_defend.obstacles.geometry import AABBObstacle, SegmentIntersection
from uav_defend.obstacles.layout import LayoutSegmentIntersection, ObstacleLayout, generate_layout

__all__ = [
    "AABBObstacle",
    "SegmentIntersection",
    "ObstacleLayout",
    "LayoutSegmentIntersection",
    "generate_layout",
]
