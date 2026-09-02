"""Deterministic 3-D obstacle geometry and layout for the journal extension.

Public API:
    AABBObstacle    -- axis-aligned rectangular-prism obstacle.
    ObstacleLayout  -- immutable collection of obstacles with query helpers.
    generate_layout -- deterministic per-episode layout builder.
"""

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout, generate_layout

__all__ = ["AABBObstacle", "ObstacleLayout", "generate_layout"]
