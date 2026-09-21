"""Deterministic 3-D obstacle geometry and layout for the journal extension.

Public API:
    AABBObstacle              -- axis-aligned rectangular-prism obstacle.
    SegmentIntersection       -- single-obstacle swept-motion query result.
    ObstacleLayout            -- immutable collection of obstacles with query helpers.
    LayoutSegmentIntersection -- layout-wide first-contact swept-motion query result.
    generate_layout           -- deterministic per-episode layout builder.
    DynamicsLimits            -- constrained-dynamics parameter bundle for prediction.
    ObstacleAvoidancePlanner  -- deterministic short-horizon local avoidance layer.
"""

from uav_defend.obstacles.avoidance import (
    BypassCandidate,
    CandidateEvaluation,
    CollisionPrediction,
    DynamicsLimits,
    ObstacleAvoidancePlanner,
    candidate_category,
    candidate_sort_key,
    evaluate_and_select_candidate,
    generate_bypass_candidates,
    predict_collision,
    predict_constrained_trajectory,
)
from uav_defend.obstacles.geometry import AABBObstacle, SegmentIntersection
from uav_defend.obstacles.layout import LayoutSegmentIntersection, ObstacleLayout, generate_layout
from uav_defend.obstacles.dynamic import (
    DYNAMIC_OBSTACLE_TYPES,
    DynamicObstacle,
    dynamic_obstacle_swept_aabb,
    dynamic_obstacle_threatens,
    generate_dynamic_obstacles,
    step_dynamic_obstacles,
    swept_sphere_vs_sphere_min_distance,
)
from uav_defend.obstacles.perception import (
    VALID_AWARENESS_MODES,
    EnemyObstaclePerception,
    PerceivedDynamicObstacle,
    PerceivedObstacleSet,
    PerceivedStaticObstacle,
)

__all__ = [
    "AABBObstacle",
    "SegmentIntersection",
    "ObstacleLayout",
    "LayoutSegmentIntersection",
    "generate_layout",
    "DynamicsLimits",
    "CollisionPrediction",
    "BypassCandidate",
    "CandidateEvaluation",
    "ObstacleAvoidancePlanner",
    "predict_constrained_trajectory",
    "predict_collision",
    "generate_bypass_candidates",
    "evaluate_and_select_candidate",
    "candidate_category",
    "candidate_sort_key",
    "DynamicObstacle",
    "DYNAMIC_OBSTACLE_TYPES",
    "generate_dynamic_obstacles",
    "step_dynamic_obstacles",
    "dynamic_obstacle_swept_aabb",
    "dynamic_obstacle_threatens",
    "swept_sphere_vs_sphere_min_distance",
    "VALID_AWARENESS_MODES",
    "EnemyObstaclePerception",
    "PerceivedStaticObstacle",
    "PerceivedDynamicObstacle",
    "PerceivedObstacleSet",
]
