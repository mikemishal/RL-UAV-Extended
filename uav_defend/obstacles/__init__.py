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
]
