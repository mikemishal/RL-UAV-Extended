"""
Fixed-size directional obstacle-sector observation (journal extension,
Phases 5-9 of the obstacle-aware-retraining sub-task).

Purpose: give PPO/PPO-Kalman/LR-PPO a FIXED-SIZE (independent of obstacle
count) summary of nearby clutter, derived STRICTLY from the defender's own
`PerceivedObstacleSet` (Level B/C perception -- never ground truth), so
that:
    - the observation dimension never changes with obstacle count/density
      (Phase 5);
    - unperceived obstacles are structurally invisible to the policy
      (Phase 9 -- same guarantee already relied on by the reactive safety
      layer);
    - the representation generalizes across unseen layouts (a sector's
      value is a scalar clearance summary, not a per-obstacle slot).

Design (Phase 6/7/8, see this module's own constants/docstrings below for
the full specification and rationale):
    - 8 horizontal azimuthal sectors (45 deg each) + 2 vertical polar caps
      ("up"/"down") = 10 sectors total, partitioning the full sphere
      around the defender with NO overlap and NO gap.
    - 2 channels per sector: static clearance, dynamic clearance.
      10 x 2 = 20 features.
    - Azimuth is measured in the DEFENDER'S OWN local horizontal frame
      (forward = current velocity direction if moving, else a fixed
      world-frame fallback -- see `_forward_axis`), not world axes, so the
      representation is rotationally consistent regardless of the
      defender's absolute heading (Phase 7).
    - Every distance is normalized by a fixed sensing/planning range
      `R_obs` and clipped to [0, 1] (Phase 8): 0.0 = touching/overlapping,
      1.0 = no obstacle in that sector within range (or beyond range).
      Never NaN/inf.

This module is PURE (no environment/gym dependency) and operates only on
already-perceived obstacle data -- see `uav_defend/envs/obstacle_observation_wrapper.py`
for the Gymnasium wrapper that calls this every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.perception import PerceivedObstacleSet

# Sector order (Phase 6, EXACT order as specified): horizontal octants
# first (starting at "front", proceeding toward "front-left" -- i.e.
# increasing azimuth is a LEFT turn, see `_azimuth_deg`), then the two
# vertical polar caps.
SECTOR_NAMES: tuple[str, ...] = (
    "front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right",
    "up", "down",
)
N_HORIZONTAL_SECTORS = 8
N_SECTORS = 10
CHANNEL_NAMES: tuple[str, ...] = ("static_clearance", "dynamic_clearance")
N_CHANNELS = len(CHANNEL_NAMES)
SECTOR_FEATURE_DIM = N_SECTORS * N_CHANNELS  # 20

VERTICAL_ELEVATION_THRESHOLD_DEG = 60.0  # |elevation| beyond this -> up/down cap
HORIZONTAL_SECTOR_WIDTH_DEG = 360.0 / N_HORIZONTAL_SECTORS  # 45.0

# Fallback forward axis (world +x) used only when the defender's
# horizontal speed is too small to define a heading (Phase 7: "unless
# inspection shows another representation is clearly preferable" -- a
# hovering/near-stationary defender has no defined heading, so falling
# back to a fixed world axis is the simplest well-defined convention).
_FALLBACK_FORWARD_WORLD = np.array([1.0, 0.0], dtype=np.float64)
_HEADING_SPEED_EPS = 1e-3


@dataclass(frozen=True)
class SectorObservationSpec:
    """Versioned metadata (Phase 12) for one obstacle-sector observation
    configuration -- store this alongside any trained checkpoint that
    consumes sector features."""

    version: str
    sensor_range: float
    vertical_elevation_threshold_deg: float = VERTICAL_ELEVATION_THRESHOLD_DEG
    sector_names: tuple[str, ...] = SECTOR_NAMES
    channel_names: tuple[str, ...] = CHANNEL_NAMES

    @property
    def feature_dim(self) -> int:
        return len(self.sector_names) * len(self.channel_names)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "sensor_range": self.sensor_range,
            "vertical_elevation_threshold_deg": self.vertical_elevation_threshold_deg,
            "sector_names": list(self.sector_names),
            "channel_names": list(self.channel_names),
            "feature_dim": self.feature_dim,
        }


SECTOR_OBSERVATION_V1 = SectorObservationSpec(version="sector_obs_v1", sensor_range=15.0)


def _forward_axis(defender_vel: np.ndarray) -> np.ndarray:
    """Unit 2-D (world x,y) forward axis for the defender-local frame:
    the current horizontal velocity direction, or a fixed world-frame
    fallback while (near-)stationary."""
    horiz = np.asarray(defender_vel, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(horiz))
    if speed <= _HEADING_SPEED_EPS:
        return _FALLBACK_FORWARD_WORLD.copy()
    return horiz / speed


def _azimuth_deg(rel_horizontal: np.ndarray, forward: np.ndarray) -> float:
    """Azimuth of `rel_horizontal` (world x,y) relative to `forward`, in
    degrees, in `[0, 360)`. 0 = front, 90 = left, 180 = rear, 270 = right
    (increasing azimuth = left turn, matching `SECTOR_NAMES`'s order)."""
    left = np.array([-forward[1], forward[0]])  # +90 deg (CCW) rotation of forward
    x = float(np.dot(rel_horizontal, forward))
    y = float(np.dot(rel_horizontal, left))
    angle = np.degrees(np.arctan2(y, x))
    return float(angle % 360.0)


def _sector_index(rel: np.ndarray, forward: np.ndarray) -> int:
    """Index into `SECTOR_NAMES` for a defender-relative vector `rel`
    (world x,y,z)."""
    horizontal_dist = float(np.linalg.norm(rel[:2]))
    elevation = float(np.degrees(np.arctan2(rel[2], max(horizontal_dist, 1e-9))))
    if elevation >= VERTICAL_ELEVATION_THRESHOLD_DEG:
        return SECTOR_NAMES.index("up")
    if elevation <= -VERTICAL_ELEVATION_THRESHOLD_DEG:
        return SECTOR_NAMES.index("down")
    azimuth = _azimuth_deg(rel[:2], forward)
    return int(round(azimuth / HORIZONTAL_SECTOR_WIDTH_DEG)) % N_HORIZONTAL_SECTORS


def compute_sector_features(
    defender_pos: np.ndarray,
    defender_vel: np.ndarray,
    perceived: PerceivedObstacleSet,
    sensor_range: float,
) -> np.ndarray:
    """Fixed-size (SECTOR_FEATURE_DIM,) float32 array: for each of the 10
    sectors, [static_clearance, dynamic_clearance] normalized to [0, 1]
    (1.0 = no perceived obstacle of that kind within `sensor_range` in
    that sector). Derived ONLY from `perceived` (Phase 9) -- never reads
    ground truth. Deterministic given identical inputs (Phase 13.12)."""
    defender_pos = np.asarray(defender_pos, dtype=np.float64)
    forward = _forward_axis(defender_vel)

    # Initialize every (sector, channel) to the max normalized clearance
    # (sensor_range itself, before normalization) -- "no obstacle" and
    # "obstacle at exactly the sensing limit" are the SAME value (1.0
    # after normalization), per Phase 8.
    min_static = np.full(N_SECTORS, sensor_range, dtype=np.float64)
    min_dynamic = np.full(N_SECTORS, sensor_range, dtype=np.float64)

    for obs in perceived.static:
        box = AABBObstacle(min_corner=obs.min_corner, max_corner=obs.max_corner)
        rel = box.center - defender_pos
        idx = _sector_index(rel, forward)
        clearance = max(box.distance_to_point(defender_pos), 0.0)
        if clearance < min_static[idx]:
            min_static[idx] = clearance

    for obs in perceived.dynamic:
        rel = np.asarray(obs.position, dtype=np.float64) - defender_pos
        idx = _sector_index(rel, forward)
        clearance = max(float(np.linalg.norm(rel)) - float(obs.radius), 0.0)
        if clearance < min_dynamic[idx]:
            min_dynamic[idx] = clearance

    static_norm = np.clip(min_static / sensor_range, 0.0, 1.0)
    dynamic_norm = np.clip(min_dynamic / sensor_range, 0.0, 1.0)
    # Interleave as [sector0_static, sector0_dynamic, sector1_static, ...]
    # -- fixed, documented feature order (Phase 12).
    features = np.stack([static_norm, dynamic_norm], axis=1).reshape(-1)
    return np.nan_to_num(features, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)
