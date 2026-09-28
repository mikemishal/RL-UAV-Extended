"""
Gymnasium observation wrapper (journal extension) that appends the
fixed-size directional obstacle-sector features (see
`uav_defend/obstacles/sector_features.py`) to `SoldierEnv`'s existing
16-D observation, for obstacle-aware RL training/evaluation.

CRITICAL (Phase 12): this wrapper is ADDITIVE and OPT-IN. It never
modifies `SoldierEnv` itself -- every existing/frozen conference checkpoint
continues to see the exact original 16-D observation via the unwrapped
`SoldierEnv`. Only code that explicitly constructs this wrapper sees the
extended observation.

Feature source (Phase 9): reads `env._defender_perception`/
`env._defender_last_perceived` -- the SAME `PerceivedObstacleSet` already
used by the shared reactive safety layer -- never ground truth. Because
`SoldierEnv._move_defender()` only updates `_defender_last_perceived`
when `defender_obstacle_avoidance_enabled` is True, this wrapper REQUIRES
that flag to be enabled on the wrapped env's config (asserted in
`__init__`); using it with obstacle avoidance disabled would silently
read stale/empty perception data.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import (
    SECTOR_OBSERVATION_V1,
    SectorObservationSpec,
    compute_sector_features,
)


class ObstacleSectorObservationWrapper(gym.ObservationWrapper):
    """Appends `spec.feature_dim` obstacle-sector features to the base
    observation: `obs_extended = concat(base_obs[16], sector_features[20])`.

    Args:
        env: a `SoldierEnv` (or something that ultimately wraps one) with
            `config.defender_obstacle_avoidance_enabled=True` -- required
            so `_defender_last_perceived` is actually maintained.
        spec: versioned sector-observation configuration (Phase 12);
            defaults to `SECTOR_OBSERVATION_V1`.
    """

    def __init__(self, env: gym.Env, spec: SectorObservationSpec = SECTOR_OBSERVATION_V1):
        super().__init__(env)
        soldier_env = env.unwrapped
        if not isinstance(soldier_env, SoldierEnv):
            raise TypeError(
                f"ObstacleSectorObservationWrapper requires a SoldierEnv (possibly wrapped), "
                f"got unwrapped type {type(soldier_env).__name__}"
            )
        if not soldier_env.config.defender_obstacle_avoidance_enabled:
            raise ValueError(
                "ObstacleSectorObservationWrapper requires "
                "config.defender_obstacle_avoidance_enabled=True (perception is only "
                "maintained by _move_defender() when the safety layer is active)."
            )
        self.sector_spec = spec
        self._soldier_env = soldier_env
        base_space = env.observation_space
        low = np.concatenate([base_space.low, np.zeros(spec.feature_dim, dtype=np.float32)])
        high = np.concatenate([base_space.high, np.ones(spec.feature_dim, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        env = self._soldier_env
        sector_features = compute_sector_features(
            defender_pos=env._defender_pos,
            defender_vel=env._defender_vel,
            perceived=env._defender_last_perceived,
            sensor_range=self.sector_spec.sensor_range,
        )
        return np.concatenate([np.asarray(observation, dtype=np.float32), sector_features])
