"""
LR-PPO obstacle-aware clutter training environment (journal extension,
Phase 10/11/14). Layers randomized-clutter resampling and sector-feature
observation augmentation on top of the EXISTING, UNCHANGED
`LeadResidualTrainingEnv` composition (canonical Lead direction + bounded
residual, theta_max=30 deg by default) -- the deterministic Lead
calculation and the residual composition math are never touched; only
new features are appended to the residual policy's OWN observation.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_defend.config import EnvConfig
from uav_defend.policies.residual.lead_residual_composition import DEFAULT_RESIDUAL_MAX_ANGLE_DEG
from uav_defend.obstacles.sector_features import SectorObservationSpec, SECTOR_OBSERVATION_V1, compute_sector_features

from experiments.post_detection_training_env import PostDetectionTrainingEnv
from experiments.post_detection_training_protocol import DEFAULT_EPISODE_SEED_POOL_SIZE
from experiments.lead_residual_training_env import LeadResidualTrainingEnv
from experiments.journal_clutter_protocol import build_randomized_training_config


class _RandomizedClutterPostDetectionTrainingEnv(PostDetectionTrainingEnv):
    """Identical to `PostDetectionTrainingEnv`, except the wrapped
    `SoldierEnv`'s obstacle-density config (Phase 14) is resampled every
    time a new internal episode seed is drawn -- BEFORE that seed is used
    to reset the canonical env -- keeping every awareness/avoidance
    CANONICAL setting fixed (only count/size/height/dynamic-count vary)."""

    def __init__(self, config: EnvConfig, estimator: str, **kwargs):
        super().__init__(config=config, **kwargs)
        self._estimator = estimator

    def _next_episode_seed(self) -> int:
        seed = super()._next_episode_seed()
        rng = np.random.default_rng(seed)
        self._env.config = build_randomized_training_config(rng, self._estimator)
        return seed


class LeadResidualClutterTrainingEnv(LeadResidualTrainingEnv):
    """`LeadResidualTrainingEnv` with its inner `PostDetectionTrainingEnv`
    replaced by the randomized-clutter variant above. The Lead direction
    computation, residual composition, and theta_max bound are inherited
    UNCHANGED from `LeadResidualTrainingEnv`."""

    def __init__(
        self,
        config: EnvConfig,
        root_seed: int,
        residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
        max_standby_steps: int | None = None,
        episode_seed_pool_size: int = DEFAULT_EPISODE_SEED_POOL_SIZE,
    ):
        super().__init__(
            config=config, root_seed=root_seed, residual_max_angle_deg=residual_max_angle_deg,
            max_standby_steps=max_standby_steps, episode_seed_pool_size=episode_seed_pool_size,
        )
        # Replace the plain inner env built by the superclass with the
        # randomized-clutter variant, reusing the SAME root_seed/pool size
        # so episode-seed reproducibility is unaffected.
        self._inner = _RandomizedClutterPostDetectionTrainingEnv(
            config=config, estimator="measurement", root_seed=root_seed,
            max_standby_steps=max_standby_steps, episode_seed_pool_size=episode_seed_pool_size,
        )


class LRPPOSectorObservationWrapper(gym.Wrapper):
    """Appends obstacle-sector features (Phase 5-9) to
    `LeadResidualClutterTrainingEnv`'s existing 19-D observation
    (concat(direct_obs[16], lead_direction[3])), giving a 39-D observation.
    Perception is read from the wrapped env's actual `SoldierEnv`
    (`self.env._inner._env`) -- the SAME perceived-obstacle-set the shared
    safety layer already uses (Phase 9: no ground-truth leakage)."""

    def __init__(self, env: LeadResidualClutterTrainingEnv, spec: SectorObservationSpec = SECTOR_OBSERVATION_V1):
        super().__init__(env)
        if not env.config.defender_obstacle_avoidance_enabled:
            raise ValueError(
                "LRPPOSectorObservationWrapper requires config.defender_obstacle_avoidance_enabled=True"
            )
        self.sector_spec = spec
        base_low = env.observation_space.low
        base_high = env.observation_space.high
        low = np.concatenate([base_low, np.zeros(spec.feature_dim, dtype=np.float32)])
        high = np.concatenate([base_high, np.ones(spec.feature_dim, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _soldier_env(self):
        return self.env._inner._env

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        se = self._soldier_env()
        features = compute_sector_features(
            defender_pos=se._defender_pos, defender_vel=se._defender_vel,
            perceived=se._defender_last_perceived, sensor_range=self.sector_spec.sensor_range,
        )
        return np.concatenate([np.asarray(obs, dtype=np.float32), features])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info


def build_lr_ppo_clutter_env(
    root_seed: int,
    residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
) -> gym.Env:
    """The canonical LR-PPO obstacle-aware clutter TRAINING environment
    (Phase 10/11): randomized clutter (Phase 14) + sector features
    (Phase 5-9), Lead+residual composition and theta_max UNCHANGED."""
    base_config = build_randomized_training_config(np.random.default_rng(root_seed), "measurement")
    env = LeadResidualClutterTrainingEnv(
        config=base_config, root_seed=root_seed, residual_max_angle_deg=residual_max_angle_deg,
    )
    return LRPPOSectorObservationWrapper(env, spec=spec)


def build_lr_ppo_clutter_eval_env(
    fixed_config: EnvConfig,
    root_seed: int,
    residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
) -> gym.Env:
    """FIXED-clutter-config counterpart of `build_lr_ppo_clutter_env`, for
    held-out evaluation at a specific canonical clutter level (C1/C2/C3).

    Bug found and fixed during Phase 15-17 (journal extension scale-up
    sub-task): `build_lr_ppo_clutter_env` ALWAYS resamples the obstacle
    config every episode (via `LeadResidualClutterTrainingEnv`'s
    `_RandomizedClutterPostDetectionTrainingEnv._next_episode_seed()`
    override) -- there was no way to evaluate LR-PPO at a FIXED canonical
    clutter level, unlike `build_ppo_clutter_env(..., randomized=False,
    fixed_config=...)` used for PPO/PPO-Kalman. That meant
    `evaluate_clutter_trained`'s LR-PPO arm was silently evaluating the
    SAME randomized-clutter distribution for every "C1"/"C2"/"C3" label,
    making its level-based comparison meaningless. This function uses the
    PLAIN (non-randomizing) `LeadResidualTrainingEnv` with the caller's
    fixed config instead of the `LeadResidualClutterTrainingEnv` subclass,
    giving genuinely fixed-clutter-level held-out evaluation while still
    using the SAME Lead+residual composition, theta_max bound, and sector
    observation wrapper as training."""
    from experiments.lead_residual_training_env import LeadResidualTrainingEnv

    env = LeadResidualTrainingEnv(
        config=fixed_config, root_seed=root_seed, residual_max_angle_deg=residual_max_angle_deg,
    )
    return LRPPOSectorObservationWrapper(env, spec=spec)
