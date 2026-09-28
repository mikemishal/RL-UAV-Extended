"""
Shared randomized-clutter training environment construction (journal
extension, Phase 14) for PPO / PPO-Kalman / LR-PPO obstacle-aware
retraining.

`RandomizedClutterWrapper` re-samples a NEW per-episode obstacle
distribution (spanning C1-C3 difficulty, static count/size/height +
dynamic count -- see `experiments.journal_clutter_protocol.build_randomized_training_config`)
on every `reset()`, using a dedicated RNG seeded from the SAME seed passed
to `reset()` (so a given training-episode seed always yields the same
sampled clutter distribution -- reproducible). The awareness/avoidance
CANONICAL settings (Level C perception, both sides' reactive avoidance,
route planning disabled) stay fixed across every resample -- only
count/size/height/dynamic-count vary, matching Phase 14's exact scope.

This module builds the environment stack shared by PPO and PPO-Kalman
directly; the LR-PPO variant additionally layers the existing
`LeadResidualTrainingEnv` composition (see `experiments.clutter_training_env.build_lr_ppo_clutter_env`).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.envs.obstacle_observation_wrapper import ObstacleSectorObservationWrapper
from uav_defend.obstacles.sector_features import SectorObservationSpec, SECTOR_OBSERVATION_V1

from experiments.journal_clutter_protocol import build_randomized_training_config


class RandomizedClutterWrapper(gym.Wrapper):
    """Resamples the wrapped `SoldierEnv`'s obstacle-density config (Phase
    14) on every `reset()`, keeping every awareness/avoidance CANONICAL
    setting fixed. Requires the wrapped env's `config` attribute to be
    reassignable (true for `SoldierEnv`, a plain instance attribute)."""

    def __init__(self, env: gym.Env, estimator: str):
        super().__init__(env)
        soldier_env = env.unwrapped
        if not isinstance(soldier_env, SoldierEnv):
            raise TypeError(f"RandomizedClutterWrapper requires a SoldierEnv, got {type(soldier_env).__name__}")
        self._soldier_env = soldier_env
        self._estimator = estimator

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._soldier_env.config = build_randomized_training_config(rng, self._estimator)
        return self.env.reset(seed=seed, options=options)


def build_ppo_clutter_env(
    estimator: str,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
    randomized: bool = True,
    fixed_config: EnvConfig | None = None,
) -> gym.Env:
    """PPO / PPO-Kalman obstacle-aware clutter training/eval environment:
    SoldierEnv -> RandomizedClutterWrapper (Phase 14, training only) ->
    ObstacleSectorObservationWrapper (Phase 5-9). `estimator` is
    "measurement" (PPO/Direct) or "kalman" (PPO-Kalman). Pass
    `randomized=False` with an explicit `fixed_config` for deterministic
    held-out evaluation (Phase 23)."""
    if randomized:
        placeholder = build_randomized_training_config(np.random.default_rng(0), estimator)
        env: gym.Env = SoldierEnv(config=placeholder)
        env = RandomizedClutterWrapper(env, estimator=estimator)
    else:
        if fixed_config is None:
            raise ValueError("fixed_config is required when randomized=False")
        env = SoldierEnv(config=fixed_config)
    return ObstacleSectorObservationWrapper(env, spec=spec)
