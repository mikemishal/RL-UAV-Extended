"""
Evaluation-time policy wrapper for a trained Lead-Residual PPO model (see
docs/lead_residual_ppo_design.md). This is NOT used for training -- see
experiments/lead_residual_training_env.py for the training-time wrapper.
Both share the SAME uav_defend.policies.residual.lead_residual_composition.
compose_lead_residual() function; there is exactly one Lead+residual
composition implementation in this repository.

Reuses the canonical LeadInterceptPolicy(state_source="measurement") verbatim
-- no duplicate/independent analytical Lead solver.

NO MODEL HAS BEEN TRAINED YET (see the current implementation-only task).
This wrapper exists so a FUTURE trained checkpoint can be evaluated with the
same act(obs, info)/reset() interface used by every other policy in this
codebase (GreedyInterceptPolicy, LeadInterceptPolicy, PPOPolicyWrapper, ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.residual.lead_residual_composition import (
    DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
    compose_lead_residual,
)
from uav_defend.policies.residual.lead_residual_observation import build_residual_observation

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class LeadResidualPPOPolicyWrapper:
    """
    Wraps a trained SB3 PPO model that outputs a bounded directional residual
    correction around the canonical Lead direction, exposing the standard
    act(obs, info) / reset() policy interface.

    `obs` passed to act() must be the environment's existing 16-D Direct
    observation (this wrapper internally builds the 19-D residual
    observation). `info` must contain only legitimate measurement-mode
    fields (defender_pos, soldier_pos, enemy_detected, enemy_measurement,
    enemy_measurement_velocity[_valid]) -- e.g. the output of
    uav_defend.policies.sanitize.build_policy_info(info, "measurement").
    NEVER pass info["enemy_pos"]/info["enemy_vel"] to act().
    """

    def __init__(
        self,
        model: "PPO",
        residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
        config: EnvConfig | None = None,
        deterministic: bool = True,
    ):
        self.model = model
        self.residual_max_angle_deg = float(residual_max_angle_deg)
        self.deterministic = deterministic
        self._lead_policy = LeadInterceptPolicy(state_source="measurement", config=config)

    @classmethod
    def load(
        cls,
        path: str | Path,
        residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
        config: EnvConfig | None = None,
        deterministic: bool = True,
        device: str = "auto",
    ) -> "LeadResidualPPOPolicyWrapper":
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for LeadResidualPPOPolicyWrapper. "
                "Install with: pip install stable-baselines3"
            ) from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        model = PPO.load(str(path), device=device)
        return cls(model, residual_max_angle_deg=residual_max_angle_deg, config=config, deterministic=deterministic)

    def act(self, obs: np.ndarray, info: dict, extra_features: np.ndarray | None = None) -> np.ndarray:
        """`extra_features` (journal extension, Adaptive Attacker v1): when
        given, appended to the 19-D residual observation before
        `model.predict()` -- required for clutter-trained LRPPO-C
        checkpoints (39-D: 19 + 20 sector features), None (default)
        preserves the exact original 19-D behavior for frozen conference
        LR-PPO checkpoints."""
        lead_direction = np.asarray(self._lead_policy.act(obs, info), dtype=np.float32)
        residual_obs = build_residual_observation(obs, lead_direction)
        if extra_features is not None:
            residual_obs = np.concatenate([residual_obs, np.asarray(extra_features, dtype=np.float32)])
        r_raw, _ = self.model.predict(residual_obs, deterministic=self.deterministic)
        return compose_lead_residual(lead_direction, r_raw, self.residual_max_angle_deg)

    def reset(self) -> None:
        self._lead_policy.reset()

    @property
    def name(self) -> str:
        return "lead_residual_ppo"

    def __repr__(self) -> str:
        mode = "deterministic" if self.deterministic else "stochastic"
        return f"LeadResidualPPOPolicyWrapper({mode}, residual_max_angle_deg={self.residual_max_angle_deg})"
