"""ANALYSIS / ORACLE controller: True-State constant-velocity Lead.

NOT a deployable policy. This controller exists SOLELY for the
learning-benefit diagnostic study (Phase 6B) to separate estimation/
sensing error from target-motion-model error: it uses the SAME
constant-velocity Lead interception model as `LeadInterceptPolicy`
(via the shared pure helper `uav_defend.policies.baseline.lead_math.
solve_cv_lead`), but is fed the hostile's TRUE position/velocity instead
of an estimate.

GROUND-TRUTH BOUNDARY (hard requirement):
    - `act()` requires the caller to explicitly pass `true_hostile_position`
      and `true_hostile_velocity` as separate keyword arguments -- NOT read
      from a sanitized `info` dict, and NOT obtainable via the standard
      `act(obs, info)` two-argument signature every deployable policy uses.
      This is a deliberate signature difference so this class can never be
      silently substituted for a deployable policy in an evaluation harness
      that only sanitizes/passes `(obs, info)`.
    - The 16-D observation space is completely unaffected; this class does
      not read `obs` at all.
    - This class must NEVER be used in the shared policy registry
      (uav_defend/policies/registry.py) or the standard evaluation harness
      that other deployable policies go through.
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline.lead_math import LeadSolution, pure_pursuit_direction, solve_cv_lead


class TrueStateLeadPolicy:
    """ANALYSIS-ONLY oracle: constant-velocity Lead fed true hostile state.

    Attributes (diagnostics, mirroring `LeadInterceptPolicy`):
        last_guidance_mode, last_solution: LeadSolution | None
    """

    def __init__(self, config: EnvConfig | None = None, v_d: float | None = None, eps: float | None = None):
        cfg = config if config is not None else EnvConfig()
        self.v_d = float(v_d) if v_d is not None else float(cfg.v_d)
        self.eps = float(eps) if eps is not None else float(cfg.eps)
        if self.v_d <= 0:
            raise ValueError(f"v_d must be > 0, got {self.v_d}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        self._init_diagnostics()

    def _init_diagnostics(self) -> None:
        self.last_guidance_mode: str | None = None
        self.last_solution: LeadSolution | None = None

    def reset(self) -> None:
        self._init_diagnostics()

    def act(
        self,
        defender_position: np.ndarray,
        true_hostile_position: np.ndarray,
        true_hostile_velocity: np.ndarray,
        soldier_position: np.ndarray | None = None,
        enemy_detected: bool = True,
    ) -> np.ndarray:
        """
        Compute the true-state constant-velocity Lead action.

        Args:
            defender_position: Defender's own true state (always legitimate).
            true_hostile_position: GROUND TRUTH hostile position -- caller's
                responsibility to source this from environment info ONLY in
                an analysis/evaluation context, never inside a deployable
                policy's act(obs, info).
            true_hostile_velocity: GROUND TRUTH hostile velocity.
            soldier_position: Escort fallback target if `enemy_detected` is
                False. If None and not detected, returns the zero vector.
            enemy_detected: Whether to use escort fallback (mirrors
                `LeadInterceptPolicy`'s detection-gated behavior so the two
                are directly comparable).

        Returns:
            action: 3-D desired-velocity-direction unit vector, shape (3,).
        """
        eps = self.eps
        defender_position = np.asarray(defender_position, dtype=np.float64)
        self._init_diagnostics()

        if not enemy_detected:
            self.last_guidance_mode = "escort"
            return pure_pursuit_direction(soldier_position, defender_position, eps)

        solution = solve_cv_lead(
            true_hostile_position, true_hostile_velocity, defender_position, self.v_d, eps,
        )
        self.last_solution = solution
        self.last_guidance_mode = solution.guidance_mode
        return solution.action

    def __repr__(self) -> str:
        return f"TrueStateLeadPolicy(ANALYSIS-ONLY, v_d={self.v_d})"
