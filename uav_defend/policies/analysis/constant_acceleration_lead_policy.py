"""ANALYSIS controller: Constant-Acceleration (CA) Lead (Phase 6C).

Practical diagnostic controller using ONLY estimated hostile history (the
same legitimate `enemy_measurement`/`enemy_measurement_velocity` or
`e_hat`/`v_hat` fields `LeadInterceptPolicy` uses -- never ground truth).

Acceleration is estimated by a simple, UNSMOOTHED finite difference of
consecutive velocity ESTIMATES:

    a_hat_t = (v_hat_t - v_hat_(t-1)) / dt

No additional filtering/tuning is applied in this first implementation
(per the Phase-6C specification). The quartic CA-Lead geometry itself
(and its fallback-to-Standard-Lead behavior) lives in
`uav_defend.policies.analysis.quartic_lead_math.solve_ca_lead` -- this
class only owns the velocity-estimate history / finite-difference
acceleration bookkeeping and the same escort/no-estimate-yet fallback
handling `LeadInterceptPolicy` uses (via the shared
`uav_defend.policies.baseline.lead_math.get_target_state_estimate` helper).
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.analysis.quartic_lead_math import CALeadSolution, solve_ca_lead
from uav_defend.policies.baseline.lead_math import get_target_state_estimate, pure_pursuit_direction, solve_cv_lead


class ConstantAccelerationLeadPolicy:
    """ANALYSIS-ONLY: constant-acceleration Lead using estimated hostile
    history. Reads only legitimate info fields (same as `LeadInterceptPolicy`)."""

    VALID_STATE_SOURCES = ("measurement", "kalman")

    def __init__(
        self,
        state_source: str = "measurement",
        config: EnvConfig | None = None,
        dt: float | None = None,
        v_d: float | None = None,
        eps: float | None = None,
    ):
        if state_source not in self.VALID_STATE_SOURCES:
            raise ValueError(f"state_source must be one of {self.VALID_STATE_SOURCES}, got '{state_source}'")

        cfg = config if config is not None else EnvConfig()
        self.state_source = state_source
        self.dt = float(dt) if dt is not None else float(cfg.dt)
        self.v_d = float(v_d) if v_d is not None else float(cfg.v_d)
        self.eps = float(eps) if eps is not None else float(cfg.eps)
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.v_d <= 0:
            raise ValueError(f"v_d must be > 0, got {self.v_d}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        self._prev_velocity_estimate: np.ndarray | None = None
        self._init_diagnostics()

    def _init_diagnostics(self) -> None:
        self.last_guidance_mode: str | None = None
        self.last_acceleration_estimate: np.ndarray | None = None
        self.last_solution: CALeadSolution | None = None

    def reset(self) -> None:
        """Reset per-episode state, including the velocity-estimate
        history used for finite-difference acceleration."""
        self._prev_velocity_estimate = None
        self._init_diagnostics()

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        eps = self.eps
        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        soldier_pos = info.get("soldier_pos")
        enemy_detected = bool(info.get("enemy_detected", False))

        self._init_diagnostics()

        if not enemy_detected:
            self.last_guidance_mode = "escort"
            return pure_pursuit_direction(soldier_pos, defender_pos, eps)

        target_pos, target_vel = get_target_state_estimate(info, self.state_source)

        if target_pos is None:
            self.last_guidance_mode = "escort"
            return pure_pursuit_direction(soldier_pos, defender_pos, eps)

        if target_vel is None:
            self.last_guidance_mode = (
                "first_measurement_fallback" if self.state_source == "measurement" else "no_velocity_fallback"
            )
            # No velocity estimate yet => no acceleration estimate either;
            # the velocity history cannot be advanced this step.
            return pure_pursuit_direction(target_pos, defender_pos, eps)

        if self._prev_velocity_estimate is None:
            # First step with a valid velocity estimate: acceleration is
            # not yet estimable (need two consecutive velocity estimates).
            self._prev_velocity_estimate = target_vel.copy()
            self.last_guidance_mode = "no_acceleration_estimate_fallback"
            cv_solution = solve_cv_lead(target_pos, target_vel, defender_pos, self.v_d, eps)
            return cv_solution.action

        accel_estimate = (target_vel - self._prev_velocity_estimate) / self.dt
        self._prev_velocity_estimate = target_vel.copy()
        self.last_acceleration_estimate = accel_estimate.astype(np.float32)

        solution = solve_ca_lead(target_pos, target_vel, accel_estimate, defender_pos, self.v_d, eps)
        self.last_solution = solution
        self.last_guidance_mode = solution.guidance_mode
        return solution.action

    def __repr__(self) -> str:
        return f"ConstantAccelerationLeadPolicy(ANALYSIS-ONLY, state_source='{self.state_source}')"
