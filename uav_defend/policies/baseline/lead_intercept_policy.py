"""
Lead Intercept Policy - 3-D constant-velocity predictive interception baseline.

=============================================================================
WHAT THIS IS
=============================================================================
A classical 3-D constant-velocity lead-intercept guidance law: the hostile
UAV's current position/velocity are estimated, its velocity is assumed
constant over a short prediction horizon, and an idealized interceptor
traveling at the defender's nominal maximum speed (config.v_d) is used to
analytically solve for the earliest future interception time and the
corresponding predicted intercept point. The controller returns only the
3-D DIRECTION from the current defender position toward that point -- the
same common desired-velocity-direction action interface used by every other
policy (GreedyInterceptPolicy, KalmanGreedyInterceptPolicy,
ProportionalNavigationPolicy, RandomPolicy, PPO).

This is "constant-velocity predictive interception" / "3-D constant-velocity
lead-intercept guidance". It is NOT:
    - globally optimal control;
    - minimum-time optimal guidance;
    - a perfect / ground-truth lead solution;
    - proportional navigation (a separate, already-implemented baseline).

The intercept-speed assumption `s = config.v_d` is an IDEALIZED kinematic
constant used only inside the analytical quadratic. The actual defender
still starts from rest, has bounded acceleration, bounded turn rate, and
bounded climb/descent rate -- it will generally NOT arrive at the predicted
point exactly at t_intercept. That mismatch is intentional: the environment
alone determines physical realizability (see _advance_velocity()). Because
the hostile UAV may maneuver/evade after the prediction is made, the
constant-velocity prediction becomes stale between steps -- the intercept
point is therefore recomputed every step from the latest legitimate target
estimate, which is the meaningful behavior this baseline is intended to
exhibit.

=============================================================================
GROUND-TRUTH INDEPENDENCE (hard requirement)
=============================================================================
This controller NEVER reads `info["enemy_pos"]` or `info["enemy_vel"]`
(ground truth, exposed only for evaluation/debugging). Guidance uses only:

    defender_pos, defender_vel   (defender's own true state -- always legitimate)
    soldier_pos                  (escort fallback target)
    enemy_detected                (detection flag)
    enemy_measurement             (raw noisy hostile position -- "measurement" mode)
    e_hat, v_hat                  (Kalman hostile position/velocity estimate -- "kalman" mode)

=============================================================================
TWO SENSING MODES, ONE INTERCEPT GEOMETRY
=============================================================================
    state_source="measurement" (canonical name "lead"):
        Hostile position is the raw noisy measurement `enemy_measurement`.
        Hostile velocity is the environment's standardized finite-difference
        measurement velocity `info["enemy_measurement_velocity"]` (valid only
        when `info["enemy_measurement_velocity_valid"]` is True). This
        estimator is computed ONCE by the environment (see
        SoldierEnv._update_detection) and shared by the Direct observation
        and every measurement-mode controller (Greedy, PN, Lead) -- no
        independent per-policy finite differencing.

    state_source="kalman" (canonical name "lead_kalman"):
        Hostile position/velocity come directly from the environment's
        existing 3-D Kalman estimator (e_hat, v_hat). No independent Kalman
        filter is instantiated here, and `enemy_measurement` is never used
        for this variant's guidance.
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline.lead_math import (
    get_target_state_estimate,
    pure_pursuit_direction,
    solve_cv_lead,
    solve_quadratic_intercept_time,
)


class LeadInterceptPolicy:
    """
    3-D constant-velocity lead-intercept guidance, adapted to the common
    desired-velocity-direction action interface.

    Intercept geometry:
        r = p_hat_e - p_d                          (relative position)
        Assumed constant target motion: p_e(t) = p_hat_e + v_hat_e * t
        Ideal interceptor at speed s = config.v_d: ||r + v_hat_e t|| = s t
        => quadratic  a t^2 + b t + c = 0  where
               a = v_hat_e . v_hat_e - s^2
               b = 2 * (r . v_hat_e)
               c = r . r
        Earliest positive real root t_intercept is selected (see
        _solve_intercept_time for the numerically stable solver).
        p_intercept = p_hat_e + v_hat_e * t_intercept
        lead_vector = p_intercept - p_d
        action      = lead_vector / ||lead_vector||

    When no valid intercept solution exists (see `act()` for the exact
    fallback conditions), the policy falls back to pure pursuit of the
    current legitimate target-position estimate/measurement -- never to
    ground truth.

    Attributes (diagnostics, NOT part of the RL observation):
        last_guidance_mode: str | None, one of:
            "escort", "first_measurement_fallback", "no_velocity_fallback",
            "no_real_solution_fallback", "no_positive_root_fallback",
            "degenerate_fallback", "lead"
        last_target_position: np.ndarray | None, shape (3,)
        last_target_velocity_estimate: np.ndarray | None, shape (3,)
        last_relative_position: np.ndarray | None, shape (3,)
        last_quadratic_a: float | None
        last_quadratic_b: float | None
        last_quadratic_c: float | None
        last_discriminant: float | None
        last_intercept_time: float | None
        last_intercept_point: np.ndarray | None, shape (3,)
        last_action: np.ndarray | None, shape (3,)
    """

    VALID_STATE_SOURCES = ("measurement", "kalman")

    def __init__(
        self,
        state_source: str = "measurement",
        config: EnvConfig | None = None,
        dt: float | None = None,
        v_d: float | None = None,
        eps: float | None = None,
    ):
        """
        Initialize the lead-intercept policy.

        Args:
            state_source: "measurement" (raw noisy measurement + finite-
                difference velocity) or "kalman" (environment's e_hat/v_hat).
            config: Optional EnvConfig used ONLY to read dt/v_d/eps defaults;
                never mutated. If None, a default EnvConfig() is constructed
                solely to source these three values.
            dt: Override for the finite-difference time step (seconds).
                Defaults to config.dt.
            v_d: Override for the assumed idealized interceptor speed `s`
                used in the intercept-time quadratic. Defaults to config.v_d.
                (Exposed as an override primarily to test intercept geometry
                in isolation from the environment's default v_d=18.0.)
            eps: Override for the numerical-stability epsilon. Defaults to
                config.eps.
        """
        if state_source not in self.VALID_STATE_SOURCES:
            raise ValueError(
                f"state_source must be one of {self.VALID_STATE_SOURCES}, got '{state_source}'"
            )

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

        self._init_diagnostics()

    def _init_diagnostics(self) -> None:
        """(Re)initialize all diagnostic attributes to their empty state."""
        self.last_guidance_mode: str | None = None
        self.last_target_position: np.ndarray | None = None
        self.last_target_velocity_estimate: np.ndarray | None = None
        self.last_relative_position: np.ndarray | None = None
        self.last_quadratic_a: float | None = None
        self.last_quadratic_b: float | None = None
        self.last_quadratic_c: float | None = None
        self.last_discriminant: float | None = None
        self.last_intercept_time: float | None = None
        self.last_intercept_point: np.ndarray | None = None
        self.last_action: np.ndarray | None = None

    def reset(self) -> None:
        """
        Reset all per-episode diagnostic state.

        The measurement-mode finite-difference velocity is owned by the
        environment (see SoldierEnv._update_detection), so there is no
        internal finite-difference history to clear here.
        """
        self._init_diagnostics()

    def _get_target_state(self, info: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return (target_position, target_velocity) using ONLY legitimate
        environment fields for the configured state_source. NEVER reads
        info["enemy_pos"] or info["enemy_vel"]. Thin delegation to the
        shared pure helper (see uav_defend.policies.baseline.lead_math.
        get_target_state_estimate), reused by other estimated-state
        Lead-family diagnostic controllers.
        """
        return get_target_state_estimate(info, self.state_source)

    def _pursue(self, target: np.ndarray | None, defender_pos: np.ndarray, eps: float) -> np.ndarray:
        """Pure-pursuit fallback direction toward `target` (never ground truth)."""
        action = pure_pursuit_direction(target, defender_pos, eps)
        self.last_action = action.copy()
        return action

    @staticmethod
    def _solve_intercept_time(
        a: float, b: float, c: float, eps: float
    ) -> tuple[float | None, float | None, str]:
        """
        Solve a t^2 + b t + c = 0 for the earliest positive real root.
        Thin delegation to the shared pure helper (see
        uav_defend.policies.baseline.lead_math.solve_quadratic_intercept_time
        for the full derivation/algorithm) -- kept as a static method here
        for backward compatibility with any existing callers.
        """
        return solve_quadratic_intercept_time(a, b, c, eps)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the lead-intercept (or fallback pure-pursuit) desired-
        direction action.

        Args:
            obs: Environment observation (unused directly; all state is read
                from `info`, matching the existing baseline-policy convention).
            info: Environment info dict. Only these keys are read:
                defender_pos, soldier_pos, enemy_detected, and (depending on
                state_source) enemy_measurement OR e_hat/v_hat.
                info["enemy_pos"] and info["enemy_vel"] (ground truth) are
                NEVER read.

        Returns:
            action: 3-D desired-velocity-direction vector in [-1, 1]^3,
                shape (3,). The environment's existing constrained dynamics
                (_advance_velocity) determine actual defender motion; this
                policy never bypasses them and never writes to
                _defender_pos / _defender_vel.
        """
        eps = self.eps
        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        soldier_pos = info.get("soldier_pos")
        enemy_detected = bool(info.get("enemy_detected", False))

        self._init_diagnostics()

        if not enemy_detected:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        target_pos, target_vel = self._get_target_state(info)

        if target_pos is None:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        self.last_target_position = target_pos.astype(np.float32)

        if target_vel is None:
            self.last_guidance_mode = (
                "first_measurement_fallback" if self.state_source == "measurement" else "no_velocity_fallback"
            )
            return self._pursue(target_pos, defender_pos, eps)

        self.last_target_velocity_estimate = target_vel.astype(np.float32)

        # Core constant-velocity Lead geometry -- delegated to the shared
        # pure helper (uav_defend.policies.baseline.lead_math.solve_cv_lead)
        # so it is not duplicated across diagnostic controllers (e.g. the
        # True-State CV Lead analysis policy reuses the SAME function).
        solution = solve_cv_lead(target_pos, target_vel, defender_pos, self.v_d, eps)
        self.last_relative_position = solution.relative_position
        self.last_quadratic_a = solution.quadratic_a
        self.last_quadratic_b = solution.quadratic_b
        self.last_quadratic_c = solution.quadratic_c
        self.last_discriminant = solution.discriminant
        self.last_intercept_time = solution.intercept_time
        self.last_intercept_point = solution.intercept_point
        self.last_guidance_mode = solution.guidance_mode
        self.last_action = solution.action.copy()
        return solution.action

    def __repr__(self) -> str:
        return f"LeadInterceptPolicy(state_source='{self.state_source}', v_d={self.v_d})"
