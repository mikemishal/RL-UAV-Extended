"""
ReachabilityAwareLeadPolicy (RA-Lead), Phase 76 -- controller ablation
using CONSTANT-VELOCITY target prediction ONLY (no learned model) combined
with the SAME dynamic-reachability selection machinery as RA-LAPG. The
only difference from `RALAPGPolicy` is the target predictor
(`p_enemy_CV(t+tau) = p_hat_e + v_hat_e*tau` vs. the learned
horizon-conditioned residual) -- everything else (candidate-time grid,
defender dynamics model, reachability metric, intercept-radius criterion,
selection rule) is reused verbatim from
`uav_defend.guidance.reachability_selection`.
"""

from __future__ import annotations

import numpy as np

from uav_defend.guidance.reachability_selection import (
    ReachabilityMode,
    ReachabilitySolution,
    evaluate_candidate,
    select_candidate,
)
from uav_defend.policies.baseline.lead_math import get_target_state_estimate, pure_pursuit_direction

CANDIDATE_TAUS: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)


class ReachabilityAwareLeadPolicy:
    def __init__(
        self, state_source: str = "measurement", config=None, dt: float | None = None,
        v_d: float | None = None, eps: float | None = None,
        candidate_taus: tuple[float, ...] = CANDIDATE_TAUS,
        intercept_radius: float | None = None, max_rollout_steps_cap: int = 40,
        reachability_mode: ReachabilityMode = ReachabilityMode.FIXED_DIRECTION,
    ):
        self.state_source = state_source
        self.config = config
        self.dt = float(dt if dt is not None else (config.dt if config is not None else 0.5))
        self.v_d = float(v_d if v_d is not None else (config.v_d if config is not None else 18.0))
        self.eps = float(eps if eps is not None else (config.eps if config is not None else 1e-8))
        self.candidate_taus = candidate_taus
        self.intercept_radius = float(
            intercept_radius if intercept_radius is not None else (config.intercept_radius if config is not None else 2.5)
        )
        self.max_rollout_steps_cap = max_rollout_steps_cap
        self.reachability_mode = reachability_mode

        self.last_action: np.ndarray | None = None
        self.last_guidance_mode: str | None = None
        self.last_solution: ReachabilitySolution | None = None

    def reset(self) -> None:
        self.last_action = None
        self.last_guidance_mode = None
        self.last_solution = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        defender_pos = np.asarray(info["defender_pos"], dtype=np.float64)
        defender_vel = np.asarray(info["defender_vel"], dtype=np.float64)
        target_pos, target_vel = get_target_state_estimate(info, self.state_source)

        if target_pos is None:
            self.last_guidance_mode = "escort"
            soldier_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
            action = pure_pursuit_direction(soldier_pos, defender_pos, self.eps)
            self.last_action = action
            return action

        if target_vel is None:
            self.last_guidance_mode = "no_velocity_fallback"
            action = pure_pursuit_direction(target_pos, defender_pos, self.eps)
            self.last_action = action
            return action

        def predict_fn(h: float) -> np.ndarray:
            return target_pos + target_vel * h  # constant-velocity prediction ONLY

        candidates = tuple(
            evaluate_candidate(
                defender_pos, defender_vel, predict_fn, tau, self.config,
                self.intercept_radius, self.max_rollout_steps_cap, self.reachability_mode,
            )
            for tau in self.candidate_taus
        )
        best, fallback_used = select_candidate(candidates)

        action = pure_pursuit_direction(best.predicted_target_point, defender_pos, self.eps)
        self.last_solution = ReachabilitySolution(
            action=action, selected_tau=best.tau, candidates=candidates,
            fallback_used=fallback_used, guidance_mode="ra_lead",
        )
        self.last_guidance_mode = "ra_lead"
        self.last_action = action
        return action
