"""
RA-LAPG (Reachability-Aware Learning-Augmented Predictive Guidance),
Phase 54-58 -- controller ablation. Contains learned target-motion
prediction + analytical dynamic-feasibility reasoning, and DELIBERATELY
NO learned action residual (Phase 59).

Uses the shared `uav_defend.guidance.reachability_selection` machinery
(Phase 76): the ONLY thing that distinguishes this controller from
`ReachabilityAwareLeadPolicy` (RA-Lead) is the target predictor -- the
candidate-time grid, defender dynamics model, reachability metric,
intercept-radius criterion, and selection rule are IDENTICAL.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, build_feature_vector
from uav_defend.guidance.reachability_selection import (
    ReachabilityMode,
    ReachabilitySolution,
    evaluate_candidate,
    select_candidate,
)
from uav_defend.policies.baseline.lead_math import get_target_state_estimate, pure_pursuit_direction

# Phase 55/75: deterministic candidate-time grid. Capped at 16s to match
# the predictor's validated training horizon (Phase 47/49) -- see Phase 75
# ("operationally relevant search horizon"): extending the search beyond
# the predictor's validated range would rely on unvalidated extrapolation.
CANDIDATE_TAUS: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)


class RALAPGPolicy:
    def __init__(
        self, predictor: HorizonConditionedPredictor, state_source: str = "measurement",
        config=None, dt: float | None = None, v_d: float | None = None, eps: float | None = None,
        history_length: int = 8, candidate_taus: tuple[float, ...] = CANDIDATE_TAUS,
        intercept_radius: float | None = None, max_rollout_steps_cap: int = 40,
        reachability_mode: ReachabilityMode = ReachabilityMode.FIXED_DIRECTION,
    ):
        self.predictor = predictor
        self.predictor.eval()
        self.state_source = state_source
        self.config = config
        self.dt = float(dt if dt is not None else (config.dt if config is not None else 0.5))
        self.v_d = float(v_d if v_d is not None else (config.v_d if config is not None else 18.0))
        self.eps = float(eps if eps is not None else (config.eps if config is not None else 1e-8))
        self.history_length = history_length
        self.candidate_taus = candidate_taus
        self.intercept_radius = float(
            intercept_radius if intercept_radius is not None else (config.intercept_radius if config is not None else 2.5)
        )
        self.max_rollout_steps_cap = max_rollout_steps_cap
        self.reachability_mode = reachability_mode

        self._history: list[np.ndarray] = []
        self._steps_since_detection = -1

        self.last_action: np.ndarray | None = None
        self.last_guidance_mode: str | None = None
        self.last_solution: ReachabilitySolution | None = None

    def reset(self) -> None:
        self._history.clear()
        self._steps_since_detection = -1
        self.last_action = None
        self.last_guidance_mode = None
        self.last_solution = None

    def _query_predictor(self, tau: float) -> np.ndarray:
        window = self._history[-self.history_length:]
        n_pad = self.history_length - len(window)
        mask = [0] * n_pad + [1] * len(window)
        padded = [np.zeros(FEATURE_DIM) for _ in range(n_pad)] + window
        history_t = torch.as_tensor(np.stack(padded, axis=0), dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        horizon_t = torch.tensor([tau], dtype=torch.float32)
        with torch.no_grad():
            delta = self.predictor(history_t, mask_t, horizon_t)[0].numpy()
        return delta

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        defender_pos = np.asarray(info["defender_pos"], dtype=np.float64)
        defender_vel = np.asarray(info["defender_vel"], dtype=np.float64)
        target_pos, target_vel = get_target_state_estimate(info, self.state_source)

        if target_pos is None:
            self._steps_since_detection = -1
            self.last_guidance_mode = "escort"
            soldier_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
            action = pure_pursuit_direction(soldier_pos, defender_pos, self.eps)
            self.last_action = action
            return action

        self._steps_since_detection = 0 if self._steps_since_detection < 0 else self._steps_since_detection + 1
        soldier_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
        est_vel_for_features = target_vel if target_vel is not None else np.zeros(3)
        features = build_feature_vector(
            target_pos, est_vel_for_features, defender_pos, defender_vel, soldier_pos,
            self._steps_since_detection * self.dt,
        )
        self._history.append(features)
        if len(self._history) > self.history_length:
            self._history.pop(0)

        if target_vel is None:
            self.last_guidance_mode = "no_velocity_fallback"
            action = pure_pursuit_direction(target_pos, defender_pos, self.eps)
            self.last_action = action
            return action

        def predict_fn(h: float) -> np.ndarray:
            delta_p = self._query_predictor(h)
            p = target_pos + target_vel * h + delta_p
            return p if np.all(np.isfinite(p)) else target_pos + target_vel * h

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
            fallback_used=fallback_used, guidance_mode="ra_lapg",
        )
        self.last_guidance_mode = "ra_lapg"
        self.last_action = action
        return action

