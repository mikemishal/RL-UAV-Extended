"""
LearnedPredictionLeadPolicy (Phase 52) -- controller ablation isolating
the value of learned target-motion prediction, with NO residual PPO, NO
learned action correction, NO feasibility gate.

At each step:
  1. Solve the existing Standard-Lead CV intercept (tau_CV, p_hat_e, v_hat_e).
  2. Query the learned horizon-conditioned predictor for
     Delta_p(history, tau_CV).
  3. p_corrected = p_hat_e + v_hat_e*tau_CV + Delta_p.
  4. Command a pure-pursuit direction toward p_corrected (same defender
     action semantics as `LeadInterceptPolicy`).

History is causal-only and reset every episode (Phase 52 "IMPORTANT
HISTORY SEMANTICS"): built from the SAME `build_feature_vector` helper
`trajectory_dataset.rollout_episode` uses, so the deployed feature layout
exactly matches what the predictor was trained on. Never reads
`info["enemy_pos"]`/`info["enemy_vel"]` (ground truth).
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, build_feature_vector
from uav_defend.policies.baseline.lead_math import get_target_state_estimate, pure_pursuit_direction, solve_cv_lead


class LearnedPredictionLeadPolicy:
    def __init__(
        self, predictor: HorizonConditionedPredictor, state_source: str = "measurement",
        config=None, dt: float | None = None, v_d: float | None = None, eps: float | None = None,
        history_length: int = 8,
    ):
        self.predictor = predictor
        self.predictor.eval()
        self.state_source = state_source
        self.config = config
        self.dt = float(dt if dt is not None else (config.dt if config is not None else 0.5))
        self.v_d = float(v_d if v_d is not None else (config.v_d if config is not None else 18.0))
        self.eps = float(eps if eps is not None else (config.eps if config is not None else 1e-8))
        self.history_length = history_length

        self._history: list[np.ndarray] = []
        self._steps_since_detection = -1

        self.last_action: np.ndarray | None = None
        self.last_guidance_mode: str | None = None
        self.last_cv_intercept_time: float | None = None
        self.last_delta_p: np.ndarray | None = None
        self.last_corrected_point: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._steps_since_detection = -1
        self.last_action = None
        self.last_guidance_mode = None
        self.last_cv_intercept_time = None
        self.last_delta_p = None
        self.last_corrected_point = None

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
        target_pos, target_vel = get_target_state_estimate(info, self.state_source)

        if target_pos is None:
            self._steps_since_detection = -1
            self.last_guidance_mode = "escort"
            soldier_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
            action = pure_pursuit_direction(soldier_pos, defender_pos, self.eps)
            self.last_action = action
            return action

        self._steps_since_detection = 0 if self._steps_since_detection < 0 else self._steps_since_detection + 1
        defender_vel = np.asarray(info["defender_vel"], dtype=np.float64)
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

        cv_solution = solve_cv_lead(target_pos, target_vel, defender_pos, self.v_d, self.eps)
        if not cv_solution.valid:
            self.last_guidance_mode = "cv_invalid_fallback"
            self.last_action = cv_solution.action
            self.last_cv_intercept_time = None
            return cv_solution.action

        tau = float(cv_solution.intercept_time)
        self.last_cv_intercept_time = tau
        delta_p = self._query_predictor(tau)
        self.last_delta_p = delta_p

        p_corrected = target_pos + target_vel * tau + delta_p
        if not np.all(np.isfinite(p_corrected)):
            p_corrected = cv_solution.intercept_point  # guard: never command toward a non-finite point

        self.last_corrected_point = p_corrected
        action = pure_pursuit_direction(p_corrected, defender_pos, self.eps)
        self.last_guidance_mode = "learned_prediction_lead"
        self.last_action = action
        return action
