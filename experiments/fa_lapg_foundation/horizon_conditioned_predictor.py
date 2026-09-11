"""
Horizon-conditioned temporal predictor (Phase 48-49).

Phase 47 found the fixed 0.5-4s GRU covers only 54-73% of valid Lead
intercept times in 5/6 scenarios and just 16% in `mobility_mismatch`
(median tau=15.3s, p95=203s -- with a heavy, almost certainly numerically
degenerate tail out to tens of thousands of seconds from near-parallel-
velocity quadratic solutions). This module replaces the fixed multi-output
head with a GRU encoder + small MLP decoder queried at an ARBITRARY
horizon, so a single model can be evaluated at whatever tau the current
Lead/RA-LAPG solution actually requires.

p_pred(t+h) = p_hat(t) + v_hat(t)*h + Delta_p_phi(history, h)

CV remains the nominal model; the network learns only the residual,
exactly as in the fixed-horizon predictor (Phase 30).
"""

from __future__ import annotations

import torch
from torch import nn

from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM

# Phase 49: training-horizon sampling range, chosen from the empirical tau
# distribution (Phase 47), NOT an arbitrary maximum. p95 across the 5
# well-behaved scenarios (clutter, clutter_strong_maneuver, nominal_open,
# strong_evasion, strong_maneuver) ranges 8.0-16.6s; 16s is used as an
# explicit cap for physically-degenerate long tails (`mobility_mismatch`'s
# p95=203s and max=56427s are near-parallel-velocity quadratic-solver
# artifacts, not actionable engagement horizons -- see
# `docs/experiment_seed_registry.md`/Phase 47 results for the full table).
TRAIN_HORIZON_MIN = 0.5
TRAIN_HORIZON_MAX = 16.0
HORIZON_NORMALIZATION_SCALE = TRAIN_HORIZON_MAX  # normalized_h = h / this


class HorizonConditionedPredictor(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_size: int = 32, decoder_hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim + 1, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 1, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 3),
        )

    def encode(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, history_mask.unsqueeze(-1).to(history.dtype)], dim=-1)
        _out, h_n = self.gru(x)
        return h_n[-1]  # (B, hidden_size)

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """
        history: (B, T, feature_dim), history_mask: (B, T)
        horizon: (B,) or (B, 1) RAW seconds (normalized internally).
        Returns: (B, 3) predicted residual Delta_p_phi(history, horizon).
        """
        z = self.encode(history, history_mask)
        h_norm = (horizon.reshape(-1, 1).to(z.dtype)) / HORIZON_NORMALIZATION_SCALE
        decoder_input = torch.cat([z, h_norm], dim=-1)
        return self.decoder(decoder_input)
