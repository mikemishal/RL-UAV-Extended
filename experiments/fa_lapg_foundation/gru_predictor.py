"""
Small GRU residual temporal-motion predictor (Phase 29C, 30, 31).

Design decisions (documented per task instructions):

Phase 29C -- model choice: a single small GRU (not a transformer, not a
large LSTM/attention model). Sequential, lightweight, a natural fit for a
short (`HISTORY_LENGTH`-step) motion-history window, and easier to train
and interpret than a larger architecture. This is sufficient to test the
hypothesis that recent motion history predicts future position better
than CV/CA.

Phase 30 -- learns RESIDUAL motion, not absolute position:
    Delta_p_phi(history, horizon) = p_pred(t+horizon) - p_CV(t+horizon)
This preserves the analytical CV model as the nominal predictor; the
network only has to learn where CV is wrong.

Phase 31 -- multi-horizon design: ONE network predicts ALL fixed horizons
in a single forward pass (a multi-output head of shape (num_horizons, 3)),
rather than five separate per-horizon networks or a horizon-conditioned
input. This was chosen because the dataset already produces one sample
per (episode, step) carrying ALL horizon targets simultaneously, so a
multi-output head requires no additional tiling/duplication of the input
history and trains all horizons jointly from a single GRU forward pass --
the simpler implementation, given the existing dataset shape.
"""

from __future__ import annotations

import torch
from torch import nn

from experiments.fa_lapg_foundation.trajectory_dataset import FEATURE_DIM, HORIZONS


class GRUResidualPredictor(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_size: int = 32, num_horizons: int = len(HORIZONS)):
        super().__init__()
        # +1 input channel for the causal history_mask (marks zero-padded
        # pre-detection rows so the network can distinguish "real but
        # zero" from "padded").
        self.gru = nn.GRU(input_size=feature_dim + 1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_horizons * 3)
        self.num_horizons = num_horizons

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """
        history: (B, T, feature_dim) float32
        history_mask: (B, T) float32/int, 1=real step, 0=padding
        Returns: (B, num_horizons, 3) predicted residual Delta_p_phi.
        """
        x = torch.cat([history, history_mask.unsqueeze(-1).to(history.dtype)], dim=-1)
        _out, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (B, hidden_size)
        delta = self.head(last_hidden)
        return delta.view(-1, self.num_horizons, 3)
