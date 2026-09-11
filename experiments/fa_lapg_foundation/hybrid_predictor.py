"""
Phase 73: deterministic hybrid CV/learned predictor.

Phase 72 acceptance test found that multiscale training (Phase 70)
substantially reduced but did NOT eliminate the short-horizon regression:
the multiscale GRU is still worse than CV at 0.5s (4.69 vs 2.95) and 1.0s
(6.25 vs 5.90), but wins from 2.0s onward (10.27 vs 15.25) with the
advantage growing sharply at long horizons. Per the explicit instruction
to not hide a persisting short-horizon regression, this hybrid predictor
is used instead of trusting the learned model at every horizon:

    p_pred(h) = p_CV(h),            h <= h_switch
              = p_CV(h) + Delta_p(h), h > h_switch

`h_switch = 2.0s` is chosen from the SAME acceptance-test evidence above
(the crossover point where the multiscale GRU first beats CV), not from
the final journal evaluation data.
"""

from __future__ import annotations

import torch
from torch import nn

# Chosen from Phase 72 acceptance-test validation results (crossover
# point where the multiscale GRU first outperforms CV).
DEFAULT_H_SWITCH = 2.0


class HybridPredictor(nn.Module):
    """Drop-in replacement for `HorizonConditionedPredictor`: returns a
    ZERO residual (pure CV) for h <= h_switch, and the wrapped model's
    learned residual for h > h_switch. Deterministic, no learned gate."""

    def __init__(self, learned_model: nn.Module, h_switch: float = DEFAULT_H_SWITCH):
        super().__init__()
        self.learned_model = learned_model
        self.learned_model.eval()
        self.h_switch = h_switch

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            delta = self.learned_model(history, history_mask, horizon)
        use_learned = (horizon > self.h_switch).reshape(-1, 1).to(delta.dtype)
        return delta * use_learned

    def eval(self):
        self.learned_model.eval()
        return super().eval()
