"""
Training loop for the GRU residual predictor (Phase 27-31 wiring).

NOT reinforcement learning: standard supervised regression on the
residual-motion targets produced by `trajectory_dataset.build_samples`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.fa_lapg_foundation.trajectory_dataset import HORIZONS


class ResidualPredictorDataset(Dataset):
    """Wraps a list of sample dicts (from `trajectory_dataset.build_samples`)
    into fixed-size tensors. Horizons with an unavailable target (beyond
    the recorded episode) are masked out of the loss, never fabricated."""

    def __init__(self, samples: list[dict], horizons: tuple[float, ...] = HORIZONS):
        self.samples = samples
        self.horizons = horizons

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        history = torch.as_tensor(s["history"], dtype=torch.float32)
        mask = torch.as_tensor(s["history_mask"], dtype=torch.float32)

        target_delta = np.zeros((len(self.horizons), 3), dtype=np.float32)
        target_valid = np.zeros(len(self.horizons), dtype=np.float32)
        for k, h in enumerate(self.horizons):
            tgt = s["targets"][h]
            if tgt is not None:
                target_delta[k] = tgt - s["cv_pred"][h]
                target_valid[k] = 1.0

        return {
            "history": history,
            "history_mask": mask,
            "target_delta": torch.as_tensor(target_delta),
            "target_valid": torch.as_tensor(target_valid),
        }


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 0


def masked_mse_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over (x,y,z), averaged ONLY over valid
    (sample, horizon) entries -- horizons with no recorded future target
    contribute nothing to the loss."""
    per_horizon_sq_err = ((pred_delta - target_delta) ** 2).sum(dim=-1)  # (B, H)
    valid_sum = target_valid.sum()
    if valid_sum <= 0:
        return per_horizon_sq_err.sum() * 0.0
    return (per_horizon_sq_err * target_valid).sum() / valid_sum


def train_predictor(model: nn.Module, train_samples: list[dict], config: TrainConfig = TrainConfig()) -> list[float]:
    torch.manual_seed(config.seed)
    dataset = ResidualPredictorDataset(train_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses = []
    model.train()
    for _epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch["history"], batch["history_mask"])
            loss = masked_mse_loss(pred, batch["target_delta"], batch["target_valid"])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses
