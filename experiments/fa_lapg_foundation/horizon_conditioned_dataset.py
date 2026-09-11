"""
Dataset + training for the horizon-conditioned predictor (Phase 48-50).

Unlike the fixed-horizon predictor, horizons here are CONTINUOUS (sampled
uniformly from `[TRAIN_HORIZON_MIN, TRAIN_HORIZON_MAX]`), so true future
hostile position is obtained via the SAME linear-interpolation machinery
the diagnostic study already uses for retrospective CV/CA error analysis
(`experiments.learning_benefit.prediction_error`), not exact dt-grid
lookup. Samples whose query time falls beyond the recorded episode are
skipped (never fabricated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.fa_lapg_foundation.horizon_conditioned_predictor import (
    TRAIN_HORIZON_MAX,
    TRAIN_HORIZON_MIN,
)
from experiments.fa_lapg_foundation.trajectory_dataset import EpisodeRecord
from experiments.learning_benefit.prediction_error import TrajectorySample, interpolate_true_position

# Phase 70: equal-count stratified horizon bands (Phase 69 found sampling
# density is uniform per unit time, but the SHORT bands are much narrower
# and therefore receive far fewer absolute training samples -- e.g. only
# 3.4% of samples land in [0.5,1)s vs 26.3% in [4,8)s -- and squared-error
# loss is dominated by the much larger long-horizon residual magnitudes.
# Sampling an EQUAL count from each band directly counteracts both.
HORIZON_BANDS: tuple[tuple[float, float], ...] = (
    (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0),
)


def _build_history_window(episode: EpisodeRecord, i: int, history_length: int) -> tuple[np.ndarray, np.ndarray]:
    step = episode.steps[i]
    window: list[np.ndarray] = []
    mask: list[int] = []
    for offset in range(history_length - 1, -1, -1):
        j = i - offset
        if j >= 0 and episode.steps[j].detected:
            window.append(episode.steps[j].features)
            mask.append(1)
        else:
            window.append(np.zeros_like(step.features))
            mask.append(0)
    return np.stack(window, axis=0), np.array(mask, dtype=np.int64)


def build_horizon_samples(
    episode: EpisodeRecord,
    rng: np.random.Generator,
    n_horizons_per_step: int = 4,
    horizon_min: float = TRAIN_HORIZON_MIN,
    horizon_max: float = TRAIN_HORIZON_MAX,
    history_length: int = 8,
) -> list[dict]:
    samples: list[dict] = []
    traj_samples = [
        TrajectorySample(s.step_index, s.sim_time, s.true_position) for s in episode.steps
    ]

    for i, step in enumerate(episode.steps):
        if not step.detected:
            continue

        window, mask = _build_history_window(episode, i, history_length)
        est_pos_now = step.features[0:3]
        est_vel_now = step.features[3:6]

        horizons = rng.uniform(horizon_min, horizon_max, size=n_horizons_per_step)
        for h in horizons:
            query_time = step.sim_time + h
            target = interpolate_true_position(traj_samples, query_time)
            if target is None:
                continue
            cv_pred = est_pos_now + est_vel_now * h
            samples.append({
                "history": window,
                "history_mask": mask,
                "horizon": float(h),
                "target_delta": target - cv_pred,
            })
    return samples


def build_stratified_horizon_samples(
    episode: EpisodeRecord,
    rng: np.random.Generator,
    n_horizons_per_band: int = 1,
    bands: tuple[tuple[float, float], ...] = HORIZON_BANDS,
    history_length: int = 8,
) -> list[dict]:
    """Phase 70 multiscale sampling: draw `n_horizons_per_band` CONTINUOUS
    horizons from EACH band per detected step (equal count per band,
    regardless of band width), instead of one uniform draw over the full
    range. Preserves arbitrary-horizon prediction (horizons within a band
    are still continuous) while giving short horizons proportionally far
    more training signal."""
    samples: list[dict] = []
    traj_samples = [
        TrajectorySample(s.step_index, s.sim_time, s.true_position) for s in episode.steps
    ]

    for i, step in enumerate(episode.steps):
        if not step.detected:
            continue

        window, mask = _build_history_window(episode, i, history_length)
        est_pos_now = step.features[0:3]
        est_vel_now = step.features[3:6]

        for lo, hi in bands:
            horizons = rng.uniform(lo, hi, size=n_horizons_per_band)
            for h in horizons:
                query_time = step.sim_time + h
                target = interpolate_true_position(traj_samples, query_time)
                if target is None:
                    continue
                cv_pred = est_pos_now + est_vel_now * h
                samples.append({
                    "history": window,
                    "history_mask": mask,
                    "horizon": float(h),
                    "target_delta": target - cv_pred,
                })
    return samples


class HorizonPredictorDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "history": torch.as_tensor(s["history"], dtype=torch.float32),
            "history_mask": torch.as_tensor(s["history_mask"], dtype=torch.float32),
            "horizon": torch.tensor(s["horizon"], dtype=torch.float32),
            "target_delta": torch.as_tensor(s["target_delta"], dtype=torch.float32),
        }


@dataclass
class HorizonTrainConfig:
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    model_seed: int = 0


def train_horizon_predictor(model: nn.Module, samples: list[dict], config: HorizonTrainConfig = HorizonTrainConfig()) -> list[float]:
    torch.manual_seed(config.model_seed)
    dataset = HorizonPredictorDataset(samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for _epoch in range(config.epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch["history"], batch["history_mask"], batch["horizon"])
            loss = loss_fn(pred, batch["target_delta"])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses
