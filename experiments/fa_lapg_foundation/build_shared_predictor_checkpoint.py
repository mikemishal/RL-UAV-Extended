"""Trains ONE horizon-conditioned predictor on the Phase 27-28 training
episodes and saves it to disk, for reuse by LearnedPredictionLeadPolicy
and RA-LAPG (Phases 52-58). Reruns are avoided by persisting the
checkpoint rather than retraining inside every controller experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.fa_lapg_foundation import seed_config
from experiments.fa_lapg_foundation.horizon_conditioned_dataset import (
    HorizonTrainConfig,
    build_horizon_samples,
    train_horizon_predictor,
)
from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import rollout_episode
from experiments.learning_benefit.scenarios import build_scenario_battery
from uav_defend.envs.soldier_env import SoldierEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "horizon_predictor" / "predictor_checkpoint.pt"
MODEL_SEED = 1  # one of the 5 Phase 50 initializations, arbitrary fixed choice


def build_and_save(model_seed: int = MODEL_SEED, epochs: int = 10) -> HorizonConditionedPredictor:
    scenarios = build_scenario_battery()
    rng = np.random.default_rng(0)
    samples = []
    for scenario in scenarios:
        for seed in seed_config.train_seeds():
            env = SoldierEnv(config=scenario.config)
            episode = rollout_episode(env, seed, scenario.name)
            samples.extend(build_horizon_samples(episode, rng, n_horizons_per_step=4))

    torch.manual_seed(model_seed)
    model = HorizonConditionedPredictor()
    train_horizon_predictor(model, samples, HorizonTrainConfig(epochs=epochs, batch_size=128, learning_rate=1e-3, model_seed=model_seed))

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    return model


def load_predictor() -> HorizonConditionedPredictor:
    model = HorizonConditionedPredictor()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    build_and_save()
    print(f"saved checkpoint to {CHECKPOINT_PATH}")
