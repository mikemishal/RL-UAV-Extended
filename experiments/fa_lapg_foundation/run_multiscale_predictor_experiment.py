"""
Phase 71-72 runner: train 5 model-initialization seeds of the SAME small
horizon-conditioned architecture (GRU hidden=32) using MULTISCALE
(stratified) horizon sampling instead of uniform sampling, on the
IDENTICAL train/test episode splits used for the original (uniformly
trained) predictor. Compares CV / old uniformly-trained GRU / new
multiscale GRU at fixed evaluation horizons (Phase 72 acceptance test).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.fa_lapg_foundation import seed_config
from experiments.fa_lapg_foundation.baseline_predictors import cv_predict
from experiments.fa_lapg_foundation.build_shared_predictor_checkpoint import load_predictor
from experiments.fa_lapg_foundation.horizon_conditioned_dataset import (
    HorizonTrainConfig,
    build_stratified_horizon_samples,
    train_horizon_predictor,
)
from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.seed_registry import RESERVED_RANGES
from experiments.fa_lapg_foundation.trajectory_dataset import build_samples, rollout_episode
from experiments.learning_benefit.prediction_error import TrajectorySample, interpolate_true_position
from experiments.learning_benefit.scenarios import build_scenario_battery
from uav_defend.envs.soldier_env import SoldierEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "multiscale_predictor"

EVAL_HORIZONS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
N_MODEL_SEEDS = 5
FRESH_TEST_N_PER_SCENARIO = 10  # SAME fresh_predictor_test_bank subset used for the original acceptance test


def _fresh_test_seeds():
    start, _end = RESERVED_RANGES["fresh_predictor_test_bank"]
    return range(start, start + FRESH_TEST_N_PER_SCENARIO)


def build_train_episodes():
    episodes = []
    for scenario in build_scenario_battery():
        for seed in seed_config.train_seeds():
            env = SoldierEnv(config=scenario.config)
            episodes.append(rollout_episode(env, seed, scenario.name))
    return episodes


def build_fresh_test_episodes():
    episodes = []
    for scenario in build_scenario_battery():
        for seed in _fresh_test_seeds():
            env = SoldierEnv(config=scenario.config)
            episodes.append(rollout_episode(env, seed, scenario.name))
    return episodes


def evaluate_models_on_episodes(models: dict[str, HorizonConditionedPredictor], episodes, horizons=EVAL_HORIZONS) -> pd.DataFrame:
    rows = []
    for episode in episodes:
        traj_samples = [TrajectorySample(s.step_index, s.sim_time, s.true_position) for s in episode.steps]
        fixed_samples = build_samples(episode, horizons=(0.5,))
        for sample in fixed_samples:
            for h in horizons:
                query_time = sample["sim_time"] + h
                target = interpolate_true_position(traj_samples, query_time)
                if target is None:
                    continue
                cv_p = cv_predict({"cv_pred": {h: sample["est_pos_now"] + sample["est_vel_now"] * h}})[h]
                row = {
                    "scenario": episode.scenario, "seed": episode.seed, "horizon": h,
                    "cv_error": float(np.linalg.norm(target - cv_p)),
                }
                history = torch.as_tensor(sample["history"], dtype=torch.float32).unsqueeze(0)
                mask = torch.as_tensor(sample["history_mask"], dtype=torch.float32).unsqueeze(0)
                horizon_t = torch.tensor([h], dtype=torch.float32)
                for name, model in models.items():
                    with torch.no_grad():
                        delta = model(history, mask, horizon_t)[0].numpy()
                    pred = sample["est_pos_now"] + sample["est_vel_now"] * h + delta
                    row[f"{name}_error"] = float(np.linalg.norm(target - pred))
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_episodes = build_train_episodes()
    print(f"train episodes: {len(train_episodes)}, {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    stratified_samples = []
    for episode in train_episodes:
        stratified_samples.extend(build_stratified_horizon_samples(episode, rng, n_horizons_per_band=1))
    print(f"stratified train samples: {len(stratified_samples)}, {time.time() - t0:.1f}s")

    test_episodes = build_fresh_test_episodes()
    print(f"fresh test episodes: {len(test_episodes)}, {time.time() - t0:.1f}s")

    old_model = load_predictor()  # existing uniformly-trained checkpoint (Phase 50/51)

    all_eval_rows = []
    for model_seed in range(N_MODEL_SEEDS):
        torch.manual_seed(model_seed)
        new_model = HorizonConditionedPredictor()
        losses = train_horizon_predictor(
            new_model, stratified_samples,
            HorizonTrainConfig(epochs=10, batch_size=128, learning_rate=1e-3, model_seed=model_seed),
        )
        print(f"model_seed={model_seed} final_loss={losses[-1]:.3f} {time.time() - t0:.1f}s")
        eval_df = evaluate_models_on_episodes({"old_uniform_gru": old_model, "new_multiscale_gru": new_model}, test_episodes)
        eval_df["model_seed"] = model_seed
        all_eval_rows.append(eval_df)

    full_eval = pd.concat(all_eval_rows, ignore_index=True)
    full_eval.to_csv(OUTPUT_ROOT / "fresh_test_errors_by_model_seed.csv", index=False)

    per_seed = full_eval.groupby(["horizon", "model_seed"]).agg(
        cv_error_mean=("cv_error", "mean"),
        old_uniform_gru_error_mean=("old_uniform_gru_error", "mean"),
        new_multiscale_gru_error_mean=("new_multiscale_gru_error", "mean"),
    ).reset_index()
    across_seeds = per_seed.groupby("horizon").agg(
        cv_error_mean=("cv_error_mean", "mean"), cv_error_std=("cv_error_mean", "std"),
        old_uniform_gru_mean=("old_uniform_gru_error_mean", "mean"), old_uniform_gru_std=("old_uniform_gru_error_mean", "std"),
        new_multiscale_gru_mean=("new_multiscale_gru_error_mean", "mean"), new_multiscale_gru_std=("new_multiscale_gru_error_mean", "std"),
    ).reset_index()
    print(across_seeds.round(3).to_string())
    across_seeds.to_csv(OUTPUT_ROOT / "acceptance_test_summary.csv", index=False)

    # Persist the best (lowest final training loss is not tracked per-seed
    # here for simplicity; use model_seed=0 as the fixed choice, consistent
    # with the original checkpoint convention) multiscale model for reuse
    # by RA-Lead/RA-LAPG/LearnedPredictionLead going forward.
    torch.manual_seed(0)
    final_model = HorizonConditionedPredictor()
    train_horizon_predictor(final_model, stratified_samples, HorizonTrainConfig(epochs=10, batch_size=128, learning_rate=1e-3, model_seed=0))
    torch.save(final_model.state_dict(), OUTPUT_ROOT / "multiscale_predictor_checkpoint.pt")


if __name__ == "__main__":
    main()
