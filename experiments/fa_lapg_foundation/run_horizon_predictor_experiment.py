"""
Phase 50-51 runner: train 5 independent model-initialization seeds of the
horizon-conditioned predictor on the SAME (already-used, Phase 27-28)
training episodes, then evaluate CV / CA / horizon-GRU on a FRESH,
previously-unused test bank (`fresh_predictor_test_bank`,
`experiments.fa_lapg_foundation.seed_registry`), reporting mean +/- std
across the 5 initializations.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.fa_lapg_foundation import seed_config
from experiments.fa_lapg_foundation.baseline_predictors import cv_predict
from experiments.fa_lapg_foundation.horizon_conditioned_dataset import (
    HorizonTrainConfig,
    build_horizon_samples,
    train_horizon_predictor,
)
from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.seed_registry import RESERVED_RANGES, assert_reserved_ranges_disjoint_from_historical
from experiments.fa_lapg_foundation.trajectory_dataset import build_samples, rollout_episode
from experiments.learning_benefit.scenarios import build_scenario_battery
from experiments.learning_benefit.prediction_error import TrajectorySample, interpolate_true_position
from uav_defend.envs.soldier_env import SoldierEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "horizon_predictor"

EVAL_HORIZONS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
N_MODEL_SEEDS = 5
FRESH_TEST_N_PER_SCENARIO = 10  # engineering-scale subset of the reserved 200-slot block


def _fresh_test_seeds():
    start, end = RESERVED_RANGES["fresh_predictor_test_bank"]
    return range(start, start + FRESH_TEST_N_PER_SCENARIO)


def build_train_episodes():
    scenarios = build_scenario_battery()
    episodes = []
    for scenario in scenarios:
        for seed in seed_config.train_seeds():
            env = SoldierEnv(config=scenario.config)
            episodes.append(rollout_episode(env, seed, scenario.name))
    return episodes


def build_fresh_test_episodes():
    scenarios = build_scenario_battery()
    episodes = []
    for scenario in scenarios:
        for seed in _fresh_test_seeds():
            env = SoldierEnv(config=scenario.config)
            episodes.append(rollout_episode(env, seed, scenario.name))
    return episodes


def evaluate_model_on_episodes(model, episodes, horizons=EVAL_HORIZONS) -> pd.DataFrame:
    rows = []
    for episode in episodes:
        traj_samples = [TrajectorySample(s.step_index, s.sim_time, s.true_position) for s in episode.steps]
        fixed_samples = build_samples(episode, horizons=(0.5,))  # reuse for history windows
        for sample in fixed_samples:
            for h in horizons:
                query_time = sample["sim_time"] + h
                target = interpolate_true_position(traj_samples, query_time)
                if target is None:
                    continue
                cv_p = cv_predict({"cv_pred": {h: sample["est_pos_now"] + sample["est_vel_now"] * h}})[h]
                with torch.no_grad():
                    history = torch.as_tensor(sample["history"], dtype=torch.float32).unsqueeze(0)
                    mask = torch.as_tensor(sample["history_mask"], dtype=torch.float32).unsqueeze(0)
                    horizon_t = torch.tensor([h], dtype=torch.float32)
                    delta = model(history, mask, horizon_t)[0].numpy()
                gru_p = sample["est_pos_now"] + sample["est_vel_now"] * h + delta
                row = {
                    "scenario": episode.scenario, "seed": episode.seed,
                    "horizon": h,
                    "cv_error": float(np.linalg.norm(target - cv_p)),
                    "gru_error": float(np.linalg.norm(target - gru_p)),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert_reserved_ranges_disjoint_from_historical()

    t0 = time.time()
    train_episodes = build_train_episodes()
    print(f"train episodes: {len(train_episodes)}, {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    train_samples = []
    for episode in train_episodes:
        train_samples.extend(build_horizon_samples(episode, rng, n_horizons_per_step=4))
    print(f"train horizon-samples: {len(train_samples)}, {time.time() - t0:.1f}s")

    test_episodes = build_fresh_test_episodes()
    print(f"fresh test episodes: {len(test_episodes)}, {time.time() - t0:.1f}s")

    all_eval_rows = []
    for model_seed in range(N_MODEL_SEEDS):
        torch.manual_seed(model_seed)
        model = HorizonConditionedPredictor()
        losses = train_horizon_predictor(
            model, train_samples, HorizonTrainConfig(epochs=10, batch_size=128, learning_rate=1e-3, model_seed=model_seed),
        )
        print(f"model_seed={model_seed} final_loss={losses[-1]:.3f} {time.time() - t0:.1f}s")
        eval_df = evaluate_model_on_episodes(model, test_episodes)
        eval_df["model_seed"] = model_seed
        all_eval_rows.append(eval_df)

    full_eval = pd.concat(all_eval_rows, ignore_index=True)
    full_eval.to_csv(OUTPUT_ROOT / "fresh_test_errors_by_model_seed.csv", index=False)

    per_seed_summary = full_eval.groupby(["horizon", "model_seed"]).agg(
        cv_error_mean=("cv_error", "mean"), gru_error_mean=("gru_error", "mean"),
    ).reset_index()
    across_seeds = per_seed_summary.groupby("horizon").agg(
        cv_error_mean=("cv_error_mean", "mean"), cv_error_std=("cv_error_mean", "std"),
        gru_error_mean=("gru_error_mean", "mean"), gru_error_std=("gru_error_mean", "std"),
    ).reset_index()
    print(across_seeds.round(3).to_string())
    across_seeds.to_csv(OUTPUT_ROOT / "fresh_test_summary_mean_std_across_seeds.csv", index=False)


if __name__ == "__main__":
    main()
