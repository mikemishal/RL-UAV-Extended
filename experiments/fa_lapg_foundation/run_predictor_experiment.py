"""
Phase 27-33 experiment runner: builds the supervised trajectory-prediction
dataset across the 6-scenario battery, trains the GRU-residual predictor,
and reports CV/CA/GRU metrics by horizon and scenario, plus a
scenario-generalization comparison (Phase 33).

NOT reinforcement learning. NOT integrated into Lead or any deployable
policy (Phase 34).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from experiments.fa_lapg_foundation import seed_config
from experiments.fa_lapg_foundation.evaluate_predictor import (
    compute_errors,
    summarize_by_horizon,
    summarize_by_scenario,
)
from experiments.fa_lapg_foundation.gru_predictor import GRUResidualPredictor
from experiments.fa_lapg_foundation.train_predictor import TrainConfig, train_predictor
from experiments.fa_lapg_foundation.trajectory_dataset import build_dataset
from experiments.learning_benefit.scenarios import build_scenario_battery

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "predictor"

GENERALIZATION_TRAIN_SCENARIOS = {"nominal_open", "strong_evasion"}
GENERALIZATION_HELD_OUT_SCENARIOS = {"strong_maneuver", "mobility_mismatch", "clutter", "clutter_strong_maneuver"}


def build_all_datasets():
    scenarios = build_scenario_battery()
    train, val, test = [], [], []
    t0 = time.time()
    for scenario in scenarios:
        train += build_dataset(seed_config.train_seeds(), scenario.name, scenario.config)
        val += build_dataset(seed_config.val_seeds(), scenario.name, scenario.config)
        test += build_dataset(seed_config.test_seeds(), scenario.name, scenario.config)
        print(f"{scenario.name}: train/val/test built, cumulative {time.time() - t0:.1f}s")
    return train, val, test


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_config.assert_disjoint_from_locked_ranges()

    train, val, test = build_all_datasets()
    print(f"train={len(train)} val={len(val)} test={len(test)} samples")

    # --- Main pooled model (Phase 29-32) ---
    torch.manual_seed(0)
    pooled_model = GRUResidualPredictor()
    losses = train_predictor(pooled_model, train, TrainConfig(epochs=15, batch_size=64, learning_rate=1e-3, seed=0))
    print("pooled model training losses:", [round(loss, 4) for loss in losses])

    dt = 0.5
    test_errors = compute_errors(test, pooled_model, dt)
    test_errors.to_csv(OUTPUT_ROOT / "pooled_test_errors.csv", index=False)

    by_horizon = summarize_by_horizon(test_errors, ["cv_error", "ca_error", "gru_error"])
    by_scenario = summarize_by_scenario(test_errors, ["cv_error", "ca_error", "gru_error"])
    by_horizon.to_csv(OUTPUT_ROOT / "pooled_by_horizon.csv", index=False)
    by_scenario.to_csv(OUTPUT_ROOT / "pooled_by_scenario.csv", index=False)
    print("\n=== Pooled model: by horizon ===")
    print(by_horizon.round(3).to_string())
    print("\n=== Pooled model: by scenario ===")
    print(by_scenario.round(3).to_string())

    # --- Generalization model (Phase 33): train ONLY on easy scenarios ---
    gen_train = [s for s in train if s["scenario"] in GENERALIZATION_TRAIN_SCENARIOS]
    torch.manual_seed(0)
    gen_model = GRUResidualPredictor()
    gen_losses = train_predictor(gen_model, gen_train, TrainConfig(epochs=15, batch_size=64, learning_rate=1e-3, seed=0))
    print("\ngeneralization model training losses:", [round(loss, 4) for loss in gen_losses])

    held_out_test = [s for s in test if s["scenario"] in GENERALIZATION_HELD_OUT_SCENARIOS]
    gen_errors = compute_errors(held_out_test, gen_model, dt)
    gen_errors.to_csv(OUTPUT_ROOT / "generalization_held_out_errors.csv", index=False)
    gen_by_scenario = summarize_by_scenario(gen_errors, ["cv_error", "ca_error", "gru_error"])
    gen_by_scenario.to_csv(OUTPUT_ROOT / "generalization_by_scenario.csv", index=False)
    print("\n=== Generalization model (trained on nominal_open+strong_evasion only), held-out scenarios ===")
    print(gen_by_scenario.round(3).to_string())

    # For comparison: how does the POOLED model (trained on all 6) do on
    # the SAME held-out scenarios?
    pooled_on_heldout = compute_errors(held_out_test, pooled_model, dt)
    pooled_on_heldout_by_scenario = summarize_by_scenario(pooled_on_heldout, ["cv_error", "ca_error", "gru_error"])
    pooled_on_heldout_by_scenario.to_csv(OUTPUT_ROOT / "pooled_on_heldout_by_scenario.csv", index=False)
    print("\n=== Pooled model (trained on all 6) on the SAME held-out scenarios, for comparison ===")
    print(pooled_on_heldout_by_scenario.round(3).to_string())

    torch.save(pooled_model.state_dict(), OUTPUT_ROOT / "pooled_model.pt")
    torch.save(gen_model.state_dict(), OUTPUT_ROOT / "generalization_model.pt")


if __name__ == "__main__":
    main()
