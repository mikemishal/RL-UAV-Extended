"""
Phase 53 + 60 combined runner: evaluate Standard Lead, True-State Lead
(analysis oracle), LR-PPO (OOD reference baseline), Learned-Prediction
Lead (Phase 52 ablation), and RA-LAPG (Phase 54-58) on the 4 OPEN
scenarios (clutter is explicitly deferred per Phase 63), using freshly
reserved development seeds (`ra_lapg_development`, NOT the RESERVED final
journal evaluation range).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fa_lapg_foundation.build_shared_predictor_checkpoint import load_predictor
from experiments.fa_lapg_foundation.seed_registry import RESERVED_RANGES, assert_no_new_range_overlaps_reserved_final
from experiments.learning_benefit.scenarios import build_scenario_battery
from experiments.eval_utils import run_episode
from experiments.learning_benefit.run_diagnostic_battery import _make_lr_ppo_policy, run_true_state_episode
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.learned_prediction_lead_policy import LearnedPredictionLeadPolicy
from uav_defend.policies.analysis.ra_lapg_policy import RALAPGPolicy
from uav_defend.policies.analysis.true_state_lead_policy import TrueStateLeadPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "controller_study"

OPEN_SCENARIOS = ("nominal_open", "strong_maneuver", "strong_evasion", "mobility_mismatch")
N_PER_SCENARIO = 25


def _dev_seeds():
    start, _end = RESERVED_RANGES["ra_lapg_development"]
    return range(start, start + N_PER_SCENARIO)


def run_scenario(scenario, predictor, lr_ppo_policy) -> pd.DataFrame:
    seeds = list(_dev_seeds())
    rows = []

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        lead_policy = LeadInterceptPolicy(state_source="measurement", config=scenario.config)
        metrics = run_episode(env, lead_policy, seed)
        metrics.update(controller="lead", scenario=scenario.name)
        rows.append(metrics)

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        ts_policy = TrueStateLeadPolicy(config=scenario.config)
        metrics = run_true_state_episode(env, ts_policy, seed)
        metrics.update(controller="true_state_lead", scenario=scenario.name)
        rows.append(metrics)

    if lr_ppo_policy is not None:
        for seed in seeds:
            env = SoldierEnv(config=scenario.config)
            metrics = run_episode(env, lr_ppo_policy, seed)
            metrics.update(controller="lr_ppo", scenario=scenario.name)
            rows.append(metrics)

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        lp_policy = LearnedPredictionLeadPolicy(predictor, config=scenario.config)
        metrics = run_episode(env, lp_policy, seed)
        metrics.update(controller="learned_prediction_lead", scenario=scenario.name)
        rows.append(metrics)

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        ra_policy = RALAPGPolicy(predictor, config=scenario.config)
        metrics = run_episode(env, ra_policy, seed)
        metrics.update(controller="ra_lapg", scenario=scenario.name)
        rows.append(metrics)

    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert_no_new_range_overlaps_reserved_final()

    predictor = load_predictor()
    lr_ppo_policy = _make_lr_ppo_policy()
    scenarios = [s for s in build_scenario_battery() if s.name in OPEN_SCENARIOS]

    t0 = time.time()
    all_rows = []
    for scenario in scenarios:
        df = run_scenario(scenario, predictor, lr_ppo_policy)
        all_rows.append(df)
        print(f"{scenario.name} done, cumulative {time.time() - t0:.1f}s")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUTPUT_ROOT / "open_scenario_controller_study.csv", index=False)
    print(f"total rows {len(full)}, wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
