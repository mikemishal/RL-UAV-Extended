"""
Phase 81: large-N (N=200) development confirmation run across the six
core controllers, on the 4 open scenarios, using ONLY
`ra_lapg_development` seeds (never touching validation/final-journal
ranges). Uses the Phase 73 hybrid predictor (multiscale GRU above
h_switch=2.0s, pure CV at/below it) for LearnedPredictionLead and
RA-LAPG, and FIXED_DIRECTION reachability mode for RA-Lead/RA-LAPG (Phase
80 default -- a dedicated fixed-vs-receding comparison study was not run
separately due to time constraints; see the final report).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from experiments.fa_lapg_foundation.build_shared_predictor_checkpoint import CHECKPOINT_PATH as MULTISCALE_UNUSED  # noqa: F401
from experiments.fa_lapg_foundation.horizon_conditioned_predictor import HorizonConditionedPredictor
from experiments.fa_lapg_foundation.hybrid_predictor import HybridPredictor
from experiments.fa_lapg_foundation.seed_registry import RESERVED_RANGES, assert_no_new_range_overlaps_reserved_final
from experiments.eval_utils import run_episode
from experiments.learning_benefit.run_diagnostic_battery import _make_lr_ppo_policy, run_true_state_episode
from experiments.learning_benefit.scenarios import build_scenario_battery
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.learned_prediction_lead_policy import LearnedPredictionLeadPolicy
from uav_defend.policies.analysis.ra_lapg_policy import RALAPGPolicy
from uav_defend.policies.analysis.reachability_aware_lead_policy import ReachabilityAwareLeadPolicy
from uav_defend.policies.analysis.true_state_lead_policy import TrueStateLeadPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "controller_study_v2"
MULTISCALE_CHECKPOINT = PROJECT_ROOT / "results" / "fa_lapg_foundation" / "multiscale_predictor" / "multiscale_predictor_checkpoint.pt"

OPEN_SCENARIOS = ("nominal_open", "strong_maneuver", "strong_evasion", "mobility_mismatch")
N_PER_SCENARIO = 200


def _dev_seeds():
    start, _end = RESERVED_RANGES["ra_lapg_development"]
    return range(start, start + N_PER_SCENARIO)


def load_hybrid_predictor() -> HybridPredictor:
    model = HorizonConditionedPredictor()
    model.load_state_dict(torch.load(MULTISCALE_CHECKPOINT, weights_only=True))
    model.eval()
    return HybridPredictor(model)


def run_scenario(scenario, predictor, lr_ppo_policy) -> pd.DataFrame:
    seeds = list(_dev_seeds())
    rows = []

    controllers = [
        ("lead", lambda: LeadInterceptPolicy(state_source="measurement", config=scenario.config)),
        ("ra_lead", lambda: ReachabilityAwareLeadPolicy(config=scenario.config)),
        ("learned_prediction_lead", lambda: LearnedPredictionLeadPolicy(predictor, config=scenario.config)),
        ("ra_lapg", lambda: RALAPGPolicy(predictor, config=scenario.config)),
    ]

    for name, make_policy in controllers:
        for seed in seeds:
            env = SoldierEnv(config=scenario.config)
            metrics = run_episode(env, make_policy(), seed)
            metrics.update(controller=name, scenario=scenario.name)
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

    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert_no_new_range_overlaps_reserved_final()

    predictor = load_hybrid_predictor()
    lr_ppo_policy = _make_lr_ppo_policy()
    scenarios = [s for s in build_scenario_battery() if s.name in OPEN_SCENARIOS]

    t0 = time.time()
    all_rows = []
    for scenario in scenarios:
        df = run_scenario(scenario, predictor, lr_ppo_policy)
        all_rows.append(df)
        print(f"{scenario.name} done, cumulative {time.time() - t0:.1f}s")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUTPUT_ROOT / "open_scenario_controller_study_v2.csv", index=False)
    print(f"total rows {len(full)}, wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
