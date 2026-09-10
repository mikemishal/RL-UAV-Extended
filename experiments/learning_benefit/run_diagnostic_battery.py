"""Phase 10-15: initial diagnostic scenario battery pilot runner.

Evaluates, on MATCHED seeds within each scenario:
  - Standard Lead (existing, unmodified deployable policy)
  - True-State constant-velocity Lead (ANALYSIS-ONLY oracle)
  - Constant-Acceleration Lead (ANALYSIS-ONLY, estimated history)
  - LR-PPO (existing residual wrapper), IF a conference-trained checkpoint
    is available -- see CHECKPOINT_PATH below. This script never trains
    or substitutes a new model; it only loads an already-trained .zip by
    absolute path (never copied into this repository).

Pilot seed range: 60000..60000+N-1 -- a NEW range, disjoint from every
locked robustness-protocol range in this repository (which top out at
35999) and from the final journal evaluation seed banks. NOT the final
evaluation seed bank.

Run directly: python experiments/learning_benefit/run_diagnostic_battery.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.eval_utils import run_episode
from experiments.learning_benefit.diagnostic_logging import run_episode_with_diagnostics, write_step_logs_csv
from experiments.learning_benefit.prediction_error import TrajectorySample, compute_ca_prediction_error, compute_cv_prediction_error
from experiments.learning_benefit.residual_analysis import correlate, residual_angle_deg, summarize_by_group
from experiments.learning_benefit.scenarios import build_scenario_battery
from experiments.learning_benefit.statistics_utils import episode_level_aggregate, paired_binary_contingency, paired_bootstrap_difference
from experiments.rl.evaluate_rl import compute_confidence_interval, compute_proportion_ci
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.analysis.constant_acceleration_lead_policy import ConstantAccelerationLeadPolicy
from uav_defend.policies.analysis.true_state_lead_policy import TrueStateLeadPolicy
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info

PILOT_SEED_START = 60000
PILOT_N_EPISODES = 200  # first engineering pilot (Phase-12 sanctioned value)
DIAGNOSTIC_LOG_N_EPISODES = 40  # smaller sub-sample for per-step logging (Phase 7-9)

# Conference-trained LR-PPO checkpoint: found on the local machine in the
# SIBLING frozen conference repository (never committed to, or copied
# into, RL-UAV-Extended) -- see the Phase-4 checkpoint-availability search
# in the task report. Loaded by absolute path only.
LR_PPO_CHECKPOINT = Path(
    r"C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\uav-defense-3d-resubmission"
    r"\models\lead_residual_post_detection\seed_55\best_model.zip"
)

OUTPUT_ROOT = PROJECT_ROOT / "results" / "learning_benefit_analysis" / "pilot"


def run_true_state_episode(env: SoldierEnv, policy: TrueStateLeadPolicy, seed: int) -> dict:
    """Mirrors `experiments.eval_utils.run_episode`'s metric computation,
    but for the ANALYSIS-ONLY True-State Lead controller, which requires
    the FULL (unsanitized) info dict -- the ONE authorized exception to
    the sanitize-before-act() rule (see TrueStateLeadPolicy's docstring)."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    min_defender_enemy_dist = info["defender_enemy_dist"]
    detection_time = None
    step = 0
    done = False
    while not done:
        action = policy.act(
            defender_position=info["defender_pos"],
            true_hostile_position=info["enemy_pos"],
            true_hostile_velocity=info["enemy_vel"],
            soldier_position=info["soldier_pos"],
            enemy_detected=bool(info["enemy_detected"]),
        )
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])
        min_defender_enemy_dist = min(min_defender_enemy_dist, info["defender_enemy_dist"])
        if info["enemy_detected"] and detection_time is None:
            detection_time = step
        if done or truncated:
            break
    intercept_time = step if info["outcome"] == "intercepted" else -1
    return {
        "seed": seed, "outcome": info["outcome"], "success": 1 if info["outcome"] == "intercepted" else 0,
        "episode_length": step, "detected": 1 if detection_time is not None else 0,
        "detection_time": detection_time if detection_time is not None else -1,
        "intercept_time": intercept_time,
        "min_enemy_soldier_dist": min_enemy_soldier_dist, "min_defender_enemy_dist": min_defender_enemy_dist,
    }


def _make_lr_ppo_policy():
    if not LR_PPO_CHECKPOINT.exists():
        return None
    from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper
    return LeadResidualPPOPolicyWrapper.load(LR_PPO_CHECKPOINT, config=EnvConfig())


def run_scenario_summary(scenario, seeds) -> pd.DataFrame:
    """Fast per-episode-summary pass (Phase 12/13): success/timeout/length/
    detection/intercept-time/min-distance metrics, matched seeds, for
    every available controller."""
    rows = []

    lead_policy = LeadInterceptPolicy(state_source="measurement", config=scenario.config)
    ca_policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=scenario.config)
    ts_policy = TrueStateLeadPolicy(config=scenario.config)
    lr_ppo_policy = _make_lr_ppo_policy()

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        metrics = run_episode(env, lead_policy, seed)
        metrics.update(controller="lead", scenario=scenario.name)
        rows.append(metrics)

        env = SoldierEnv(config=scenario.config)
        metrics = run_episode(env, ca_policy, seed)
        metrics.update(controller="ca_lead", scenario=scenario.name)
        rows.append(metrics)

        env = SoldierEnv(config=scenario.config)
        metrics = run_true_state_episode(env, ts_policy, seed)
        metrics.update(controller="true_state_lead", scenario=scenario.name)
        rows.append(metrics)

        if lr_ppo_policy is not None:
            env = SoldierEnv(config=scenario.config)
            metrics = run_episode(env, lr_ppo_policy, seed)
            metrics.update(controller="lr_ppo", scenario=scenario.name)
            rows.append(metrics)

    return pd.DataFrame(rows)


def run_scenario_diagnostic_logs(scenario, seeds) -> list[dict]:
    """Per-step diagnostic-logging pass (Phase 7-9) on a smaller seed
    sub-sample, for Lead/CA-Lead prediction-error and LR-PPO residual
    analysis."""
    all_rows = []
    lr_ppo_policy = _make_lr_ppo_policy()

    for seed in seeds:
        env = SoldierEnv(config=scenario.config)
        lead_policy = LeadInterceptPolicy(state_source="measurement", config=scenario.config)
        all_rows += run_episode_with_diagnostics(env, lead_policy, seed, "lead", scenario.name)

        env = SoldierEnv(config=scenario.config)
        ca_policy = ConstantAccelerationLeadPolicy(state_source="measurement", config=scenario.config)
        all_rows += run_episode_with_diagnostics(env, ca_policy, seed, "ca_lead", scenario.name)

        if lr_ppo_policy is not None:
            env = SoldierEnv(config=scenario.config)
            all_rows += run_episode_with_diagnostics(env, lr_ppo_policy, seed, "lr_ppo", scenario.name)

    return all_rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "summaries").mkdir(exist_ok=True)

    lr_ppo_available = LR_PPO_CHECKPOINT.exists()
    print(f"LR-PPO checkpoint available: {lr_ppo_available} ({LR_PPO_CHECKPOINT})")

    seeds = list(range(PILOT_SEED_START, PILOT_SEED_START + PILOT_N_EPISODES))
    diagnostic_seeds = seeds[:DIAGNOSTIC_LOG_N_EPISODES]

    scenarios = build_scenario_battery()
    all_summaries = []
    all_step_rows = []
    t0 = time.time()
    for scenario in scenarios:
        print(f"--- scenario: {scenario.name} ---")
        summary_df = run_scenario_summary(scenario, seeds)
        all_summaries.append(summary_df)

        step_rows = run_scenario_diagnostic_logs(scenario, diagnostic_seeds)
        all_step_rows += step_rows
        print(f"  done in {time.time() - t0:.1f}s cumulative")

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(OUTPUT_ROOT / "summaries" / "scenario_summary.csv", index=False)
    write_step_logs_csv(all_step_rows, OUTPUT_ROOT / "logs" / "step_diagnostics.csv")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    print(f"Scenario summary rows: {len(summary_df)}, step-diagnostic rows: {len(all_step_rows)}")
    return summary_df, all_step_rows


if __name__ == "__main__":
    main()
