"""
Locked final nominal-evaluation protocol -- SINGLE SOURCE OF TRUTH.

This module defines the FIRST use of the reserved final-test seed range
(>= 20000). It must exist in committed source BEFORE any seed in
FINAL_NOMINAL_SEEDS is evaluated (see experiments/run_final_nominal_evaluation.py).

Once these seeds are evaluated, they are OPENED test data: no controller,
PPO, Kalman, reward, or environment tuning may be based on them afterward.

Seed ranges (disjoint from all prior protocol ranges -- see
test_final_experiment_protocol.py):

    Training seeds (experiments/training_protocol.py) : 42, 43, 44
    Validation seeds                                   : 10000..10099
    Diagnostic seeds                                   : 12000..12199
    FINAL_NOMINAL_SEEDS  (this module, OPENED here)     : 20000..24999
    Reserved gap (untouched in this task)               : 25000..29999
    ROBUSTNESS_SEED_OFFSET (reserved for later tasks)   : 30000+
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.artifact_paths import resolve_artifact_or_local
from experiments.method_specs import get_method_spec
from experiments.training_protocol import (
    TRAINING_SEEDS as PPO_TRAINING_SEEDS,
    VALIDATION_SEEDS,
    DIAGNOSTIC_SEEDS,
    CAMPAIGN_400K_DIRNAME,
    run_output_dir,
)

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# FINAL NOMINAL TEST SEEDS -- first use of the reserved final-test range
# =============================================================================
FINAL_NOMINAL_SEED_OFFSET = 20_000
FINAL_NOMINAL_EPISODES = 5_000
FINAL_NOMINAL_SEEDS: range = range(
    FINAL_NOMINAL_SEED_OFFSET, FINAL_NOMINAL_SEED_OFFSET + FINAL_NOMINAL_EPISODES
)

# Reserved for LATER robustness sweeps -- do NOT use in this task.
ROBUSTNESS_SEED_OFFSET = 30_000
# 25000..29999 is a reserved gap, untouched in this task (and not part of
# FINAL_NOMINAL_SEEDS or ROBUSTNESS_SEED_OFFSET+).
RESERVED_GAP_SEEDS: range = range(25_000, ROBUSTNESS_SEED_OFFSET)

# Must match experiments/training_protocol.TRAINING_SEEDS exactly.
PPO_TRAINING_SEEDS = PPO_TRAINING_SEEDS

# =============================================================================
# STATISTICS PROTOCOL
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816
CONFIDENCE_LEVEL = 0.95


def assert_final_nominal_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked final nominal test range."""
    if seed not in FINAL_NOMINAL_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in FINAL_NOMINAL_SEEDS "
            f"({FINAL_NOMINAL_SEED_OFFSET}..{FINAL_NOMINAL_SEED_OFFSET + FINAL_NOMINAL_EPISODES - 1})."
        )


# =============================================================================
# FROZEN 400k MODEL LOCATIONS (results/training_400k/<track>/seed_<N>/best_model.zip)
# =============================================================================
def frozen_model_path(track: str, seed: int) -> Path:
    """Path to the frozen 400k best_model.zip for a given track/seed.

    NEVER final_model.zip and NEVER a 200k model -- this is the SOLE source
    of PPO/PPO-Kalman weights for the final nominal evaluation.
    """
    local_path = run_output_dir(PROJECT_ROOT / "results" / CAMPAIGN_400K_DIRNAME, track, seed) / "best_model.zip"
    relative_path = local_path.relative_to(PROJECT_ROOT)
    return resolve_artifact_or_local(relative_path)


# =============================================================================
# 8 paper-level methods -> 12 policy instances
# =============================================================================
@dataclass(frozen=True)
class PolicyInstance:
    """One concrete, runnable policy instance evaluated on all final seeds."""

    paper_method: str            # one of the 8 paper-level method keys
    policy_instance: str          # unique id, e.g. "greedy" or "ppo_seed_42"
    estimator: str                # "measurement" | "kalman"
    training_seed: int | None     # None for scripted; 42/43/44 for PPO
    policy_name: str              # registry key for uav_defend.policies.registry.get_policy()
    model_path: Path | None       # None for scripted; frozen 400k best_model.zip for PPO


# Six scripted controllers (one fixed instance each) + 2 PPO tracks (3 seeds each).
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "pn", "lead", "kalman_greedy", "pn_kalman", "lead_kalman")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")
PAPER_METHODS: tuple[str, ...] = SCRIPTED_METHODS + PPO_METHODS  # 8 total, canonical order


def _track_for_ppo_method(method: str) -> str:
    return "kalman" if method == "ppo_kalman" else "direct"


def build_policy_instances() -> tuple[PolicyInstance, ...]:
    """Build all 12 policy instances (6 scripted + 3 PPO + 3 PPO-Kalman)."""
    instances: list[PolicyInstance] = []
    for method in SCRIPTED_METHODS:
        spec = get_method_spec(method)
        instances.append(PolicyInstance(
            paper_method=method,
            policy_instance=method,
            estimator=spec.estimator,
            training_seed=None,
            policy_name=spec.policy_name,
            model_path=None,
        ))
    for method in PPO_METHODS:
        spec = get_method_spec(method)
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(PolicyInstance(
                paper_method=method,
                policy_instance=f"{method}_seed_{seed}",
                estimator=spec.estimator,
                training_seed=seed,
                policy_name=spec.policy_name,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


POLICY_INSTANCES: tuple[PolicyInstance, ...] = build_policy_instances()

EXPECTED_N_POLICY_INSTANCES = 12
EXPECTED_RAW_ROW_COUNT = EXPECTED_N_POLICY_INSTANCES * FINAL_NOMINAL_EPISODES  # 60,000

# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE viewing final-test results)
# =============================================================================
# (comparison_name, method_a, method_b) -- reported as A minus B (success-rate
# percentage-point difference), per experiments/run_final_nominal_evaluation.py.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("A", "ppo", "greedy"),
    ("B", "ppo", "pn"),
    ("C", "ppo", "lead"),
    ("D", "ppo_kalman", "kalman_greedy"),
    ("E", "ppo_kalman", "pn_kalman"),
    ("F", "ppo_kalman", "lead_kalman"),
    ("G", "ppo_kalman", "ppo"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("H", "kalman_greedy", "greedy"),
    ("I", "pn_kalman", "pn"),
    ("J", "lead_kalman", "lead"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS

# Comparisons where BOTH methods are scripted/fixed (single instance each) --
# use the paired-bootstrap + exact McNemar test (experiments/run_final_nominal_evaluation.py).
FIXED_PAIRED_COMPARISONS: tuple[str, ...] = ("H", "I", "J")
# Comparisons involving a PPO (3-seed) method -- use the hierarchical paired bootstrap.
HIERARCHICAL_PAIRED_COMPARISONS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
FINAL_NOMINAL_OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_nominal"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
METHOD_SUMMARY_FILENAME = "method_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PAIRWISE_COMPARISONS_FILENAME = "pairwise_comparisons.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
FINAL_NOMINAL_SUMMARY_FILENAME = "final_nominal_summary.json"
