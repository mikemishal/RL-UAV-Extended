"""
Post-detection diagnostic (200-seed) evaluation protocol -- SINGLE SOURCE OF TRUTH.

ONE-TIME diagnostic/integrity experiment on the previously-reserved diagnostic
seed range (40100..40299, 200 seeds), NOT the final paper evaluation. Uses the
revised five paper-facing methods only (Greedy, PN, Lead, PPO, PPO-Kalman);
Kalman-Greedy/PN-Kalman/Lead-Kalman/Escorted-PPO/legacy-42-43-44/Random are
explicitly excluded.

training_source_commit  = 4b0a903bf801c7b439be4671ed822e2f853d72ba (all six
    frozen models were trained from this commit -- see
    results/training_post_detection_400k/*/seed_*/run_manifest.json)
evaluation_source_commit = recorded at evaluation time (see
    run_post_detection_diagnostic.py); must be identical before/after.
"""

from __future__ import annotations

from pathlib import Path

from experiments.artifact_paths import resolve_artifact_or_local

PROJECT_ROOT = Path(__file__).parent.parent

DIAGNOSTIC_SEED_OFFSET = 40_100
DIAGNOSTIC_EPISODES = 200
DIAGNOSTIC_SEEDS: range = range(DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES)

FINAL_TEST_SEED_START = 41_000
FINAL_TEST_SEED_END = 45_999
FUTURE_ROBUSTNESS_SEED_START = 46_000

TRAINING_SEEDS: tuple[int, ...] = (45, 46, 47)

# Frozen model checkpoints -- selection is COMPLETE and LOCKED (see
# results/training_post_detection_400k/*/seed_*/run_manifest.json). This
# diagnostic evaluation must NEVER alter this table regardless of outcome.
FROZEN_MODELS = {
    ("direct", 45): {
        "best_step": 350_000,
        "sha256": "3b7e37dcf94a90c484364127c8a594c06386fba389bed1735cac316ed34b6595",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/direct/seed_45/best_model.zip"),
        "validation_best_success_rate": 0.92,
    },
    ("direct", 46): {
        "best_step": 200_000,
        "sha256": "ceb4c652a84ab88616e8389ae4aa80b58ae5e323b3496ea2103dfe8c85c1b37c",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/direct/seed_46/best_model.zip"),
        "validation_best_success_rate": 0.94,
    },
    ("direct", 47): {
        "best_step": 100_000,
        "sha256": "ea38fe085ec6ea1cefa051ba08ee80d749f35b87b699a129bf9157d82ed4a334",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/direct/seed_47/best_model.zip"),
        "validation_best_success_rate": 0.89,
    },
    ("kalman", 45): {
        "best_step": 225_000,
        "sha256": "ddf16641a71b102cfb01d9928597ab0c43d11598cd2d0f4eb63932d80490f98e",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/kalman/seed_45/best_model.zip"),
        "validation_best_success_rate": 0.89,
    },
    ("kalman", 46): {
        "best_step": 200_000,
        "sha256": "b22116eb659552a94c28bfb32cdef58d5630267eaa4654a7a87b4ab802234f07",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/kalman/seed_46/best_model.zip"),
        "validation_best_success_rate": 0.83,
    },
    ("kalman", 47): {
        "best_step": 175_000,
        "sha256": "a9d7a3f39c3c63dab98cfcfbb68c117844c3ef1884a23a242a9d38cfa4f2152d",
        "path": resolve_artifact_or_local("results/training_post_detection_400k/kalman/seed_47/best_model.zip"),
        "validation_best_success_rate": 0.88,
    },
}
TRAINING_SOURCE_COMMIT = "4b0a903bf801c7b439be4671ed822e2f853d72ba"

PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "pn", "lead", "ppo", "ppo_kalman")
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "pn", "lead")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")
EXCLUDED_METHODS = ("kalman_greedy", "pn_kalman", "lead_kalman", "escorted_ppo", "random")

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816

OUTPUT_ROOT = PROJECT_ROOT / "results" / "post_detection_diagnostic_200"


def assert_not_reserved_seed(seed: int) -> None:
    if FINAL_TEST_SEED_START <= seed <= FINAL_TEST_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED final-test range; must not be used in a diagnostic.")
    if seed >= FUTURE_ROBUSTNESS_SEED_START:
        raise ValueError(f"Seed {seed} is >= FUTURE_ROBUSTNESS_SEED_START ({FUTURE_ROBUSTNESS_SEED_START}); reserved.")
