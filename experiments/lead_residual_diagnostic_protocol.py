"""
Lead-Residual PPO 200-seed independent diagnostic protocol -- SINGLE SOURCE
OF TRUTH. ONE-TIME diagnostic/generalization-evidence experiment on the
previously-reserved LR-PPO diagnostic range (60100..60299, 200 seeds).
NOT the final expanded-paper test (61000..65999, still sealed).

Evaluates 12 method instances: Greedy, PN, Lead (classical), PPO-45/46/47
(standalone RL), PPOK-45/46/47 (estimator ablation), LR-PPO-55/56/57
(proposed method). Excludes Kalman-Greedy and any historical/legacy policy.

training_source_commit (old 6-model PPO/PPO-Kalman campaign) =
    4b0a903bf801c7b439be4671ed822e2f853d72ba
lr_ppo_training_source_commit (LR-PPO-55/56/57 campaign) =
    3b5bc9a48c0d3d52bcf369aea2d97de4c85d6445
diagnostic_source_commit = recorded at evaluation time (see
    run_lead_residual_diagnostic.py); must be identical before/after.
"""

from __future__ import annotations

from pathlib import Path

from experiments.artifact_paths import resolve_artifact_or_local
from experiments.post_detection_diagnostic_protocol import FROZEN_MODELS as _OLD_FROZEN_MODELS

PROJECT_ROOT = Path(__file__).parent.parent

DIAGNOSTIC_SEED_START = 60_100
DIAGNOSTIC_SEED_END = 60_299
DIAGNOSTIC_EPISODES = DIAGNOSTIC_SEED_END - DIAGNOSTIC_SEED_START + 1  # 200
DIAGNOSTIC_SEEDS: range = range(DIAGNOSTIC_SEED_START, DIAGNOSTIC_SEED_END + 1)
assert DIAGNOSTIC_EPISODES == 200

EXPANDED_FINAL_SEED_START = 61_000
EXPANDED_FINAL_SEED_END = 65_999
ROBUSTNESS_SEED_START = 66_000
ROBUSTNESS_SEED_END = 99_999

TRAINING_SEEDS: tuple[int, ...] = (45, 46, 47)
LR_TRAINING_ROOTS: tuple[int, ...] = (55, 56, 57)

TRAINING_SOURCE_COMMIT = "4b0a903bf801c7b439be4671ed822e2f853d72ba"
LR_PPO_TRAINING_SOURCE_COMMIT = "3b5bc9a48c0d3d52bcf369aea2d97de4c85d6445"

# Reuse the old 6-model table verbatim (single source of truth) -- NEVER
# duplicated/retyped, avoiding hash-transcription risk.
FROZEN_MODELS = dict(_OLD_FROZEN_MODELS)

# LR-PPO models (verified against the training campaign's run_manifest.json
# hashes; see docs/lead_residual_training_protocol.md).
FROZEN_MODELS[("lr_ppo", 55)] = {
    "best_step": 250_000,
    "sha256": "40c64c46cad7f97da8ecf1f59fcec856e68b3633a557c133116b64759933fc7b",
    "path": resolve_artifact_or_local("models/lead_residual_post_detection/seed_55/best_model.zip"),
    "validation_best_success_rate": 0.94,
}
FROZEN_MODELS[("lr_ppo", 56)] = {
    "best_step": 200_000,
    "sha256": "e12753eafd357dbeb849126997373c4bd963d79079636a30a945c2fc3ae2c343",
    "path": resolve_artifact_or_local("models/lead_residual_post_detection/seed_56/best_model.zip"),
    "validation_best_success_rate": 0.96,
}
FROZEN_MODELS[("lr_ppo", 57)] = {
    "best_step": 375_000,
    "sha256": "f7b54cdeaa2a4e760399df41a915cc7a5f5f134e567599f0e8a59a59427ff29a",
    "path": resolve_artifact_or_local("models/lead_residual_post_detection/seed_57/best_model.zip"),
    "validation_best_success_rate": 0.97,
}

PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "pn", "lead", "ppo", "ppo_kalman", "lr_ppo")
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "pn", "lead")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")
LR_PPO_METHOD = "lr_ppo"
EXCLUDED_METHODS = ("kalman_greedy", "pn_kalman", "lead_kalman", "escorted_ppo", "random")

RESIDUAL_MAX_ANGLE_DEG = 30.0

# Frozen BEFORE opening the diagnostic range (item 12) -- deliberately
# DIFFERENT from the 20260816 seed used by the prior 5-method diagnostic/
# final-nominal evaluations, per this task's explicit instruction.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260818

# Primary comparisons frozen BEFORE viewing diagnostic results (item 13) --
# must not be reclassified after seeing outcomes.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("LR-PPO family - Lead", "lr_ppo", "lead"),
    ("LR-PPO family - PPO family", "lr_ppo", "ppo"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("PPO family - Greedy", "ppo", "greedy"),
    ("PPO family - PN", "ppo", "pn"),
    ("PPO family - Lead", "ppo", "lead"),
)

# Validation success rates recorded BEFORE this diagnostic (item 24) -- for
# a purely descriptive generalization-check delta, never re-selection.
VALIDATION_BEST_SUCCESS_RATES = {
    ("lr_ppo", 55): 0.94,
    ("lr_ppo", 56): 0.96,
    ("lr_ppo", 57): 0.97,
}

OUTPUT_ROOT = PROJECT_ROOT / "results" / "lead_residual_diagnostic_200"


def assert_not_reserved_seed(seed: int) -> None:
    if EXPANDED_FINAL_SEED_START <= seed <= EXPANDED_FINAL_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (sealed) expanded-final range; must not be used here.")
    if ROBUSTNESS_SEED_START <= seed <= ROBUSTNESS_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (sealed) robustness range; must not be used here.")
