"""
Seed ranges for the FA-LAPG temporal-prediction dataset (Phase 28).

Disjoint from every previously locked seed range in this repository,
including:
  - training seeds 42-44
  - validation 10000-10099
  - diagnostic 12000-12199
  - final-nominal 20000-24999
  - acquisition-ablation 25000-25999
  - detection-radius 30000-30999
  - measurement-noise 31000-31999
  - 1-D mobility 32000-32999
  - hostile-evasion 33000-33999
  - maneuverability 34000-34999
  - mobility-grid 35000-35499
  - learning-benefit diagnostic pilot 60000-60199

These ranges are used ONLY for the supervised trajectory-prediction dataset
introduced in this task (Phases 27-34) and must not be reused as a
model-training seed range for any other study, nor treated as the final
journal evaluation seed bank.
"""

from __future__ import annotations

PREDICTOR_TRAIN_SEED_START = 70000
PREDICTOR_TRAIN_N_EPISODES = 20  # per scenario

PREDICTOR_VAL_SEED_START = 71000
PREDICTOR_VAL_N_EPISODES = 8  # per scenario

PREDICTOR_TEST_SEED_START = 72000
PREDICTOR_TEST_N_EPISODES = 8  # per scenario


def train_seeds() -> range:
    return range(PREDICTOR_TRAIN_SEED_START, PREDICTOR_TRAIN_SEED_START + PREDICTOR_TRAIN_N_EPISODES)


def val_seeds() -> range:
    return range(PREDICTOR_VAL_SEED_START, PREDICTOR_VAL_SEED_START + PREDICTOR_VAL_N_EPISODES)


def test_seeds() -> range:
    return range(PREDICTOR_TEST_SEED_START, PREDICTOR_TEST_SEED_START + PREDICTOR_TEST_N_EPISODES)


def assert_disjoint_from_locked_ranges() -> None:
    """Defensive check: the three predictor seed blocks must never overlap
    each other or any previously locked range in this repository."""
    locked = [
        range(42, 45), range(10000, 10100), range(12000, 12200),
        range(20000, 25000), range(25000, 26000), range(30000, 31000),
        range(31000, 32000), range(32000, 33000), range(33000, 34000),
        range(34000, 35000), range(35000, 35500), range(60000, 60200),
    ]
    blocks = [train_seeds(), val_seeds(), test_seeds()]
    for i, a in enumerate(blocks):
        for b in locked:
            if set(a) & set(b):
                raise AssertionError(f"Predictor seed block {a} overlaps locked range {b}")
        for j, other in enumerate(blocks):
            if i != j and set(a) & set(other):
                raise AssertionError(f"Predictor seed blocks {a} and {other} overlap")
