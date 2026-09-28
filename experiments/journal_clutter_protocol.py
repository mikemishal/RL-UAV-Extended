"""
Canonical journal clutter protocol (Phase 2/3) -- SINGLE SOURCE OF TRUTH for
all obstacle-aware training/evaluation from this sub-task forward.

IMPORTANT (per this sub-task's own finding): `experiments/clutter_benchmark_protocol.py`
is a PRE-EXISTING, historical artifact whose `build_env_config()` never
enables the attacker's obstacle avoidance (`enemy_obstacle_avoidance_enabled`
stays at its environment default, False) -- documented explicitly in that
module's own docstring, written back when obstacle collisions were purely
diagnostic. Now that collisions are physically consequential
(`journal-physical-collision-semantics-v1`), that configuration causes the
attacker to crash into obstacles very frequently at C2/C3 (measured:
53-97% attacker-environmental-failure rate), which would confound ANY
defender-training/evaluation signal. That module is LEFT UNCHANGED (it is
historical evidence for an earlier benchmark), but it MUST NOT be reused
for the training/evaluation work in this sub-task -- use ONLY this module.

This module defines, explicitly (never relying on EnvConfig defaults):
    - CANONICAL_CLUTTER_LEVELS: static+dynamic obstacle density table
      (C1-C3), reusing the C1-C3 static-obstacle parameters from
      clutter_benchmark_protocol.CLUTTER_LEVELS unchanged (already
      validated placement-success), extended with modest dynamic-obstacle
      counts per level (absent from the historical table).
    - build_canonical_env_config(level, estimator, ...): the ONE function
      that must be used to build an EnvConfig for canonical journal
      clutter work. Every attacker/defender obstacle-awareness/avoidance
      field is set EXPLICITLY (Phase 2: "do NOT rely on defaults").
    - build_randomized_training_config(rng, ...): Phase 14's per-episode
      randomized training-clutter distribution (spans C1-C3 difficulty).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from uav_defend.config.env_config import EnvConfig

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS as _LEGACY_CLUTTER_LEVELS

# =============================================================================
# Phase 2/3: explicit canonical attacker/defender obstacle settings.
# =============================================================================
# Primary awareness mode for BOTH sides: Level C ("map_local") -- mapped
# static obstacles known from episode start, plus local sensing of
# everything else (unmapped static + all dynamic). Level B ("local") is
# retained as an explicit, equally-supported ablation mode (Phase 26), not
# the default.
CANONICAL_AWARENESS_MODE = "map_local"

# Shared reactive-avoidance geometry (identical defaults already used by
# the existing, tested safety layer -- restated here explicitly, per
# Phase 2's "do not rely on defaults").
CANONICAL_OBSTACLE_CLEARANCE = 2.0
CANONICAL_OBSTACLE_PREDICTION_HORIZON_STEPS = 12
CANONICAL_OBSTACLE_RELEASE_STEPS = 4
CANONICAL_STATIC_SENSOR_RANGE = 15.0
CANONICAL_DYNAMIC_SENSOR_RANGE = 15.0
CANONICAL_DYNAMIC_MEMORY_STEPS = 6

# Phase 3: route planning is an ablation, NOT the canonical default -- the
# canonical attacker is the reactive-only (Level C perception + reactive
# static/dynamic avoidance) hostile, exactly as already validated by
# `journal-behavior-arbitration-v1`/`journal-defender-obstacle-awareness-v1`.
CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED = False

# Dynamic-obstacle density per level (absent from the historical static-only
# CLUTTER_LEVELS table -- Phase 2 requires BOTH static and dynamic clutter).
# Chosen conservatively (few, slow-moving obstacles) so placement/detection
# remain reliable; not yet tuned against any performance target.
_DYNAMIC_OBSTACLE_COUNTS: dict[str, int] = {"C0": 0, "C1": 1, "C2": 2, "C3": 4}

# Bounds for Phase 14's per-episode randomized training distribution.
RANDOMIZED_STATIC_COUNT_RANGE = (3, 15)          # spans C1.count .. C3.count
RANDOMIZED_STATIC_SIZE_RANGE = (2.0, 7.0)        # spans C1..C3 min/max size
RANDOMIZED_STATIC_HEIGHT_RANGE = (2.0, 12.0)     # spans C1..C3 min/max height
RANDOMIZED_DYNAMIC_COUNT_RANGE = (0, 4)          # spans C0..C3 dynamic counts

# =============================================================================
# Phase 16/18/23: seed namespaces, kept entirely separate from (a) the
# conference TRAINING_SEEDS/VALIDATION_SEEDS in training_protocol.py, (b)
# the historical clutter_benchmark_protocol's CLUTTER_SEED_START=80_000
# range, and (c) each other -- so training episodes, validation episodes,
# and held-out evaluation episodes can never silently collide.
# =============================================================================
CLUTTER_TRAINING_SEEDS: dict[str, tuple[int, ...]] = {
    "ppo": (71, 72, 73),
    "ppo_kalman": (74, 75, 76),
    "lr_ppo": (77, 78, 79),
}
CLUTTER_EVAL_SEED_START = 95_000  # held-out; NEVER used for training


def _canonical_awareness_fields(role: str, mode: str) -> dict:
    """Explicit awareness/avoidance fields for `role` in {"enemy","defender"}."""
    prefix = role
    return {
        f"{prefix}_obstacle_avoidance_enabled": True,
        f"{prefix}_obstacle_awareness_mode": mode,
        f"{prefix}_obstacle_clearance": CANONICAL_OBSTACLE_CLEARANCE,
        f"{prefix}_obstacle_prediction_horizon_steps": CANONICAL_OBSTACLE_PREDICTION_HORIZON_STEPS,
        f"{prefix}_obstacle_release_steps": CANONICAL_OBSTACLE_RELEASE_STEPS,
        f"{prefix}_obstacle_sensor_range": CANONICAL_STATIC_SENSOR_RANGE,
        f"{prefix}_dynamic_obstacle_sensor_range": CANONICAL_DYNAMIC_SENSOR_RANGE,
        f"{prefix}_static_obstacle_memory_enabled": True,
        f"{prefix}_dynamic_obstacle_memory_steps": CANONICAL_DYNAMIC_MEMORY_STEPS,
    }


def build_canonical_clutter_levels(awareness_mode: str = CANONICAL_AWARENESS_MODE) -> dict[str, dict]:
    """C0-C3 obstacle-density table: reuses the legacy static-obstacle
    parameters verbatim (already placement-validated), adds dynamic-obstacle
    counts, and returns per-level EnvConfig-kwargs dicts. `awareness_mode`
    is NOT included here -- see `build_canonical_env_config`, which applies
    it uniformly on top of whichever level dict is selected."""
    levels: dict[str, dict] = {}
    for level, legacy_kwargs in _LEGACY_CLUTTER_LEVELS.items():
        kwargs = dict(legacy_kwargs)
        kwargs["dynamic_obstacles_enabled"] = _DYNAMIC_OBSTACLE_COUNTS[level] > 0
        kwargs["dynamic_obstacle_count"] = _DYNAMIC_OBSTACLE_COUNTS[level]
        levels[level] = kwargs
    return levels


CANONICAL_CLUTTER_LEVELS = build_canonical_clutter_levels()
CANONICAL_CLUTTER_LEVEL_ORDER = ("C0", "C1", "C2", "C3")


def build_canonical_env_config(
    level: str,
    estimator: str,
    *,
    awareness_mode: str = CANONICAL_AWARENESS_MODE,
    enemy_route_planning_enabled: bool = CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
    extra_overrides: dict | None = None,
) -> EnvConfig:
    """THE canonical journal clutter EnvConfig builder (Phase 2/3). Every
    attacker/defender obstacle field is set explicitly -- never left at an
    EnvConfig default. Use this for ALL clutter training/evaluation in this
    sub-task; do NOT reuse `clutter_benchmark_protocol.build_env_config`
    (see module docstring)."""
    if level not in CANONICAL_CLUTTER_LEVELS:
        raise ValueError(f"Unknown clutter level: {level!r}. Expected one of {CANONICAL_CLUTTER_LEVEL_ORDER}.")
    if estimator not in ("measurement", "kalman"):
        raise ValueError(f"Unknown estimator track: {estimator!r}")

    kwargs = dict(CANONICAL_CLUTTER_LEVELS[level])
    kwargs.update(_canonical_awareness_fields("enemy", awareness_mode))
    kwargs.update(_canonical_awareness_fields("defender", awareness_mode))
    kwargs["enemy_route_planning_enabled"] = enemy_route_planning_enabled
    # Behavior arbitration stays at its own validated default (True) --
    # dormant while route planning is disabled; explicitly restated so this
    # is never silently dependent on the EnvConfig default (Phase 2).
    kwargs["enemy_route_arbitration_enabled"] = True
    if extra_overrides:
        kwargs.update(extra_overrides)
    return EnvConfig(use_kalman_tracking=(estimator == "kalman"), **kwargs)


def build_randomized_training_config(
    rng: np.random.Generator,
    estimator: str,
    *,
    awareness_mode: str = CANONICAL_AWARENESS_MODE,
    enemy_route_planning_enabled: bool = CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
) -> EnvConfig:
    """Phase 14: one randomly-sampled training-clutter EnvConfig spanning
    C1-C3 difficulty (never C0 -- training always includes obstacles).
    `rng` must be a caller-owned `numpy.random.Generator` (e.g. seeded from
    the training seed), so the SAME seed always yields the SAME sampled
    clutter distribution (reproducible curriculum), independent of the
    environment's own internal obstacle-placement RNG stream."""
    static_count = int(rng.integers(RANDOMIZED_STATIC_COUNT_RANGE[0], RANDOMIZED_STATIC_COUNT_RANGE[1] + 1))
    min_size = float(rng.uniform(*RANDOMIZED_STATIC_SIZE_RANGE))
    max_size = float(min_size + rng.uniform(0.5, 3.0))
    min_height = float(rng.uniform(*RANDOMIZED_STATIC_HEIGHT_RANGE))
    max_height = float(min_height + rng.uniform(1.0, 4.0))
    dynamic_count = int(rng.integers(RANDOMIZED_DYNAMIC_COUNT_RANGE[0], RANDOMIZED_DYNAMIC_COUNT_RANGE[1] + 1))

    kwargs = dict(
        obstacles_enabled=True,
        obstacle_layout_mode="random",
        obstacle_count=static_count,
        obstacle_min_size=min_size,
        obstacle_max_size=max_size,
        obstacle_min_height=min_height,
        obstacle_max_height=max_height,
        # Clearance/separation held at the C2 (mid-difficulty) values --
        # randomizing count/size/height already spans C1-C3 difficulty;
        # randomizing clearance too risks unreliable placement.
        obstacle_clearance_from_asset=CANONICAL_CLUTTER_LEVELS["C2"]["obstacle_clearance_from_asset"],
        obstacle_clearance_from_spawn=CANONICAL_CLUTTER_LEVELS["C2"]["obstacle_clearance_from_spawn"],
        obstacle_min_separation=CANONICAL_CLUTTER_LEVELS["C2"]["obstacle_min_separation"],
        dynamic_obstacles_enabled=dynamic_count > 0,
        dynamic_obstacle_count=dynamic_count,
    )
    kwargs.update(_canonical_awareness_fields("enemy", awareness_mode))
    kwargs.update(_canonical_awareness_fields("defender", awareness_mode))
    kwargs["enemy_route_planning_enabled"] = enemy_route_planning_enabled
    kwargs["enemy_route_arbitration_enabled"] = True
    return EnvConfig(use_kalman_tracking=(estimator == "kalman"), **kwargs)
