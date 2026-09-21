"""
Journal clutter-benchmark protocol -- SINGLE SOURCE OF TRUTH for the C0-C3
clutter benchmark of the existing, frozen, accepted-paper LR-PPO conference
controller set (Greedy, PN, Lead, RMPC, PPO, PPO-Kalman, LR-PPO).

This module does NOT train, retrain, or modify any controller. It only:
    1. loads the existing frozen checkpoints/configs exactly as used by the
       accepted resubmission evidence (see experiments/build_resubmission_evidence.py
       FAMILY_ORDER / LEARNED_ROOTS -- the canonical controller/seed inventory);
    2. defines the C0-C3 obstacle-density parameter table;
    3. defines a single deterministic seed contract shared by every
       controller/instance/clutter level (matched-scenario benchmarking).

=============================================================================
OBSTACLE SEMANTICS (Phase 3 audit -- read before interpreting results)
=============================================================================
Under the current environment (uav_defend/envs/soldier_env.py):
    - Defender-obstacle and hostile-obstacle collisions are DIAGNOSTIC ONLY:
      they never stop motion, never alter position/velocity, never set
      `terminated`, and never alter `reward`/`outcome`. Episode outcome
      ("intercepted" / "soldier_caught" / "unsafe_intercept" / "timeout")
      is computed purely from distances and step count, exactly as in the
      no-clutter (C0) environment.
    - The SOLDIER's random-walk motion IS obstacle-blocked (a candidate
      step that would cross an obstacle footprint is rejected), the only
      entity whose actual kinematics change due to obstacle presence.
    - The hostile UAV's own obstacle-avoidance planner
      (`EnvConfig.enemy_obstacle_avoidance_enabled`) is left at its
      environment default (False) for every clutter level in this
      benchmark, to strictly avoid modifying hostile behavior. The hostile
      therefore flies through obstacles exactly as it does in C0; only its
      diagnostic collision/clearance fields differ.
    - Detection is a pure 3-D distance-radius check, unaffected by obstacles
      (no line-of-sight blocking).
    - Initial entity positions can never spawn inside an obstacle
      (enforced by rejection sampling in uav_defend/obstacles/layout.py);
      if a requested layout cannot be placed within
      `obstacle_max_placement_attempts`, `generate_layout` raises
      `RuntimeError` -- treated as a SEPARATE "placement-infeasible scenario"
      category (see clutter_eval_utils.run_clutter_episode), never silently
      discarded or worked around.
    - Consequently: this benchmark measures (a) whether/how much each
      controller's SUCCESS RATE changes due to the soldier's
      obstacle-constrained motion altering engagement geometry, and (b)
      each controller's own diagnostic OBSTACLE-COLLISION RATE (how often
      its frozen, obstacle-unaware flight path geometrically intersects an
      obstacle) -- NOT a measurement of controllers being kinematically
      blocked, since no controller is currently blocked by obstacles.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.mpc.rmpc_policy import RMPCConfig, RMPCPolicy
from uav_defend.policies.registry import get_policy
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.artifact_paths import resolve_artifact_or_local
from experiments.lead_residual_diagnostic_protocol import FROZEN_MODELS, RESIDUAL_MAX_ANGLE_DEG

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Seed contract (Phase 5) -- reserved sub-range of the generic
# ROBUSTNESS_SEED_START..ROBUSTNESS_SEED_END (66000..99999) range declared in
# experiments/lead_residual_diagnostic_protocol.py. The SAME seed is used
# for a given evaluation index across EVERY clutter level (C0..C3) and
# EVERY controller/instance: env.reset(seed=seed) alone determines initial
# entity states, hostile stochastic behavior, soldier behavior, measurement
# noise, and obstacle layout (a dedicated, independent RNG stream spawned
# from the same seed -- see uav_defend/envs/soldier_env.py reset()).
# Consequently identical seeds give byte-identical scenarios across
# controllers, and near-identical scenarios (differing only in
# obstacle-related effects) across clutter levels.
# =============================================================================
CLUTTER_SEED_START = 80_000
CLUTTER_SMOKE_EPISODES = 20

# RUNTIME NOTE (Phase 8 smoke-test finding): RMPC's CEM optimizer costs
# ~24s/episode (C0, no obstacles) to ~58s/episode (C1-C3, with obstacles) --
# roughly 150-400x slower than every other controller (<0.5s/episode). At
# 500-1000 episodes/level this alone would take 27-54 hours, infeasible for
# this benchmarking session. CLUTTER_MAIN_EPISODES is therefore reduced to
# 50 (a documented, measured-runtime-justified deviation from the "prefer
# 500, extend to 1000" guidance -- see the Phase-13 final report's
# recommendation to re-run at full N with a faster CEM config or dedicated
# offline/HPC compute). All 13 controller instances use the SAME N so no
# asymmetric-sample-size complication is introduced into the analysis.
CLUTTER_MAIN_EPISODES = 50  # see run_clutter_benchmark.py --episodes to override

SMOKE_SEEDS: tuple[int, ...] = tuple(range(CLUTTER_SEED_START, CLUTTER_SEED_START + CLUTTER_SMOKE_EPISODES))


def main_seeds(n_episodes: int = CLUTTER_MAIN_EPISODES) -> tuple[int, ...]:
    """Deterministic seed set shared by every controller/level for the main benchmark."""
    return tuple(range(CLUTTER_SEED_START, CLUTTER_SEED_START + n_episodes))


# =============================================================================
# C0-C3 clutter-level parameter table (Phase 4)
# =============================================================================
# C0 reproduces the accepted conference (no-clutter) environment exactly:
# obstacles_enabled=False, no obstacle RNG stream consumed, no other field
# read. C1-C3 progressively increase obstacle count/size/height while
# keeping clearance-from-asset/spawn large enough that placement succeeds
# with high probability within obstacle_max_placement_attempts (500); see
# tests/test_clutter_benchmark_protocol.py for empirical placement-success
# verification on the full main-benchmark seed set.
CLUTTER_LEVELS: dict[str, dict] = {
    "C0": dict(
        obstacles_enabled=False,
    ),
    "C1": dict(
        obstacles_enabled=True,
        obstacle_layout_mode="random",
        obstacle_count=3,
        obstacle_min_size=2.0,
        obstacle_max_size=4.0,
        obstacle_min_height=2.0,
        obstacle_max_height=8.0,
        obstacle_clearance_from_asset=8.0,
        obstacle_clearance_from_spawn=5.0,
        obstacle_min_separation=3.0,
    ),
    "C2": dict(
        obstacles_enabled=True,
        obstacle_layout_mode="random",
        obstacle_count=8,
        obstacle_min_size=3.0,
        obstacle_max_size=6.0,
        obstacle_min_height=3.0,
        obstacle_max_height=10.0,
        obstacle_clearance_from_asset=6.0,
        obstacle_clearance_from_spawn=4.0,
        obstacle_min_separation=2.0,
    ),
    "C3": dict(
        obstacles_enabled=True,
        obstacle_layout_mode="random",
        obstacle_count=15,
        obstacle_min_size=3.0,
        obstacle_max_size=7.0,
        obstacle_min_height=4.0,
        obstacle_max_height=12.0,
        obstacle_clearance_from_asset=5.0,
        obstacle_clearance_from_spawn=3.0,
        obstacle_min_separation=1.0,
    ),
}
CLUTTER_LEVEL_ORDER: tuple[str, ...] = ("C0", "C1", "C2", "C3")


def build_env_config(level: str, estimator: str) -> EnvConfig:
    """Build the EnvConfig for a given clutter level and estimator track.

    `estimator` is "measurement" (Direct) or "kalman" -- everything else
    (dynamics, rewards, geometry thresholds, hostile evasion) is left at
    the frozen conference/journal defaults; only `use_kalman_tracking` and
    the clutter-level's obstacle fields differ.
    """
    if level not in CLUTTER_LEVELS:
        raise ValueError(f"Unknown clutter level: {level!r}. Expected one of {CLUTTER_LEVEL_ORDER}.")
    if estimator not in ("measurement", "kalman"):
        raise ValueError(f"Unknown estimator track: {estimator!r}")
    return EnvConfig(use_kalman_tracking=(estimator == "kalman"), **CLUTTER_LEVELS[level])


# =============================================================================
# Controller inventory (Phase 2) -- exactly the accepted-paper family order
# and checkpoint seeds from experiments/build_resubmission_evidence.py
# (FAMILY_ORDER, LEARNED_ROOTS). No controller is retrained or modified;
# LR-PPO uses the unmodified 30-degree bounded residual composition.
# =============================================================================
LEARNED_ROOTS: dict[str, tuple[int, ...]] = {"PPO": (45, 46, 47), "PPO-Kalman": (45, 46, 47), "LR-PPO": (55, 56, 57)}


@dataclasses.dataclass(frozen=True)
class ControllerInstance:
    """One concrete, runnable controller instance for the clutter benchmark."""

    family: str  # display family name, e.g. "PPO" (pools instances for analysis)
    instance_id: str  # unique instance name, e.g. "PPO-45" or "Greedy"
    estimator: str  # "measurement" | "kalman"
    build: Callable[[EnvConfig], object]  # config -> policy with act()/reset()
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None


_FROZEN_RMPC_CONFIG: RMPCConfig | None = None


def _frozen_rmpc_config() -> RMPCConfig:
    """Load the frozen, selected RMPC hyperparameters (results/rmpc_tuning/
    selected_rmpc_config.json), exactly as used by run_rmpc_final_nominal.py.
    Never retunes; raises if the frozen selection marker is missing/false."""
    global _FROZEN_RMPC_CONFIG
    if _FROZEN_RMPC_CONFIG is not None:
        return _FROZEN_RMPC_CONFIG
    import json

    config_path = resolve_artifact_or_local("results/rmpc_tuning/selected_rmpc_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"frozen RMPC config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if not data.get("frozen_for_final_evaluation"):
        raise RuntimeError("selected_rmpc_config.json: frozen_for_final_evaluation is not true -- STOP")
    _FROZEN_RMPC_CONFIG = RMPCConfig(
        horizon=data["horizon"],
        population_size=data["population_size"],
        elite_fraction=data["elite_fraction"],
        cem_iterations=data["cem_iterations"],
        controller_seed=data["controller_seed"],
        cem_initial_std=data["cem_initial_std"],
        cem_min_std=data["cem_min_std"],
        cem_sample_clip=data["cem_sample_clip"],
        enable_timing=False,
    )
    return _FROZEN_RMPC_CONFIG


def controller_instances() -> list[ControllerInstance]:
    """The 13 controller instances (4 scripted/optimization-based + 3x3
    learned-family seeds) exercised at every clutter level."""
    instances: list[ControllerInstance] = []

    instances.append(ControllerInstance("Greedy", "Greedy", "measurement", lambda cfg: get_policy("greedy")))
    instances.append(ControllerInstance("PN", "PN", "measurement", lambda cfg: get_policy("pn", config=cfg)))
    instances.append(ControllerInstance("Lead", "Lead", "measurement", lambda cfg: get_policy("lead", config=cfg)))
    instances.append(
        ControllerInstance(
            "RMPC", "RMPC", "measurement",
            lambda cfg: RMPCPolicy(config=cfg, rmpc_config=_frozen_rmpc_config()),
        )
    )

    for seed in LEARNED_ROOTS["PPO"]:
        spec = FROZEN_MODELS[("direct", seed)]
        instances.append(
            ControllerInstance(
                "PPO", f"PPO-{seed}", "measurement",
                (lambda cfg, p=spec["path"]: get_policy("ppo", model_path=str(p), deterministic=True)),
                checkpoint_path=spec["path"], checkpoint_sha256=spec["sha256"],
            )
        )

    for seed in LEARNED_ROOTS["PPO-Kalman"]:
        spec = FROZEN_MODELS[("kalman", seed)]
        instances.append(
            ControllerInstance(
                "PPO-Kalman", f"PPOK-{seed}", "kalman",
                (lambda cfg, p=spec["path"]: get_policy("ppo_kalman", model_path=str(p), deterministic=True)),
                checkpoint_path=spec["path"], checkpoint_sha256=spec["sha256"],
            )
        )

    for seed in LEARNED_ROOTS["LR-PPO"]:
        spec = FROZEN_MODELS[("lr_ppo", seed)]
        instances.append(
            ControllerInstance(
                "LR-PPO", f"LRPPO-{seed}", "measurement",
                (lambda cfg, p=spec["path"]: LeadResidualPPOPolicyWrapper.load(
                    p, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=cfg, deterministic=True,
                )),
                checkpoint_path=spec["path"], checkpoint_sha256=spec["sha256"],
            )
        )

    return instances


FAMILY_ORDER: tuple[str, ...] = ("Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman", "LR-PPO")
