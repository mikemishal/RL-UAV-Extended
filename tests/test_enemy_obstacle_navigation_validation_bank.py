"""Held-out hostile obstacle-navigation validation bank (journal
extension, Phase 3B, Step 10).

Independent of the 75-config development regression set used while tuning
`enemy_obstacle_prediction_horizon_steps`/`enemy_obstacle_release_steps`
(see test_enemy_obstacle_avoidance.py's docstring and the Phase-3B
report). Uses its own deterministic seed range (900-909) -- NOT any seed
used by defender evaluation/experiment pipelines -- and varies hostile
lateral offset, hostile altitude, and obstacle width/depth/height
(short/tall) independently of that tuning set, to avoid overfitting the
reported reliability number to the exact scenarios used for tuning.

Metrics reported (per the specification):
  A. first-pass obstacle avoidance rate
  B. complete obstacle-passage collision-free rate
  C. progress toward asset after clearing obstacle
  D. timeout/stall rate
  E. horizontal-bypass count
  F. vertical-bypass count
  G. fallback/no-feasible-candidate count

Run directly for the full report: python tests/test_enemy_obstacle_navigation_validation_bank.py
A lightweight pytest test also asserts the overall collision-free rate
meets the >=95% acceptance target as a regression guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout

_SEEDS = (900, 901, 902, 903, 904)  # held-out: never used for tuning or by defender evaluation pipelines
_LATERAL_OFFSETS = (0.0, 3.0, -3.0, 6.0, -6.0)
_ALTITUDES = (1.0, 5.0, 10.0)

# (label, center_z, half_extents) -- center_xy is fixed at (15, 0) for all;
# "short" obstacles are climb-bypassable, "tall" ones are not.
_OBSTACLE_GEOMETRIES = (
    ("medium_cube", 5.0, (4.0, 4.0, 5.0)),
    ("short_wide", 1.0, (4.0, 4.0, 1.0)),
    ("tall_narrow", 8.0, (2.0, 2.0, 8.0)),
    ("deep_wall", 5.0, (2.0, 8.0, 5.0)),
)

_HOSTILE_START_X = -20.0
_SOLDIER_POS = np.array([40.0, 0.0, 0.0], dtype=np.float32)
_HORIZON = 80


def _make_env(config: EnvConfig, obstacle: AABBObstacle) -> SoldierEnv:
    env = SoldierEnv(config=config)
    env.reset(seed=0)  # obstacle_layout_mode="none": no obstacle RNG stream consumed
    env._obstacle_layout = ObstacleLayout(obstacles=(obstacle,))
    return env


def _run_case(seed: int, y0: float, z0: float, geometry) -> dict:
    _label, center_z, half_extents = geometry
    obstacle = AABBObstacle.from_center_half_extents((15.0, 0.0, center_z), half_extents)
    config = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="none",
        enemy_obstacle_avoidance_enabled=True,
        defender_standby_until_detection=False,
    )
    env = _make_env(config, obstacle)
    env.reset(seed=seed)
    env._obstacle_layout = ObstacleLayout(obstacles=(obstacle,))
    env._enemy_pos = np.array([_HOSTILE_START_X, y0, z0], dtype=np.float32)
    env._enemy_vel = np.zeros(3, dtype=np.float32)
    env._soldier_pos = _SOLDIER_POS.copy()
    env._defender_pos = np.array([1000.0, 1000.0, 0.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    initial_dist = float(np.linalg.norm(env._enemy_pos.astype(np.float64) - _SOLDIER_POS.astype(np.float64)))
    ever_active = False
    first_pass_collision = False
    any_collision = False
    horizontal_bypass_used = False
    vertical_bypass_used = False
    fallback_used = False
    reached_or_progressed = False
    final_pos = env._enemy_pos.copy()
    steps_run = 0
    for _ in range(_HORIZON):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        steps_run += 1
        final_pos = info["enemy_pos"]
        if info["enemy_obstacle_avoidance_active"]:
            if not ever_active:
                ever_active = True
            mode = info["enemy_obstacle_avoidance_mode"]
            if mode in ("left", "right"):
                horizontal_bypass_used = True
            elif mode == "climb":
                vertical_bypass_used = True
        if info["enemy_obstacle_fallback_used"]:
            fallback_used = True
        if info["enemy_obstacle_collision"]:
            any_collision = True
            if not first_pass_collision and steps_run <= 30:
                first_pass_collision = True
        if term or trunc:
            break

    final_dist = float(np.linalg.norm(final_pos.astype(np.float64) - _SOLDIER_POS.astype(np.float64)))
    progress = initial_dist - final_dist
    stalled = steps_run >= _HORIZON and progress < initial_dist * 0.3

    return {
        "collision_free": not any_collision,
        "first_pass_collision_free": not first_pass_collision,
        "progress": progress,
        "initial_dist": initial_dist,
        "stalled": stalled,
        "horizontal_bypass_used": horizontal_bypass_used,
        "vertical_bypass_used": vertical_bypass_used,
        "fallback_used": fallback_used,
        "ever_active": ever_active,
    }


def _run_validation_bank() -> dict:
    results = []
    for geometry in _OBSTACLE_GEOMETRIES:
        for y0 in _LATERAL_OFFSETS:
            for z0 in _ALTITUDES:
                for seed in _SEEDS:
                    results.append(_run_case(seed, y0, z0, geometry))

    total = len(results)
    engaged = [r for r in results if r["ever_active"]]
    return {
        "total_configs": total,
        "engaged_count": len(engaged),
        "first_pass_rate": sum(r["first_pass_collision_free"] for r in results) / total,
        "complete_collision_free_rate": sum(r["collision_free"] for r in results) / total,
        "mean_progress_fraction": float(np.mean([r["progress"] / r["initial_dist"] for r in results])),
        "stall_rate": sum(r["stalled"] for r in results) / total,
        "horizontal_bypass_count": sum(r["horizontal_bypass_used"] for r in results),
        "vertical_bypass_count": sum(r["vertical_bypass_used"] for r in results),
        "fallback_count": sum(r["fallback_used"] for r in results),
        "results": results,
    }


def test_held_out_validation_bank_meets_acceptance_target():
    report = _run_validation_bank()
    print(f"\nheld-out validation bank: {report['total_configs']} configs, "
          f"{report['engaged_count']} engaged avoidance, "
          f"complete collision-free rate = {report['complete_collision_free_rate']:.2%}")
    # Acceptance target (Step 11): >= 95% collision-free passage on the
    # broader held-out bank. If this ever regresses, investigate rather
    # than raise gains/radius (there are no such parameters anymore).
    assert report["complete_collision_free_rate"] >= 0.95


if __name__ == "__main__":
    report = _run_validation_bank()
    print("Held-out obstacle-navigation validation bank")
    print("=" * 60)
    print(f"Total configurations:              {report['total_configs']}")
    print(f"Configurations that engaged avoidance: {report['engaged_count']}")
    print(f"A. First-pass avoidance rate:       {report['first_pass_rate']:.2%}")
    print(f"B. Complete collision-free rate:    {report['complete_collision_free_rate']:.2%}")
    print(f"C. Mean progress-toward-asset frac: {report['mean_progress_fraction']:.2%}")
    print(f"D. Timeout/stall rate:              {report['stall_rate']:.2%}")
    print(f"E. Horizontal-bypass count:         {report['horizontal_bypass_count']}")
    print(f"F. Vertical-bypass count:           {report['vertical_bypass_count']}")
    print(f"G. Fallback/no-feasible count:      {report['fallback_count']}")
