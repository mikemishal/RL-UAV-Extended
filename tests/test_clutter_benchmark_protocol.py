"""
Lightweight regression tests for experiments/clutter_benchmark_protocol.py
(Phase 4/5 seed contract and parameter table) and
experiments/clutter_eval_utils.py (Phase 6/7 episode runner).

No training; no environment/controller modification.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.envs import SoldierEnv

from experiments.clutter_benchmark_protocol import (
    CLUTTER_LEVEL_ORDER,
    CLUTTER_LEVELS,
    SMOKE_SEEDS,
    build_env_config,
    controller_instances,
    main_seeds,
)
from experiments.clutter_eval_utils import run_clutter_episode


def test_c0_reproduces_conference_baseline_defaults():
    cfg = build_env_config("C0", "measurement")
    assert cfg.obstacles_enabled is False


def test_c1_c2_c3_strictly_increase_obstacle_count():
    counts = [CLUTTER_LEVELS[level]["obstacle_count"] for level in ("C1", "C2", "C3")]
    assert counts == sorted(counts)
    assert len(set(counts)) == 3


def test_seed_contract_is_shared_across_levels_and_deterministic():
    assert main_seeds(500)[:20] == SMOKE_SEEDS
    assert main_seeds(10) == main_seeds(10)  # deterministic, pure function


def test_controller_inventory_has_13_instances_across_7_families():
    instances = controller_instances()
    assert len(instances) == 13
    families = {inst.family for inst in instances}
    assert families == {"Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman", "LR-PPO"}


def test_checkpoint_instances_have_sha256_and_existing_paths():
    for inst in controller_instances():
        if inst.checkpoint_path is not None:
            assert inst.checkpoint_path.exists(), f"{inst.instance_id}: missing {inst.checkpoint_path}"
            assert inst.checkpoint_sha256 and len(inst.checkpoint_sha256) == 64


@pytest.mark.parametrize("level", CLUTTER_LEVEL_ORDER)
def test_layout_is_placeable_for_first_20_smoke_seeds(level):
    """Placement-feasibility smoke check for the C0-C3 parameter table."""
    config = build_env_config(level, "measurement")
    env = SoldierEnv(config=config)
    infeasible = 0
    for seed in SMOKE_SEEDS:
        try:
            env.reset(seed=seed)
        except RuntimeError:
            infeasible += 1
    assert infeasible == 0, f"{level}: {infeasible}/{len(SMOKE_SEEDS)} seeds failed obstacle placement"


def test_same_seed_gives_identical_scenario_across_two_controllers():
    """Matched-scenario contract (Phase 5): same seed -> same obstacle layout
    and identical hostile/soldier initial state regardless of controller."""
    config = build_env_config("C2", "measurement")
    env_a = SoldierEnv(config=config)
    env_b = SoldierEnv(config=config)
    _, info_a = env_a.reset(seed=SMOKE_SEEDS[0])
    _, info_b = env_b.reset(seed=SMOKE_SEEDS[0])
    assert info_a["obstacle_layout"] == info_b["obstacle_layout"]
    assert np.array_equal(info_a["soldier_pos"], info_b["soldier_pos"])
    assert np.array_equal(info_a["enemy_pos"], info_b["enemy_pos"])


def test_obstacle_collision_is_diagnostic_not_terminal():
    """Journal extension update: obstacle collision now DOES terminate the
    episode with its own dedicated outcome (physical-collision-semantics
    phase) -- it is no longer folded into the four original mission
    outcomes, but it also must never be MISCLASSIFIED as one of them."""
    config = build_env_config("C3", "measurement")
    env = SoldierEnv(config=config)
    from uav_defend.policies.registry import get_policy

    policy = get_policy("lead", config=config)
    metrics = run_clutter_episode(env, policy, SMOKE_SEEDS[0])
    assert metrics["feasible"] is True
    assert metrics["outcome"] in (
        "intercepted", "soldier_caught", "unsafe_intercept", "timeout",
        "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
        "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
    )


def test_run_clutter_episode_reports_feasible_flag():
    config = build_env_config("C0", "measurement")
    env = SoldierEnv(config=config)
    from uav_defend.policies.registry import get_policy

    policy = get_policy("greedy")
    metrics = run_clutter_episode(env, policy, SMOKE_SEEDS[0])
    assert metrics["feasible"] is True
    assert metrics["obstacles_enabled"] is False
    assert metrics["defender_obstacle_collision"] == 0
