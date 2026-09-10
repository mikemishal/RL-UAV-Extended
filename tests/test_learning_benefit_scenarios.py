"""Tests for the Phase-10 diagnostic scenario battery definitions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.learning_benefit.scenarios import build_scenario_battery
from uav_defend.config.env_config import EnvConfig


def test_battery_has_six_scenarios():
    scenarios = build_scenario_battery()
    assert len(scenarios) == 6
    names = [s.name for s in scenarios]
    assert names == [
        "nominal_open", "strong_maneuver", "strong_evasion",
        "mobility_mismatch", "clutter", "clutter_strong_maneuver",
    ]


def test_nominal_scenario_matches_default_env_config():
    scenarios = build_scenario_battery()
    nominal = next(s for s in scenarios if s.name == "nominal_open")
    default = EnvConfig()
    assert nominal.config.v_d == default.v_d
    assert nominal.config.v_e == default.v_e
    assert nominal.config.obstacles_enabled is False


def test_clutter_scenarios_enable_obstacles_and_hostile_avoidance():
    scenarios = build_scenario_battery()
    for name in ("clutter", "clutter_strong_maneuver"):
        scenario = next(s for s in scenarios if s.name == name)
        assert scenario.config.obstacles_enabled is True
        assert scenario.config.enemy_obstacle_avoidance_enabled is True
        # Frozen clutter-env-v1 obstacle-navigation defaults untouched.
        assert scenario.config.enemy_obstacle_clearance == 2.0
        assert scenario.config.enemy_obstacle_prediction_horizon_steps == 12
        assert scenario.config.enemy_obstacle_release_steps == 4


def test_strong_maneuver_uses_restrictive_dynamics_grid_values():
    scenarios = build_scenario_battery()
    scenario = next(s for s in scenarios if s.name == "strong_maneuver")
    assert scenario.config.defender_max_accel == 4.0
    assert scenario.config.defender_max_turn_rate_deg == 45.0


def test_mobility_mismatch_uses_established_grid_extremes():
    scenarios = build_scenario_battery()
    scenario = next(s for s in scenarios if s.name == "mobility_mismatch")
    assert scenario.config.v_d == 12.0
    assert scenario.config.v_e == 16.0


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
