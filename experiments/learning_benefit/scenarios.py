"""Phase 10: diagnostic scenario battery.

Reuses ALREADY-ESTABLISHED parameter values from the locked robustness
protocols in this repository (never invents new parameter values):
  - experiments/robustness_hostile_evasion_protocol.py (EVASION_GAIN_VALUES)
  - experiments/robustness_maneuverability_protocol.py (DEFENDER_MAX_ACCEL/TURN_RATE_VALUES)
  - experiments/robustness_mobility_protocol.py (DEFENDER/HOSTILE_SPEED_VALUES)

Clutter scenarios use the FROZEN `clutter-env-v1` obstacle-navigation
defaults (uav_defend/config/env_config.py: enemy_obstacle_clearance=2.0,
enemy_obstacle_prediction_horizon_steps=12, enemy_obstacle_release_steps=4)
-- only `obstacles_enabled`/`obstacle_layout_mode`/`obstacle_count`/
`enemy_obstacle_avoidance_enabled` are set here; every hostile/obstacle
BEHAVIOR parameter remains untouched, per the environment-freeze directive.
"""

from __future__ import annotations

from dataclasses import dataclass

from uav_defend.config.env_config import EnvConfig

# Reused, not reinvented -- see module docstring.
_STRONG_EVASION_GAIN = 1.00      # max of EVASION_GAIN_VALUES = (0.00, 0.25, 0.50, 0.75, 1.00)
_RESTRICTIVE_ACCEL = 4.0         # min of DEFENDER_MAX_ACCEL_VALUES = (4.0, 8.0, 12.0)
_RESTRICTIVE_TURN_RATE_DEG = 45.0  # min of DEFENDER_MAX_TURN_RATE_VALUES = (45.0, 90.0, 135.0)
_MOBILITY_MISMATCH_DEFENDER_SPEED = 12.0  # min of DEFENDER_SPEED_VALUES
_MOBILITY_MISMATCH_HOSTILE_SPEED = 16.0   # max of HOSTILE_SPEED_VALUES
_CLUTTER_OBSTACLE_COUNT = 8


@dataclass(frozen=True)
class DiagnosticScenario:
    name: str
    description: str
    config: EnvConfig


def build_scenario_battery() -> tuple[DiagnosticScenario, ...]:
    """The 6 conceptual scenarios from the Phase-10 specification."""
    nominal = DiagnosticScenario("nominal_open", "Nominal parameters, no obstacles", EnvConfig())

    strong_maneuver = DiagnosticScenario(
        "strong_maneuver",
        "Restrictive defender acceleration + turn-rate (hardest defender-dynamics point on the established grid)",
        EnvConfig(defender_max_accel=_RESTRICTIVE_ACCEL, defender_max_turn_rate_deg=_RESTRICTIVE_TURN_RATE_DEG),
    )

    strong_evasion = DiagnosticScenario(
        "strong_evasion",
        "Maximum hostile reactive-evasion gain from the established evasion-gain grid",
        EnvConfig(enemy_evasion_gain=_STRONG_EVASION_GAIN),
    )

    mobility_mismatch = DiagnosticScenario(
        "mobility_mismatch",
        "Slowest defender / fastest hostile from the established mobility grid",
        EnvConfig(v_d=_MOBILITY_MISMATCH_DEFENDER_SPEED, v_e=_MOBILITY_MISMATCH_HOSTILE_SPEED),
    )

    clutter = DiagnosticScenario(
        "clutter",
        "Frozen clutter-env-v1 obstacle navigation, random layout, hostile obstacle avoidance enabled",
        EnvConfig(
            obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=_CLUTTER_OBSTACLE_COUNT,
            enemy_obstacle_avoidance_enabled=True,
        ),
    )

    clutter_strong_maneuver = DiagnosticScenario(
        "clutter_strong_maneuver",
        "Clutter combined with restrictive defender dynamics",
        EnvConfig(
            obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=_CLUTTER_OBSTACLE_COUNT,
            enemy_obstacle_avoidance_enabled=True,
            defender_max_accel=_RESTRICTIVE_ACCEL, defender_max_turn_rate_deg=_RESTRICTIVE_TURN_RATE_DEG,
        ),
    )

    return (nominal, strong_maneuver, strong_evasion, mobility_mismatch, clutter, clutter_strong_maneuver)
