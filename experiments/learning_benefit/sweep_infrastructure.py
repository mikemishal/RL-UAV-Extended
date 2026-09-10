"""Phase 16: reusable sweep-grid SCAFFOLDING for later learning-benefit
heatmap experiments. NOT executed as a large sweep in this task -- only
the grid-point definition / single-point runner interface is provided so
a future task can drive it.

Supported grid axes (paired, per the Phase-16 specification):
  - hostile weave amplitude x hostile turn rate
  - evasion gain x defender/hostile speed ratio
  - measurement noise x hostile maneuverability
  - obstacle density x mobility ratio
  - detection radius x maneuver strength
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from uav_defend.config.env_config import EnvConfig


@dataclass(frozen=True)
class SweepAxis:
    """One named grid axis: a config-field name and the values to sweep."""

    name: str
    config_field: str
    values: tuple


@dataclass(frozen=True)
class SweepGridPoint:
    axis_a: SweepAxis
    value_a: float
    axis_b: SweepAxis
    value_b: float

    def build_config(self, base_config: EnvConfig | None = None) -> EnvConfig:
        import dataclasses
        base = base_config if base_config is not None else EnvConfig()
        return dataclasses.replace(base, **{self.axis_a.config_field: self.value_a, self.axis_b.config_field: self.value_b})


# Paired grid axis definitions (values reused from existing protocols where
# established; NOT executed here -- see module docstring).
WEAVE_AMPLITUDE_AXIS = SweepAxis("hostile_weave_amplitude", "weave_amplitude", (0.5, 1.0, 1.5, 2.0))
HOSTILE_TURN_RATE_AXIS = SweepAxis("hostile_turn_rate_deg", "enemy_max_turn_rate_deg", (45.0, 75.0, 105.0))
EVASION_GAIN_AXIS = SweepAxis("evasion_gain", "enemy_evasion_gain", (0.0, 0.25, 0.5, 0.75, 1.0))
MEASUREMENT_NOISE_AXIS = SweepAxis("measurement_var", "measurement_var", (0.05, 0.25, 0.5, 1.0, 2.0, 4.0))
OBSTACLE_COUNT_AXIS = SweepAxis("obstacle_density", "obstacle_count", (0, 4, 8, 12, 16))
DETECTION_RADIUS_AXIS = SweepAxis("detection_radius", "detection_radius", (5.0, 8.0, 10.0, 12.0, 15.0))


def build_grid(axis_a: SweepAxis, axis_b: SweepAxis) -> list[SweepGridPoint]:
    """Cartesian product of two sweep axes -- scaffolding only, not executed."""
    return [
        SweepGridPoint(axis_a, va, axis_b, vb)
        for va in axis_a.values
        for vb in axis_b.values
    ]


def evaluate_grid_point(
    grid_point: SweepGridPoint,
    run_controller_fn: Callable[[EnvConfig, str], float],
    controller_lr_ppo: str = "lr_ppo",
    controller_lead: str = "lead",
    base_config: EnvConfig | None = None,
) -> dict:
    """
    Interface stub for a future sweep runner: `run_controller_fn(config,
    controller_name) -> success_rate` is supplied by the caller (e.g. a
    matched-seed evaluation over N episodes); this function only wires the
    grid-point config and computes:

        Delta_LR-Lead = success_LR-PPO - success_Lead

    NOT executed as part of this task (Phase 16 is scaffolding only).
    """
    config = grid_point.build_config(base_config)
    success_lead = run_controller_fn(config, controller_lead)
    success_lr_ppo = run_controller_fn(config, controller_lr_ppo)
    return {
        "axis_a": grid_point.axis_a.name, "value_a": grid_point.value_a,
        "axis_b": grid_point.axis_b.name, "value_b": grid_point.value_b,
        "success_lead": success_lead, "success_lr_ppo": success_lr_ppo,
        "delta_lr_lead": success_lr_ppo - success_lead,
    }
