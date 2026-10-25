"""Pure-math unit tests for the reactive terminal reacquisition diagnostic
(experiments/terminal_reacquisition_diagnostic.py). See
docs/journal_v2_terminal_reacquisition.md for why this design was ultimately rejected (not wired
into production); these tests cover only the detector/curvature-law geometry itself, which is
correct and kept as a diagnostic."""

from __future__ import annotations

import numpy as np

from experiments.terminal_reacquisition_diagnostic import (
    closing_velocity,
    curvature_speed,
    curvature_speed_decel_aware,
    flyby_detected,
    heading_error_degrees,
    reacquisition_achieved,
    turning_radius,
)
from uav_defend.obstacles.avoidance import DynamicsLimits

LIMITS = DynamicsLimits(
    max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0), max_climb_rate=5.0, max_descent_rate=5.0,
    dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0,
)


def test_closing_velocity_sign_matches_approach_and_recede():
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([10.0, 0.0, 0.0])
    approaching = closing_velocity(position, np.array([5.0, 0.0, 0.0]), asset, 1e-8)
    receding = closing_velocity(position, np.array([-5.0, 0.0, 0.0]), asset, 1e-8)
    assert approaching > 0 and receding < 0


def test_heading_error_zero_when_pointed_directly_at_the_asset():
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([10.0, 0.0, 0.0])
    assert heading_error_degrees(np.array([5.0, 0.0, 0.0]), position, asset, 1e-8) < 1e-6
    assert heading_error_degrees(np.array([-5.0, 0.0, 0.0]), position, asset, 1e-8) > 179.0


def test_turning_radius_scales_linearly_with_speed():
    assert turning_radius(12.0, LIMITS.max_turn_rate_rad, LIMITS.eps) == 2 * turning_radius(6.0, LIMITS.max_turn_rate_rad, LIMITS.eps)


def test_flyby_requires_a_closing_to_opening_transition():
    # was closing (prev>0), now opening (now<=0), close enough to be relevant -> detected
    assert flyby_detected(5.0, -1.0, range_now=3.0, speed_now=10.0, capture_radius=2.0, limits=LIMITS)
    # still closing -> not a fly-by
    assert not flyby_detected(5.0, 1.0, range_now=3.0, speed_now=10.0, capture_radius=2.0, limits=LIMITS)
    # no prior closing step recorded -> cannot detect a transition
    assert not flyby_detected(None, -1.0, range_now=3.0, speed_now=10.0, capture_radius=2.0, limits=LIMITS)


def test_flyby_ignores_geometrically_irrelevant_distant_misses():
    # local minimum far beyond the vehicle's own turning radius: an ordinary wide pass, not a fly-by
    r_turn = turning_radius(10.0, LIMITS.max_turn_rate_rad, LIMITS.eps)
    assert not flyby_detected(5.0, -1.0, range_now=2.0 + r_turn + 5.0, speed_now=10.0, capture_radius=2.0, limits=LIMITS)


def test_curvature_speed_is_zero_at_the_capture_boundary_and_grows_with_range():
    assert curvature_speed(2.0, capture_radius=2.0, max_speed=12.0, limits=LIMITS) == 0.0
    near = curvature_speed(3.0, capture_radius=2.0, max_speed=12.0, limits=LIMITS)
    far = curvature_speed(6.0, capture_radius=2.0, max_speed=12.0, limits=LIMITS)
    assert 0.0 < near < far <= 12.0


def test_curvature_speed_never_exceeds_max_speed():
    assert curvature_speed(1000.0, capture_radius=2.0, max_speed=12.0, limits=LIMITS) == 12.0


def test_decel_aware_speed_is_never_higher_than_the_instantaneous_law():
    """Variant B accounts for extra distance consumed while slowing down, so it can only be more
    conservative (lower) than Variant A's instantaneous law, never higher."""
    range_now = 5.0
    instantaneous = curvature_speed(range_now, 2.0, 12.0, LIMITS)
    decel_aware = curvature_speed_decel_aware(range_now, 2.0, current_speed=10.5, max_speed=12.0, limits=LIMITS)
    assert decel_aware <= instantaneous + 1e-9


def test_decel_aware_speed_never_exceeds_current_speed():
    decel_aware = curvature_speed_decel_aware(3.0, 2.0, current_speed=10.5, max_speed=12.0, limits=LIMITS)
    assert decel_aware <= 10.5 + 1e-9


def test_reacquisition_exit_requires_closing_and_facing_the_asset():
    assert reacquisition_achieved(closing_now=1.0, heading_error_deg=10.0)
    assert not reacquisition_achieved(closing_now=-1.0, heading_error_deg=10.0)
    assert not reacquisition_achieved(closing_now=1.0, heading_error_deg=150.0)
