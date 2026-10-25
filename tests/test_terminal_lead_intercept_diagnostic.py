"""Pure-math unit tests for the capture-assured terminal lead-intercept diagnostic
(experiments/terminal_lead_intercept_diagnostic.py). See
docs/journal_v2_terminal_lead_intercept.md for why this design was ultimately rejected (not
wired into production); these tests cover only the interceptor geometry itself, which is correct
and kept as a diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.terminal_lead_intercept_diagnostic import lead_direction, solve_lead_intercept_time


def test_stationary_asset_reduces_to_pure_pursuit():
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([10.0, 0.0, 0.0])
    direction = lead_direction(position, np.zeros(3), asset, speed=5.0, fallback=np.array([1.0, 0.0, 0.0]), eps=1e-8)
    assert np.allclose(direction, [1.0, 0.0, 0.0])


def test_solved_time_satisfies_the_intercept_equation():
    r = np.array([20.0, 0.0, 0.0])
    v_asset = np.array([0.0, 2.0, 0.0])  # moving perpendicular to the current LOS
    speed = 8.0
    t = solve_lead_intercept_time(r, v_asset, speed, eps=1e-8)
    assert t is not None and t > 0
    assert np.linalg.norm(r + v_asset * t) == pytest.approx(speed * t, rel=1e-6)


def test_no_solution_when_the_asset_outruns_the_attacker_directly_away():
    r = np.array([10.0, 0.0, 0.0])
    v_asset = np.array([20.0, 0.0, 0.0])  # fleeing directly away faster than the attacker can fly
    assert solve_lead_intercept_time(r, v_asset, speed=8.0, eps=1e-8) is None


def test_lead_direction_falls_back_to_pursuit_when_no_solution_exists():
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([10.0, 0.0, 0.0])
    v_asset = np.array([20.0, 0.0, 0.0])
    fallback = np.array([1.0, 0.0, 0.0])
    direction = lead_direction(position, v_asset, asset, speed=8.0, fallback=fallback, eps=1e-8)
    assert np.array_equal(direction, fallback)


def test_zero_speed_is_a_degenerate_hold():
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([10.0, 0.0, 0.0])
    direction = lead_direction(position, np.zeros(3), asset, speed=0.0, fallback=np.array([1.0, 0.0, 0.0]), eps=1e-8)
    assert np.array_equal(direction, np.zeros(3))


def test_lead_point_is_reached_simultaneously_by_both_bodies():
    """The classic lead-pursuit sanity check: attacker and asset arrive at the intercept point at
    the same predicted time t."""
    position = np.array([0.0, 0.0, 0.0])
    asset = np.array([30.0, -10.0, 0.0])
    v_asset = np.array([-1.0, 1.5, 0.0])
    speed = 9.0
    r = asset - position
    t = solve_lead_intercept_time(r, v_asset, speed, eps=1e-8)
    assert t is not None
    intercept_point = asset + v_asset * t
    assert np.linalg.norm(intercept_point - position) == pytest.approx(speed * t, rel=1e-6)
    assert np.linalg.norm((asset + v_asset * t) - intercept_point) == pytest.approx(0.0, abs=1e-9)
