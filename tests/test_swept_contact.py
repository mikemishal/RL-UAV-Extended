"""Independent tests for the swept moving-sphere contact helper."""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.obstacles.swept_contact import CONTACT_DISTANCE_TOLERANCE, first_contact


def _v(*xyz):
    return np.array(xyz, dtype=np.float64)


def test_high_speed_pass_through_both_endpoints_outside():
    hit = first_contact(_v(-3, 0, 0), _v(3, 0, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0)
    assert hit is not None
    assert hit.fraction == pytest.approx(1.0 / 6.0)
    assert hit.distance == pytest.approx(2.0)
    assert np.allclose(hit.position_a, [-2, 0, 0])


def test_no_intersection_offset_path():
    assert first_contact(_v(-3, 3, 0), _v(3, 3, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0) is None


def test_already_inside_at_tau_zero():
    hit = first_contact(_v(1, 0, 0), _v(10, 0, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0)
    assert hit is not None and hit.fraction == 0.0
    assert hit.distance == pytest.approx(1.0)


def test_zero_relative_displacement_outside_and_inside():
    p, q = _v(5, 0, 0), _v(0, 0, 0)
    assert first_contact(p, p, q, q, 2.0) is None
    inside = _v(1, 0, 0)
    assert first_contact(inside, inside, q, q, 2.0).fraction == 0.0
    # Both move identically: relative displacement is zero even though each moves.
    assert first_contact(_v(5, 0, 0), _v(8, 0, 0), _v(0, 0, 0), _v(3, 0, 0), 2.0) is None


def test_tangent_contact_is_detected():
    hit = first_contact(_v(-3, 2, 0), _v(3, 2, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0)
    assert hit is not None
    assert hit.fraction == pytest.approx(0.5)
    assert hit.distance == pytest.approx(2.0, abs=1e-9)


def test_tangent_within_floating_point_noise():
    offset = 2.0 + 0.5 * CONTACT_DISTANCE_TOLERANCE
    assert first_contact(_v(-3, offset, 0), _v(3, offset, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0) is not None
    assert first_contact(_v(-3, 2.0 + 1e-6, 0), _v(3, 2.0 + 1e-6, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0) is None


def test_target_moving_creates_contact_a_static_target_would_miss():
    # Attacker passes 2.5 m from the asset's START position (a miss against
    # a static target) but the asset moves toward the path.
    a0, a1 = _v(-3, 2.5, 0), _v(3, 2.5, 0)
    b0, b1 = _v(0, 0, 0), _v(0, 1.5, 0)
    assert first_contact(a0, a1, b0, b0, 2.0) is None
    assert first_contact(a0, a1, b0, b1, 2.0) is not None
    hit = first_contact(a0, a1, b0, b1, 2.0)
    assert 0.0 < hit.fraction < 1.0
    assert hit.distance == pytest.approx(2.0, abs=1e-9)


def test_both_moving_head_on():
    hit = first_contact(_v(-10, 0, 0), _v(0, 0, 0), _v(10, 0, 0), _v(0, 0, 0), 2.0)
    assert hit.fraction == pytest.approx(0.9)  # relative separation 20 -> 2 m over tau in [0, 1]


def test_contact_beyond_end_of_step_is_not_a_contact():
    assert first_contact(_v(-10, 0, 0), _v(-5, 0, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0) is None


def test_moving_away_never_contacts():
    assert first_contact(_v(5, 0, 0), _v(20, 0, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0) is None


def test_endpoint_exactly_on_sphere_surface():
    hit = first_contact(_v(-5, 0, 0), _v(-2, 0, 0), _v(0, 0, 0), _v(0, 0, 0), 2.0)
    assert hit is not None and hit.fraction == pytest.approx(1.0)


def test_three_dimensional_geometry_and_fraction_is_first_entry():
    hit = first_contact(_v(0, 0, 20), _v(0, 0, -20), _v(0, 0, 0), _v(0, 0, 0), 2.5)
    assert hit.fraction == pytest.approx((20 - 2.5) / 40.0)
    assert np.allclose(hit.position_a, [0, 0, 2.5])


def test_fraction_is_within_unit_interval_for_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p0a, p1a, p0b, p1b = (rng.uniform(-10, 10, 3) for _ in range(4))
        hit = first_contact(p0a, p1a, p0b, p1b, 2.0)
        if hit is not None:
            assert 0.0 <= hit.fraction <= 1.0
            assert hit.distance <= 2.0 + 1e-7


def test_matches_dense_sampling_reference():
    rng = np.random.default_rng(1)
    taus = np.linspace(0.0, 1.0, 20001)
    for _ in range(200):
        p0a, p1a, p0b, p1b = (rng.uniform(-8, 8, 3) for _ in range(4))
        rel = (p0a - p0b)[None, :] + taus[:, None] * ((p1a - p0a) - (p1b - p0b))[None, :]
        inside = np.linalg.norm(rel, axis=1) <= 2.0
        hit = first_contact(p0a, p1a, p0b, p1b, 2.0)
        if inside.any():
            assert hit is not None
            assert hit.fraction == pytest.approx(taus[np.argmax(inside)], abs=1e-4)
        else:
            # A grazing contact can fall between samples; it must be within noise of the surface.
            assert hit is None or hit.distance == pytest.approx(2.0, abs=1e-3)


def test_negative_radius_rejected():
    with pytest.raises(ValueError):
        first_contact(_v(0, 0, 0), _v(1, 0, 0), _v(5, 0, 0), _v(5, 0, 0), -1.0)
