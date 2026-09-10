"""Tests for LR residual-angle analysis (Phase 9 / 19).

Covers:
 - zero residual angle (u_final == u_Lead)
 - known angular difference (90 deg, 180 deg)
 - clipping / numerical stability for near-parallel vectors with roundoff
 - degenerate (zero-norm) inputs return None rather than fabricating
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.learning_benefit.residual_analysis import correlate, residual_angle_deg, summarize_by_group


def test_zero_residual_angle():
    u = np.array([1.0, 0.0, 0.0])
    assert abs(residual_angle_deg(u, u) - 0.0) < 1e-9


def test_known_90_degree_difference():
    u_lead = np.array([1.0, 0.0, 0.0])
    u_final = np.array([0.0, 1.0, 0.0])
    assert abs(residual_angle_deg(u_lead, u_final) - 90.0) < 1e-6


def test_known_180_degree_difference():
    u_lead = np.array([1.0, 0.0, 0.0])
    u_final = np.array([-1.0, 0.0, 0.0])
    assert abs(residual_angle_deg(u_lead, u_final) - 180.0) < 1e-6


def test_clipping_handles_roundoff_beyond_unit_range():
    # Slightly-more-than-parallel due to floating point should clip to 0 deg, not NaN.
    u = np.array([1.0, 0.0, 0.0])
    u_scaled = u * (1.0 + 1e-16)
    angle = residual_angle_deg(u, u_scaled)
    assert angle is not None
    assert angle < 1e-3


def test_degenerate_zero_vector_returns_none():
    assert residual_angle_deg(np.zeros(3), np.array([1.0, 0.0, 0.0])) is None
    assert residual_angle_deg(np.array([1.0, 0.0, 0.0]), np.zeros(3)) is None


def test_correlate_perfect_positive():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    result = correlate(x, y)
    assert result["n"] == 4
    assert abs(result["r"] - 1.0) < 1e-9


def test_correlate_drops_nan_pairs():
    x = pd.Series([1.0, 2.0, np.nan, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, np.nan])
    result = correlate(x, y)
    assert result["n"] == 2


def test_summarize_by_group():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]})
    summary = summarize_by_group(df, "g", "v")
    row_a = summary[summary["g"] == "a"].iloc[0]
    assert row_a["count"] == 2
    assert abs(row_a["mean"] - 2.0) < 1e-9


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
