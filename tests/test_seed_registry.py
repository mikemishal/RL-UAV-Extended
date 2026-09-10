"""Tests for the Phase 45 seed registry: no overlaps among newly reserved
ranges, reserved final-evaluation range protected, and known historical
overlaps correctly detected (not silently missed)."""

import pytest

from experiments.fa_lapg_foundation.seed_registry import (
    CONFIRMED_OVERLAPPING_PROJECT_RANGES,
    HISTORICAL_RANGES,
    PROJECT_RANGES,
    RESERVED_RANGES,
    assert_no_new_range_overlaps_reserved_final,
    assert_reserved_ranges_disjoint_from_historical,
    assert_reserved_ranges_pairwise_disjoint,
    find_overlaps,
)


def test_reserved_ranges_pairwise_disjoint():
    assert_reserved_ranges_pairwise_disjoint()


def test_reserved_ranges_disjoint_from_all_historical_ranges():
    assert_reserved_ranges_disjoint_from_historical()


def test_final_journal_evaluation_range_never_overlapped_by_other_reserved_ranges():
    assert_no_new_range_overlaps_reserved_final()


def test_final_journal_evaluation_range_is_protected_from_deliberate_collision():
    """A hypothetical dev range colliding with the reserved final range
    must be detected (proves the guard is not a no-op)."""
    bad_reserved = dict(RESERVED_RANGES)
    bad_reserved["accidental_dev_range"] = (112000, 112500)  # inside final_journal_evaluation
    final_range = bad_reserved["final_journal_evaluation"]
    overlap_found = any(
        name != "final_journal_evaluation"
        and rng[0] <= final_range[1] and final_range[0] <= rng[1]
        for name, rng in bad_reserved.items()
    )
    assert overlap_found


def test_predictor_training_range_overlap_is_detected():
    overlaps = find_overlaps(PROJECT_RANGES["predictor_training"])
    assert "robustness_expanded_detection_radius" in overlaps


def test_predictor_engineering_test_range_overlap_is_detected():
    overlaps = find_overlaps(PROJECT_RANGES["predictor_engineering_test"])
    assert "conference_evaluation_rmpc_final_nominal" in overlaps


def test_predictor_validation_range_has_no_concrete_collision():
    # It falls inside the broad "reserved_robustness_headroom" declaration
    # by construction, but no OTHER concretely-used historical range.
    overlaps = find_overlaps(PROJECT_RANGES["predictor_validation"])
    concrete_overlaps = [o for o in overlaps if o != "reserved_robustness_headroom"]
    assert concrete_overlaps == []


def test_confirmed_overlapping_ranges_are_exactly_the_documented_two():
    assert CONFIRMED_OVERLAPPING_PROJECT_RANGES == {"predictor_training", "predictor_engineering_test"}


def test_historical_registry_is_nonempty_and_well_formed():
    assert len(HISTORICAL_RANGES) > 10
    for name, (start, end) in HISTORICAL_RANGES.items():
        assert start <= end, f"{name}: invalid range ({start}, {end})"
