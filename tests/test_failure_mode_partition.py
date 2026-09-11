"""Tests for Phase 84 failure-mode classification."""

import pandas as pd

from experiments.fa_lapg_foundation.failure_mode_partition import classify_failure, failure_mode_partition


def test_classify_failure_success_row():
    row = pd.Series({"success": True, "outcome": "intercepted"})
    assert classify_failure(row) == "success"


def test_classify_failure_timeout():
    row = pd.Series({"success": False, "outcome": "timeout"})
    assert classify_failure(row) == "G_timeout"


def test_classify_failure_unsafe_intercept():
    row = pd.Series({"success": False, "outcome": "unsafe_intercept"})
    assert classify_failure(row) == "H_unsafe_intercept"


def test_classify_failure_soldier_caught_maps_to_deadline_category():
    row = pd.Series({"success": False, "outcome": "soldier_caught"})
    assert classify_failure(row) == "E_no_reachable_intercept_before_deadline"


def test_classify_failure_unknown_outcome_falls_back_to_unknown():
    row = pd.Series({"success": False, "outcome": "something_new"})
    assert classify_failure(row) == "I_unknown"


def test_failure_mode_partition_excludes_successes_and_groups_correctly():
    df = pd.DataFrame([
        {"controller": "lead", "scenario": "s", "success": True, "outcome": "intercepted"},
        {"controller": "lead", "scenario": "s", "success": False, "outcome": "timeout"},
        {"controller": "lead", "scenario": "s", "success": False, "outcome": "timeout"},
        {"controller": "ra_lapg", "scenario": "s", "success": False, "outcome": "unsafe_intercept"},
    ])
    partition = failure_mode_partition(df)
    assert partition.loc[("lead", "s"), "G_timeout"] == 2
    assert partition.loc[("ra_lapg", "s"), "H_unsafe_intercept"] == 1
