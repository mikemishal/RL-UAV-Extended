"""Campaign tooling (evaluation tracing, integrity check, analysis helpers). No PPO training is executed."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

import experiments.journal_v2_a1_campaign_analysis as AN
import experiments.journal_v2_a1_campaign_eval as E
from experiments import journal_v2_a1 as A


def synthetic_records(a1_capture=0.5, a0_capture=0.5, n_seeds=4):
    records = []
    for attacker, rate in (("A0", a0_capture), *((a, a1_capture) for a in AN.A1_IDS)):
        for level in A.A1_V2_CLUTTER_LEVELS:
            for family in A.A1_V2_DEFENDER_FAMILIES:
                for seed in range(n_seeds):
                    captured = seed < rate * n_seeds
                    outcome = "soldier_caught" if captured else "enemy_endurance_exhausted"
                    records.append(dict(
                        attacker=attacker, level=level, defender=family, seed=140000 + seed, outcome=outcome, steps=40, flight_time_s=20.0,
                        path_length_m=60.0, chord_m=50.0, path_excess=1.2, initial_range_m=50.0, min_defender_separation_m=20.0,
                        obstacle_collision=False, endurance_exhausted=not captured, captured=captured, overshoots=0,
                        orbit_revolutions_max=0.0, avoidance_steps=0, return_undiscounted=11.0 if captured else -10.0, reward_components={}))
    return records


def test_angle_deg_basic_cases():
    assert AN.angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert AN.angle_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert math.isnan(AN.angle_deg([0, 0, 0], [1, 0, 0]))


def test_group_metrics_and_cell_table_shapes():
    df = AN.frame(synthetic_records(0.5, 0.5))
    table = AN.cell_table(df)
    assert len(table) == 4 * 5 * 4 and set(table.attacker) == {"A0", *AN.A1_IDS}
    metrics = AN.group_metrics(df[(df.attacker == "A0") & (df.level == "C0")])
    assert metrics["capture_rate"] == pytest.approx(0.5) and metrics["endurance_exhaustion_rate"] == pytest.approx(0.5)
    assert metrics["gt1.3x"] == 0 and metrics["median_path_excess"] == pytest.approx(1.2)


def test_paired_bootstrap_uses_matched_pairs_and_repo_convention():
    assert AN.BOOTSTRAP_REPLICATES == 10_000 and AN.BOOTSTRAP_SEED == 20260816
    df = AN.frame(synthetic_records(a1_capture=0.75, a0_capture=0.25))
    table = AN.paired_table(df)
    overall = table[(table.scope == "overall") & (table.attacker == "A1-201")].iloc[0]
    assert overall.pairs == 4 * 5 * 4 and overall.diff_pp == pytest.approx(50.0)
    assert {"cell", "by_level_all_defenders", "by_defender_all_levels", "overall"} <= set(table.scope)
    assert len(table[(table.scope == "cell") & (table.attacker == "A1-203")]) == 20


def test_integrity_check_rejects_missing_or_unmatched_episodes():
    records = []
    for attacker in E.ATTACKERS:
        for level in E.LEVELS:
            for family in E.FAMILIES:
                for seed in E.SEEDS:
                    records.append(dict(attacker=attacker, level=level, defender=family, seed=seed, outcome="timeout", initial_range_m=50.0))
    assert E.integrity_check(records)["episodes"] == 8000
    with pytest.raises(AssertionError):
        E.integrity_check(records[:-1])
    records[5]["initial_range_m"] = 51.0
    with pytest.raises(AssertionError):
        E.integrity_check(records)


def test_trace_env_records_a0_episode_without_changing_the_result(tmp_path):
    task = ("A0", "C1", "greedy", (140000, 140001), str(tmp_path))
    result = E.run_task(task)
    assert result["episodes"] == 2
    with open(tmp_path / "A0__C1__greedy.pkl", "rb") as handle:
        payload = pickle.load(handle)
    plain = E.run_episode(None, "A0", "C1", "greedy", 140000)
    record = payload["records"][0]
    assert record["outcome"] == plain["outcome"] and record["steps"] == plain["steps"]
    trace = payload["traces"][140000]
    assert len(trace) == record["steps"] and trace[-1]["outcome"] == record["outcome"]
    assert trace[0]["obs_before"].shape == (37,) and "perceived_static_ids" in trace[0] and trace[-1]["reward"] == pytest.approx(
        trace[-1]["reward"])
    assert E.run_task(task)["skipped"] is True  # resumable: finished tasks are not recomputed


def test_derive_and_discounted_return_by_outcome():
    gamma = A.A1_V2_PPO_HYPERPARAMS["gamma"]
    trace = [{"reward": 1.0, "enemy_pos": [i, 0.0, 0.0], "actual_min_clearance": 5.0 - i} for i in range(3)]
    derived = AN.derive({}, trace)
    assert derived["discounted_return"] == pytest.approx(1 + gamma + gamma ** 2)
    assert derived["lateral_excursion_m"] == pytest.approx(0.0) and derived["min_clearance_all_m"] == 3.0
    df = AN.frame([dict(attacker="A1-201", level="C0", defender="greedy", seed=1, outcome="soldier_caught", steps=3, flight_time_s=1.0,
                        path_length_m=1.0, chord_m=1.0, path_excess=1.0, initial_range_m=1.0, min_defender_separation_m=1.0,
                        obstacle_collision=False, endurance_exhausted=False, captured=True, overshoots=0, orbit_revolutions_max=0.0,
                        avoidance_steps=0, return_undiscounted=3.0, reward_components={}, **derived)])
    row = AN.discounted_return_by_outcome(df)[0]
    assert row["mean_undiscounted"] == pytest.approx(3.0) and row["mean_discounted"] == pytest.approx(1 + gamma + gamma ** 2)


def test_late_failure_is_cheaper_than_early_failure_under_the_frozen_gamma():
    """Documents a property of the frozen reward: a -10 terminal 150 steps away is discounted to about -2."""
    gamma = A.A1_V2_PPO_HYPERPARAMS["gamma"]
    assert -10 * gamma ** 150 > -2.5 > -10 * gamma ** 12
