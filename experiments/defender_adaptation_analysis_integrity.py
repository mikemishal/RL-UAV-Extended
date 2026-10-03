"""
Phase 1-3 analysis-input integrity/audit for the defender-adaptation
journal results (journal extension, publication sub-task). Pure
read-only verification of the ALREADY-COMPLETE `episodes.jsonl`/
`episodes.csv` -- never modifies episode data, never reruns episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.defender_adaptation_evaluation import (
    ARCHITECTURES, ATTACKER_CONDITIONS, EVAL_LEVELS, EVAL_SEEDS, OUT_DIR, VALID_OUTCOMES,
    build_evaluation_manifest, load_completed_episodes,
)

OUTCOME_COLUMNS = (
    "defender_safe_intercept", "attacker_asset_reach", "unsafe_intercept", "timeout",
    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
)


def phase1_integrity_check() -> dict:
    jsonl_path = OUT_DIR / "episodes.jsonl"
    manifest = build_evaluation_manifest()
    expected_keys = {e["key"] for e in manifest["entries"]}
    completed = load_completed_episodes(jsonl_path)  # raises on conflicting duplicates / non-final malformed lines

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    n_raw_lines = len([l for l in lines if l.strip()])

    missing = expected_keys - set(completed.keys())
    extra = set(completed.keys()) - expected_keys

    df = pd.read_csv(OUT_DIR / "episodes.csv")

    # 27 defender instances present.
    defender_keys_present = set(zip(df.architecture, df.defender_seed, df.generation))
    expected_defender_keys = {
        (arch, seed, gen) for arch, spec in ARCHITECTURES.items() for seed in spec["seeds"] for gen in ("d0", "d0_plus", "d1")
    }

    # 4 attacker conditions present.
    attacker_types_present = set(df.attacker_type.unique())
    expected_attacker_types = {name for name, _ in ATTACKER_CONDITIONS}

    # C1/C2/C3 complete.
    levels_present = set(df.level.unique())

    # Exactly 100 eval seeds per (architecture, defender_seed, generation, attacker_type, level) cell.
    cell_counts = df.groupby(["architecture", "defender_seed", "generation", "attacker_type", "level"]).size()
    cells_not_100 = cell_counts[cell_counts != 100]

    report = {
        "total_expected_keys": len(expected_keys),
        "total_unique_completed_keys": len(completed),
        "n_raw_jsonl_lines": n_raw_lines,
        "missing_keys_count": len(missing),
        "unexpected_extra_keys_count": len(extra),
        "no_duplicate_conflicting_keys": True,  # load_completed_episodes() would have raised otherwise
        "no_malformed_non_final_records": True,  # ditto
        "episodes_csv_row_count": len(df),
        "defender_instances_present": len(defender_keys_present),
        "defender_instances_expected": len(expected_defender_keys),
        "defender_instances_match": defender_keys_present == expected_defender_keys,
        "attacker_conditions_present": sorted(attacker_types_present),
        "attacker_conditions_expected": sorted(expected_attacker_types),
        "attacker_conditions_match": attacker_types_present == expected_attacker_types,
        "levels_present": sorted(levels_present),
        "levels_expected": sorted(EVAL_LEVELS),
        "levels_match": levels_present == set(EVAL_LEVELS),
        "cells_with_wrong_episode_count": len(cells_not_100),
        "all_cells_have_exactly_100_episodes": len(cells_not_100) == 0,
        "outcome_values_all_valid": bool(df["outcome"].isin(VALID_OUTCOMES).all()),
        "outcome_sum_equals_n_check": bool((df[list(OUTCOME_COLUMNS)].sum(axis=1) == 1).all()),
    }
    return report


def phase2_factorial_audit_table() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "episodes.csv")
    rows = []
    for arch, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            for generation in ("d0", "d0_plus", "d1"):
                for attacker_name, _ in ATTACKER_CONDITIONS:
                    for level in EVAL_LEVELS:
                        n = len(df[(df.architecture == arch) & (df.defender_seed == seed) & (df.generation == generation)
                                   & (df.attacker_type == attacker_name) & (df.level == level)])
                        rows.append({
                            "architecture": arch, "defender_seed": seed, "generation": generation,
                            "attacker_type": attacker_name, "level": level,
                            "expected_n": 100, "observed_n": n, "match": n == 100,
                        })
    return pd.DataFrame(rows)


def phase3_outcome_integrity() -> dict:
    df = pd.read_csv(OUT_DIR / "episodes.csv")
    sums = df[list(OUTCOME_COLUMNS)].sum(axis=1)
    return {
        "n_episodes": len(df),
        "all_rows_sum_to_1": bool((sums == 1).all()),
        "rows_summing_to_not_1": int((sums != 1).sum()),
        "outcome_category_totals": {col: int(df[col].sum()) for col in OUTCOME_COLUMNS},
        "outcome_category_totals_sum": int(sum(int(df[col].sum()) for col in OUTCOME_COLUMNS)),
        "expected_total": len(df),
    }


if __name__ == "__main__":
    integrity = phase1_integrity_check()
    (OUT_DIR / "analysis_input_integrity.json").write_text(json.dumps(integrity, indent=2, default=str))
    print(json.dumps(integrity, indent=2))

    factorial = phase2_factorial_audit_table()
    factorial.to_csv(OUT_DIR / "factorial_design_audit.csv", index=False)
    print("factorial audit: all match =", bool(factorial["match"].all()), "rows =", len(factorial))

    outcome = phase3_outcome_integrity()
    (OUT_DIR / "outcome_integrity.json").write_text(json.dumps(outcome, indent=2, default=str))
    print(json.dumps(outcome, indent=2))
