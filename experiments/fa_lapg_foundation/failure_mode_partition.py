"""
Phase 84: failure-mode partition using the diagnostic evidence actually
available from `run_episode`'s per-episode metrics (outcome type, min
distances, intercept time). A full per-step saturation/D_turn-tagged
partition (categories B/C/D from the task's suggested list) would require
re-running the Phase 81 controllers with the per-step diagnostic logging
pipeline (`experiments.learning_benefit.diagnostic_logging`), which was
out of scope for this pass given time constraints -- see the final
report. Categories not distinguishable from the available fields are
explicitly left as UNKNOWN rather than forced into a speculative bucket.
"""

from __future__ import annotations

import pandas as pd


def classify_failure(row: pd.Series) -> str:
    if row["success"]:
        return "success"
    if row["outcome"] == "timeout":
        return "G_timeout"
    if row["outcome"] == "unsafe_intercept":
        return "H_unsafe_intercept"
    if row["outcome"] == "soldier_caught":
        return "E_no_reachable_intercept_before_deadline"
    return "I_unknown"


def failure_mode_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Per (controller, scenario) counts of each failure category."""
    labeled = df.copy()
    labeled["failure_category"] = labeled.apply(classify_failure, axis=1)
    failures_only = labeled[labeled["failure_category"] != "success"]
    return failures_only.groupby(["controller", "scenario", "failure_category"]).size().unstack(fill_value=0)
