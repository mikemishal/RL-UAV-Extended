"""Phase 14/15: paired statistical comparison utilities.

Reuses the SAME `BOOTSTRAP_SEED = 20260816` convention already established
elsewhere in this repository's robustness protocols, for a fixed,
reproducible paired bootstrap.

Episode-level clustering (Phase 15): per-step diagnostic rows are
correlated within an episode, so treating every row as an independent
Bernoulli/continuous sample would understate variance (pseudoreplication).
`episode_level_aggregate()` collapses each episode to ONE summary value
first; only those per-episode values are then treated as i.i.d. samples
for confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPLICATES = 10_000


def episode_level_aggregate(df: pd.DataFrame, episode_column: str, value_column: str, how: str = "mean") -> np.ndarray:
    """Collapse a per-step DataFrame to ONE value per episode (default:
    mean of `value_column` within each `episode_column` group, dropping
    NaNs), returning the resulting per-episode array. Use THIS array (not
    the raw per-step values) for confidence intervals / hypothesis tests."""
    valid = df[df[value_column].notna()]
    grouped = valid.groupby(episode_column)[value_column].agg(how)
    return grouped.to_numpy(dtype=np.float64)


def paired_bootstrap_difference(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """
    Deterministic paired bootstrap for the difference in means
    mean(outcomes_b) - mean(outcomes_a), where `outcomes_a`/`outcomes_b`
    are MATCHED (same length, same episode/seed order -- e.g. success
    indicators or continuous metrics for the SAME seeds under two
    controllers).

    Returns {"mean_diff", "ci_lower", "ci_upper", "n"}.
    """
    a = np.asarray(outcomes_a, dtype=np.float64)
    b = np.asarray(outcomes_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n == 0:
        return {"mean_diff": None, "ci_lower": None, "ci_upper": None, "n": 0}

    diffs = b - a
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    replicate_means = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        replicate_means[i] = float(np.mean(diffs[idx]))

    alpha = 1.0 - confidence
    lo = float(np.percentile(replicate_means, 100 * (alpha / 2)))
    hi = float(np.percentile(replicate_means, 100 * (1 - alpha / 2)))
    return {"mean_diff": mean_diff, "ci_lower": lo, "ci_upper": hi, "n": n}


def paired_binary_contingency(outcomes_a: np.ndarray, outcomes_b: np.ndarray) -> dict:
    """2x2 matched-pairs contingency counts for two binary (0/1) matched
    outcome arrays, plus a McNemar continuity-corrected chi-square
    statistic (no external stats package required)."""
    a = np.asarray(outcomes_a, dtype=bool)
    b = np.asarray(outcomes_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape, got {a.shape} vs {b.shape}")
    both = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    discordant = only_a + only_b
    chi2 = None if discordant == 0 else float((abs(only_b - only_a) - 1) ** 2 / discordant)
    return {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither, "mcnemar_chi2": chi2}
