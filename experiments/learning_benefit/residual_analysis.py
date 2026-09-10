"""Phase 9: LR-PPO residual-angle analysis and correlational diagnostics.

    theta_residual = arccos( clip( dot(u_Lead, u_LR-PPO), -1, 1 ) )

using normalized directions; degenerate (near-zero-norm) actions return
`None` (angle undefined) rather than fabricating a value.

This module is purely descriptive/correlational (Phase 9 explicitly does
NOT claim causality): `correlate` returns a Pearson correlation
coefficient plus the sample size, nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_angle_deg(u_lead: np.ndarray, u_final: np.ndarray, eps: float = 1e-8) -> float | None:
    """theta_residual in degrees, or None if either direction is
    (near-)zero (angle undefined)."""
    u_lead = np.asarray(u_lead, dtype=np.float64)
    u_final = np.asarray(u_final, dtype=np.float64)
    lead_norm = float(np.linalg.norm(u_lead))
    final_norm = float(np.linalg.norm(u_final))
    if lead_norm <= eps or final_norm <= eps:
        return None
    cos_angle = float(np.clip(np.dot(u_lead, u_final) / (lead_norm * final_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def correlate(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict:
    """Pearson correlation between two equal-length series, dropping any
    row where either value is NaN/None. Returns {"r", "n"}; `r` is None
    if fewer than 2 valid paired samples remain. DESCRIPTIVE ONLY -- no
    causal claim."""
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()
    x_valid, y_valid = x[mask], y[mask]
    n = int(mask.sum())
    if n < 2 or x_valid.std() == 0 or y_valid.std() == 0:
        return {"r": None, "n": n}
    r = float(np.corrcoef(x_valid, y_valid)[0, 1])
    return {"r": r, "n": n}


def summarize_by_group(df: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    """Groupby summary (count, mean, median, std) of `value_column` by
    `group_column`, dropping NaNs in `value_column` per group."""
    valid = df[df[value_column].notna()]
    return valid.groupby(group_column)[value_column].agg(["count", "mean", "median", "std"]).reset_index()
