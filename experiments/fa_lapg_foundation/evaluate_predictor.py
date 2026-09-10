"""
Prediction-error metrics (Phase 32) and generalization evaluation
(Phase 33) for CV, CA, and the GRU-residual predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.fa_lapg_foundation.baseline_predictors import ca_predict, cv_predict
from experiments.fa_lapg_foundation.gru_predictor import GRUResidualPredictor
from experiments.fa_lapg_foundation.trajectory_dataset import HORIZONS


def gru_predict(model: GRUResidualPredictor, sample: dict, horizons: tuple[float, ...] = HORIZONS) -> dict[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        history = torch.as_tensor(sample["history"], dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(sample["history_mask"], dtype=torch.float32).unsqueeze(0)
        delta = model(history, mask)[0].numpy()
    cv = sample["cv_pred"]
    return {h: cv[h] + delta[k] for k, h in enumerate(horizons)}


def compute_errors(samples: list[dict], model: GRUResidualPredictor | None, dt: float) -> pd.DataFrame:
    """Per-(sample, horizon) position error for CV, CA, and (if provided)
    the GRU-residual predictor. Rows with no available target are
    omitted (never fabricated)."""
    rows = []
    for s in samples:
        cv_pred = cv_predict(s)
        ca_pred = ca_predict(s, dt)
        gru_pred = gru_predict(model, s) if model is not None else None
        for h, target in s["targets"].items():
            if target is None:
                continue
            row = {
                "seed": s["seed"], "scenario": s["scenario"], "step_index": s["step_index"],
                "horizon": h,
                "cv_error": float(np.linalg.norm(target - cv_pred[h])),
                "ca_error": float(np.linalg.norm(target - ca_pred[h])),
            }
            if gru_pred is not None:
                row["gru_error"] = float(np.linalg.norm(target - gru_pred[h]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_horizon(error_df: pd.DataFrame, error_cols: list[str]) -> pd.DataFrame:
    """Phase 32: mean/median/RMSE/p90/p95 per horizon, for each predictor
    error column."""
    records = []
    for h, sub in error_df.groupby("horizon"):
        row = {"horizon": h, "n": len(sub)}
        for col in error_cols:
            vals = sub[col].values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_rmse"] = float(np.sqrt(np.mean(vals ** 2)))
            row[f"{col}_p90"] = float(np.quantile(vals, 0.90))
            row[f"{col}_p95"] = float(np.quantile(vals, 0.95))
        records.append(row)
    return pd.DataFrame(records).sort_values("horizon").reset_index(drop=True)


def summarize_by_scenario(error_df: pd.DataFrame, error_cols: list[str]) -> pd.DataFrame:
    """Phase 32: same metrics, broken down per scenario (pooled over
    horizons) -- used for the generalization comparison (Phase 33)."""
    records = []
    for scenario, sub in error_df.groupby("scenario"):
        row = {"scenario": scenario, "n": len(sub)}
        for col in error_cols:
            vals = sub[col].values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_rmse"] = float(np.sqrt(np.mean(vals ** 2)))
            row[f"{col}_p90"] = float(np.quantile(vals, 0.90))
            row[f"{col}_p95"] = float(np.quantile(vals, 0.95))
        records.append(row)
    return pd.DataFrame(records)
