"""
pipeline.py – Orchestrator for the R6 Two-Stage Predictive Pipeline.

Executes four stages in sequence:

1. **Load Models** – Load pre-trained R4 (Keras) and R5 (joblib) artefacts.
2. **R4 Fatigue Prediction** – Generate Dynamic Fatigue Index (DFI) for all
   participant-days using the R4 Bi-LSTM + Attention model.
3. **Feature Engineering Handoff** – Merge DFI into the R5 feature matrix
   with cold-start imputation, then split by participant.
4. **R5 Injury Risk Prediction** – Predict injury probability on the test
   split and compute evaluation metrics.

Public API
----------
run(cfg) -> IntegrationReport
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import IntegrationConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Result containers
# ────────────────────────────────────────────────────────────

@dataclass
class StageReport:
    name: str
    duration_s: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationReport:
    """Summary of a full R6 pipeline execution."""

    stages: List[StageReport] = field(default_factory=list)
    total_duration_s: float = 0.0
    n_predictions: int = 0
    fatigue_summary: Dict[str, float] = field(default_factory=dict)
    injury_metrics: Dict[str, float] = field(default_factory=dict)
    output_csv: Optional[str] = None
    fatigue_model_path: Optional[str] = None
    injury_model_path: Optional[str] = None


# ────────────────────────────────────────────────────────────
# Stage 1 – Load pre-trained models
# ────────────────────────────────────────────────────────────

def load_models(cfg: IntegrationConfig):
    """
    Load R4 (Keras) and R5 (joblib) pre-trained models.

    Returns
    -------
    (fatigue_model, injury_model)

    Raises
    ------
    RuntimeError
        If either model file does not exist or cannot be loaded.
    """
    import tensorflow as tf
    from src.fatigue.model import TemporalAttention

    # --- R4 Fatigue model ---
    if not os.path.isfile(cfg.fatigue_model_path):
        raise RuntimeError(
            f"R4 fatigue model not found: {cfg.fatigue_model_path}"
        )
    fatigue_model = tf.keras.models.load_model(
        cfg.fatigue_model_path,
        custom_objects={"TemporalAttention": TemporalAttention},
    )
    logger.info("R4 fatigue model loaded from %s (%d params)",
                cfg.fatigue_model_path, fatigue_model.count_params())

    # --- R5 Injury model ---
    if not os.path.isfile(cfg.injury_model_path):
        raise RuntimeError(
            f"R5 injury model not found: {cfg.injury_model_path}"
        )
    injury_model = joblib.load(cfg.injury_model_path)
    logger.info("R5 injury model loaded from %s (%s)",
                cfg.injury_model_path, type(injury_model).__name__)

    return fatigue_model, injury_model


# ────────────────────────────────────────────────────────────
# Stage 2 – R4 Fatigue Prediction (DFI generation)
# ────────────────────────────────────────────────────────────

def generate_dfi(fatigue_model, cfg: IntegrationConfig) -> pd.DataFrame:
    """
    Generate DFI predictions for every participant-day that has a
    complete lookback window.

    Reuses ``src.fatigue.predict.predict_all`` so all scaler-fitting
    and windowing logic is consistent with standalone R4 runs.

    Returns
    -------
    pd.DataFrame
        Columns: ``[participant_id, date, dfi_predicted, dfi_actual]``.
    """
    from src.fatigue.predict import predict_all

    dfi_df = predict_all(fatigue_model, cfg.fatigue_cfg)
    logger.info("DFI predictions generated: %d rows, "
                "mean=%.3f, std=%.3f",
                len(dfi_df),
                dfi_df["dfi_predicted"].mean(),
                dfi_df["dfi_predicted"].std())
    return dfi_df


# ────────────────────────────────────────────────────────────
# Stage 3 – Feature engineering handoff
# ────────────────────────────────────────────────────────────

def merge_dfi_features(
    dfi_df: pd.DataFrame,
    cfg: IntegrationConfig,
) -> pd.DataFrame:
    """
    Load the un-normalised feature CSV, LEFT-JOIN the in-memory DFI
    predictions, and apply cold-start imputation.

    This mirrors the logic in ``src.injury.dataset.load_and_merge``
    but receives DFI as a DataFrame instead of reading a CSV.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``dfi_predicted`` column filled.
    """
    icfg = cfg.injury_cfg

    df = pd.read_csv(icfg.input_csv, parse_dates=["date"])
    df.sort_values(["participant_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded feature CSV: %d rows × %d cols from %s",
                len(df), len(df.columns), icfg.input_csv)

    # Left-join DFI predictions (in-memory handoff)
    dfi_cols = dfi_df[["participant_id", "date", "dfi_predicted"]].copy()
    df = df.merge(dfi_cols, on=["participant_id", "date"], how="left")

    # Cold-start imputation: per-participant median → global median
    n_missing = df["dfi_predicted"].isna().sum()
    if n_missing > 0:
        medians = df.groupby("participant_id")["dfi_predicted"].transform(
            "median")
        df["dfi_predicted"] = df["dfi_predicted"].fillna(medians)
        global_median = df["dfi_predicted"].median()
        df["dfi_predicted"] = df["dfi_predicted"].fillna(global_median)
        logger.info("Filled %d cold-start DFI values with "
                     "participant/global median", n_missing)

    logger.info("Merged dataset: %d rows × %d cols", len(df), len(df.columns))
    return df


# ────────────────────────────────────────────────────────────
# Stage 4 – R5 Injury Risk Prediction
# ────────────────────────────────────────────────────────────

def predict_injury(
    injury_model,
    merged_df: pd.DataFrame,
    cfg: IntegrationConfig,
) -> Dict[str, Any]:
    """
    Split by participant, predict injury probability on the test set,
    compute evaluation metrics, and build the results DataFrame.

    Returns
    -------
    dict with keys:
        ``metrics``  – evaluation metrics dict
        ``results``  – DataFrame with predictions
        ``n_test``   – number of test rows
    """
    from src.injury.dataset import prepare_features, split_participants
    from src.injury.evaluate import evaluate_model

    icfg = cfg.injury_cfg
    X, y, meta = prepare_features(merged_df, icfg)
    train_pids, val_pids, test_pids = split_participants(merged_df, icfg)

    # Test-set subset
    test_mask = meta["participant_id"].isin(test_pids)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)

    logger.info("Test split: %d rows, %d features, participants=%s",
                len(X_test), X_test.shape[1], test_pids)

    # Evaluate
    eval_result = evaluate_model(
        injury_model, X_test, y_test, meta_test, icfg,
        model_name="XGBoost_integrated",
    )

    # Build results DataFrame
    y_prob = injury_model.predict_proba(X_test)[:, 1]
    threshold = eval_result.optimal_threshold
    results = meta_test.copy()
    results["dfi_predicted"] = X_test["dfi_predicted"].values
    results["injury_probability"] = y_prob
    results["injury_predicted"] = (y_prob >= threshold).astype(int)
    results["injury_actual"] = y_test.values

    return {
        "metrics": eval_result.metrics,
        "results": results,
        "n_test": len(X_test),
    }


# ────────────────────────────────────────────────────────────
# Main orchestrator
# ────────────────────────────────────────────────────────────

def run(cfg: Optional[IntegrationConfig] = None) -> IntegrationReport:
    """
    Execute the full R6 two-stage predictive pipeline.

    Parameters
    ----------
    cfg : IntegrationConfig, optional
        If *None* a default config is used.

    Returns
    -------
    IntegrationReport
    """
    if cfg is None:
        cfg = IntegrationConfig()

    stages: List[StageReport] = []
    t_global = time.perf_counter()

    # ── STAGE 1: LOAD MODELS ─────────────────────────────
    logger.info("═" * 60)
    logger.info("R6 STAGE 1 / 4 — LOAD PRE-TRAINED MODELS")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    fatigue_model, injury_model = load_models(cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="LoadModels", duration_s=round(dt, 2),
        details={
            "fatigue_model": cfg.fatigue_model_path,
            "injury_model": cfg.injury_model_path,
        },
    ))

    # ── STAGE 2: R4 FATIGUE PREDICTION ───────────────────
    logger.info("═" * 60)
    logger.info("R6 STAGE 2 / 4 — R4 FATIGUE PREDICTION (DFI)")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    dfi_df = generate_dfi(fatigue_model, cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="FatiguePrediction", duration_s=round(dt, 2),
        details={
            "n_predictions": len(dfi_df),
            "dfi_mean": round(float(dfi_df["dfi_predicted"].mean()), 4),
            "dfi_std": round(float(dfi_df["dfi_predicted"].std()), 4),
        },
    ))

    # ── STAGE 3: FEATURE ENGINEERING HANDOFF ─────────────
    logger.info("═" * 60)
    logger.info("R6 STAGE 3 / 4 — FEATURE ENGINEERING HANDOFF")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    merged_df = merge_dfi_features(dfi_df, cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="FeatureHandoff", duration_s=round(dt, 2),
        details={
            "n_rows": len(merged_df),
            "n_cols": len(merged_df.columns),
        },
    ))

    # ── STAGE 4: R5 INJURY RISK PREDICTION ───────────────
    logger.info("═" * 60)
    logger.info("R6 STAGE 4 / 4 — R5 INJURY RISK PREDICTION")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    injury_out = predict_injury(injury_model, merged_df, cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="InjuryPrediction", duration_s=round(dt, 2),
        details=injury_out["metrics"],
    ))

    # ── SAVE RESULTS ─────────────────────────────────────
    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "integration_predictions.csv"
    injury_out["results"].to_csv(csv_path, index=False)
    logger.info("Integration results saved: %s", csv_path)

    # Also save DFI predictions for reproducibility
    dfi_csv_path = out_dir / "fatigue_index_predictions.csv"
    dfi_df.to_csv(dfi_csv_path, index=False)
    logger.info("DFI predictions cached: %s", dfi_csv_path)

    total_duration = time.perf_counter() - t_global

    report = IntegrationReport(
        stages=stages,
        total_duration_s=round(total_duration, 2),
        n_predictions=injury_out["n_test"],
        fatigue_summary={
            "n_dfi": len(dfi_df),
            "dfi_mean": round(float(dfi_df["dfi_predicted"].mean()), 4),
            "dfi_std": round(float(dfi_df["dfi_predicted"].std()), 4),
        },
        injury_metrics=injury_out["metrics"],
        output_csv=str(csv_path),
        fatigue_model_path=cfg.fatigue_model_path,
        injury_model_path=cfg.injury_model_path,
    )

    logger.info("R6 pipeline complete in %.2fs", total_duration)
    return report
