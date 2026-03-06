"""
predict.py – Generate DFI predictions for *all* participants.

This module runs inference on every available sequence (train + val + test)
so that the resulting CSV can feed into R5 (injury prediction model).

Public API
----------
predict_all(model, cfg) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import FatigueConfig
from .dataset import (
    load_dataframe,
    compute_dfi,
    split_participants,
    fit_scaler,
    apply_scaler,
    create_sequences,
)

logger = logging.getLogger(__name__)


def predict_all(
    model,
    cfg: Optional[FatigueConfig] = None,
) -> pd.DataFrame:
    """
    Generate DFI predictions for every participant-day that has a
    complete 14-day lookback window.

    The scaler is fitted on training participants (same split seed)
    to maintain consistency with the training pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: ``[participant_id, date, dfi_predicted, dfi_actual]``.
    """
    if cfg is None:
        cfg = FatigueConfig()

    # --- Load & prepare data (mirrors dataset.py logic) ---------------
    df = load_dataframe(cfg)
    feature_cols = [c for c in cfg.objective_features if c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(0)
    df["dfi"] = compute_dfi(df[cfg.target_raw_col])

    # Fit scaler on train participants only
    train_pids, _, _ = split_participants(df, cfg)
    scaler = fit_scaler(df, train_pids, feature_cols)
    df = apply_scaler(df, scaler, feature_cols)

    # --- Sequences for ALL participants --------------------------------
    X_all, y_all, meta_all = create_sequences(df, feature_cols, cfg.window_size)

    # --- Predict -------------------------------------------------------
    y_pred = model.predict(X_all, batch_size=cfg.batch_size, verbose=0).squeeze()

    result = meta_all.copy()
    result["dfi_predicted"] = y_pred
    result["dfi_actual"] = y_all

    # --- Save -----------------------------------------------------------
    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fatigue_index_predictions.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Full predictions saved: %s (%d rows)", csv_path, len(result))

    return result
