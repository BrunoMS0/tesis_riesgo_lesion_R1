"""
train.py – Training pipeline for R5 Injury Risk Prediction models.

Handles fitting XGBoost with early stopping on validation data
and saving model artifacts.

Public API
----------
train_injury_model(model, X_train, y_train, X_val, y_val, cfg) -> trained model
compute_scale_pos_weight(y) -> float
save_model(model, path, name) -> str
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import InjuryConfig

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight = n_negative / n_positive."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        logger.warning("No positive samples; scale_pos_weight set to 1.0")
        return 1.0
    weight = n_neg / n_pos
    logger.info("scale_pos_weight = %.2f  (neg=%d, pos=%d)", weight, n_neg, n_pos)
    return weight


def train_injury_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: InjuryConfig,
):
    """
    Fit a classifier with early stopping on validation PR-AUC.

    For XGBoost models, uses ``eval_set`` + ``early_stopping_rounds``.
    For scikit-learn models (e.g. RandomForest), fits directly.

    Returns the fitted model.
    """
    model_type = type(model).__name__

    if model_type == "XGBClassifier":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Retrieve best iteration info
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            logger.info("XGBoost training done — best iteration: %d", best_iter)
        else:
            logger.info("XGBoost training done — %d estimators", cfg.xgb_n_estimators)
    else:
        # scikit-learn style (RF, etc.)
        model.fit(X_train, y_train)
        logger.info("%s training done", model_type)

    return model


def save_model(model, output_path: str, name: str) -> str:
    """Save a trained model to disk using joblib. Returns the file path."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(output_path, f"{name}.joblib")
    joblib.dump(model, fpath)
    logger.info("Model saved to %s", fpath)
    return fpath
