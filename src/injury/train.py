"""
train.py – Training pipeline for R5 Injury Risk Prediction models.

Handles fitting Logistic Regression with optional hyper-parameter
search (LogisticRegressionCV) and saving model artifacts.

Public API
----------
train_injury_model(model, X_train, y_train, cfg) -> trained model
train_with_cv(X_train, y_train, cfg) -> LogisticRegressionCV
save_model(model, path, name) -> str
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from typing import Tuple

from .config import InjuryConfig

logger = logging.getLogger(__name__)


def train_injury_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: InjuryConfig,
):
    """
    Fit a classifier on training data.

    Returns the fitted model.
    """
    model_type = type(model).__name__
    model.fit(X_train, y_train)
    logger.info("%s training done (%d samples, %d features)",
                model_type, len(X_train), X_train.shape[1])
    return model


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: InjuryConfig,
) -> LogisticRegressionCV:
    """
    Train LogisticRegressionCV to automatically select C via cross-validation.

    Uses 5-fold stratified CV, scoring by AUC-ROC.

    Returns the fitted model with optimal C.
    """
    model = LogisticRegressionCV(
        Cs=10,
        penalty=cfg.lr_penalty,
        solver=cfg.lr_solver,
        max_iter=cfg.lr_max_iter,
        class_weight=cfg.lr_class_weight,
        scoring="roc_auc",
        cv=5,
        random_state=cfg.seed,
    )
    model.fit(X_train, y_train)
    best_C = float(model.C_[0])
    logger.info("LogisticRegressionCV done — best C=%.6f (5-fold AUC-ROC)",
                best_C)
    return model


def save_model(model, output_path: str, name: str) -> str:
    """Save a trained model to disk using joblib. Returns the file path."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(output_path, f"{name}.joblib")
    joblib.dump(model, fpath)
    logger.info("Model saved to %s", fpath)
    return fpath


def grid_search_C(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: InjuryConfig,
) -> Tuple[float, pd.DataFrame]:
    """
    Simple grid search over ``cfg.c_grid`` evaluated on a held-out
    validation set.

    Returns
    -------
    best_C : float
        The value of C with the highest validation ROC-AUC.
    results : pd.DataFrame
        Table with columns [C, roc_auc] for every candidate.
    """
    rows = []
    for C in cfg.c_grid:
        model = LogisticRegression(
            C=C,
            penalty=cfg.lr_penalty,
            solver=cfg.lr_solver,
            max_iter=cfg.lr_max_iter,
            class_weight=cfg.lr_class_weight,
            random_state=cfg.seed,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.0
        rows.append({"C": C, "roc_auc": round(auc, 4)})
        logger.info("Grid search C=%.4f → val ROC-AUC=%.4f", C, auc)

    results = pd.DataFrame(rows)
    best_idx = results["roc_auc"].idxmax()
    best_C = float(results.loc[best_idx, "C"])
    logger.info("Grid search complete — best C=%.4f (ROC-AUC=%.4f)",
                best_C, results.loc[best_idx, "roc_auc"])
    return best_C, results


def grid_search_RF(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: InjuryConfig,
):
    """
    Grid search over ``cfg.rf_param_grid`` evaluated on the validation set.

    Returns
    -------
    best_params : dict
        Best (max_depth, min_samples_leaf) found.
    results : pd.DataFrame
        Table with one row per candidate.
    """
    from sklearn.ensemble import RandomForestClassifier

    rows = []
    for params in cfg.rf_param_grid:
        model = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_features=cfg.rf_max_features,
            class_weight=cfg.rf_class_weight,
            random_state=cfg.seed,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.0
        row = {**params, "roc_auc": round(auc, 4)}
        rows.append(row)
        logger.info(
            "Grid search RF max_depth=%s min_samples_leaf=%s → val ROC-AUC=%.4f",
            params.get("max_depth", "None"),
            params.get("min_samples_leaf", 1),
            auc,
        )

    results = pd.DataFrame(rows)
    best_idx = int(results["roc_auc"].idxmax())
    best_params = {
        k: v for k, v in results.iloc[best_idx].to_dict().items()
        if k != "roc_auc"
    }
    # Restore None for max_depth if pandas stored it as NaN
    if "max_depth" in best_params and pd.isna(best_params["max_depth"]):
        best_params["max_depth"] = None
    elif "max_depth" in best_params and best_params["max_depth"] is not None:
        best_params["max_depth"] = int(best_params["max_depth"])
    if "min_samples_leaf" in best_params:
        best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

    logger.info(
        "Grid search RF complete — best params=%s (ROC-AUC=%.4f)",
        best_params,
        results.iloc[best_idx]["roc_auc"],
    )
    return best_params, results
