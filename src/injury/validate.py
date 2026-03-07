"""
validate.py – Leave-One-Subject-Out (LOSO) cross-validation for R5.

Implements the gold-standard validation approach for multi-subject
sports science where each fold holds out one real participant.
Synthetic data is regenerated from the remaining participants per fold.

Public API
----------
loso_cross_validation(X, y, meta, cfg) -> LOSOResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from .config import InjuryConfig
from .model import build_xgboost
from .train import compute_scale_pos_weight

logger = logging.getLogger(__name__)


@dataclass
class LOSOFoldResult:
    """Metrics from a single LOSO fold."""

    participant_id: str
    n_samples: int
    n_injuries: int
    pr_auc: float
    roc_auc: float
    f1: float


@dataclass
class LOSOResult:
    """Aggregated LOSO cross-validation results."""

    folds: List[LOSOFoldResult] = field(default_factory=list)
    mean_pr_auc: float = 0.0
    mean_roc_auc: float = 0.0
    mean_f1: float = 0.0
    std_pr_auc: float = 0.0


def loso_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cfg: InjuryConfig,
    *,
    use_augmentation: bool = True,
) -> LOSOResult:
    """
    Leave-One-Subject-Out cross-validation.

    For each of the N real participants:
      1. Hold out that participant as the test fold.
      2. Optionally augment the remaining training data with synthetic athletes.
      3. Train an XGBoost model and evaluate on the held-out participant.

    Parameters
    ----------
    X : DataFrame of shape (n, n_features)
    y : Series of shape (n,) — binary target
    meta : DataFrame with [participant_id, date]
    cfg : InjuryConfig
    use_augmentation : bool
        If True, generate synthetic athletes per fold (slower but better).

    Returns
    -------
    LOSOResult
    """
    pids = sorted(meta["participant_id"].unique())
    # Filter out synthetic participants if any are mixed in
    pids = [p for p in pids if not str(p).startswith("synth_")]
    logger.info("Starting LOSO cross-validation with %d participants", len(pids))

    folds: List[LOSOFoldResult] = []

    for i, held_out_pid in enumerate(pids, 1):
        mask_test = meta["participant_id"] == held_out_pid
        mask_train = ~mask_test & ~meta["participant_id"].str.startswith("synth_")

        X_fold_train = X.loc[mask_train].reset_index(drop=True)
        y_fold_train = y.loc[mask_train].reset_index(drop=True)
        meta_fold_train = meta.loc[mask_train].reset_index(drop=True)

        X_fold_test = X.loc[mask_test].reset_index(drop=True)
        y_fold_test = y.loc[mask_test].reset_index(drop=True)

        # Optionally augment training data
        if use_augmentation and cfg.n_synthetic_athletes > 0:
            try:
                from .augment import generate_synthetic_athletes
                X_fold_train, y_fold_train, _ = generate_synthetic_athletes(
                    X_fold_train, y_fold_train, meta_fold_train, cfg,
                )
            except Exception as exc:
                logger.warning("Augmentation failed for fold %d (%s): %s",
                               i, held_out_pid, exc)

        # Compute class weight and build model
        spw = compute_scale_pos_weight(y_fold_train)
        model = build_xgboost(cfg, scale_pos_weight=spw)

        # Train (use a small held-out slice from training for early stopping)
        model.fit(X_fold_train, y_fold_train, verbose=False)

        # Predict probabilities on held-out participant
        y_prob = model.predict_proba(X_fold_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        n_injuries = int(y_fold_test.sum())

        # Compute metrics (handle edge cases where fold has no injuries)
        try:
            pr_auc = average_precision_score(y_fold_test, y_prob)
        except ValueError:
            pr_auc = 0.0

        try:
            roc_auc = roc_auc_score(y_fold_test, y_prob)
        except ValueError:
            roc_auc = 0.0

        f1 = f1_score(y_fold_test, y_pred, zero_division=0.0)

        fold_result = LOSOFoldResult(
            participant_id=held_out_pid,
            n_samples=len(y_fold_test),
            n_injuries=n_injuries,
            pr_auc=round(pr_auc, 4),
            roc_auc=round(roc_auc, 4),
            f1=round(f1, 4),
        )
        folds.append(fold_result)
        logger.info("Fold %d/%d [%s]: PR-AUC=%.4f, ROC-AUC=%.4f, F1=%.4f "
                     "(n=%d, injuries=%d)",
                     i, len(pids), held_out_pid,
                     fold_result.pr_auc, fold_result.roc_auc, fold_result.f1,
                     fold_result.n_samples, fold_result.n_injuries)

    # Aggregate
    pr_aucs = [f.pr_auc for f in folds]
    roc_aucs = [f.roc_auc for f in folds]
    f1s = [f.f1 for f in folds]

    result = LOSOResult(
        folds=folds,
        mean_pr_auc=round(float(np.mean(pr_aucs)), 4),
        mean_roc_auc=round(float(np.mean(roc_aucs)), 4),
        mean_f1=round(float(np.mean(f1s)), 4),
        std_pr_auc=round(float(np.std(pr_aucs)), 4),
    )
    logger.info("LOSO complete — mean PR-AUC=%.4f (±%.4f), "
                "mean ROC-AUC=%.4f, mean F1=%.4f",
                result.mean_pr_auc, result.std_pr_auc,
                result.mean_roc_auc, result.mean_f1)
    return result
