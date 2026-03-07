"""
model.py – Model factories for R5 Injury Risk Prediction.

Provides factory functions that build configured XGBoost and
Random Forest classifiers ready for training.

Public API
----------
build_xgboost(cfg, scale_pos_weight) -> XGBClassifier
build_random_forest(cfg) -> RandomForestClassifier
"""

from __future__ import annotations

import logging

from .config import InjuryConfig

logger = logging.getLogger(__name__)


def build_xgboost(
    cfg: InjuryConfig,
    scale_pos_weight: float = 1.0,
):
    """
    Construct and return a configured XGBClassifier.

    Parameters
    ----------
    cfg : InjuryConfig
    scale_pos_weight : float
        Ratio of negative to positive class.  Pass ``(n_neg / n_pos)``
        to handle residual imbalance after augmentation.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        eval_metric="aucpr",
        random_state=cfg.seed,
    )
    logger.info("Built XGBClassifier (n_estimators=%d, max_depth=%d, "
                "scale_pos_weight=%.2f)",
                cfg.xgb_n_estimators, cfg.xgb_max_depth, scale_pos_weight)
    return model


def build_random_forest(cfg: InjuryConfig):
    """Construct and return a configured RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        class_weight="balanced",
        min_samples_leaf=cfg.rf_min_samples_leaf,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    logger.info("Built RandomForestClassifier (n_estimators=%d, max_depth=%d)",
                cfg.rf_n_estimators, cfg.rf_max_depth)
    return model
