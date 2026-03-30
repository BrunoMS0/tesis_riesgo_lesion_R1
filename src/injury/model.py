"""
model.py – Model factories for R5 Injury Risk Prediction.

Provides factory functions that build a configured Logistic Regression
classifier and a dummy baseline for comparison.

Public API
----------
build_logistic_regression(cfg) -> LogisticRegression
build_baseline_model(cfg) -> DummyClassifier
"""

from __future__ import annotations

import logging

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from .config import InjuryConfig

logger = logging.getLogger(__name__)


def build_logistic_regression(cfg: InjuryConfig) -> LogisticRegression:
    """
    Construct and return a configured LogisticRegression.

    Parameters
    ----------
    cfg : InjuryConfig
        Configuration with LR hyper-parameters (C, penalty, solver, etc.).
    """
    model = LogisticRegression(
        C=cfg.lr_C,
        penalty=cfg.lr_penalty,
        solver=cfg.lr_solver,
        max_iter=cfg.lr_max_iter,
        class_weight=cfg.lr_class_weight,
        random_state=cfg.seed,
    )
    logger.info("Built LogisticRegression (C=%.4f, penalty=%s, solver=%s, "
                "class_weight=%s)",
                cfg.lr_C, cfg.lr_penalty, cfg.lr_solver, cfg.lr_class_weight)
    return model


def build_baseline_model(cfg: InjuryConfig) -> DummyClassifier:
    """Construct a DummyClassifier (stratified) as a baseline comparator."""
    model = DummyClassifier(
        strategy="stratified",
        random_state=cfg.seed,
    )
    logger.info("Built DummyClassifier (strategy=stratified)")
    return model
