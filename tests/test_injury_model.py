"""
test_injury_model.py – Tests for R5 model factories.

Validates:
- Logistic Regression and Baseline models build correctly
- Predictions are probabilities in [0, 1]
- class_weight is applied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.model import build_baseline_model, build_logistic_regression


@pytest.fixture()
def cfg():
    return InjuryConfig()


@pytest.fixture()
def synthetic_data():
    """Small synthetic training data."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        f"feat_{i}": rng.randn(n) for i in range(5)
    })
    y = pd.Series((rng.rand(n) > 0.9).astype(int))  # ~10% positive
    return X, y


class TestBuildLogisticRegression:
    def test_builds(self, cfg):
        model = build_logistic_regression(cfg)
        assert model is not None
        assert model.get_params()["C"] == cfg.lr_C

    def test_class_weight_balanced(self, cfg):
        model = build_logistic_regression(cfg)
        assert model.get_params()["class_weight"] == "balanced"

    def test_predictions_are_probabilities(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_logistic_regression(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_sums_to_one(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_logistic_regression(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestBuildBaseline:
    def test_builds(self, cfg):
        model = build_baseline_model(cfg)
        assert model is not None

    def test_predictions_are_probabilities(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_baseline_model(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
