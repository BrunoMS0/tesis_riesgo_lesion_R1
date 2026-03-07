"""
test_injury_model.py – Tests for R5 model factories.

Validates:
- XGBoost and RF models build correctly
- Predictions are probabilities in [0, 1]
- scale_pos_weight is applied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.model import build_random_forest, build_xgboost


@pytest.fixture()
def cfg():
    return InjuryConfig(
        xgb_n_estimators=10,  # small for fast tests
        rf_n_estimators=10,
    )


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


class TestBuildXGBoost:
    def test_builds(self, cfg):
        model = build_xgboost(cfg)
        assert model is not None
        assert model.get_params()["max_depth"] == cfg.xgb_max_depth

    def test_scale_pos_weight(self, cfg):
        model = build_xgboost(cfg, scale_pos_weight=10.0)
        assert model.get_params()["scale_pos_weight"] == 10.0

    def test_predictions_are_probabilities(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_xgboost(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_sums_to_one(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_xgboost(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestBuildRandomForest:
    def test_builds(self, cfg):
        model = build_random_forest(cfg)
        assert model is not None
        assert model.get_params()["max_depth"] == cfg.rf_max_depth

    def test_class_weight_balanced(self, cfg):
        model = build_random_forest(cfg)
        assert model.get_params()["class_weight"] == "balanced"

    def test_predictions_are_probabilities(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_random_forest(cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
