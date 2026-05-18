"""
test_injury_model.py – Tests for M2 model factory (Runner Dataset).

Validates:
- RandomForestClassifier builds with Runner hyperparams (n_estimators=200,
  max_features='sqrt', class_weight='balanced', random_state=42)
- Predictions are probabilities in [0, 1]
- class_weight is 'balanced'
- Baseline model builds correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.model import build_baseline_model, build_random_forest


@pytest.fixture()
def cfg():
    """InjuryConfig con hiperparámetros del Runner M2."""
    return InjuryConfig(
        model_type="rf",
        rf_n_estimators=200,
        rf_max_features="sqrt",
        rf_class_weight="balanced",
        seed=42,
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


class TestBuildRandomForestClassifier:
    def test_builds(self, cfg):
        model = build_random_forest(cfg)
        assert model is not None
        assert model.get_params()["n_estimators"] == 200

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

    def test_predict_proba_sums_to_one(self, cfg, synthetic_data):
        X, y = synthetic_data
        model = build_random_forest(cfg)
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
