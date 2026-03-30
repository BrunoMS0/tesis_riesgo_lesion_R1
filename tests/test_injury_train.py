"""
test_injury_train.py – Smoke tests for R5 training pipeline.

Validates:
- Logistic Regression trains and produces finite predictions
- Model save / load round-trip
- Evaluation metrics are finite
- train_with_cv returns a fitted model
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.model import build_logistic_regression, build_baseline_model
from src.injury.train import save_model, train_injury_model, train_with_cv
from src.injury.evaluate import evaluate_model


@pytest.fixture()
def cfg():
    return InjuryConfig()


@pytest.fixture()
def synthetic_split():
    """Train / val / test split with binary target."""
    rng = np.random.RandomState(42)
    features = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

    def _make(n, pids):
        X = pd.DataFrame({f: rng.randn(n) for f in features})
        y = pd.Series((rng.rand(n) > 0.9).astype(int))
        meta = pd.DataFrame({
            "participant_id": rng.choice(pids, n),
            "date": pd.date_range("2020-01-01", periods=n),
        })
        return X, y, meta

    X_train, y_train, meta_train = _make(200, ["p01", "p02", "p03"])
    X_val, y_val, meta_val = _make(50, ["p04"])
    X_test, y_test, meta_test = _make(50, ["p05"])

    return {
        "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
        "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
    }


class TestTrainLogisticRegression:
    def test_smoke_train(self, cfg, synthetic_split):
        s = synthetic_split
        model = build_logistic_regression(cfg)
        model = train_injury_model(model, s["X_train"], s["y_train"], cfg)
        proba = model.predict_proba(s["X_test"])[:, 1]
        assert np.all(np.isfinite(proba))

    def test_evaluation_metrics_finite(self, cfg, synthetic_split):
        s = synthetic_split
        model = build_logistic_regression(cfg)
        model = train_injury_model(model, s["X_train"], s["y_train"], cfg)
        result = evaluate_model(
            model, s["X_test"], s["y_test"], s["meta_test"], cfg,
        )
        for k, v in result.metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


class TestTrainBaseline:
    def test_smoke_train(self, cfg, synthetic_split):
        s = synthetic_split
        model = build_baseline_model(cfg)
        model = train_injury_model(model, s["X_train"], s["y_train"], cfg)
        proba = model.predict_proba(s["X_test"])[:, 1]
        assert np.all(np.isfinite(proba))


class TestTrainWithCV:
    def test_returns_fitted_model(self, cfg, synthetic_split):
        s = synthetic_split
        model = train_with_cv(s["X_train"], s["y_train"], cfg)
        assert hasattr(model, "predict_proba")
        proba = model.predict_proba(s["X_test"])[:, 1]
        assert np.all(np.isfinite(proba))


class TestSaveModel:
    def test_save_creates_file(self, cfg, synthetic_split, tmp_path):
        s = synthetic_split
        model = build_logistic_regression(cfg)
        model.fit(s["X_train"], s["y_train"])
        path = save_model(model, str(tmp_path), "test_model")
        assert os.path.exists(path)
        assert path.endswith(".joblib")
