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
from src.injury.model import build_logistic_regression, build_baseline_model, build_random_forest
from src.injury.train import save_model, train_injury_model, train_with_cv, grid_search_C
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


class TestGridSearchC:
    def test_returns_valid_result(self, cfg, synthetic_split):
        s = synthetic_split
        best_C, results = grid_search_C(
            s["X_train"], s["y_train"], s["X_val"], s["y_val"], cfg,
        )
        assert best_C in cfg.c_grid
        assert list(results.columns) == ["C", "roc_auc"]
        assert len(results) == len(cfg.c_grid)
        assert all(results["roc_auc"].between(0.0, 1.0))


# ────────────────────────────────────────────────────────────
# Tests — RF-11 cross-domain model (wellness+load, z-scored)
# ────────────────────────────────────────────────────────────

class TestRF11CrossDomainModel:
    """
    Smoke tests for the RF-11 model used in cross-domain SoccerMon evaluation.
    Verifies that a RandomForest trained on only the 11 shared features
    (no wearables) produces finite AUC on a held-out set.
    """

    @pytest.fixture()
    def cfg_rf(self):
        from src.injury.config import SOCCERMON_SHARED_FEATURES
        return InjuryConfig(
            model_type="rf",
            feature_columns=list(SOCCERMON_SHARED_FEATURES),
            use_per_athlete_zscore=True,
            zscore_features=list(SOCCERMON_SHARED_FEATURES),
        )

    @pytest.fixture()
    def split_11(self):
        """Synthetic split using exactly the 11 shared features."""
        from src.injury.config import SOCCERMON_SHARED_FEATURES
        rng = np.random.RandomState(55)
        features = list(SOCCERMON_SHARED_FEATURES)
        n_tr, n_v, n_te = 200, 50, 50

        def _make(n, pids):
            X = pd.DataFrame({f: rng.randn(n) for f in features})
            y = pd.Series((rng.rand(n) > 0.88).astype(int))
            meta = pd.DataFrame({
                "participant_id": np.random.choice(pids, n),
                "date": pd.date_range("2020-01-01", periods=n),
            })
            return X, y, meta

        X_tr, y_tr, m_tr = _make(n_tr, ["p01", "p02", "p03"])
        X_v, y_v, m_v   = _make(n_v, ["p04"])
        X_te, y_te, m_te = _make(n_te, ["p05"])

        return {
            "X_train": X_tr, "y_train": y_tr, "meta_train": m_tr,
            "X_val": X_v, "y_val": y_v, "meta_val": m_v,
            "X_test": X_te, "y_test": y_te, "meta_test": m_te,
        }

    def test_rf11_trains_and_predicts(self, cfg_rf, split_11):
        """RF-11 must fit without error and produce finite probabilities."""
        s = split_11
        model = build_random_forest(cfg_rf)
        model = train_injury_model(model, s["X_train"], s["y_train"], cfg_rf)
        proba = model.predict_proba(s["X_test"])[:, 1]
        assert np.all(np.isfinite(proba))

    def test_rf11_auc_finite(self, cfg_rf, split_11):
        """RF-11 evaluation metrics (incl. roc_auc) must all be finite."""
        s = split_11
        model = build_random_forest(cfg_rf)
        model = train_injury_model(model, s["X_train"], s["y_train"], cfg_rf)
        result = evaluate_model(model, s["X_test"], s["y_test"], s["meta_test"], cfg_rf)
        for k, v in result.metrics.items():
            assert np.isfinite(v), f"RF-11 metric '{k}' is not finite: {v}"

    def test_rf11_feature_count(self, cfg_rf, split_11):
        """Trained RF-11 must use exactly 11 input features."""
        from src.injury.config import SOCCERMON_SHARED_FEATURES
        s = split_11
        model = build_random_forest(cfg_rf)
        model.fit(s["X_train"], s["y_train"])
        assert model.n_features_in_ == len(SOCCERMON_SHARED_FEATURES)
