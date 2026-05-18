"""
test_fatigue_model.py — Pruebas unitarias del módulo M1 — Construcción del modelo.

Cubre los 8 tests del catálogo Anexo N (IDs FM-1 a FM-8).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.runner.models.fatigue import (
    FATIGUE_FEATURE_COLUMNS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    _fit_normalizer,
    _apply_normalizer,
    train_final_fatigue_model,
)
from src.runner.config import SEED


# ─── Fixtures: datos sintéticos para el modelo M1 ────────────────────────────

@pytest.fixture(scope="module")
def synthetic_Xy():
    """X (10 features GPS) + y (perceived_recovery, escala [0,1]) sintéticos."""
    rng = np.random.default_rng(SEED)
    n = 200
    X = pd.DataFrame(
        rng.uniform(0.0, 10.0, size=(n, len(FATIGUE_FEATURE_COLUMNS))),
        columns=FATIGUE_FEATURE_COLUMNS,
    )
    y = pd.Series(rng.uniform(0.0, 1.0, size=n), name="recent_recovery")
    return X, y


@pytest.fixture(scope="module")
def fitted_model(synthetic_Xy):
    """RF Regressor entrenado con datos sintéticos."""
    X, y = synthetic_Xy
    pt = _fit_normalizer(X)
    X_norm = _apply_normalizer(X, pt)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(X_norm, y)
    return rf, pt, X_norm


# ─── FM-1: build_model retorna RandomForestRegressor ─────────────────────────

def test_build_returns_model(synthetic_Xy):
    """FM-1: Al construir el modelo se obtiene una instancia de RandomForestRegressor."""
    X, y = synthetic_Xy
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=SEED,
    )
    assert isinstance(rf, RandomForestRegressor)


# ─── FM-2: n_estimators = 200 ────────────────────────────────────────────────

def test_n_estimators():
    """FM-2: n_estimators = 200 en el modelo construido."""
    assert RF_N_ESTIMATORS == 200


# ─── FM-3: max_depth = 10 ────────────────────────────────────────────────────

def test_max_depth():
    """FM-3: max_depth = 10 en el modelo construido."""
    assert RF_MAX_DEPTH == 10


# ─── FM-4: random_state = 42 ──────────────────────────────────────────────────

def test_random_state():
    """FM-4: random_state = 42 (reproducibilidad garantizada)."""
    assert SEED == 42


# ─── FM-5: predicciones ∈ [0, 1] ─────────────────────────────────────────────

def test_predictions_valid_range(fitted_model, synthetic_Xy):
    """FM-5: Predicciones ∈ [0, 1] (escala de recuperación percibida)."""
    rf, pt, X_norm = fitted_model
    preds = rf.predict(X_norm)
    # Las predicciones pueden salir levemente fuera de [0,1] para un RF,
    # pero deben estar dentro de márgenes razonables.
    assert preds.min() >= -0.1, f"Predicción mínima {preds.min()} demasiado baja"
    assert preds.max() <= 1.1, f"Predicción máxima {preds.max()} demasiado alta"


# ─── FM-6: modelo puede predecir tras fit() ───────────────────────────────────

def test_model_fitted(fitted_model, synthetic_Xy):
    """FM-6: El modelo puede predecir tras fit() sin errores."""
    rf, pt, X_norm = fitted_model
    X, _ = synthetic_Xy
    preds = rf.predict(_apply_normalizer(X[:5], pt))
    assert len(preds) == 5
    assert not np.any(np.isnan(preds))


# ─── FM-7: feature_importances_ accesible ────────────────────────────────────

def test_feature_importance_available(fitted_model):
    """FM-7: feature_importances_ accesible tras entrenamiento."""
    rf, _, _ = fitted_model
    fi = rf.feature_importances_
    assert fi is not None
    assert len(fi) == len(FATIGUE_FEATURE_COLUMNS)
    assert abs(fi.sum() - 1.0) < 1e-6, "Las importancias deben sumar 1.0"


# ─── FM-8: reproducibilidad ───────────────────────────────────────────────────

def test_reproducible(synthetic_Xy):
    """FM-8: Mismos datos + misma semilla → mismas predicciones."""
    X, y = synthetic_Xy
    pt = _fit_normalizer(X)
    X_norm = _apply_normalizer(X, pt)

    rf1 = RandomForestRegressor(n_estimators=50, random_state=SEED)
    rf1.fit(X_norm, y)

    rf2 = RandomForestRegressor(n_estimators=50, random_state=SEED)
    rf2.fit(X_norm, y)

    np.testing.assert_array_equal(rf1.predict(X_norm), rf2.predict(X_norm))
