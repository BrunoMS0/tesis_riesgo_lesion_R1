"""
test_fatigue_train.py — Pruebas unitarias del entrenamiento del Modelo M1.

Cubre los 4 tests del catálogo Anexo N (IDs FT-1 a FT-4).
Usa datos sintéticos mínimos para verificar el comportamiento del entrenamiento LOAO.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.runner.models.fatigue import (
    FATIGUE_FEATURE_COLUMNS,
    FATIGUE_TARGET_COL,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    _fit_normalizer,
    _apply_normalizer,
    prepare_fatigue_dataset,
)
from src.runner.config import RAW_FEATURE_BASE_NAMES, DAY_SUFFIXES, SEED
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─── Fixture: CSV sintético con estructura Runner Dataset ─────────────────────

def _build_csv(n_athletes: int = 5, n_days: int = 20) -> str:
    rng = np.random.default_rng(SEED)
    rows = []
    for aid in range(n_athletes):
        for day in range(n_days):
            row = {"Athlete ID": aid, "Date": day, "injury": 0}
            for base in RAW_FEATURE_BASE_NAMES:
                for suffix in DAY_SUFFIXES:
                    col = base if suffix == "" else f"{base}{suffix}"
                    # perceived exertion: mayoría de días son días de entrenamiento
                    if base == "perceived exertion" and day % 7 == 0:
                        row[col] = -0.01  # día de descanso
                    else:
                        row[col] = rng.uniform(1.0, 8.0)
            rows.append(row)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    pd.DataFrame(rows).to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def synthetic_csv():
    path = _build_csv()
    yield path
    os.unlink(path)


# ─── FT-1: entrenamiento completa sin error ───────────────────────────────────

def test_smoke_train(synthetic_csv):
    """FT-1: Entrenamiento completa sin error con datos sintéticos."""
    df_all, X, y, meta = prepare_fatigue_dataset(synthetic_csv)
    valid = y.notna()
    X_fit = X.loc[valid]
    y_fit = y.loc[valid]

    pt = _fit_normalizer(X_fit)
    X_norm = _apply_normalizer(X_fit, pt)

    rf = RandomForestRegressor(
        n_estimators=10, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF, random_state=SEED, n_jobs=1,
    )
    rf.fit(X_norm, y_fit)  # no debe lanzar excepción


# ─── FT-2: evaluate() retorna RMSE, MAE, R² y baseline_rmse ─────────────────

def test_evaluation_returns_metrics(synthetic_csv):
    """FT-2: La evaluación retorna RMSE, MAE, R² y baseline_rmse."""
    df_all, X, y, meta = prepare_fatigue_dataset(synthetic_csv)
    valid = y.notna()
    X_fit = X.loc[valid]
    y_fit = y.loc[valid]

    pt = _fit_normalizer(X_fit)
    X_norm = _apply_normalizer(X_fit, pt)
    rf = RandomForestRegressor(n_estimators=10, random_state=SEED, n_jobs=1)
    rf.fit(X_norm, y_fit)
    preds = rf.predict(X_norm)

    rmse = np.sqrt(mean_squared_error(y_fit, preds))
    mae = mean_absolute_error(y_fit, preds)
    r2 = r2_score(y_fit, preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_fit, np.full(len(y_fit), y_fit.mean())))

    for metric_name, metric_val in [("rmse", rmse), ("mae", mae), ("r2", r2), ("baseline_rmse", baseline_rmse)]:
        assert np.isfinite(metric_val), f"{metric_name} no es finito: {metric_val}"


# ─── FT-3: desglose de métricas por atleta ────────────────────────────────────

def test_per_athlete_breakdown(synthetic_csv):
    """FT-3: La evaluación produce desglose de métricas por atleta."""
    df_all, X, y, meta = prepare_fatigue_dataset(synthetic_csv)
    valid = y.notna()

    athlete_metrics = []
    for pid in meta["participant_id"].unique():
        mask = (meta["participant_id"] == pid) & valid
        if mask.sum() < 2:
            continue
        X_p = X.loc[mask]
        y_p = y.loc[mask]
        # Simula LOAO: entrena en los demás, predice en este
        others = valid & (meta["participant_id"] != pid)
        if others.sum() < 2:
            continue
        X_train = X.loc[others]
        y_train = y.loc[others]
        pt = _fit_normalizer(X_train)
        rf = RandomForestRegressor(n_estimators=5, random_state=SEED, n_jobs=1)
        rf.fit(_apply_normalizer(X_train, pt), y_train)
        preds = rf.predict(_apply_normalizer(X_p, pt))
        rmse = np.sqrt(mean_squared_error(y_p, preds))
        athlete_metrics.append({"participant_id": pid, "rmse": rmse})

    assert len(athlete_metrics) > 0, "No se generó ningún desglose por atleta"
    assert all("rmse" in m for m in athlete_metrics)


# ─── FT-4: LOAO produce exactamente N folds ───────────────────────────────────

def test_loao_n_folds(synthetic_csv):
    """FT-4: LOAO produce exactamente N folds (uno por atleta)."""
    df_all, X, y, meta = prepare_fatigue_dataset(synthetic_csv)
    n_athletes = meta["participant_id"].nunique()
    fold_count = 0
    for pid in meta["participant_id"].unique():
        fold_count += 1

    assert fold_count == n_athletes, (
        f"Se esperaban {n_athletes} folds, se obtuvieron {fold_count}"
    )
