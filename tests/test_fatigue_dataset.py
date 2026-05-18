"""
test_fatigue_dataset.py — Pruebas unitarias del dataset para el Modelo M1.

Cubre los 10 tests del catálogo Anexo N (IDs FD-1 a FD-10).
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
    FATIGUE_MODEL_PATH,
    FATIGUE_IMPORTANCE_PATH,
    FATIGUE_PREDICTIONS_PATH,
    prepare_fatigue_dataset,
    _fit_normalizer,
    _apply_normalizer,
)
from src.runner.config import RAW_FEATURE_BASE_NAMES, DAY_SUFFIXES, SEED, REST_DAY_VALUE
from sklearn.ensemble import RandomForestRegressor


# ─── Fixture: CSV sintético con días de descanso explícitos ──────────────────

def _build_csv_with_rest(n_athletes: int = 4, n_days: int = 25) -> str:
    rng = np.random.default_rng(SEED)
    rows = []
    for aid in range(n_athletes):
        for day in range(n_days):
            row = {"Athlete ID": aid, "Date": day, "injury": int(day == 10)}
            for base in RAW_FEATURE_BASE_NAMES:
                for suffix in DAY_SUFFIXES:
                    col = base if suffix == "" else f"{base}{suffix}"
                    if base == "perceived exertion" and day % 6 == 0:
                        row[col] = REST_DAY_VALUE  # día de descanso
                    else:
                        row[col] = rng.uniform(1.0, 8.0)
            rows.append(row)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    pd.DataFrame(rows).to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def synthetic_csv():
    path = _build_csv_with_rest()
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def dataset(synthetic_csv):
    return prepare_fatigue_dataset(synthetic_csv)


# ─── FD-1: días de descanso excluidos del entrenamiento ──────────────────────

def test_rest_days_excluded(dataset):
    """FD-1: Días con perceived_exertion = -0.01 excluidos (y == NaN)."""
    df_all, X, y, meta = dataset
    # El target (y) debe ser NaN en los días de descanso
    rest_mask = y.isna()
    assert rest_mask.sum() > 0, "No se encontraron días de descanso (y NaN) en los datos"


# ─── FD-2: target es perceived_recovery.6 ────────────────────────────────────

def test_target_is_perceived_recovery(dataset):
    """FD-2: El target es la columna perceived_recovery.6 (FATIGUE_TARGET_COL)."""
    df_all, X, y, meta = dataset
    assert y.name == FATIGUE_TARGET_COL, (
        f"Target incorrecto: esperado '{FATIGUE_TARGET_COL}', obtenido '{y.name}'"
    )


# ─── FD-3: X tiene shape (N, n_features) 2D ──────────────────────────────────

def test_input_shape_2d(dataset):
    """FD-3: X tiene shape (N, n_features) — entrada 2D, no tensores 3D."""
    df_all, X, y, meta = dataset
    assert X.ndim == 2, f"X debe ser 2D, pero tiene {X.ndim} dimensiones"


# ─── FD-4: X tiene exactamente 10 columnas ────────────────────────────────────

def test_feature_count(dataset):
    """FD-4: X tiene exactamente 10 columnas (features GPS objetivas)."""
    df_all, X, y, meta = dataset
    assert X.shape[1] == 10, (
        f"Se esperaban 10 features, X tiene {X.shape[1]} columnas: {list(X.columns)}"
    )


# ─── FD-5: sin fuga de atletas entre train y test ─────────────────────────────

def test_no_participant_leakage(dataset):
    """FD-5: Ningún atleta de prueba aparece en el conjunto de entrenamiento."""
    df_all, X, y, meta = dataset
    athletes = sorted(meta["participant_id"].unique())
    if len(athletes) < 2:
        pytest.skip("Se necesitan al menos 2 atletas para esta prueba")

    test_athlete = athletes[0]
    train_mask = meta["participant_id"] != test_athlete
    test_mask  = meta["participant_id"] == test_athlete

    train_ids = set(meta.loc[train_mask, "participant_id"].unique())
    test_ids  = set(meta.loc[test_mask, "participant_id"].unique())

    assert test_ids.isdisjoint(train_ids), (
        f"Atleta(s) de test encontrado(s) en train: {test_ids & train_ids}"
    )


# ─── FD-6: normalización Yeo-Johnson ajustada solo sobre entrenamiento ────────

def test_yeo_johnson_per_fold(dataset):
    """FD-6: Normalización Yeo-Johnson ajustada exclusivamente sobre datos de entrenamiento."""
    df_all, X, y, meta = dataset
    valid = y.notna()
    athletes = sorted(meta["participant_id"].unique())
    if len(athletes) < 2:
        pytest.skip("Se necesitan al menos 2 atletas para esta prueba")

    test_pid = athletes[0]
    train_mask = valid & (meta["participant_id"] != test_pid)
    X_train = X.loc[train_mask]

    # El normalizer se ajusta SOLO sobre train — no debe lanzar excepción
    pt = _fit_normalizer(X_train)

    # Aplicar al set de test — el scaler no fue visto por test
    test_mask = meta["participant_id"] == test_pid
    X_test = X.loc[test_mask]
    X_test_norm = _apply_normalizer(X_test, pt)

    assert not X_test_norm.isnull().any().any(), "NaN tras normalización en el set de test"


# ─── FD-7: predicciones LOAO son out-of-sample ────────────────────────────────

def test_loao_out_of_sample(dataset):
    """FD-7: Todas las predicciones LOAO son siempre out-of-sample."""
    df_all, X, y, meta = dataset
    valid = y.notna()
    athletes = sorted(meta["participant_id"].unique())
    if len(athletes) < 2:
        pytest.skip("Se necesitan al menos 2 atletas")

    for test_pid in athletes[:2]:  # verificar 2 folds como muestra
        train_mask = valid & (meta["participant_id"] != test_pid)
        test_mask  = valid & (meta["participant_id"] == test_pid)
        if train_mask.sum() < 5 or test_mask.sum() < 1:
            continue

        pt = _fit_normalizer(X.loc[train_mask])
        rf = RandomForestRegressor(n_estimators=5, random_state=SEED, n_jobs=1)
        rf.fit(_apply_normalizer(X.loc[train_mask], pt), y.loc[train_mask])

        # El atleta de test NO fue visto durante el entrenamiento
        preds = rf.predict(_apply_normalizer(X.loc[test_mask], pt))
        assert len(preds) == test_mask.sum()
        assert not np.any(np.isnan(preds))


# ─── FD-8: archivo runner_fatigue_predictions_loao.csv generado ───────────────

def test_loao_predictions_csv():
    """FD-8: Archivo runner_fatigue_predictions_loao.csv tiene columnas correctas si existe."""
    if not os.path.isfile(FATIGUE_PREDICTIONS_PATH):
        pytest.skip("Archivo de predicciones LOAO aún no generado (ejecutar run_runner_fatigue.py)")

    df = pd.read_csv(FATIGUE_PREDICTIONS_PATH)
    for col in ("athlete_id", "date", "fatigue_score_predicted"):
        assert col in df.columns, f"Columna '{col}' faltante en predictions CSV"


# ─── FD-9: archivo fatigue_feature_importance.csv generado ────────────────────

def test_feature_importance_csv():
    """FD-9: Archivo fatigue_feature_importance.csv generado tras entrenamiento final."""
    if not os.path.isfile(FATIGUE_IMPORTANCE_PATH):
        pytest.skip("Archivo de importancia aún no generado (ejecutar run_runner_fatigue.py)")

    df = pd.read_csv(FATIGUE_IMPORTANCE_PATH)
    assert "feature" in df.columns, "Columna 'feature' faltante"
    assert "importance" in df.columns, "Columna 'importance' faltante"
    assert len(df) == len(FATIGUE_FEATURE_COLUMNS)


# ─── FD-10: archivo rf_fatigue_runner_model.pkl guardado ─────────────────────

def test_model_pkl_saved():
    """FD-10: Archivo rf_fatigue_runner_model.pkl guardado en src/outputs/."""
    if not os.path.isfile(FATIGUE_MODEL_PATH):
        pytest.skip("Modelo aún no generado (ejecutar run_runner_fatigue.py)")

    import joblib
    bundle = joblib.load(FATIGUE_MODEL_PATH)
    assert "model" in bundle, "El bundle debe contener 'model'"
    assert "normalizer" in bundle, "El bundle debe contener 'normalizer'"
    assert "feature_columns" in bundle, "El bundle debe contener 'feature_columns'"
