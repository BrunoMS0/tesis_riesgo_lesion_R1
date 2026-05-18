"""
test_transform.py — Pruebas unitarias del módulo ETL Transform (Runner Dataset).

Cubre los 13 tests del catálogo Anexo G (IDs T-1 a T-13).
Usa datos sintéticos pasados por el extractor y luego transformados.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.runner.etl.extract import load_runner_csv
from src.runner.etl.transform import compute_features
from src.runner.config import RAW_FEATURE_BASE_NAMES, DAY_SUFFIXES, RUNNER_FEATURE_COLUMNS


# ─── Fixture: DataFrame ya transformado ──────────────────────────────────────

def _build_synthetic_df(n_athletes: int = 3, n_days: int = 30) -> pd.DataFrame:
    """Genera DataFrame sintético con la estructura wide del Runner Dataset."""
    rows = []
    rng = np.random.default_rng(42)
    for aid in range(n_athletes):
        for day in range(n_days):
            row = {"Athlete ID": aid, "Date": day, "injury": int(day % 20 == 0)}
            for base in RAW_FEATURE_BASE_NAMES:
                for suffix in DAY_SUFFIXES:
                    col = base if suffix == "" else f"{base}{suffix}"
                    row[col] = rng.uniform(0.5, 8.0)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def raw_df():
    return _build_synthetic_df()


@pytest.fixture(scope="module")
def raw_csv_path(raw_df):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        raw_df.to_csv(tmp.name, index=False)
        path = tmp.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def transformed_df(raw_csv_path):
    raw = load_runner_csv(raw_csv_path)
    return compute_features(raw)


# ─── T-1: sin valores nulos en columnas numéricas ─────────────────────────────

def test_no_numeric_nulls(transformed_df):
    """T-1: Sin valores nulos en columnas numéricas tras transformación."""
    num_cols = transformed_df.select_dtypes(include=[np.number]).columns
    null_counts = transformed_df[num_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, f"Columnas con nulos: {cols_with_nulls.to_dict()}"


# ─── T-2: columna acwr presente ──────────────────────────────────────────────

def test_acwr_column_exists(transformed_df):
    """T-2: Columna acwr presente en el DataFrame transformado."""
    assert "acwr" in transformed_df.columns


# ─── T-3: acwr finito donde chronic_load_28d > 0 ─────────────────────────────

def test_acwr_finite(transformed_df):
    """T-3: acwr es finito en todas las filas donde chronic_load_28d > 0."""
    mask = transformed_df["chronic_load_28d"] > 0
    if mask.any():
        assert np.isfinite(transformed_df.loc[mask, "acwr"]).all(), (
            "acwr tiene valores no-finitos donde chronic_load_28d > 0"
        )


# ─── T-4: acute_load_7d presente ─────────────────────────────────────────────

def test_acute_load_7d_exists(transformed_df):
    """T-4: Columna acute_load_7d presente."""
    assert "acute_load_7d" in transformed_df.columns


# ─── T-5: chronic_load_28d presente ──────────────────────────────────────────

def test_chronic_load_28d_exists(transformed_df):
    """T-5: Columna chronic_load_28d presente."""
    assert "chronic_load_28d" in transformed_df.columns


# ─── T-6: session_load_proxy presente ────────────────────────────────────────

def test_session_load_proxy_exists(transformed_df):
    """T-6: Columna session_load_proxy presente."""
    assert "session_load_proxy" in transformed_df.columns


# ─── T-7: wellness_score presente ────────────────────────────────────────────

def test_wellness_score_exists(transformed_df):
    """T-7: Columna wellness_score presente (proxy subjetivo)."""
    assert "wellness_score" in transformed_df.columns


# ─── T-8: se generan ≥ 18 features derivadas ─────────────────────────────────

def test_derived_features_count(transformed_df):
    """T-8: Se generan ≥ 18 features derivadas (RUNNER_FEATURE_COLUMNS)."""
    derived = [c for c in RUNNER_FEATURE_COLUMNS if c in transformed_df.columns]
    assert len(derived) >= 18, (
        f"Solo {len(derived)} de {len(RUNNER_FEATURE_COLUMNS)} features derivadas presentes"
    )


# ─── T-9: columna injury preservada ──────────────────────────────────────────

def test_target_preserved(transformed_df):
    """T-9: Columna injury preservada tras transformación."""
    assert "injury" in transformed_df.columns
    assert transformed_df["injury"].isin([0, 1]).all()


# ─── T-10: no se crea ningún archivo .tfrecord ───────────────────────────────

def test_no_tfrecord_created(raw_csv_path):
    """T-10: NO se crea ningún archivo .tfrecord durante la transformación."""
    raw = load_runner_csv(raw_csv_path)
    compute_features(raw)
    # Verificar que no hay tfrecord en outputs/
    outputs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "outputs"
    )
    if os.path.isdir(outputs_dir):
        tfrecord_files = [f for f in os.listdir(outputs_dir) if f.endswith(".tfrecord")]
        assert len(tfrecord_files) == 0, f"TFRecord inesperado: {tfrecord_files}"


# ─── T-11: participant_id preservado ─────────────────────────────────────────

def test_athlete_id_preserved(transformed_df):
    """T-11: Columna participant_id preservada tras transformación."""
    assert "participant_id" in transformed_df.columns
    assert transformed_df["participant_id"].notna().all()


# ─── T-12: date preservada ───────────────────────────────────────────────────

def test_date_preserved(transformed_df):
    """T-12: Columna date preservada tras transformación."""
    assert "date" in transformed_df.columns
    assert transformed_df["date"].notna().all()


# ─── T-13: compute_features retorna DataFrame ────────────────────────────────

def test_returns_transform_result(raw_csv_path):
    """T-13: compute_features() retorna un DataFrame válido (no None, no vacío)."""
    raw = load_runner_csv(raw_csv_path)
    result = compute_features(raw)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert len(result.columns) > 3
