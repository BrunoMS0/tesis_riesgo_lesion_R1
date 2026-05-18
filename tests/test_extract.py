"""
test_extract.py — Pruebas unitarias del módulo ETL Extract (Runner Dataset).

Cubre los 9 tests del catálogo Anexo G (IDs E-1 a E-9).
Usa datos sintéticos: CSV temporal con 3 atletas × 10 días.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.runner.etl.extract import load_runner_csv
from src.runner.config import RAW_FEATURE_BASE_NAMES, DAY_SUFFIXES

# ─── Fixture: CSV sintético ────────────────────────────────────────────────────

def _build_synthetic_csv(n_athletes: int = 3, n_days: int = 10) -> str:
    """Genera un CSV temporal con formato wide idéntico al Runner Dataset."""
    rows = []
    for aid in range(n_athletes):
        for day in range(n_days):
            row = {"Athlete ID": aid, "Date": day, "injury": int(day == 5)}
            for base in RAW_FEATURE_BASE_NAMES:
                for suffix in DAY_SUFFIXES:
                    col = base if suffix == "" else f"{base}{suffix}"
                    row[col] = np.random.uniform(0.5, 8.0)
            rows.append(row)

    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def synthetic_csv():
    path = _build_synthetic_csv()
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def loaded_df(synthetic_csv):
    return load_runner_csv(synthetic_csv)


# ─── E-1: retorna DataFrame no vacío ─────────────────────────────────────────

def test_returns_dataframe(loaded_df):
    """E-1: extract() retorna un DataFrame no vacío."""
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) > 0


# ─── E-2: columnas obligatorias presentes ────────────────────────────────────

def test_required_columns(loaded_df):
    """E-2: Columnas obligatorias presentes: participant_id, date, injury."""
    for col in ("participant_id", "date", "injury"):
        assert col in loaded_df.columns, f"Columna '{col}' faltante"


# ─── E-3: conteo de atletas ≥ 1 ──────────────────────────────────────────────

def test_athlete_count(loaded_df):
    """E-3: Se detectan N atletas ≥ 1 en el dataset sintético."""
    n = loaded_df["participant_id"].nunique()
    assert n >= 1


# ─── E-4: tipo de columna date ───────────────────────────────────────────────

def test_date_column_type(loaded_df):
    """E-4: Columna date es numérica (int o float — índice de día por atleta)."""
    assert pd.api.types.is_numeric_dtype(loaded_df["date"]), (
        "La columna 'date' debe ser numérica (índice de día)"
    )


# ─── E-5: sin duplicados por (athlete_id, date) ───────────────────────────────

def test_no_duplicate_dates(loaded_df):
    """E-5: Sin duplicados por (participant_id, date)."""
    dupes = loaded_df.duplicated(subset=["participant_id", "date"]).sum()
    assert dupes == 0, f"Se encontraron {dupes} duplicados por (participant_id, date)"


# ─── E-6: columnas con sufijos .1 a .6 presentes ────────────────────────────

def test_wide_format_columns(loaded_df):
    """E-6: Columnas con sufijos .1 a .6 presentes (ej. 'total km.1')."""
    sample_base = "total km"
    for suffix in [".1", ".2", ".3", ".4", ".5", ".6"]:
        col = f"{sample_base}{suffix}"
        assert col in loaded_df.columns, f"Columna ancha '{col}' faltante"


# ─── E-7: 10 variables GPS base presentes ────────────────────────────────────

def test_ten_base_features(loaded_df):
    """E-7: 10 variables GPS base presentes en el DataFrame."""
    assert len(RAW_FEATURE_BASE_NAMES) == 10
    for base in RAW_FEATURE_BASE_NAMES:
        assert base in loaded_df.columns, f"Feature base '{base}' faltante"


# ─── E-8: columna injury binaria ─────────────────────────────────────────────

def test_injury_column_binary(loaded_df):
    """E-8: Columna injury contiene solo valores 0 y 1."""
    unique_vals = set(loaded_df["injury"].unique())
    assert unique_vals.issubset({0, 1}), f"Valores inesperados en injury: {unique_vals}"


# ─── E-9: CSV vacío no interrumpe la carga ───────────────────────────────────

def test_handles_missing_athletes():
    """E-9: Un CSV vacío (solo encabezado) no lanza excepción durante la carga."""
    cols = ["Athlete ID", "Date", "injury"]
    for base in RAW_FEATURE_BASE_NAMES:
        for suffix in DAY_SUFFIXES:
            cols.append(base if suffix == "" else f"{base}{suffix}")

    df_empty = pd.DataFrame(columns=cols)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df_empty.to_csv(tmp.name, index=False)
        path = tmp.name

    try:
        result = load_runner_csv(path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    finally:
        os.unlink(path)
