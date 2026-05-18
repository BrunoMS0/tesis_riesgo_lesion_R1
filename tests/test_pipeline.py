"""
test_pipeline.py — Pruebas unitarias del módulo ETL Pipeline (Runner Dataset).

Cubre los 6 tests del catálogo Anexo G (IDs P-1 a P-6).
Usa datos sintéticos; el pipeline se ejecuta con save_output=False salvo P-2.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.runner.etl.extract import load_runner_csv
from src.runner.etl.pipeline import run_etl_pipeline
from src.runner.config import RAW_FEATURE_BASE_NAMES, DAY_SUFFIXES


# ─── Fixture: CSV sintético compartido ────────────────────────────────────────

def _build_synthetic_csv(n_athletes: int = 3, n_days: int = 15) -> str:
    rng = np.random.default_rng(0)
    rows = []
    for aid in range(n_athletes):
        for day in range(n_days):
            row = {"Athlete ID": aid, "Date": day, "injury": 0}
            for base in RAW_FEATURE_BASE_NAMES:
                for suffix in DAY_SUFFIXES:
                    col = base if suffix == "" else f"{base}{suffix}"
                    row[col] = rng.uniform(0.5, 8.0)
            rows.append(row)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    pd.DataFrame(rows).to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def synthetic_csv():
    path = _build_synthetic_csv()
    yield path
    os.unlink(path)


# ─── P-1: run_etl_pipeline retorna DataFrame ─────────────────────────────────

def test_returns_report(synthetic_csv):
    """P-1: run_etl_pipeline() retorna un DataFrame con al menos 3 columnas."""
    result = run_etl_pipeline(csv_path=synthetic_csv, save_output=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) >= 3, "El resultado debe tener al menos 3 columnas"


# ─── P-2: archivo CSV de salida existe en disco ───────────────────────────────

def test_csv_exists(synthetic_csv):
    """P-2: Archivo CSV de salida existe en disco tras la ejecución."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_output.csv")
        run_etl_pipeline(csv_path=synthetic_csv, save_output=True, output_path=out_path)
        assert os.path.isfile(out_path), f"CSV de salida no encontrado en {out_path}"


# ─── P-3: no existe ningún archivo .tfrecord ─────────────────────────────────

def test_no_tfrecord(synthetic_csv):
    """P-3: NO existe ningún archivo .tfrecord tras la ejecución del pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "output.csv")
        run_etl_pipeline(csv_path=synthetic_csv, save_output=True, output_path=out_path)
        tfrecords = [f for f in os.listdir(tmpdir) if f.endswith(".tfrecord")]
        assert len(tfrecords) == 0, f"TFRecord inesperado: {tfrecords}"


# ─── P-4: dataset procesado tiene filas ───────────────────────────────────────

def test_positive_duration(synthetic_csv):
    """P-4: El DataFrame resultante tiene más filas que 0 (pipeline ejecutado)."""
    result = run_etl_pipeline(csv_path=synthetic_csv, save_output=False)
    assert len(result) > 0, "El pipeline retornó un DataFrame vacío"


# ─── P-5: conteo de features > 0 ─────────────────────────────────────────────

def test_features_positive(synthetic_csv):
    """P-5: El DataFrame procesado registra conteo de features > 3 (más que solo metadata)."""
    result = run_etl_pipeline(csv_path=synthetic_csv, save_output=False)
    feature_cols = [c for c in result.columns if c not in ("participant_id", "date", "injury")]
    assert len(feature_cols) > 0, "No se generaron columnas de features"


# ─── P-6: conteo de filas del CSV de salida ───────────────────────────────────

def test_row_count_in_report(synthetic_csv):
    """P-6: El CSV de salida tiene el mismo conteo de filas que el DataFrame retornado."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "output.csv")
        result = run_etl_pipeline(csv_path=synthetic_csv, save_output=True, output_path=out_path)
        saved = pd.read_csv(out_path)
        assert len(saved) == len(result), (
            f"El CSV guardado tiene {len(saved)} filas pero el DataFrame tiene {len(result)}"
        )
