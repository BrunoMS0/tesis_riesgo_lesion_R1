"""
pipeline.py — Orquestador ETL del Runner Dataset (Löwdal 2021).

Ejecuta las etapas del pipeline de datos en orden secuencial:

  Paso 1 — extract.py   : Carga el CSV fuente, estandariza columnas de identidad
  Paso 2 — transform.py : Ingeniería de features (ACWR, cargas, wellness, D-1)
  Paso 3 — [Persistencia]: Guarda el dataset procesado en src/outputs/ (opcional)

Este módulo es el punto de entrada único para reproducir el dataset procesado
que alimenta los modelos M1 (fatigue.py) y M2 (injury.py).

Reproducibilidad
----------------
    python -c "from src.runner.etl.pipeline import run_etl_pipeline; run_etl_pipeline()"

O desde un script:
    python run_runner.py   # llama internamente a build_runner_datasets() → run_etl_pipeline()

Public API
----------
    run_etl_pipeline(csv_path, save_output, output_path) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import RUNNER_CSV, RUNNER_OUTPUT_CSV
from .extract import load_runner_csv
from .transform import compute_features

logger = logging.getLogger(__name__)


def run_etl_pipeline(
    csv_path: str = RUNNER_CSV,
    *,
    save_output: bool = True,
    output_path: str = RUNNER_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Ejecutar el pipeline ETL completo: extract → transform → [persistencia].

    Parameters
    ----------
    csv_path    : Ruta al CSV fuente day_approach_maskedID_timeseries.csv.
                  Por defecto usa la ruta definida en config.py.
    save_output : Si True, guarda el dataset procesado en output_path.
    output_path : Ruta donde guardar el CSV procesado.
                  Por defecto: src/outputs/runner_dataset_processed.csv

    Returns
    -------
    pd.DataFrame con columnas:
      [participant_id, date, injury] + RUNNER_FEATURE_COLUMNS

    Ejemplo
    -------
        from src.runner.etl.pipeline import run_etl_pipeline
        df = run_etl_pipeline(save_output=True)
        print(df.shape)  # (N_rows, 22)
    """
    logger.info("=" * 55)
    logger.info("ETL — Runner Dataset (Löwdal 2021)")
    logger.info("=" * 55)

    # ── Paso 1: Extracción ─────────────────────────────────────────────────────
    logger.info("Paso 1/2: Extracción → cargando CSV fuente...")
    raw_df = load_runner_csv(csv_path)
    logger.info(
        "  %d atletas | %d filas | %d columnas en formato ancho",
        raw_df["participant_id"].nunique(), len(raw_df), len(raw_df.columns),
    )

    # ── Paso 2: Transformación + Ingeniería de Features ───────────────────────
    logger.info("Paso 2/2: Transformación → ingeniería de features...")
    df = compute_features(raw_df)
    feature_cols = [c for c in df.columns if c not in ("participant_id", "date", "injury")]
    logger.info(
        "  %d features derivadas | prevalencia lesión=%.2f%% (%d eventos)",
        len(feature_cols),
        100.0 * df["injury"].mean(), int(df["injury"].sum()),
    )

    # ── Paso 3: Persistencia (opcional) ───────────────────────────────────────
    if save_output:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Dataset procesado guardado → %s (%d filas)", output_path, len(df))

    logger.info("=" * 55)
    logger.info("ETL completado: %d atletas, %d filas totales", df["participant_id"].nunique(), len(df))
    logger.info("=" * 55)

    return df
