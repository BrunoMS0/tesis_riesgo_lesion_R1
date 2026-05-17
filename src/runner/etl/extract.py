"""
extract.py — Paso 1 del pipeline ETL: carga y estandarización del CSV fuente.

Carga el CSV del Runner Dataset (Löwdal 2021) y estandariza las columnas de
identidad para que sean compatibles con el resto del pipeline:
  - 'Athlete ID' → 'participant_id'  (str, prefijo 'runner_')
  - 'Date'       → 'date'            (int, índice de día por atleta)
  - 'injury'     → sin cambio        (int, etiqueta prospectiva ya correcta)

Las 70 columnas de features en formato ancho se conservan intactas para
su procesamiento posterior en transform.py.

Módulo siguiente en el pipeline
---------------------------------
  src/runner/etl/transform.py → compute_features(df)

Public API
----------
    load_runner_csv(csv_path) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import RUNNER_CSV

logger = logging.getLogger(__name__)


def load_runner_csv(csv_path: str = RUNNER_CSV) -> pd.DataFrame:
    """
    Cargar el CSV del Runner Dataset y renombrar columnas de identidad.

    Parameters
    ----------
    csv_path : Ruta a day_approach_maskedID_timeseries.csv.

    Returns
    -------
    pd.DataFrame con columnas:
      - participant_id : str ('runner_0' … 'runner_73')
      - date           : int  (índice de día secuencial por atleta, desde 0)
      - injury         : int  (0 / 1, ya prospectivo)
      - 70 columnas de features en formato ancho (nombres originales del CSV,
        e.g. 'total km', 'total km.1', ..., 'perceived recovery.6')

    Ordenado por (participant_id, date).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Runner CSV no encontrado: {csv_path}\n"
            f"Ruta esperada: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    logger.info("Cargadas %d filas × %d columnas desde %s", len(df), len(df.columns), csv_path)

    # Renombrar columnas de identidad a convención del pipeline
    df = df.rename(columns={"Athlete ID": "participant_id", "Date": "date"})

    # Convertir IDs numéricos a strings con prefijo 'runner_'.
    # Requerido por loso_cross_validation, que filtra por str.startswith().
    df["participant_id"] = "runner_" + df["participant_id"].astype(str)

    # Ordenar cronológicamente por atleta para secuencias temporales correctas
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)

    n_athletes = df["participant_id"].nunique()
    n_injuries = int(df["injury"].sum())
    n_injured_athletes = int((df.groupby("participant_id")["injury"].max() > 0).sum())

    logger.info(
        "Runner Dataset: %d atletas (%d con ≥1 lesión), "
        "%d eventos de lesión (%.2f%% prevalencia), rango de fechas %d–%d",
        n_athletes, n_injured_athletes,
        n_injuries, 100.0 * df["injury"].mean(),
        int(df["date"].min()), int(df["date"].max()),
    )
    return df
