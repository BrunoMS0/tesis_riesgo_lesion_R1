"""
week_transform.py — Variante semanal del ETL para validación cruzada de granularidad.

Carga el CSV de granularidad semanal (week_approach_maskedID_timeseries.csv) y
renombra sus columnas a los mismos nombres de feature que el pipeline diario.
Esto permite evaluar si el modelo entrenado en granularidad diaria generaliza
a granularidad semanal (validación cruzada de granularidad).

Mapeo semanal → diario
-----------------------
  total kms                → acute_load_7d
  rel total kms week 0_1   → acwr
  nr. sessions             → nr_sessions_7d
  nr. rest days            → nr_rest_days_7d
  total km Z3-Z4-Z5-T1-T2  → high_intensity_km_7d
  nr. strength trainings   → strength_days_7d
  avg recovery             → mean_perceived_recovery
  avg exertion             → mean_perceived_exertion
  avg training success     → mean_perceived_success

Estructura del dataset semanal
--------------------------------
  - Cada fila es una ventana de predicción semanal
  - Columnas sin sufijo = semana más reciente (W0)
  - Columnas con '.1'   = semana anterior (W-1)
  - Columnas con '.2'   = dos semanas atrás (W-2)
  - 'rel total kms week 0_1' = ratio W0/W-1 (proxy del ACWR)
  - 'injury' = etiqueta prospectiva (ya correcta según autores del dataset)

Public API
----------
    load_week_approach(csv_path) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

WEEK_CSV: str = os.path.join(
    _WORKSPACE_ROOT, "Runner dataset", "week_approach_maskedID_timeseries.csv"
)

# Mapeo: nombre de columna semanal → nombre de feature del pipeline diario
WEEK_TO_DAILY_MAP: dict = {
    "total kms":               "acute_load_7d",
    "rel total kms week 0_1":  "acwr",
    "nr. sessions":            "nr_sessions_7d",
    "nr. rest days":           "nr_rest_days_7d",
    "total km Z3-Z4-Z5-T1-T2": "high_intensity_km_7d",
    "nr. strength trainings":  "strength_days_7d",
    "avg recovery":            "mean_perceived_recovery",
    "avg exertion":            "mean_perceived_exertion",
    "avg training success":    "mean_perceived_success",
}

# Lista ordenada de los 9 nombres de features comunes (en nomenclatura del pipeline diario)
WEEK_COMMON_FEATURES: List[str] = list(WEEK_TO_DAILY_MAP.values())


def load_week_approach(csv_path: str = WEEK_CSV) -> pd.DataFrame:
    """
    Cargar el CSV semanal y devolver con nombres de columna estandarizados.

    Columnas de identidad:
      participant_id  : str — 'runner_0' … 'runner_73' (coincide con pipeline diario)
      date            : int — índice de semana secuencial por atleta (desde 'Date')
      injury          : int — 0/1, etiqueta prospectiva

    Columnas de features (9, renombradas de semanal a diario):
      acute_load_7d, acwr, nr_sessions_7d, nr_rest_days_7d,
      high_intensity_km_7d, strength_days_7d,
      mean_perceived_recovery, mean_perceived_exertion, mean_perceived_success

    Parameters
    ----------
    csv_path : Ruta a week_approach_maskedID_timeseries.csv

    Returns
    -------
    pd.DataFrame ordenado por (participant_id, date)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV semanal no encontrado: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    logger.info(
        "Cargado week_approach: %d filas × %d columnas desde %s",
        len(df), len(df.columns), csv_path,
    )

    missing = [c for c in WEEK_TO_DAILY_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en el CSV semanal: {missing}\n"
            f"Disponibles: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["participant_id"] = df["Athlete ID"].apply(lambda x: f"runner_{int(x)}")
    out["date"]           = df["Date"].astype(int)
    out["injury"]         = df["injury"].astype(int)

    for raw_col, daily_col in WEEK_TO_DAILY_MAP.items():
        series = pd.to_numeric(df[raw_col], errors="coerce")
        # Recortar ACWR a [0, 4.0] — el ratio semanal tiene outliers por div/0
        # cuando la semana anterior tiene km == 0 (semana de descanso / inicio).
        if daily_col == "acwr":
            series = series.clip(lower=0.0, upper=4.0)
        out[daily_col] = series.fillna(0.0)

    out = out.sort_values(["participant_id", "date"]).reset_index(drop=True)

    n_athletes = out["participant_id"].nunique()
    n_injured  = out[out["injury"] == 1]["participant_id"].nunique()
    n_injuries = int(out["injury"].sum())

    logger.info(
        "Dataset semanal preparado: %d filas | %d atletas (%d con lesiones) | "
        "%d eventos de lesión (%.2f%%) | %d features comunes",
        len(out), n_athletes, n_injured, n_injuries,
        100.0 * out["injury"].mean(), len(WEEK_COMMON_FEATURES),
    )
    return out
