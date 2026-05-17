"""
transform.py — Paso 2 del pipeline ETL: ingeniería de features.

Convierte el formato ancho del Runner Dataset (10 variables × 7 sufijos de día
por fila) en un conjunto compacto de features derivadas listas para modelar.

Convención de sufijos en el CSV fuente
---------------------------------------
  Sin sufijo   = D-7  (día más antiguo de la ventana)
  '.1'         = D-6
  '.2'         = D-5
  '.3'         = D-4
  '.4'         = D-3
  '.5'         = D-2
  '.6'         = D-1  (ayer, día más reciente)

Los días de descanso se marcan con perceived exertion == -0.01 (README del dataset).
Se excluyen del cómputo de medias de wellness pero no de las sumas de carga.

Módulo anterior en el pipeline
---------------------------------
  src/runner/etl/extract.py → load_runner_csv()

Public API
----------
    compute_features(df) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import REST_DAY_VALUE, SUFFIX_TO_OFFSET

logger = logging.getLogger(__name__)

# Sufijos ordenados de D-7 (más antiguo) a D-1 (más reciente)
_SUFFIXES = ["", ".1", ".2", ".3", ".4", ".5", ".6"]


# ─── Utilidades internas ──────────────────────────────────────────────────────

def _col(base: str, suffix: str) -> str:
    """Devuelve el nombre completo de columna para un feature base y un sufijo de día."""
    return base if suffix == "" else f"{base}{suffix}"


def _sum_across_days(df: pd.DataFrame, base: str) -> pd.Series:
    """Suma un feature a través de los 7 días de la ventana (D-7 a D-1)."""
    return sum(df[_col(base, s)] for s in _SUFFIXES)


def _mean_excl_rest(df: pd.DataFrame, base: str) -> pd.Series:
    """
    Media de un feature subjetivo a través de 7 días, excluyendo días de descanso.

    Los días de descanso se identifican por perceived exertion == REST_DAY_VALUE (-0.01).
    La misma máscara se aplica a los features de recovery y trainingSuccess.
    """
    exertion_cols = [df[_col("perceived exertion", s)] for s in _SUFFIXES]
    feature_cols  = [df[_col(base, s)] for s in _SUFFIXES]

    masked = [
        feat.where(ex != REST_DAY_VALUE, other=np.nan)
        for ex, feat in zip(exertion_cols, feature_cols)
    ]
    stacked = pd.concat(masked, axis=1)
    return stacked.mean(axis=1)


def _count_rest_days(df: pd.DataFrame) -> pd.Series:
    """Cuenta días en la ventana de 7 días donde perceived exertion == REST_DAY_VALUE."""
    flags = [
        (df[_col("perceived exertion", s)] == REST_DAY_VALUE).astype(int)
        for s in _SUFFIXES
    ]
    return sum(flags)


# ─── ACWR: reconstrucción de la serie temporal diaria de km ──────────────────

def _compute_chronic_load_28d(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula chronic_load_28d para cada fila reconstruyendo la serie diaria de km por atleta.

    Para una fila en la fecha D:
      - Las 7 columnas anchas codifican km para los días D-7 a D-1.
      - Se reconstruye la serie diaria de km por atleta, rellenando huecos con 0.
      - La suma rolling de 28 días en D-1 da la carga crónica.

    Returns
    -------
    np.ndarray de forma (len(df),) con chronic_load_28d por fila.
    """
    pieces = []
    for suffix, offset in SUFFIX_TO_OFFSET.items():
        col = _col("total km", suffix)
        tmp = df[["participant_id", "date", col]].copy()
        tmp["actual_day"] = df["date"] - offset
        tmp = tmp.rename(columns={col: "km"})
        pieces.append(tmp[["participant_id", "actual_day", "km"]])

    daily_km = pd.concat(pieces, ignore_index=True)
    daily_km = daily_km.drop_duplicates(["participant_id", "actual_day"])
    daily_km = daily_km.sort_values(["participant_id", "actual_day"]).reset_index(drop=True)

    result_parts = []
    for pid, group in daily_km.groupby("participant_id"):
        day_min = int(group["actual_day"].min())
        day_max = int(group["actual_day"].max())

        full_days = pd.DataFrame({
            "actual_day": range(day_min, day_max + 1),
            "participant_id": pid,
        })
        merged = full_days.merge(group[["actual_day", "km"]], on="actual_day", how="left")
        merged["km"] = merged["km"].fillna(0.0)
        merged["rolling_28d"] = merged["km"].rolling(window=28, min_periods=1).sum()
        result_parts.append(merged[["participant_id", "actual_day", "rolling_28d"]])

    daily_rolling = pd.concat(result_parts, ignore_index=True)

    lookup = daily_rolling.rename(
        columns={"actual_day": "lookup_day", "rolling_28d": "chronic_load_28d"}
    )
    df_lookup = df[["participant_id", "date"]].copy()
    df_lookup["lookup_day"] = df_lookup["date"] - 1
    df_lookup = df_lookup.merge(lookup, on=["participant_id", "lookup_day"], how="left")

    # Fallback para filas de inicio de temporada sin historial de 28 días
    acute_fallback = _sum_across_days(df, "total km").values
    chronic = df_lookup["chronic_load_28d"].fillna(pd.Series(acute_fallback)).values

    return chronic


# ─── Función principal de ingeniería de features ──────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcular todas las features derivadas desde el DataFrame en formato ancho de 7 días.

    Entrada : DataFrame de extract.load_runner_csv()
    Salida  : DataFrame con [participant_id, date, injury] + RUNNER_FEATURE_COLUMNS.
              Las 70 columnas anchas del CSV fuente son eliminadas.

    Manejo de NaN
    -------------
    - La exclusión de días de descanso puede dejar NaN para atletas con todas las
      ventanas de descanso.
    - Se aplica forward/backward fill por atleta, luego relleno con 0.
    """
    out = df[["participant_id", "date", "injury"]].copy()

    # ── Carga de entrenamiento ─────────────────────────────────────────────────
    out["acute_load_7d"] = _sum_across_days(df, "total km")
    out["chronic_load_28d"] = _compute_chronic_load_28d(df)

    # ACWR = aguda / (crónica/4).  Crónica/4 = carga semanal promedio.
    # Recortado a [0, 5] para limitar valores extremos en filas de inicio.
    weekly_chronic_avg = (out["chronic_load_28d"] / 4.0).clip(lower=0.01)
    out["acwr"] = (out["acute_load_7d"] / weekly_chronic_avg).clip(upper=5.0)

    out["high_intensity_km_7d"] = (
        _sum_across_days(df, "km Z3-4") + _sum_across_days(df, "km Z5-T1-T2")
    )
    out["km_sprint_7d"]     = _sum_across_days(df, "km sprinting")
    out["nr_sessions_7d"]   = _sum_across_days(df, "nr. sessions")
    out["nr_rest_days_7d"]  = _count_rest_days(df)
    out["strength_days_7d"] = _sum_across_days(df, "strength training")
    out["alt_hours_7d"]     = _sum_across_days(df, "hours alternative")

    # ── Wellness subjetivo (medias excluyendo días de descanso) ───────────────
    out["mean_perceived_exertion"] = _mean_excl_rest(df, "perceived exertion")
    out["mean_perceived_recovery"] = _mean_excl_rest(df, "perceived recovery")
    out["mean_perceived_success"]  = _mean_excl_rest(df, "perceived trainingSuccess")

    # wellness_score: compuesto de recovery + success (ambas señales positivas)
    out["wellness_score"] = (
        out["mean_perceived_recovery"].fillna(0) +
        out["mean_perceived_success"].fillna(0)
    ) / 2.0

    # session_load_proxy = volumen agudo × esfuerzo percibido medio (análogo sRPE)
    out["session_load_proxy"] = (
        out["acute_load_7d"] * out["mean_perceived_exertion"].fillna(0)
    )

    # ── Features del día más reciente (D-1, sufijo '.6') ──────────────────────
    out["recent_exertion"] = df["perceived exertion.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_recovery"] = df["perceived recovery.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_success"]  = df["perceived trainingSuccess.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_km"]       = df["total km.6"]

    # ── Imputar NaN residuales (ventanas todo-descanso, filas de arranque) ────
    float_cols = [c for c in out.columns if c not in ("participant_id", "date", "injury")]
    out[float_cols] = (
        out.groupby("participant_id")[float_cols]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0.0)
    )

    logger.info(
        "Ingeniería de features completa: %d filas, %d features derivadas, "
        "prevalencia lesión=%.2f%% (%d eventos)",
        len(out), len(float_cols),
        100.0 * out["injury"].mean(), int(out["injury"].sum()),
    )
    return out
