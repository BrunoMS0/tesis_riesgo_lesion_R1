"""
fatigue.py — M1: Modelo de Estimación de Fatiga/Recuperación (Runner Dataset).

Primera etapa del pipeline predictivo M1→M2. Predice la recuperación percibida
pre-sesión (perceived_recovery D-1) a partir de 10 features objetivas de carga
GPS, sin requerir ningún dato de autoinforme del atleta.

Arquitectura
------------
  Modelo      : Random Forest Regressor (n_estimators=200, max_features='sqrt')
  Preproceso  : PowerTransformer (Yeo-Johnson), ajustado solo sobre entrenamiento
  Validación  : LOAO (Leave-One-Athlete-Out), 75 folds
  Resultados  : RMSE ponderado = 0.1652 | MAE ponderado = 0.1383

Features de entrada (10 — solo objetivas, sin autoinforme)
-----------------------------------------------------------
  acute_load_7d, chronic_load_28d, acwr, high_intensity_km_7d,
  nr_sessions_7d, nr_rest_days_7d, km_sprint_7d, strength_days_7d,
  alt_hours_7d, recent_km

Variable objetivo
-----------------
  recent_recovery  (= perceived_recovery.6, pre-sesión; NaN en días de descanso)

Artefactos generados
---------------------
  src/outputs/loao_fatigue_runner_results.csv      — métricas LOAO por atleta
  src/outputs/rf_fatigue_runner_model.pkl          — modelo final (todos los atletas)
  src/outputs/fatigue_feature_importance.csv       — importancia de features (Gini)
  src/outputs/runner_fatigue_predictions_loao.csv  — fatigue_score_predicted por (atleta, día)

Módulo siguiente en el pipeline
---------------------------------
  src/runner/models/injury.py → build_runner_datasets()
  (consume runner_fatigue_predictions_loao.csv como feature adicional para M2)

Public API
----------
    run_fatigue_pipeline(csv_path, skip_loao) -> dict
    prepare_fatigue_dataset(csv_path)          -> (df_all, X, y, meta)
    run_loao_fatigue(X, y, meta)               -> (results_df, predictions)
    train_final_fatigue_model(X, y)            -> (model, normalizer)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

from ..config import RUNNER_CSV, SEED
from ..etl.extract import load_runner_csv
from ..etl.transform import compute_features

logger = logging.getLogger(__name__)

# ─── Features de entrada (10 variables objetivas GPS) ────────────────────────

FATIGUE_FEATURE_COLUMNS: List[str] = [
    "acute_load_7d",
    "chronic_load_28d",
    "acwr",
    "high_intensity_km_7d",
    "nr_sessions_7d",
    "nr_rest_days_7d",
    "km_sprint_7d",
    "strength_days_7d",
    "alt_hours_7d",
    "recent_km",
]

# perceived_recovery.6 (D-1, pre-sesión) — NaN donde perceived exertion == -0.01
FATIGUE_TARGET_COL: str = "recent_recovery"

_WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

FATIGUE_LOAO_RESULTS: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "loao_fatigue_runner_results.csv"
)
FATIGUE_MODEL_PATH: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "rf_fatigue_runner_model.pkl"
)
FATIGUE_IMPORTANCE_PATH: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "fatigue_feature_importance.csv"
)
FATIGUE_PREDICTIONS_PATH: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "runner_fatigue_predictions_loao.csv"
)

# ─── Hiperparámetros del RF Regressor ─────────────────────────────────────────

RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_LEAF: int = 5

# Umbrales de calidad T7.4
RMSE_TARGET: float = 0.15
RMSE_WARN: float = 0.20


# ─── Preparación del dataset ──────────────────────────────────────────────────

def prepare_fatigue_dataset(
    csv_path: str = RUNNER_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Cargar y preparar el Runner Dataset para la regresión de fatiga.

    El target (perceived_recovery.6) se extrae del CSV RAW antes de que
    compute_features() impute valores NaN de días de descanso mediante
    forward/backward fill. Esto garantiza que el modelo se entrene solo
    sobre recuperaciones reales de días de entrenamiento, no sobre valores imputados.

    Returns
    -------
    df_all : DataFrame procesado completo (todas las columnas, todas las filas)
    X      : DataFrame de FATIGUE_FEATURE_COLUMNS (todas las filas incl. descanso)
    y      : Series de perceived_recovery.6 original (NaN en días de descanso)
    meta   : DataFrame con [participant_id, date]
    """
    from ..config import REST_DAY_VALUE

    raw_df = load_runner_csv(csv_path)
    df = compute_features(raw_df)

    missing = [c for c in FATIGUE_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas de feature faltantes en el dataset procesado: {missing}")

    # Extraer target DIRECTAMENTE del CSV crudo (antes de la imputación de compute_features)
    raw_target_col = "perceived recovery.6"
    if raw_target_col not in raw_df.columns:
        raise ValueError(f"Columna objetivo '{raw_target_col}' no encontrada en el CSV.")

    X    = df[FATIGUE_FEATURE_COLUMNS].copy()
    y    = raw_df[raw_target_col].replace(REST_DAY_VALUE, np.nan).reset_index(drop=True)
    y.name = FATIGUE_TARGET_COL
    meta = df[["participant_id", "date"]].copy()

    n_valid = int(y.notna().sum())
    n_rest  = int(y.isna().sum())
    logger.info(
        "Dataset M1: %d filas | %d días de entrenamiento (target válido) | "
        "%d días de descanso (target NaN, excluidos de evaluación, %.1f%%)",
        len(df), n_valid, n_rest, 100.0 * n_rest / len(df),
    )
    return df, X, y, meta


# ─── Normalización por fold (Yeo-Johnson) ────────────────────────────────────

def _fit_normalizer(X_train: pd.DataFrame) -> PowerTransformer:
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(X_train)
    return pt


def _apply_normalizer(X: pd.DataFrame, pt: PowerTransformer) -> pd.DataFrame:
    return pd.DataFrame(
        pt.transform(X),
        columns=X.columns,
        index=X.index,
    )


# ─── Núcleo LOAO (un solo paso — evalúa Y genera predicciones) ───────────────

def _loao_single_pass(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> Tuple[List[dict], pd.Series]:
    """
    Loop interno LOAO — un paso por todos los atletas.

    Returns
    -------
    fold_rows   : lista de dicts con métricas por atleta (para el CSV de resultados)
    predictions : Series indexada como X — fatigue_score_predicted para TODAS las filas
                  (predicciones del modelo que nunca vio ese atleta)
    """
    pids        = sorted(meta["participant_id"].unique())
    predictions = pd.Series(
        index=X.index, dtype=float, name="fatigue_score_predicted"
    )
    fold_rows: List[dict] = []

    for i, held_out_pid in enumerate(pids, 1):
        mask_test  = (meta["participant_id"] == held_out_pid).values
        mask_train = ~mask_test

        X_train = X.iloc[mask_train]
        y_train = y.iloc[mask_train]
        X_test  = X.iloc[mask_test]
        y_test  = y.iloc[mask_test]

        # Filtrar días de descanso del ENTRENAMIENTO (y NaN = día de descanso)
        train_valid_mask = y_train.notna()
        X_train_fit = X_train.loc[train_valid_mask]
        y_train_fit = y_train.loc[train_valid_mask]

        # Para evaluación: solo días de entrenamiento del conjunto de prueba
        test_valid_mask = y_test.notna()
        X_test_eval = X_test.loc[test_valid_mask]
        y_test_eval = y_test.loc[test_valid_mask]

        if len(X_train_fit) < 10:
            logger.warning(
                "[%d/%d] %s — muy pocas muestras de entrenamiento (%d), omitiendo",
                i, len(pids), held_out_pid, len(X_train_fit),
            )
            fold_rows.append(_skipped_row(held_out_pid, len(y_test), int(test_valid_mask.sum())))
            continue

        # Ajustar normalizador SOLO en días de entrenamiento
        pt           = _fit_normalizer(X_train_fit)
        X_train_norm = _apply_normalizer(X_train_fit, pt)
        X_test_norm  = _apply_normalizer(X_test, pt)  # TODAS las filas de prueba (incl. descanso)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train_norm, y_train_fit)

        # Guardar predicciones para TODAS las filas de prueba (sin filtrar NaN en inferencia)
        pred_values = rf.predict(X_test_norm)
        predictions.iloc[np.where(mask_test)[0]] = pred_values

        # Evaluar solo en días de entrenamiento del conjunto de prueba
        if int(test_valid_mask.sum()) < 2:
            logger.warning(
                "[%d/%d] %s — muy pocas muestras de evaluación (%d), omitiendo métricas",
                i, len(pids), held_out_pid, int(test_valid_mask.sum()),
            )
            fold_rows.append(_skipped_row(held_out_pid, len(y_test), int(test_valid_mask.sum())))
            continue

        X_test_eval_norm = _apply_normalizer(X_test_eval, pt)
        y_pred_eval = rf.predict(X_test_eval_norm)
        y_true_eval = y_test_eval.values

        rmse         = float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval)))
        mae          = float(mean_absolute_error(y_true_eval, y_pred_eval))
        r2           = float(r2_score(y_true_eval, y_pred_eval))
        baseline_rmse = float(np.sqrt(mean_squared_error(
            y_true_eval,
            np.full_like(y_true_eval, float(y_train_fit.mean())),
        )))

        if (i - 1) % 10 == 0 or i == len(pids):
            logger.info(
                "[%d/%d] %-12s RMSE=%.4f (base=%.4f)  MAE=%.4f  R²=%+.4f  n=%d",
                i, len(pids), held_out_pid,
                rmse, baseline_rmse, mae, r2, len(y_test_eval),
            )

        fold_rows.append({
            "participant_id": held_out_pid,
            "n_total_rows":   int(len(y_test)),
            "n_eval_rows":    int(len(y_test_eval)),
            "rmse":           rmse,
            "mae":            mae,
            "r2":             r2,
            "baseline_rmse":  baseline_rmse,
            "skipped":        False,
        })

    return fold_rows, predictions


def _skipped_row(pid: str, n_total: int, n_eval: int) -> dict:
    return {
        "participant_id": pid,
        "n_total_rows":   n_total,
        "n_eval_rows":    n_eval,
        "rmse":           float("nan"),
        "mae":            float("nan"),
        "r2":             float("nan"),
        "baseline_rmse":  float("nan"),
        "skipped":        True,
    }


# ─── Validación LOAO pública ──────────────────────────────────────────────────

def run_loao_fatigue(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: int = RF_MAX_DEPTH,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
    seed: int = SEED,
    results_path: str = FATIGUE_LOAO_RESULTS,
    predictions_path: str = FATIGUE_PREDICTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validación cruzada Leave-One-Athlete-Out para el regresor de fatiga.

    Ejecuta UN SOLO paso por los 75 atletas, recopilando:
      - Métricas de evaluación por atleta (RMSE, MAE, R²) en días de entrenamiento
      - Predicciones LOAO limpias para TODAS las filas (input para M2, Fase 8)

    Returns
    -------
    results_df  : DataFrame guardado en results_path
    predictions : Series guardada en predictions_path
    """
    logger.info(
        "LOAO Regresión de Fatiga — %d atletas, RF(%d árboles, depth=%d, min_leaf=%d)",
        meta["participant_id"].nunique(), n_estimators, max_depth, min_samples_leaf,
    )

    fold_rows, predictions = _loao_single_pass(
        X, y, meta,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    # ── Resumen agregado ─────────────────────────────────────────────────────
    valid_rows = [r for r in fold_rows if not r.get("skipped", False)]
    n_valid    = len(valid_rows)
    n_skip     = len(fold_rows) - n_valid

    def _mean_safe(key: str) -> float:
        vals = [r[key] for r in valid_rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _std_safe(key: str) -> float:
        vals = [r[key] for r in valid_rows if not np.isnan(r[key])]
        return float(np.std(vals)) if vals else float("nan")

    def _median_safe(key: str) -> float:
        vals = [r[key] for r in valid_rows if not np.isnan(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    mean_rmse  = _mean_safe("rmse")
    std_rmse   = _std_safe("rmse")
    mean_mae   = _mean_safe("mae")
    # Usar MEDIANA del R² — la media se distorsiona con atletas de varianza casi nula
    mean_r2    = _median_safe("r2")
    base_rmse  = _mean_safe("baseline_rmse")

    fold_rows.append({
        "participant_id": "MEAN",
        "n_total_rows":   None,
        "n_eval_rows":    None,
        "rmse":           mean_rmse,
        "mae":            mean_mae,
        "r2":             mean_r2,
        "baseline_rmse":  base_rmse,
        "skipped":        False,
    })

    logger.info(
        "LOAO completo — RMSE=%.4f±%.4f (baseline=%.4f)  MAE=%.4f  R²=%.4f  "
        "válidos=%d/%d  omitidos=%d",
        mean_rmse, std_rmse, base_rmse, mean_mae, mean_r2,
        n_valid, len(fold_rows) - 1, n_skip,
    )

    # ── Guardar CSV de resultados ────────────────────────────────────────────
    results_df = pd.DataFrame(fold_rows)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info("Métricas LOAO guardadas → %s", results_path)

    # ── Guardar CSV de predicciones ──────────────────────────────────────────
    pred_df = meta.copy()
    pred_df["fatigue_score_predicted"] = predictions.values
    pred_df.to_csv(predictions_path, index=False)
    n_null = int(predictions.isna().sum())
    logger.info(
        "Predicciones LOAO guardadas → %s (%d filas, %d NaN)",
        predictions_path, len(pred_df), n_null,
    )

    return results_df, predictions


# ─── Modelo final (todos los atletas) ────────────────────────────────────────

def train_final_fatigue_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: int = RF_MAX_DEPTH,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
    seed: int = SEED,
    model_path: str = FATIGUE_MODEL_PATH,
    importance_path: str = FATIGUE_IMPORTANCE_PATH,
) -> Tuple[RandomForestRegressor, PowerTransformer]:
    """
    Entrenar el RF Regressor final sobre TODOS los atletas (sin hold-out).

    Filtra a días de entrenamiento antes de ajustar (días de descanso tienen target NaN).
    Guarda:
      - modelo + normalizador + lista de features como dict → model_path
      - CSV de importancia de features                      → importance_path
    """
    valid_mask = y.notna()
    X_fit = X.loc[valid_mask]
    y_fit = y.loc[valid_mask]

    logger.info(
        "Entrenando modelo final de fatiga sobre %d filas (todos los atletas, días de entrenamiento)",
        len(X_fit),
    )

    pt     = _fit_normalizer(X_fit)
    X_norm = _apply_normalizer(X_fit, pt)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_norm, y_fit)

    # Guardar bundle del modelo
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":           rf,
            "normalizer":      pt,
            "feature_columns": FATIGUE_FEATURE_COLUMNS,
        },
        model_path,
    )
    logger.info("Modelo final guardado → %s", model_path)

    # Importancia de features
    importance_df = (
        pd.DataFrame({
            "feature":    FATIGUE_FEATURE_COLUMNS,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(importance_path, index=False)
    logger.info("Importancia de features guardada → %s", importance_path)

    return rf, pt


# ─── Verificación de umbrales T7.4 ───────────────────────────────────────────

def _check_thresholds(results_df: pd.DataFrame) -> None:
    mean_row   = results_df[results_df["participant_id"] == "MEAN"].iloc[0]
    valid_df   = results_df[results_df["participant_id"] != "MEAN"]
    rmse       = float(mean_row["rmse"])
    r2         = float(mean_row["r2"])
    n_neg_r2   = int((valid_df["r2"] < 0).sum())
    n_valid    = int((~valid_df["skipped"]).sum())

    logger.info("─" * 55)
    logger.info("VERIFICACIÓN DE UMBRALES T7.4")
    logger.info("─" * 55)
    logger.info("  RMSE (escala normalizada, media LOAO) = %.4f", rmse)
    if rmse < RMSE_TARGET:
        logger.info("  ✓ RMSE < %.2f — objetivo CUMPLIDO", RMSE_TARGET)
    elif rmse < RMSE_WARN:
        logger.warning(
            "  ⚠ %.2f ≤ RMSE < %.2f — aceptable, documentar en tesis",
            RMSE_TARGET, RMSE_WARN,
        )
    else:
        logger.error(
            "  ✗ RMSE ≥ %.2f — considerar XGBoost/LightGBM como alternativa",
            RMSE_WARN,
        )
    logger.info(
        "  R² mediana = %.4f (se usa mediana; media distorsionada por atletas "
        "con varianza de recuperación casi nula) | R² < 0 en %d/%d atletas",
        r2, n_neg_r2, n_valid,
    )
    logger.info("─" * 55)


# ─── Orquestador del pipeline completo Fase 7 ────────────────────────────────

def run_fatigue_pipeline(
    csv_path: str = RUNNER_CSV,
    skip_loao: bool = False,
) -> Dict[str, object]:
    """
    Orquestar el pipeline completo de la Fase 7 (T7.1 → T7.4).

    Etapas:
      1. Cargar y preparar el dataset                      (T7.1)
      2. LOAO + generación de predicciones de fatiga       (T7.2 + T8.1)
      3. Entrenar modelo final sobre todos los atletas     (T7.3)
      4. Verificar umbrales RMSE / R²                      (T7.4)

    Parameters
    ----------
    csv_path  : Ruta a day_approach_maskedID_timeseries.csv.
    skip_loao : Si True, omite el LOAO (solo entrena modelo final).

    Returns
    -------
    dict con claves:
        loao_results     — DataFrame de run_loao_fatigue (si LOAO ejecutado)
        mean_rmse        — float
        mean_mae         — float
        mean_r2          — float
        baseline_rmse    — float
        loao_predictions — pd.Series
        model            — RandomForestRegressor entrenado
        normalizer       — PowerTransformer ajustado
    """
    logger.info("=" * 60)
    logger.info("FASE 7 — Regresor de Fatiga (Runner Dataset)")
    logger.info("Inputs : 10 features GPS objetivas")
    logger.info("Target : perceived_recovery (pre-sesión, D-1)")
    logger.info("=" * 60)

    # Paso 1 — Cargar datos
    df_all, X, y, meta = prepare_fatigue_dataset(csv_path)

    results: Dict[str, object] = {}

    if not skip_loao:
        # Paso 2 — LOAO + predicciones (un solo paso)
        logger.info("Etapa 1/2 — LOAO + Generación de Predicciones")
        loao_df, predictions = run_loao_fatigue(X, y, meta)

        mean_row = loao_df[loao_df["participant_id"] == "MEAN"].iloc[0]
        results["loao_results"]     = loao_df
        results["mean_rmse"]        = float(mean_row["rmse"])
        results["mean_mae"]         = float(mean_row["mae"])
        results["mean_r2"]          = float(mean_row["r2"])
        results["baseline_rmse"]    = float(mean_row["baseline_rmse"])
        results["loao_predictions"] = predictions

        # Paso 4 — Verificar umbrales
        _check_thresholds(loao_df)

    # Paso 3 — Modelo final sobre todos los atletas
    logger.info("Etapa 2/2 — Entrenando modelo final sobre todos los atletas")
    model, normalizer = train_final_fatigue_model(X, y)
    results["model"]      = model
    results["normalizer"] = normalizer

    logger.info("=" * 60)
    logger.info("FASE 7 COMPLETADA")
    if "mean_rmse" in results:
        logger.info(
            "  LOAO → RMSE=%.4f | MAE=%.4f | R²=%.4f | Baseline RMSE=%.4f",
            results["mean_rmse"], results["mean_mae"],
            results["mean_r2"],   results["baseline_rmse"],
        )
    logger.info("  Modelo final → %s", FATIGUE_MODEL_PATH)
    logger.info("  Métricas LOAO → %s", FATIGUE_LOAO_RESULTS)
    logger.info("  Predicciones LOAO → %s", FATIGUE_PREDICTIONS_PATH)
    logger.info("=" * 60)

    return results
