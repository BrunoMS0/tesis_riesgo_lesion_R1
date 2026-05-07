"""
fatigue.py — Modelo 1 de Fatiga: RF Regressor sobre Runner Dataset (Fase 7).

Predice `perceived_recovery` (pre-sesión, D-1) a partir de 10 features
objetivas de carga GPS. Forma la primera etapa del pipeline M1→M2.

Inputs (10 features objetivas — sin dato subjetivo):
    acute_load_7d, chronic_load_28d, acwr, high_intensity_km_7d,
    nr_sessions_7d, nr_rest_days_7d, km_sprint_7d, strength_days_7d,
    alt_hours_7d, recent_km

Target:
    recent_recovery  (= perceived_recovery.6, pre-sesión; NaN en días de descanso)

Outputs generados:
    src/outputs/loao_fatigue_runner_results.csv  — métricas LOAO por atleta (T7.2)
    src/outputs/rf_fatigue_runner_model.pkl      — modelo final en todos los atletas (T7.3)
    src/outputs/fatigue_feature_importance.csv   — importancia de features (T7.3)
    src/outputs/runner_fatigue_predictions_loao.csv — predicciones LOAO (T8.1)

Public API
----------
run_fatigue_pipeline(csv_path, skip_loao) -> dict
prepare_fatigue_dataset(csv_path)          -> (df_all, X, y, meta)
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

from .config import RUNNER_CSV, SEED
from .extract import load_runner_csv
from .transform import compute_features

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# RF Regressor hyperparameters
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_LEAF: int = 5

# T7.4 thresholds (raw perceived_recovery scale, which is 0–1 in this dataset)
RMSE_TARGET: float = 0.15
RMSE_WARN: float = 0.20


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_fatigue_dataset(
    csv_path: str = RUNNER_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and prepare the Runner dataset for fatigue regression.

    The target (perceived_recovery.6) is extracted from the RAW CSV before
    compute_features() imputes rest-day NaN values via forward/backward fill.
    This ensures the model is only trained on genuine training-day recoveries,
    not on imputed rest-day values.

    Returns
    -------
    df_all : full processed DataFrame (all columns, all rows)
    X      : DataFrame of FATIGUE_FEATURE_COLUMNS (all rows incl. rest days)
    y      : Series of raw perceived_recovery.6 (NaN on rest days, -0.01 → NaN)
    meta   : DataFrame with [participant_id, date]
    """
    from .config import REST_DAY_VALUE

    raw_df = load_runner_csv(csv_path)
    df = compute_features(raw_df)

    missing = [c for c in FATIGUE_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in processed dataset: {missing}")

    # Extract target DIRECTLY from raw CSV (before compute_features imputation)
    # 'perceived recovery.6' = D-1 pre-session recovery; -0.01 marks rest days.
    raw_target_col = "perceived recovery.6"
    if raw_target_col not in raw_df.columns:
        raise ValueError(f"Raw target column '{raw_target_col}' not found in CSV.")

    X    = df[FATIGUE_FEATURE_COLUMNS].copy()
    y    = raw_df[raw_target_col].replace(REST_DAY_VALUE, np.nan).reset_index(drop=True)
    y.name = FATIGUE_TARGET_COL
    meta = df[["participant_id", "date"]].copy()

    n_valid = int(y.notna().sum())
    n_rest  = int(y.isna().sum())
    logger.info(
        "Fatigue dataset: %d rows | %d training days (target valid) | "
        "%d rest days (target NaN, excluded from evaluation, %.1f%%)",
        len(df), n_valid, n_rest, 100.0 * n_rest / len(df),
    )
    return df, X, y, meta


# ─── Per-fold Yeo-Johnson normalisation ──────────────────────────────────────

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


# ─── LOAO core (single pass — evaluates AND generates predictions) ────────────

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
    Internal LOAO loop — one pass through all athletes.

    Returns
    -------
    fold_rows   : list of per-athlete metric dicts (for results CSV)
    predictions : Series indexed like X — fatigue_score_predicted for ALL rows
                  (predictions made by the model that never saw that athlete)
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

        # Filter rest days from TRAIN (y NaN = rest day)
        train_valid_mask = y_train.notna()
        X_train_fit = X_train.loc[train_valid_mask]
        y_train_fit = y_train.loc[train_valid_mask]

        # For evaluation: only non-rest test days
        test_valid_mask = y_test.notna()
        X_test_eval = X_test.loc[test_valid_mask]
        y_test_eval = y_test.loc[test_valid_mask]

        if len(X_train_fit) < 10:
            logger.warning(
                "[%d/%d] %s — too few train samples (%d), skipping",
                i, len(pids), held_out_pid, len(X_train_fit),
            )
            fold_rows.append(_skipped_row(held_out_pid, len(y_test), int(test_valid_mask.sum())))
            continue

        # Fit normalizer on training days only
        pt           = _fit_normalizer(X_train_fit)
        X_train_norm = _apply_normalizer(X_train_fit, pt)
        X_test_norm  = _apply_normalizer(X_test, pt)  # ALL test rows (incl. rest days)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train_norm, y_train_fit)

        # Store predictions for ALL test rows (no NaN filter for inference)
        pred_values = rf.predict(X_test_norm)
        predictions.iloc[np.where(mask_test)[0]] = pred_values

        # Evaluate only on non-rest test rows
        if int(test_valid_mask.sum()) < 2:
            logger.warning(
                "[%d/%d] %s — too few eval samples (%d), skipping metrics",
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


# ─── Public LOAO function ─────────────────────────────────────────────────────

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
    Leave-One-Athlete-Out cross-validation for the fatigue regressor.

    Runs a SINGLE pass through all 74 athletes, collecting both:
      - Per-athlete evaluation metrics (RMSE, MAE, R²) on non-rest days
      - LOAO-clean fatigue predictions for ALL rows (for Phase 8)

    Returns
    -------
    results_df  : DataFrame saved to results_path (T7.2)
    predictions : Series saved to predictions_path (T8.1 input)
    """
    logger.info(
        "LOAO Fatigue Regression — %d athletes, RF(%d trees, depth=%d, min_leaf=%d)",
        meta["participant_id"].nunique(), n_estimators, max_depth, min_samples_leaf,
    )

    fold_rows, predictions = _loao_single_pass(
        X, y, meta,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    # ── Aggregate summary ────────────────────────────────────────────────────
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
    # Use MEDIAN R² — arithmetic mean is dominated by near-zero-variance outlier
    # athletes that produce astronomically negative R² (SS_tot ≈ 0, SS_res > 0).
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
        "LOAO complete — RMSE=%.4f±%.4f (baseline=%.4f)  MAE=%.4f  R²=%.4f  "
        "valid=%d/%d  skipped=%d",
        mean_rmse, std_rmse, base_rmse, mean_mae, mean_r2,
        n_valid, len(fold_rows) - 1, n_skip,
    )

    # ── Save results CSV ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(fold_rows)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info("LOAO metrics saved → %s", results_path)

    # ── Save predictions CSV (T8.1) ──────────────────────────────────────────
    pred_df = meta.copy()
    pred_df["fatigue_score_predicted"] = predictions.values
    pred_df.to_csv(predictions_path, index=False)
    n_null = int(predictions.isna().sum())
    logger.info(
        "LOAO predictions saved → %s (%d rows, %d NaN)",
        predictions_path, len(pred_df), n_null,
    )

    return results_df, predictions


# ─── Final model (all athletes) ───────────────────────────────────────────────

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
    Train the final RF Regressor on ALL athletes (no hold-out).

    Filters to non-rest days before training (rest days have NaN target).
    Saves:
      - model + normalizer + feature list as a dict → model_path  (T7.3)
      - feature importance CSV                      → importance_path (T7.3)
    """
    valid_mask = y.notna()
    X_fit = X.loc[valid_mask]
    y_fit = y.loc[valid_mask]

    n_athletes = X.loc[valid_mask, FATIGUE_FEATURE_COLUMNS[0]].shape[0]
    logger.info(
        "Training final fatigue model on %d rows (all athletes, non-rest days)",
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

    # Save model bundle
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":           rf,
            "normalizer":      pt,
            "feature_columns": FATIGUE_FEATURE_COLUMNS,
        },
        model_path,
    )
    logger.info("Final fatigue model saved → %s", model_path)

    # Feature importance
    importance_df = (
        pd.DataFrame({
            "feature":    FATIGUE_FEATURE_COLUMNS,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(importance_path, index=False)
    logger.info("Feature importance saved → %s", importance_path)

    return rf, pt


# ─── T7.4 threshold check ─────────────────────────────────────────────────────

def _check_thresholds(results_df: pd.DataFrame) -> None:
    mean_row   = results_df[results_df["participant_id"] == "MEAN"].iloc[0]
    valid_df   = results_df[results_df["participant_id"] != "MEAN"]
    rmse       = float(mean_row["rmse"])
    r2         = float(mean_row["r2"])
    n_neg_r2   = int((valid_df["r2"] < 0).sum())
    n_valid    = int((~valid_df["skipped"]).sum())

    logger.info("─" * 55)
    logger.info("T7.4 THRESHOLD CHECK")
    logger.info("─" * 55)
    logger.info("  RMSE (raw scale, LOAO mean) = %.4f", rmse)
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
        "con varianza de recuperacion casi nula) | R² < 0 en %d/%d atletas",
        r2, n_neg_r2, n_valid,
    )
    if n_valid > 0 and n_neg_r2 > n_valid // 2:
        logger.warning(
            "  Mayoría de atletas con R² < 0 — variabilidad individual alta. "
            "Documentar en tesis (esperado en recuperación subjetiva LOAO)."
        )
    logger.info("─" * 55)


# ─── Pipeline orchestrator ────────────────────────────────────────────────────

def run_fatigue_pipeline(
    csv_path: str = RUNNER_CSV,
    skip_loao: bool = False,
) -> Dict[str, object]:
    """
    Orchestrate the complete Fase 7 pipeline (T7.1 → T7.4).

    Steps:
      1. Load and prepare dataset                   (T7.1)
      2. Run LOAO regression + generate predictions  (T7.2 + T8.1)
      3. Train final model on all athletes            (T7.3)
      4. Check RMSE / R² thresholds                  (T7.4)

    Parameters
    ----------
    csv_path  : Path to day_approach_maskedID_timeseries.csv.
    skip_loao : If True, skip LOAO (only train final model).

    Returns
    -------
    dict with keys:
        loao_results    — DataFrame from run_loao_fatigue (if loao run)
        mean_rmse       — float (if loao run)
        mean_mae        — float (if loao run)
        mean_r2         — float (if loao run)
        baseline_rmse   — float (if loao run)
        loao_predictions — pd.Series (if loao run)
        model           — trained RandomForestRegressor
        normalizer      — fitted PowerTransformer
    """
    logger.info("=" * 60)
    logger.info("FASE 7 — Fatigue Regressor (Runner Dataset)")
    logger.info("Inputs: 10 features GPS objetivas")
    logger.info("Target: perceived_recovery (pre-sesion, D-1)")
    logger.info("=" * 60)

    # Step 1 — Load data
    df_all, X, y, meta = prepare_fatigue_dataset(csv_path)

    results: Dict[str, object] = {}

    if not skip_loao:
        # Step 2 — LOAO regression + predictions (single pass)
        logger.info("Stage 1/2 — LOAO Regression + Prediction Generation")
        loao_df, predictions = run_loao_fatigue(X, y, meta)

        mean_row = loao_df[loao_df["participant_id"] == "MEAN"].iloc[0]
        results["loao_results"]     = loao_df
        results["mean_rmse"]        = float(mean_row["rmse"])
        results["mean_mae"]         = float(mean_row["mae"])
        results["mean_r2"]          = float(mean_row["r2"])
        results["baseline_rmse"]    = float(mean_row["baseline_rmse"])
        results["loao_predictions"] = predictions

        # Step 4 — Threshold check
        _check_thresholds(loao_df)

    # Step 3 — Final model on all athletes
    logger.info("Stage 2/2 — Training final model on all athletes")
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
    logger.info("  LOAO metricas → %s", FATIGUE_LOAO_RESULTS)
    logger.info("  LOAO predicciones → %s", FATIGUE_PREDICTIONS_PATH)
    logger.info("=" * 60)

    return results
