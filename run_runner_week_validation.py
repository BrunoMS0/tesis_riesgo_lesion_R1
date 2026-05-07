#!/usr/bin/env python
"""
run_runner_week_validation.py — Fase 9: Validación cross-granularity con week_approach.

Entrena un RF Clasificador con las 9 features COMUNES (disponibles tanto en
day_approach diario como en week_approach semanal), y valida con un protocolo
LOAO cross-granularity:

  Para cada atleta i (74 folds):
    - Entrenar RF en day_approach de los 73 atletas restantes (granularidad diaria)
    - Evaluar sobre week_approach del atleta i (granularidad semanal)
    - Las 9 features semanales se renombran a los nombres de features diarias

  Esto prueba la robustez del pipeline ante cambios de granularidad temporal.

Outputs
-------
  src/outputs/rf_runner_weekly_model.pkl        — modelo final (todos los atletas, daily)
  src/outputs/week_validation_results.csv       — AUC LOAO cross-granularity

Usage
-----
  python run_runner_week_validation.py          # pipeline completo
  python run_runner_week_validation.py -v       # verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import PowerTransformer

DAILY_COMMON_FEATURES = [
    "acute_load_7d",
    "acwr",
    "nr_sessions_7d",
    "nr_rest_days_7d",
    "high_intensity_km_7d",
    "strength_days_7d",
    "mean_perceived_recovery",
    "mean_perceived_exertion",
    "mean_perceived_success",
]

PROCESSED_CSV   = "src/outputs/runner_dataset_processed.csv"
WEEK_MODEL_PATH = "src/outputs/rf_runner_weekly_model.pkl"
WEEK_RESULTS    = "src/outputs/week_validation_results.csv"

RF_PARAMS = {
    "n_estimators":    200,
    "max_features":    "sqrt",
    "class_weight":    "balanced",
    "min_samples_leaf": 1,
    "max_depth":       None,
    "random_state":    42,
    "n_jobs":          -1,
}

logger = logging.getLogger(__name__)


def _fit_normalizer(X: pd.DataFrame) -> PowerTransformer:
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(X)
    return pt


def _apply_normalizer(X: pd.DataFrame, pt: PowerTransformer) -> pd.DataFrame:
    return pd.DataFrame(pt.transform(X), columns=X.columns, index=X.index)


def _load_daily_data() -> tuple:
    """Load processed day_approach data with only the 9 common features."""
    df = pd.read_csv(PROCESSED_CSV)
    missing = [c for c in DAILY_COMMON_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in processed CSV: {missing}")

    X    = df[DAILY_COMMON_FEATURES].copy()
    y    = df["injury"].astype(int)
    meta = df[["participant_id", "date"]].copy()
    return X, y, meta


def _load_weekly_data() -> tuple:
    """Load week_approach data with the 9 common features (renamed from weekly names)."""
    from src.runner.week_transform import load_week_approach, WEEK_COMMON_FEATURES
    df   = load_week_approach()
    X    = df[WEEK_COMMON_FEATURES].copy()
    y    = df["injury"].astype(int)
    meta = df[["participant_id", "date"]].copy()
    return X, y, meta


def run_cross_granularity_loao(
    X_daily:      pd.DataFrame,
    y_daily:      pd.Series,
    meta_daily:   pd.DataFrame,
    X_weekly:     pd.DataFrame,
    y_weekly:     pd.Series,
    meta_weekly:  pd.DataFrame,
    *,
    results_path: str = WEEK_RESULTS,
) -> pd.DataFrame:
    """
    Cross-granularity LOAO:
      For each athlete i:
        - Train RF on day_approach of athletes ≠ i  (daily granularity)
        - Evaluate on week_approach of athlete i    (weekly granularity)

    The 9 features are identically named in both DataFrames after renaming.

    Returns
    -------
    pd.DataFrame with per-athlete metrics and MEAN summary row.
    """
    pids_daily  = sorted(meta_daily["participant_id"].unique())
    pids_weekly = sorted(meta_weekly["participant_id"].unique())

    # Athletes must be present in both datasets
    pids = sorted(set(pids_daily) & set(pids_weekly))
    logger.info(
        "Cross-granularity LOAO: %d athletes (daily ∩ weekly)", len(pids)
    )

    fold_rows = []

    for i, pid in enumerate(pids, 1):
        # Train on ALL daily data except this athlete
        mask_train = meta_daily["participant_id"] != pid
        X_train = X_daily.loc[mask_train]
        y_train = y_daily.loc[mask_train]

        # Test on this athlete's WEEKLY data
        mask_test_w = meta_weekly["participant_id"] == pid
        X_test  = X_weekly.loc[mask_test_w]
        y_test  = y_weekly.loc[mask_test_w]

        n_injuries = int(y_test.sum())

        if n_injuries == 0:
            if i % 10 == 0 or i == len(pids):
                logger.info("[%d/%d] %-12s SKIPPED — 0 injuries (n=%d)", i, len(pids), pid, len(y_test))
            fold_rows.append({
                "participant_id": pid,
                "n_samples": int(len(y_test)),
                "n_injuries": 0,
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "f1": float("nan"),
                "skipped": True,
            })
            continue

        # Per-fold normalisation: fit on daily training data
        pt           = _fit_normalizer(X_train)
        X_train_norm = _apply_normalizer(X_train, pt)
        X_test_norm  = _apply_normalizer(X_test,  pt)

        # SMOTE augmentation (same as day_approach LOAO)
        try:
            from src.runner.config import AUGMENTATION_METHOD, TARGET_RATIO, SMOTE_K_NEIGHBORS, SEED
            from src.runner.dataset import make_runner_injury_config
            from src.injury.augment import augment_training_data
            cfg = make_runner_injury_config(feature_cols=DAILY_COMMON_FEATURES)
            meta_train = meta_daily.loc[mask_train].reset_index(drop=True)
            X_tr_aug, y_tr_aug = augment_training_data(
                X_train_norm.reset_index(drop=True),
                y_train.reset_index(drop=True),
                meta_train,
                cfg,
            )
        except Exception as exc:
            logger.debug("Augmentation failed for %s: %s — using original", pid, exc)
            X_tr_aug = X_train_norm.reset_index(drop=True)
            y_tr_aug = y_train.reset_index(drop=True)

        # Train RF
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr_aug, y_tr_aug)

        # Evaluate
        y_prob = rf.predict_proba(X_test_norm)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            roc_auc = 0.0

        try:
            pr_auc = float(average_precision_score(y_test, y_prob))
        except ValueError:
            pr_auc = 0.0

        f1 = float(f1_score(y_test, y_pred, zero_division=0.0))

        if i % 10 == 1 or i == len(pids):
            logger.info(
                "[%d/%d] %-12s AUC=%.4f  PR-AUC=%.4f  F1=%.4f  n=%d  inj=%d",
                i, len(pids), pid, roc_auc, pr_auc, f1, len(y_test), n_injuries,
            )

        fold_rows.append({
            "participant_id": pid,
            "n_samples":      int(len(y_test)),
            "n_injuries":     n_injuries,
            "roc_auc":        round(roc_auc, 4),
            "pr_auc":         round(pr_auc, 4),
            "f1":             round(f1, 4),
            "skipped":        False,
        })

    # Summary
    valid   = [r for r in fold_rows if not r["skipped"]]
    n_valid = len(valid)
    n_skip  = len(fold_rows) - n_valid

    mean_auc  = float(np.mean([r["roc_auc"] for r in valid])) if valid else float("nan")
    std_auc   = float(np.std([r["roc_auc"] for r in valid]))  if valid else float("nan")
    mean_pr   = float(np.mean([r["pr_auc"]  for r in valid])) if valid else float("nan")
    mean_f1   = float(np.mean([r["f1"]      for r in valid])) if valid else float("nan")

    fold_rows.append({
        "participant_id": "MEAN",
        "n_samples": None,
        "n_injuries": None,
        "roc_auc": mean_auc,
        "pr_auc":  mean_pr,
        "f1":      mean_f1,
        "skipped": False,
    })

    logger.info(
        "Cross-granularity LOAO complete — AUC=%.4f±%.4f  PR-AUC=%.4f  F1=%.4f  "
        "valid=%d/%d  skipped=%d",
        mean_auc, std_auc, mean_pr, mean_f1, n_valid, len(pids), n_skip,
    )

    results_df = pd.DataFrame(fold_rows)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info("Week validation results saved -> %s", results_path)

    return results_df


def train_weekly_model(
    X_daily: pd.DataFrame,
    y_daily: pd.Series,
    *,
    model_path: str = WEEK_MODEL_PATH,
) -> tuple:
    """
    Train the final weekly model on ALL day_approach data (9 features).
    Saved for deployment: use daily training data, apply to weekly test data.
    """
    pt     = _fit_normalizer(X_daily)
    X_norm = _apply_normalizer(X_daily, pt)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_norm, y_daily)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":           rf,
            "normalizer":      pt,
            "feature_columns": DAILY_COMMON_FEATURES,
            "trained_on":      "day_approach (9 common features)",
        },
        model_path,
    )
    logger.info("Weekly model saved -> %s", model_path)
    return rf, pt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fase 9 — Validación cross-granularity (daily→weekly)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Logging DEBUG.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("FASE 9 — Validación Cross-Granularity (day→week)")
    logger.info("Train: day_approach (9 features) | Test: week_approach")
    logger.info("=" * 60)

    # T9.1 / T9.2 — Load and map both datasets
    logger.info("Stage 1/3 — Loading datasets")
    X_daily,  y_daily,  meta_daily  = _load_daily_data()
    X_weekly, y_weekly, meta_weekly = _load_weekly_data()

    logger.info(
        "Daily:  %d rows | %d athletes | %d injuries (%.2f%%)",
        len(X_daily),  meta_daily["participant_id"].nunique(),
        int(y_daily.sum()), 100.0 * y_daily.mean(),
    )
    logger.info(
        "Weekly: %d rows | %d athletes | %d injuries (%.2f%%)",
        len(X_weekly), meta_weekly["participant_id"].nunique(),
        int(y_weekly.sum()), 100.0 * y_weekly.mean(),
    )

    # T9.3 — Cross-granularity LOAO
    logger.info("Stage 2/3 — Cross-granularity LOAO (74 folds: train daily, eval weekly)")
    t0 = time.time()
    results_df = run_cross_granularity_loao(
        X_daily, y_daily, meta_daily,
        X_weekly, y_weekly, meta_weekly,
    )
    elapsed_loao = time.time() - t0

    # Final model (trained on all daily data)
    logger.info("Stage 3/3 — Training final weekly model (all daily data, 9 features)")
    rf, pt = train_weekly_model(X_daily, y_daily)

    # Summary and T9.4 — Day vs Week comparison
    mean_row  = results_df[results_df["participant_id"] == "MEAN"].iloc[0]
    auc_week  = float(mean_row["roc_auc"])
    auc_daily = 0.9074   # Condition A from ablation (10 daily features, same 9 + nr_rest_days)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"FASE 9 — RESUMEN  ({elapsed:.0f}s)")
    print("=" * 60)
    print(f"  AUC day_approach  (10 feat, LOAO): 0.9074  [referencia Fase 8 Cond A]")
    print(f"  AUC week_approach (9 feat, LOAO) : {auc_week:.4f}  <- resultado Fase 9")

    delta = auc_week - auc_daily
    if delta >= 0:
        print(f"  Δ = +{delta:.4f} — AUC semanal ≥ AUC diario (robustez a granularidad)")
    elif delta >= -0.05:
        print(f"  Δ = {delta:.4f} — AUC semanal ligeramente menor (esperado, -1 feature)")
    else:
        print(f"  Δ = {delta:.4f} — AUC semanal claramente menor (pérdida de resolución)")

    valid_folds = results_df[results_df["participant_id"] != "MEAN"]
    n_valid     = int((~valid_folds["skipped"]).sum())
    n_skipped   = int(valid_folds["skipped"].sum())
    print(f"  Folds válidos: {n_valid}/74 (skipped: {n_skipped} — 0 injuries)")
    print("=" * 60)

    # T9.4 documentation notes
    logger.info("─" * 55)
    logger.info("T9.4 — BRECHA DAY→WEEK")
    logger.info("  AUC diario  (Cond A, 10 features): %.4f", auc_daily)
    logger.info("  AUC semanal (9 features):          %.4f", auc_week)
    logger.info("  Δ AUC = %.4f", delta)
    if abs(delta) < 0.02:
        logger.info(
            "  ✓ Brecha < 2%% — Las 9 features comunes son suficientes para "
            "predicción robusta independiente de granularidad temporal."
        )
    else:
        logger.info(
            "  ⚠ Brecha = %.1f%% — Reducir a granularidad semanal tiene costo "
            "documentable (pérdida de resolución temporal).",
            abs(delta) * 100,
        )
    logger.info("─" * 55)


if __name__ == "__main__":
    main()
