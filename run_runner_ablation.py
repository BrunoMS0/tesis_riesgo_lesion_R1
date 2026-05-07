#!/usr/bin/env python
"""
run_runner_ablation.py — Fase 8: Experimento de ablación M1→M2.

Compara tres condiciones de Modelo 2 (RF Clasificador de lesión), usando
siempre las mismas 10 features GPS objetivas como base:

  A) Solo GPS       : 10 features objetivas
                      → Escenario: atleta con reloj, sin cuestionario, sin M1
  B) GPS + M1       : 10 objetivas + fatigue_score_predicted (output de M1 LOAO)
                      → Escenario: nuestro sistema completo en producción
  C) GPS + real     : 10 objetivas + recent_recovery real
                      → Upper bound: atleta con reloj + cuestionario diario

Pregunta central: ¿GPS+M1 ≈ GPS+real? (M1 aproxima el cuestionario subjetivo)

Outputs
-------
  src/outputs/loao_runner_gps_only.csv          — AUC LOAO condición A
  src/outputs/loao_runner_v2_results.csv        — AUC LOAO condición B (resultado principal)
  src/outputs/loao_runner_gps_real_fatigue.csv  — AUC LOAO condición C
  src/outputs/ablation_fatigue_runner.csv       — tabla resumen de ablación

Usage
-----
  python run_runner_ablation.py              # pipeline completo (3 × LOAO)
  python run_runner_ablation.py -v           # verbose
  python run_runner_ablation.py --condition A   # ejecutar solo una condición
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

GPS_FEATURE_COLUMNS = [
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

PROCESSED_CSV      = "src/outputs/runner_dataset_processed.csv"
FATIGUE_PREDS_CSV  = "src/outputs/runner_fatigue_predictions_loao.csv"

OUTPUT_A = "src/outputs/loao_runner_gps_only.csv"
OUTPUT_B = "src/outputs/loao_runner_v2_results.csv"
OUTPUT_C = "src/outputs/loao_runner_gps_real_fatigue.csv"
OUTPUT_ABLATION = "src/outputs/ablation_fatigue_runner.csv"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_data() -> dict:
    """
    Load and join processed dataset + LOAO fatigue predictions.

    Returns a dict with X_A, X_B, X_C, y, meta for the three ablation
    conditions.  All DataFrames are aligned by row index.
    """
    df = pd.read_csv(PROCESSED_CSV)
    preds = pd.read_csv(FATIGUE_PREDS_CSV)

    # Merge fatigue predictions onto processed dataset by participant_id + date
    df = df.merge(
        preds[["participant_id", "date", "fatigue_score_predicted"]],
        on=["participant_id", "date"],
        how="left",
    )

    n_null = int(df["fatigue_score_predicted"].isna().sum())
    if n_null > 0:
        logger.warning(
            "%d rows have no fatigue prediction (skipped LOAO athlete folds). "
            "Filling with column mean.", n_null,
        )
        df["fatigue_score_predicted"] = df["fatigue_score_predicted"].fillna(
            df["fatigue_score_predicted"].mean()
        )

    y    = df["injury"].astype(int)
    meta = df[["participant_id", "date"]].copy()

    # Condition A — pure GPS (10 features)
    X_A = df[GPS_FEATURE_COLUMNS].copy()

    # Condition B — GPS + fatigue predicted by M1 (11 features)
    X_B = df[GPS_FEATURE_COLUMNS + ["fatigue_score_predicted"]].copy()

    # Condition C — GPS + real recovery (upper bound, 11 features)
    # recent_recovery = processed perceived_recovery.6 (forward-filled on rest days)
    X_C = df[GPS_FEATURE_COLUMNS + ["recent_recovery"]].copy()

    logger.info(
        "Ablation data: %d rows | %d injuries (%.2f%%) | %d athletes",
        len(df), int(y.sum()), 100.0 * y.mean(),
        df["participant_id"].nunique(),
    )
    return {
        "X_A": X_A, "X_B": X_B, "X_C": X_C,
        "y": y, "meta": meta,
    }


# ─── LOAO runner (wraps loso_cross_validation) ────────────────────────────────

def _run_condition_loao(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    condition_name: str,
    output_path: str,
) -> dict:
    """
    Run LOAO (74 folds) for one ablation condition.

    Uses the same protocol as the original runner LOAO:
      - Per-fold Yeo-Johnson normalisation
      - RF Classifier (n_estimators=200, class_weight=balanced, min_samples_leaf=5)
      - SMOTE augmentation (target_ratio=0.15)
      - Skip folds where test athlete has no injuries (AUC undefined)

    Returns dict with mean_auc, std_auc, n_valid, n_skipped, results_df.
    """
    from src.runner.config import (
        AUGMENTATION_METHOD, RF_CLASS_WEIGHT, RF_MAX_FEATURES,
        RF_N_ESTIMATORS, SEED, SMOTE_K_NEIGHBORS, TARGET_RATIO,
    )
    from src.runner.dataset import make_runner_injury_config
    from src.injury.validate import loso_cross_validation

    cfg = make_runner_injury_config(feature_cols=list(X.columns))

    logger.info(
        "[Condition %s] Starting LOAO — %d features: %s%s",
        condition_name, len(X.columns),
        str(list(X.columns)[:5])[:-1],
        "…]" if len(X.columns) > 5 else "]",
    )

    result = loso_cross_validation(
        X, y, meta, cfg, use_augmentation=True,
    )

    # Save per-fold CSV
    rows = []
    for f in result.folds:
        rows.append({
            "participant_id": f.participant_id,
            "n_samples":      f.n_samples,
            "n_injuries":     f.n_injuries,
            "roc_auc":        f.roc_auc,
            "pr_auc":         f.pr_auc,
            "f1":             f.f1,
            "skipped":        f.skipped,
        })
    rows.append({
        "participant_id": "MEAN",
        "n_samples":      None,
        "n_injuries":     None,
        "roc_auc":        result.mean_roc_auc,
        "pr_auc":         result.mean_pr_auc,
        "f1":             result.mean_f1,
        "skipped":        False,
    })
    results_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    n_valid   = sum(1 for f in result.folds if not f.skipped)
    n_skipped = result.n_skipped_folds

    logger.info(
        "[Condition %s] AUC=%.4f ± %.4f | PR-AUC=%.4f | F1=%.4f "
        "| valid=%d/%d (skipped=%d) → %s",
        condition_name,
        result.mean_roc_auc, result.std_roc_auc,
        result.mean_pr_auc, result.mean_f1,
        n_valid, len(result.folds), n_skipped,
        output_path,
    )

    return {
        "condition":  condition_name,
        "mean_auc":   result.mean_roc_auc,
        "std_auc":    result.std_roc_auc,
        "mean_prauc": result.mean_pr_auc,
        "mean_f1":    result.mean_f1,
        "n_valid":    n_valid,
        "n_skipped":  n_skipped,
        "results_df": results_df,
    }


# ─── Ablation summary ─────────────────────────────────────────────────────────

def _save_ablation_table(conditions: list, output_path: str) -> pd.DataFrame:
    """Build and save the ablation comparison table (T8.4)."""
    rows = []
    for c in conditions:
        rows.append({
            "condition":         c["condition"],
            "features":          c["features_desc"],
            "n_features":        c["n_features"],
            "auc_loao_mean":     round(c["mean_auc"],   4),
            "auc_loao_std":      round(c["std_auc"],    4),
            "prauc_loao_mean":   round(c["mean_prauc"], 4),
            "f1_loao_mean":      round(c["mean_f1"],    4),
            "valid_folds":       c["n_valid"],
            "skipped_folds":     c["n_skipped"],
        })
    ablation_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(output_path, index=False)
    logger.info("Ablation table saved → %s", output_path)
    return ablation_df


# ─── T8.5 Threshold check ─────────────────────────────────────────────────────

def _check_ablation_results(conditions: list) -> None:
    by_name = {c["condition"]: c for c in conditions}

    auc_A = by_name.get("A_GPS_only",   {}).get("mean_auc", float("nan"))
    auc_B = by_name.get("B_GPS_M1",     {}).get("mean_auc", float("nan"))
    auc_C = by_name.get("C_GPS_real",   {}).get("mean_auc", float("nan"))

    logger.info("─" * 60)
    logger.info("T8.5 ABLATION RESULTS")
    logger.info("─" * 60)
    logger.info("  A) Solo GPS       AUC = %.4f  (baseline sin fatiga)", auc_A)
    logger.info("  B) GPS + M1       AUC = %.4f  ← resultado principal", auc_B)
    logger.info("  C) GPS + real     AUC = %.4f  (upper bound cuestionario)", auc_C)
    logger.info("─" * 60)

    if not np.isnan(auc_A) and not np.isnan(auc_B):
        delta_AB = auc_B - auc_A
        if delta_AB >= 0:
            logger.info(
                "  ✓ META MÍNIMA CUMPLIDA: GPS+M1 (%.4f) ≥ Solo GPS (%.4f) "
                "[Δ=+%.4f — M1 aporta señal discriminativa]",
                auc_B, auc_A, delta_AB,
            )
        else:
            logger.warning(
                "  ⚠ GPS+M1 (%.4f) < Solo GPS (%.4f) [Δ=%.4f] — "
                "M1 no mejora. El RF de lesión puede capturar la fatiga "
                "implícitamente en las features de carga.",
                auc_B, auc_A, delta_AB,
            )

    if not np.isnan(auc_B) and not np.isnan(auc_C):
        delta_BC = auc_C - auc_B
        gap_pct  = 100.0 * delta_BC / max(auc_C, 1e-9)
        if gap_pct < 1.0:
            logger.info(
                "  ✓ META IDEAL CUMPLIDA: GPS+M1 ≈ GPS+real "
                "(brecha=%.4f, %.1f%%) — M1 aproxima el cuestionario",
                delta_BC, gap_pct,
            )
        elif gap_pct < 5.0:
            logger.info(
                "  ~ GPS+M1 cercano a GPS+real "
                "(brecha=%.4f, %.1f%%) — M1 captura la mayor parte de la señal",
                delta_BC, gap_pct,
            )
        else:
            logger.warning(
                "  ⚠ GPS+M1 lejos de GPS+real "
                "(brecha=%.4f, %.1f%%) — M1 aproxima parcialmente el cuestionario",
                delta_BC, gap_pct,
            )

    if not np.isnan(auc_B) and auc_B >= 0.92:
        logger.info("  ✓ META IDEAL AUC ≥ 0.92 CUMPLIDA: GPS+M1 AUC = %.4f", auc_B)
    elif not np.isnan(auc_B) and auc_B >= 0.91:
        logger.info("  ✓ META MÍNIMA AUC ≥ 0.91 CUMPLIDA: GPS+M1 AUC = %.4f", auc_B)

    logger.info("─" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fase 8 — Experimento de ablación M1→M2 (3 × LOAO)",
    )
    p.add_argument(
        "--condition", choices=["A", "B", "C"], default=None,
        help="Ejecutar solo una condición (por defecto: las tres).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Activar logging DEBUG.",
    )
    return p.parse_args()


logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("FASE 8 — Ablación M1→M2 (Runner Dataset)")
    logger.info("Argumento: ¿GPS+M1 ≈ GPS+cuestionario_real?")
    logger.info("=" * 60)

    data   = _load_data()
    t0     = time.time()
    done   = []
    run_A  = args.condition in (None, "A")
    run_B  = args.condition in (None, "B")
    run_C  = args.condition in (None, "C")

    # ── Condition A: Solo GPS ──────────────────────────────────────────────────
    if run_A:
        logger.info("=" * 60)
        logger.info("CONDICIÓN A — Solo GPS (10 features objetivas)")
        logger.info("=" * 60)
        r = _run_condition_loao(
            data["X_A"], data["y"], data["meta"],
            condition_name="A_GPS_only",
            output_path=OUTPUT_A,
        )
        r["features_desc"] = "10 GPS objetivas"
        r["n_features"]    = 10
        done.append(r)

    # ── Condition B: GPS + M1 (resultado principal) ────────────────────────────
    if run_B:
        logger.info("=" * 60)
        logger.info("CONDICIÓN B — GPS + fatiga predicha por M1 (resultado principal)")
        logger.info("=" * 60)
        r = _run_condition_loao(
            data["X_B"], data["y"], data["meta"],
            condition_name="B_GPS_M1",
            output_path=OUTPUT_B,
        )
        r["features_desc"] = "10 GPS + fatigue_score_predicted (M1)"
        r["n_features"]    = 11
        done.append(r)

    # ── Condition C: GPS + real (upper bound) ─────────────────────────────────
    if run_C:
        logger.info("=" * 60)
        logger.info("CONDICIÓN C — GPS + recuperación real (upper bound)")
        logger.info("=" * 60)
        r = _run_condition_loao(
            data["X_C"], data["y"], data["meta"],
            condition_name="C_GPS_real",
            output_path=OUTPUT_C,
        )
        r["features_desc"] = "10 GPS + recent_recovery (real)"
        r["n_features"]    = 11
        done.append(r)

    # ── Ablation table ────────────────────────────────────────────────────────
    if len(done) > 1:
        ablation_df = _save_ablation_table(done, OUTPUT_ABLATION)

    # ── T8.5 threshold check (only when all 3 conditions run) ─────────────────
    if len(done) == 3:
        _check_ablation_results(done)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"FASE 8 — RESUMEN  ({elapsed:.0f}s)")
    print("=" * 60)
    for c in done:
        print(f"  {c['condition']:<15} AUC={c['mean_auc']:.4f} ± {c['std_auc']:.4f}  "
              f"PR-AUC={c['mean_prauc']:.4f}  F1={c['mean_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
