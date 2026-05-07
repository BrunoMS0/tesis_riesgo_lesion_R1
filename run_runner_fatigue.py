#!/usr/bin/env python
"""
run_runner_fatigue.py — Fase 7: Modelo 1 de Fatiga sobre Runner Dataset.

Entrena un RF Regressor que predice `perceived_recovery` (fatiga del atleta)
a partir de 10 features objetivas de carga GPS. Valida con LOAO (74 folds).

Outputs
-------
  src/outputs/loao_fatigue_runner_results.csv   — RMSE / MAE / R² por atleta
  src/outputs/rf_fatigue_runner_model.pkl       — modelo final (todos los atletas)
  src/outputs/fatigue_feature_importance.csv    — importancia de features
  src/outputs/runner_fatigue_predictions_loao.csv — fatigue_score_predicted (T8.1)

Usage
-----
  python run_runner_fatigue.py              # pipeline completo
  python run_runner_fatigue.py --no-loao   # solo entrenar modelo final (más rápido)
  python run_runner_fatigue.py -v          # verbose (DEBUG)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fase 7 — Fatigue Regressor sobre Runner Dataset",
    )
    p.add_argument(
        "--no-loao", action="store_true",
        help="Saltar LOAO (solo entrenar modelo final). Útil para pruebas rápidas.",
    )
    p.add_argument(
        "--csv", type=str, default=None,
        help="Ruta al CSV day_approach_maskedID_timeseries.csv (por defecto: config).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Activar logging DEBUG.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from src.runner.fatigue import run_fatigue_pipeline, RUNNER_CSV

    csv_path = args.csv or RUNNER_CSV

    t0 = time.time()
    results = run_fatigue_pipeline(csv_path=csv_path, skip_loao=args.no_loao)
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"FASE 7 — RESUMEN  ({elapsed:.0f}s)")
    print("=" * 60)
    if "mean_rmse" in results:
        print(f"  RMSE (LOAO)      : {results['mean_rmse']:.4f}")
        print(f"  MAE  (LOAO)      : {results['mean_mae']:.4f}")
        print(f"  R²   (LOAO)      : {results['mean_r2']:.4f}")
        print(f"  Baseline RMSE    : {results['baseline_rmse']:.4f}")
    else:
        print("  LOAO omitido (--no-loao)")
    print("=" * 60)


if __name__ == "__main__":
    main()
