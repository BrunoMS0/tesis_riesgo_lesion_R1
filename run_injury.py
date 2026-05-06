#!/usr/bin/env python
"""
run_injury.py – CLI entry-point for the R5 Injury Risk Prediction pipeline.

Usage
-----
    python run_injury.py                             # default settings
    python run_injury.py --input-csv /path/data.csv  # override input
    python run_injury.py --dfi-csv /path/dfi.csv     # override DFI predictions
    python run_injury.py --output /out               # override output path
    python run_injury.py --augmentation smote         # SMOTE augmentation
    python run_injury.py --augmentation copula        # Gaussian Copula augmentation
    python run_injury.py -v                          # verbose (DEBUG) logging
"""

from __future__ import annotations

# ── Windows / OpenBLAS threading fix ────────────────────────────────────────
# NearestNeighbors.kneighbors() deadlocks on Windows when the BLAS thread pool
# spins up multiple threads inside a script context.  Forcing single-threaded
# BLAS here (before numpy is imported) prevents the deadlock in SMOTE.
import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys

from src.injury.config import InjuryConfig
from src.injury.pipeline import run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="R5 – ML Injury Risk Prediction Pipeline",
    )
    p.add_argument(
        "--input-csv", type=str, default=None,
        help="Path to the un-normalised feature CSV (default from config).",
    )
    p.add_argument(
        "--dfi-csv", type=str, default=None,
        help="Path to R4 fatigue index predictions CSV (default from config).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to store pipeline outputs (default from config).",
    )
    p.add_argument(
        "--augmentation", type=str, default=None,
        choices=["smote", "copula"],
        help="Data augmentation method: 'smote' or 'copula' (default from config).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--lstm", action="store_true",
        help="Enable Stage 5 LSTM temporal model LOAO (slow).",
    )
    p.add_argument(
        "--lstm-window", type=int, default=None,
        help="Sequence window size in days for LSTM (default: 14).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {}
    if args.input_csv:
        overrides["input_csv"] = args.input_csv
    if args.dfi_csv:
        overrides["dfi_csv"] = args.dfi_csv
    if args.output:
        overrides["output_path"] = args.output
    if args.augmentation:
        overrides["augmentation_method"] = args.augmentation
        # Changing augmentation invalidates cached Stage 4b results
        overrides["rerun_combined"] = True
    if args.lstm:
        overrides["use_lstm"] = True
    if args.lstm_window:
        overrides["lstm_window_size"] = args.lstm_window
    cfg = InjuryConfig(**overrides)

    report = run(cfg)

    print("\n" + "=" * 60)
    print("R5 PIPELINE COMPLETE")
    print("=" * 60)
    for stage in report.stages:
        print(f"  {stage.name:<30s}  {stage.duration_s:>7.2f}s")
    print(f"  {'TOTAL':<30s}  {report.total_duration_s:>7.2f}s")
    print(f"\nTrain: {report.n_train} (augmented: {report.n_train_augmented})")
    print(f"Val: {report.n_val}  |  Test: {report.n_test}  |  Features: {report.n_features}")
    print(f"\nLogistic Regression  ROC-AUC: {report.lr_metrics.get('roc_auc', 'N/A')}")
    print(f"Baseline             ROC-AUC: {report.baseline_metrics.get('roc_auc', 'N/A')}")
    print(f"LOSO                 ROC-AUC: {report.loso_roc_auc}")
    if report.soccermon_metrics:
        sm = report.soccermon_metrics
        print("\n" + "-" * 60)
        print("  SoccerMon External Test Set  (real injury labels)")
        print("-" * 60)
        print(f"  ROC-AUC  : {sm.get('roc_auc', 'N/A'):.4f}")
        print(f"  PR-AUC   : {sm.get('pr_auc', 'N/A'):.4f}")
        print(f"  F1       : {sm.get('f1', 'N/A'):.4f}")
        print(f"  Brier    : {sm.get('brier_score', 'N/A'):.4f}")
        print("-" * 60)
    if report.comparison_table is not None:
        print(f"\n{report.comparison_table.to_string()}")


if __name__ == "__main__":
    main()
