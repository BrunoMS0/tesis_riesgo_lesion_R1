#!/usr/bin/env python
"""
run_injury.py – CLI entry-point for the R5 Injury Risk Prediction pipeline.

Usage
-----
    python run_injury.py                             # default settings
    python run_injury.py --input-csv /path/data.csv  # override input
    python run_injury.py --dfi-csv /path/dfi.csv     # override DFI predictions
    python run_injury.py --output /out               # override output path
    python run_injury.py -v                          # verbose (DEBUG) logging
"""

from __future__ import annotations

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
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
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
    cfg = InjuryConfig(**overrides)

    report = run(cfg)

    print("\n" + "═" * 60)
    print("R5 PIPELINE COMPLETE")
    print("═" * 60)
    for stage in report.stages:
        print(f"  {stage.name:<30s}  {stage.duration_s:>7.2f}s")
    print(f"  {'TOTAL':<30s}  {report.total_duration_s:>7.2f}s")
    print(f"\nTrain: {report.n_train} (augmented: {report.n_train_augmented})")
    print(f"Val: {report.n_val}  |  Test: {report.n_test}  |  Features: {report.n_features}")
    print(f"\nXGBoost  PR-AUC: {report.xgb_metrics.get('pr_auc', 'N/A')}")
    print(f"RF       PR-AUC: {report.rf_metrics.get('pr_auc', 'N/A')}")
    print(f"LOSO     PR-AUC: {report.loso_pr_auc}")
    if report.comparison_table is not None:
        print(f"\n{report.comparison_table.to_string()}")


if __name__ == "__main__":
    main()
