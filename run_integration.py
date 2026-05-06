#!/usr/bin/env python
"""
run_integration.py – CLI entry-point for the R6 Two-Stage Pipeline.

Loads pre-trained R4 (fatigue) and R5 (injury) models, runs them in
sequence, and outputs integrated predictions.

Usage
-----
    python run_integration.py
    python run_integration.py --fatigue-model path/best_weights.keras
    python run_integration.py --injury-model path/xgboost_injury.joblib
    python run_integration.py --input-csv path/features.csv
    python run_integration.py --output path/out -v
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.fatigue.config import FatigueConfig
from src.injury.config import InjuryConfig
from src.integration.config import IntegrationConfig
from src.integration.pipeline import run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="R6 – Two-Stage Predictive Integration Pipeline",
    )
    p.add_argument(
        "--input-csv", type=str, default=None,
        help="Path to the un-normalised feature CSV (default from config).",
    )
    p.add_argument(
        "--fatigue-model", type=str, default=None,
        help="Path to R4 fatigue model (.keras) file.",
    )
    p.add_argument(
        "--injury-model", type=str, default=None,
        help="Path to R5 injury model (.joblib) file.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to store integration outputs (default from config).",
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

    # Build sub-configs with optional input-csv override
    fatigue_overrides = {}
    injury_overrides = {}
    if args.input_csv:
        fatigue_overrides["input_csv"] = args.input_csv
        injury_overrides["input_csv"] = args.input_csv

    integration_kwargs = {
        "fatigue_cfg": FatigueConfig(**fatigue_overrides),
        "injury_cfg": InjuryConfig(**injury_overrides),
    }
    if args.fatigue_model:
        integration_kwargs["fatigue_model_path"] = args.fatigue_model
    if args.injury_model:
        integration_kwargs["injury_model_path"] = args.injury_model
    if args.output:
        integration_kwargs["output_path"] = args.output

    cfg = IntegrationConfig(**integration_kwargs)
    report = run(cfg)

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("  R6 TWO-STAGE PIPELINE -- EXECUTION SUMMARY")
    print(sep)
    for s in report.stages:
        print(f"  {s.name:<22s}  {s.duration_s:>7.2f}s")
    print("-" * 60)
    print(f"  Total time        : {report.total_duration_s:.2f} s")
    print(f"  DFI predictions   : {report.fatigue_summary.get('n_dfi', 0)}")
    print(f"  DFI mean / std    : "
          f"{report.fatigue_summary.get('dfi_mean', 0):.4f} / "
          f"{report.fatigue_summary.get('dfi_std', 0):.4f}")
    print(f"  Test predictions  : {report.n_predictions}")
    print(f"  Injury PR-AUC     : "
          f"{report.injury_metrics.get('pr_auc', 'N/A')}")
    print(f"  Injury ROC-AUC    : "
          f"{report.injury_metrics.get('roc_auc', 'N/A')}")
    print(f"  Injury F1         : "
          f"{report.injury_metrics.get('f1', 'N/A')}")
    print(f"  Output CSV        : {report.output_csv}")
    print(f"  Fatigue model     : {report.fatigue_model_path}")
    print(f"  Injury model      : {report.injury_model_path}")
    print(sep)


if __name__ == "__main__":
    main()
