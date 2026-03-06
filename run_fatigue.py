#!/usr/bin/env python
"""
run_fatigue.py – CLI entry-point for the R4 Fatigue model pipeline.

Usage
-----
    python run_fatigue.py                             # default settings
    python run_fatigue.py --input-csv /path/data.csv  # override input
    python run_fatigue.py --output /out               # override output path
    python run_fatigue.py -v                          # verbose (DEBUG) logging
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.fatigue.config import FatigueConfig
from src.fatigue.pipeline import run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="R4 – Deep Learning Fatigue Estimation Pipeline",
    )
    p.add_argument(
        "--input-csv", type=str, default=None,
        help="Path to the un-normalised feature CSV (default from config).",
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
    if args.output:
        overrides["output_path"] = args.output
    cfg = FatigueConfig(**overrides)

    report = run(cfg)

    # Summary
    print("\n" + "=" * 60)
    print("  R4 FATIGUE MODEL – EXECUTION SUMMARY")
    print("=" * 60)
    for s in report.stages:
        print(f"  {s.name:<18s}  {s.duration_s:>7.2f}s  {s.details}")
    print("-" * 60)
    print(f"  Total time     : {report.total_duration_s:.2f} s")
    print(f"  Sequences      : train={report.n_train}, "
          f"val={report.n_val}, test={report.n_test}")
    print(f"  Features       : {report.n_features}")
    print(f"  Test metrics   : {report.test_metrics}")
    print(f"  Predictions CSV: {report.predictions_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
