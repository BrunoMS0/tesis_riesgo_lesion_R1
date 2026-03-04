#!/usr/bin/env python
"""
run_pipeline.py – CLI entry‑point for the ETL pipeline.

Usage
-----
    python run_pipeline.py                     # default settings
    python run_pipeline.py --raw-data /path    # override raw‑data path
    python run_pipeline.py --output /out       # override output path
    python run_pipeline.py -v                  # verbose (DEBUG) logging
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.etl.config import PipelineConfig
from src.etl.pipeline import run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PMData ETL Pipeline – Tesis Riesgo de Lesión",
    )
    p.add_argument(
        "--raw-data", type=str, default=None,
        help="Path to the raw PMData directory (default from config).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to store pipeline outputs (default from config).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG‑level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config (override only what the user specified)
    overrides = {}
    if args.raw_data:
        overrides["raw_data_path"] = args.raw_data
    if args.output:
        overrides["output_path"] = args.output
    cfg = PipelineConfig(**overrides)

    # Run
    report = run(cfg)

    # Summary
    print("\n" + "=" * 60)
    print("  ETL PIPELINE – EXECUTION SUMMARY")
    print("=" * 60)
    for s in report.stages:
        print(f"  {s.name:<12s}  {s.duration_s:>6.2f}s  "
              f"->  {s.rows_out} rows x {s.cols_out} cols")
    print("-" * 60)
    print(f"  Total time     : {report.total_duration_s:.2f} s")
    print(f"  Final CSV      : {report.final_csv}")
    print(f"  Final features : {report.final_features}")
    print(f"  tf.data ready  : {report.tf_datasets_created}")
    if report.tfrecord_paths:
        for split, p in report.tfrecord_paths.items():
            print(f"  TFRecord {split:<5s} : {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
