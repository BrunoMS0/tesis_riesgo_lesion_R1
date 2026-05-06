#!/usr/bin/env python
"""
run_soccermon.py – CLI entry point for the SoccerMon ETL pipeline.

Converts the raw SoccerMon dataset into an R5-compatible player-day CSV
with real injury labels, ready for external evaluation of the injury model.

Usage
-----
    python run_soccermon.py
    python run_soccermon.py --base-path "C:/path/to/subjective/subjective"
    python run_soccermon.py --recovery-window 14
    python run_soccermon.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SoccerMon ETL — build R5-compatible dataset with real injury labels",
    )
    p.add_argument(
        "--base-path", type=str, default=None,
        help="Path to the SoccerMon 'subjective' folder (default from config).",
    )
    p.add_argument(
        "--output-csv", type=str, default=None,
        help="Output CSV path (default: src/outputs/soccermon_dataset_final.csv).",
    )
    p.add_argument(
        "--recovery-window", type=int, default=None,
        help="Days after injury onset marked as is_injured=1 (default: 7).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.soccermon.config import SoccerMonConfig, SOCCERMON_FEATURES
    from src.soccermon.pipeline import run
    from src.injury.config import FEATURE_COLUMNS

    cfg_kwargs = {}
    if args.base_path:
        cfg_kwargs["base_path"] = args.base_path
    if args.output_csv:
        cfg_kwargs["output_csv"] = args.output_csv
    if args.recovery_window is not None:
        cfg_kwargs["recovery_window_days"] = args.recovery_window

    cfg = SoccerMonConfig(**cfg_kwargs)

    print(f"\nSoccerMon ETL Pipeline")
    print(f"  Base path      : {cfg.base_path}")
    print(f"  Output CSV     : {cfg.output_csv}")
    print(f"  Injury window  : onset + {cfg.recovery_window_days} days")
    print()

    df = run(cfg)

    # ── Summary ───────────────────────────────────────────
    n_players  = df["participant_id"].nunique()
    n_rows     = len(df)
    n_injured  = int(df["is_injured"].sum())
    prevalence = 100.0 * df["is_injured"].mean()

    real_features    = [f for f in FEATURE_COLUMNS if f in SOCCERMON_FEATURES]
    imputed_features = [f for f in FEATURE_COLUMNS if f not in SOCCERMON_FEATURES]

    print("\n" + "=" * 60)
    print("SOCCERMON ETL COMPLETE")
    print("=" * 60)
    print(f"  Players          : {n_players}")
    print(f"  Total rows       : {n_rows:,}")
    print(f"  Injury positives : {n_injured} ({prevalence:.1f}% prevalence)")
    print(f"\n  Real/computable  : {len(real_features)}/{len(FEATURE_COLUMNS)} features")
    print(f"  Imputed (median) : {len(imputed_features)}/{len(FEATURE_COLUMNS)} features")
    print(f"\n  Real features    : {real_features}")
    print(f"  Imputed features : {imputed_features}")
    print(f"\n  Saved to: {cfg.output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
