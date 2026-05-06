"""
extract.py — Load and minimally clean the Runner Dataset CSV.

Renames identity columns to match the pipeline convention:
  - 'Athlete ID' → 'participant_id' (str, prefixed with 'runner_')
  - 'Date'       → 'date' (int, per-athlete sequential day index)
  - 'injury'     → unchanged (int, already prospective label)

All 70 raw feature columns are preserved unchanged for downstream
feature engineering in transform.py.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .config import RUNNER_CSV

logger = logging.getLogger(__name__)


def load_runner_csv(csv_path: str = RUNNER_CSV) -> pd.DataFrame:
    """
    Load the Runner dataset CSV and rename identity columns.

    Parameters
    ----------
    csv_path : Path to day_approach_maskedID_timeseries.csv.

    Returns
    -------
    pd.DataFrame with columns:
      - participant_id : str ('runner_0' … 'runner_73')
      - date           : int  (per-athlete sequential day index, starts at 0)
      - injury         : int  (0 / 1, already prospective)
      - 70 raw feature columns (unchanged names, e.g. 'total km', 'total km.1', ...)

    Sorted by (participant_id, date).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Runner CSV not found: {csv_path}\n"
            f"Expected at: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), csv_path)

    # Rename identity columns to pipeline convention
    df = df.rename(columns={"Athlete ID": "participant_id", "Date": "date"})

    # Convert integer athlete IDs to strings with 'runner_' prefix.
    # Required by loso_cross_validation which uses str.startswith() filtering.
    df["participant_id"] = "runner_" + df["participant_id"].astype(str)

    # Sort chronologically per athlete for correct temporal ordering
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)

    n_athletes = df["participant_id"].nunique()
    n_injuries = int(df["injury"].sum())
    n_injured_athletes = int((df.groupby("participant_id")["injury"].max() > 0).sum())

    logger.info(
        "Runner dataset: %d athletes (%d with >=1 injury), "
        "%d injury events (%.2f%% prevalence), date range %d-%d",
        n_athletes, n_injured_athletes,
        n_injuries, 100.0 * df["injury"].mean(),
        int(df["date"].min()), int(df["date"].max()),
    )
    return df
