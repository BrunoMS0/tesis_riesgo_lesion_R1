"""
week_transform.py — Feature mapping from week_approach to daily feature names.

Extracts the 9 common features from week_approach_maskedID_timeseries.csv
and renames them to match the daily pipeline's feature names, enabling
cross-granularity validation (train daily → evaluate weekly).

Weekly dataset structure:
  - Each row is a weekly prediction window
  - Columns with no suffix = most recent week (W0)
  - Columns with '.1' = previous week (W-1)
  - Columns with '.2' = two weeks ago (W-2)
  - 'rel total kms week 0_1' = ratio W0/W-1 (ACWR proxy)
  - 'injury' = prospective label (already windowed by dataset authors)

Feature mapping (weekly → daily equivalent):
  total kms             → acute_load_7d
  rel total kms week 0_1 → acwr
  nr. sessions          → nr_sessions_7d
  nr. rest days         → nr_rest_days_7d
  total km Z3-Z4-Z5-T1-T2 → high_intensity_km_7d
  nr. strength trainings → strength_days_7d
  avg recovery          → mean_perceived_recovery
  avg exertion          → mean_perceived_exertion
  avg training success  → mean_perceived_success

Public API
----------
load_week_approach(csv_path) -> pd.DataFrame
    Returns DataFrame with [participant_id, date, injury, *9 daily feature names]
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

WEEK_CSV: str = os.path.join(
    _WORKSPACE_ROOT, "Runner dataset", "week_approach_maskedID_timeseries.csv"
)

# Mapping: raw weekly column name → daily feature name used in the pipeline
WEEK_TO_DAILY_MAP: dict = {
    "total kms":               "acute_load_7d",
    "rel total kms week 0_1":  "acwr",
    "nr. sessions":            "nr_sessions_7d",
    "nr. rest days":           "nr_rest_days_7d",
    "total km Z3-Z4-Z5-T1-T2": "high_intensity_km_7d",
    "nr. strength trainings":  "strength_days_7d",
    "avg recovery":            "mean_perceived_recovery",
    "avg exertion":            "mean_perceived_exertion",
    "avg training success":    "mean_perceived_success",
}

# Ordered list of the 9 common feature names (in daily pipeline naming)
WEEK_COMMON_FEATURES: List[str] = list(WEEK_TO_DAILY_MAP.values())


def load_week_approach(csv_path: str = WEEK_CSV) -> pd.DataFrame:
    """
    Load week_approach CSV and return with standardised column names.

    Identity columns:
      participant_id  : str — 'runner_0' … 'runner_73' (matches daily pipeline)
      date            : int — per-athlete sequential week index (from 'Date')
      injury          : int — 0/1, prospective label

    Feature columns (9, renamed from weekly to daily names):
      acute_load_7d, acwr, nr_sessions_7d, nr_rest_days_7d,
      high_intensity_km_7d, strength_days_7d,
      mean_perceived_recovery, mean_perceived_exertion, mean_perceived_success

    Parameters
    ----------
    csv_path : path to week_approach_maskedID_timeseries.csv

    Returns
    -------
    pd.DataFrame sorted by (participant_id, date)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Week approach CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    logger.info(
        "Loaded week_approach: %d rows × %d cols from %s",
        len(df), len(df.columns), csv_path,
    )

    # Validate required columns
    missing = [c for c in WEEK_TO_DAILY_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in week_approach CSV: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    # Build output DataFrame
    out = pd.DataFrame()
    out["participant_id"] = df["Athlete ID"].apply(lambda x: f"runner_{int(x)}")
    out["date"]           = df["Date"].astype(int)
    out["injury"]         = df["injury"].astype(int)

    for raw_col, daily_col in WEEK_TO_DAILY_MAP.items():
        series = pd.to_numeric(df[raw_col], errors="coerce")
        # Clip ACWR to [0, 4.0] — weekly rel_kms has div-by-zero outliers up to 2e8
        # when the previous week km == 0 (rest week / start of training).
        # The daily pipeline caps acwr at 4.0 (99th-pct = 4.0 in processed CSV).
        if daily_col == "acwr":
            series = series.clip(lower=0.0, upper=4.0)
        out[daily_col] = series.fillna(0.0)

    out = out.sort_values(["participant_id", "date"]).reset_index(drop=True)

    n_athletes = out["participant_id"].nunique()
    n_injured  = out[out["injury"] == 1]["participant_id"].nunique()
    n_injuries = int(out["injury"].sum())

    logger.info(
        "Week approach prepared: %d rows | %d athletes (%d with injuries) | "
        "%d injury events (%.2f%%) | %d features",
        len(out), n_athletes, n_injured, n_injuries,
        100.0 * out["injury"].mean(), len(WEEK_COMMON_FEATURES),
    )
    return out
