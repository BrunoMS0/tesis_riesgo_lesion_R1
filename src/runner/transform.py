"""
transform.py — Feature engineering for the Runner Dataset (Löwdal 2021).

Converts the 7-day wide format (10 features × 7 day-columns per row) into
a compact set of derived features suitable for the Random Forest injury model.

Column naming convention (raw CSV):
  - No suffix  = D-7 (oldest day in window)
  - '.1'       = D-6
  - '.2'       = D-5
  - '.3'       = D-4
  - '.4'       = D-3
  - '.5'       = D-2
  - '.6'       = D-1  (yesterday, most recent)

Rest days are marked with perceived exertion == -0.01 (per dataset README).
These are excluded from mean-based wellness computations.

Public API
----------
compute_features(df) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import REST_DAY_VALUE, SUFFIX_TO_OFFSET

logger = logging.getLogger(__name__)

# Ordered suffixes from D-7 (oldest) to D-1 (most recent)
_SUFFIXES = ["", ".1", ".2", ".3", ".4", ".5", ".6"]


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _col(base: str, suffix: str) -> str:
    """Return the full column name for a base feature and day suffix."""
    return base if suffix == "" else f"{base}{suffix}"


def _sum_across_days(df: pd.DataFrame, base: str) -> pd.Series:
    """Sum a feature across all 7 day-columns (D-7 to D-1)."""
    return sum(df[_col(base, s)] for s in _SUFFIXES)


def _mean_excl_rest(df: pd.DataFrame, base: str) -> pd.Series:
    """
    Mean of a subjective feature across 7 days, excluding rest days.

    Rest days have perceived exertion == REST_DAY_VALUE (-0.01).
    The same rest-day mask is applied to recovery and trainingSuccess columns.
    """
    exertion_cols = [df[_col("perceived exertion", s)] for s in _SUFFIXES]
    feature_cols  = [df[_col(base, s)] for s in _SUFFIXES]

    # Mask rest days with NaN
    masked = [
        feat.where(ex != REST_DAY_VALUE, other=np.nan)
        for ex, feat in zip(exertion_cols, feature_cols)
    ]
    stacked = pd.concat(masked, axis=1)
    return stacked.mean(axis=1)


def _count_rest_days(df: pd.DataFrame) -> pd.Series:
    """Count days in the 7-day window where perceived exertion == REST_DAY_VALUE."""
    flags = [
        (df[_col("perceived exertion", s)] == REST_DAY_VALUE).astype(int)
        for s in _SUFFIXES
    ]
    return sum(flags)


# ─── ACWR: chronic load via daily time series reconstruction ──────────────────

def _compute_chronic_load_28d(df: pd.DataFrame) -> np.ndarray:
    """
    Compute chronic_load_28d for each row by reconstructing the per-athlete
    daily km time series from the wide format.

    For a row at event date D:
      - The 7 wide columns encode km for days D-7 to D-1.
      - We reconstruct a per-athlete daily km series, filling gaps with 0.
      - Rolling 28-day sum at day D-1 gives the chronic load.

    Returns
    -------
    np.ndarray of shape (len(df),) with chronic_load_28d per row.
    """
    # Build all (participant, actual_day, km) tuples from the wide columns
    pieces = []
    for suffix, offset in SUFFIX_TO_OFFSET.items():
        col = _col("total km", suffix)
        tmp = df[["participant_id", "date", col]].copy()
        tmp["actual_day"] = df["date"] - offset
        tmp = tmp.rename(columns={col: "km"})
        pieces.append(tmp[["participant_id", "actual_day", "km"]])

    daily_km = pd.concat(pieces, ignore_index=True)

    # Deduplicate: the same (athlete, day) appears in up to 7 rows with identical km
    daily_km = daily_km.drop_duplicates(["participant_id", "actual_day"])
    daily_km = daily_km.sort_values(["participant_id", "actual_day"]).reset_index(drop=True)

    # Per athlete: fill missing days with 0, compute rolling 28-day sum
    result_parts = []
    for pid, group in daily_km.groupby("participant_id"):
        day_min = int(group["actual_day"].min())
        day_max = int(group["actual_day"].max())

        full_days = pd.DataFrame({
            "actual_day": range(day_min, day_max + 1),
            "participant_id": pid,
        })
        merged = full_days.merge(group[["actual_day", "km"]], on="actual_day", how="left")
        merged["km"] = merged["km"].fillna(0.0)
        merged["rolling_28d"] = merged["km"].rolling(window=28, min_periods=1).sum()
        result_parts.append(merged[["participant_id", "actual_day", "rolling_28d"]])

    daily_rolling = pd.concat(result_parts, ignore_index=True)

    # For event date D, look up rolling_28d at day D-1 (the last training day)
    lookup = daily_rolling.rename(
        columns={"actual_day": "lookup_day", "rolling_28d": "chronic_load_28d"}
    )
    df_lookup = df[["participant_id", "date"]].copy()
    df_lookup["lookup_day"] = df_lookup["date"] - 1
    df_lookup = df_lookup.merge(lookup, on=["participant_id", "lookup_day"], how="left")

    # Fallback for earliest rows where 28d history doesn't exist:
    # use the 7-day acute load as the chronic estimate
    acute_fallback = _sum_across_days(df, "total km").values
    chronic = df_lookup["chronic_load_28d"].fillna(pd.Series(acute_fallback)).values

    return chronic


# ─── Main feature engineering function ────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from the 7-day wide format DataFrame.

    Input  : raw DataFrame from extract.load_runner_csv()
    Output : DataFrame with [participant_id, date, injury] + RUNNER_FEATURE_COLUMNS.
             The 70 raw wide-format columns are dropped.

    NaN handling
    ------------
    - Rest-day exclusion can leave NaN for athletes with all-rest windows.
    - These are forward/backward filled per athlete, then zero-filled.
    """
    out = df[["participant_id", "date", "injury"]].copy()

    # ── Training load ──────────────────────────────────────────────────────────
    out["acute_load_7d"] = _sum_across_days(df, "total km")
    out["chronic_load_28d"] = _compute_chronic_load_28d(df)

    # ACWR = acute / (chronic/4).  Chronic/4 = average weekly load.
    # Clip to [0, 5] to bound extreme values at early-season rows.
    weekly_chronic_avg = (out["chronic_load_28d"] / 4.0).clip(lower=0.01)
    out["acwr"] = (out["acute_load_7d"] / weekly_chronic_avg).clip(upper=5.0)

    out["high_intensity_km_7d"] = (
        _sum_across_days(df, "km Z3-4") + _sum_across_days(df, "km Z5-T1-T2")
    )
    out["km_sprint_7d"]     = _sum_across_days(df, "km sprinting")
    out["nr_sessions_7d"]   = _sum_across_days(df, "nr. sessions")
    out["nr_rest_days_7d"]  = _count_rest_days(df)
    out["strength_days_7d"] = _sum_across_days(df, "strength training")
    out["alt_hours_7d"]     = _sum_across_days(df, "hours alternative")

    # ── Subjective wellness (rest-day-excluded means) ──────────────────────────
    out["mean_perceived_exertion"] = _mean_excl_rest(df, "perceived exertion")
    out["mean_perceived_recovery"] = _mean_excl_rest(df, "perceived recovery")
    out["mean_perceived_success"]  = _mean_excl_rest(df, "perceived trainingSuccess")

    # wellness_score: composite of recovery + success (both positive signals)
    out["wellness_score"] = (
        out["mean_perceived_recovery"].fillna(0) +
        out["mean_perceived_success"].fillna(0)
    ) / 2.0

    # session_load_proxy = acute volume × mean perceived effort  (sRPE analogue)
    out["session_load_proxy"] = (
        out["acute_load_7d"] * out["mean_perceived_exertion"].fillna(0)
    )

    # ── Most recent day features (D-1, suffix '.6') ────────────────────────────
    out["recent_exertion"] = df["perceived exertion.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_recovery"] = df["perceived recovery.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_success"]  = df["perceived trainingSuccess.6"].replace(REST_DAY_VALUE, np.nan)
    out["recent_km"]       = df["total km.6"]

    # ── Impute residual NaNs (all-rest windows, cold-start rows) ──────────────
    float_cols = [c for c in out.columns if c not in ("participant_id", "date", "injury")]
    out[float_cols] = (
        out.groupby("participant_id")[float_cols]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0.0)
    )

    logger.info(
        "Feature engineering complete: %d rows, %d derived features, "
        "injury prevalence=%.2f%% (%d events)",
        len(out), len(float_cols),
        100.0 * out["injury"].mean(), int(out["injury"].sum()),
    )
    return out
