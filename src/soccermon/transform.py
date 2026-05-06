"""
transform.py – TRANSFORM stage of the SoccerMon ETL pipeline.

Converts raw long-format DataFrames into a single player-day DataFrame
with all R5-compatible features and binary injury labels.

Feature engineering:
    - acute_load_7d    : 7-day rolling sum of session_load (per player)
    - chronic_load_28d : 28-day rolling mean of session_load (per player)
    - wellness_score   : row-wise mean of the 6 wellness variables
    - is_injured       : 1 for injury onset day + recovery_window_days (vectorised merge)

Missing R5 features (HR, sleep-device, activity, exercise, TRIMP, DFI) are
imputed with PMData training-set medians; constant-valued columns are ignored
by the Random Forest at prediction time.

Public API
----------
transform(result, cfg, pmdata_medians, r5_feature_cols) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Returns (df_raw, df_z, df_z_prospective) where df_z_prospective also has
    injury_next{window}d for use in combined PMData+SoccerMon training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SoccerMonConfig
from .extract import ExtractResult

logger = logging.getLogger(__name__)

_WELLNESS_VARS = ["fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress"]


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _compute_load_features(load_long: pd.DataFrame) -> pd.DataFrame:
    """Add acute_load_7d and chronic_load_28d rolling features to load_long."""
    df = load_long.sort_values(["player_id", "date"]).copy()
    df["acute_load_7d"] = (
        df.groupby("player_id")["session_load"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    df["chronic_load_28d"] = (
        df.groupby("player_id")["session_load"]
        .transform(lambda s: s.rolling(28, min_periods=1).mean())
    )
    return df


def _build_injury_labels(
    df: pd.DataFrame,
    injuries: pd.DataFrame,
    window: int = 7,
) -> pd.Series:
    """
    Vectorised construction of is_injured labels.

    For each injury event (player_id, injury_date):
        days [injury_date, injury_date + window] → is_injured = 1

    Parameters
    ----------
    df : DataFrame with columns 'player_id' and 'date'.
    injuries : DataFrame with columns 'player_id' and 'injury_date'.
    window : int — days after onset also marked as injured (default 7).

    Returns
    -------
    pd.Series of dtype int, same index as df.
    """
    if injuries.empty:
        return pd.Series(0, index=df.index)

    # Build all (player_id, date) pairs that should be labelled as injured
    injured_rows = []
    for _, row in injuries.iterrows():
        for i in range(window + 1):          # onset + window recovery days
            injured_rows.append({
                "player_id": row["player_id"],
                "date": row["injury_date"] + pd.Timedelta(days=i),
            })

    injured_df = (
        pd.DataFrame(injured_rows)
        .drop_duplicates()
        .assign(is_injured=1)
    )

    # Merge to assign labels — left join preserves all rows in df
    merged = (
        df[["player_id", "date"]]
        .reset_index(drop=True)
        .merge(injured_df, on=["player_id", "date"], how="left")
    )
    return merged["is_injured"].fillna(0).astype(int)


def _create_prospective_target(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Create ``injury_next{window}d`` column per player.

    Label at day T is 1 if any ``is_injured``==1 falls within [T+1, T+window].
    The last ``window`` rows per player are dropped (undefined future window).
    ``is_injured`` is preserved unchanged.
    """
    df_out = df.copy().sort_values(["participant_id", "date"]).reset_index(drop=True)
    col_name = f"injury_next{window}d"
    df_out[col_name] = df_out.groupby("participant_id")["is_injured"].transform(
        lambda x: x.rolling(window, min_periods=1).max().shift(-window)
    )
    n_before = len(df_out)
    df_out = df_out.dropna(subset=[col_name]).reset_index(drop=True)
    df_out[col_name] = df_out[col_name].astype(int)
    n_dropped = n_before - len(df_out)
    logger.info(
        "Prospective target '%s' (window=%d): %d rows (dropped %d tail rows), "
        "prevalence=%.1f%%",
        col_name, window, len(df_out), n_dropped,
        100.0 * df_out[col_name].mean(),
    )
    return df_out


def _compute_wellness_score(df: pd.DataFrame) -> pd.Series:
    """Row-wise mean of available wellness variables (mirrors PMData wellness_score)."""
    available = [c for c in _WELLNESS_VARS if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    return df[available].mean(axis=1)


def _apply_per_athlete_zscore(
    df: pd.DataFrame,
    player_col: str,
    features: List[str],
) -> pd.DataFrame:
    """
    Apply per-athlete z-score normalisation to ``features``.

    For each athlete i and feature f:
        z_{i,f} = (x_{i,f} - mean_i_f) / std_i_f

    Edge cases:
        - std == 0 (constant feature for an athlete) → z = 0.0, no exception.
        - NaN values are preserved as NaN (not imputed here).

    Parameters
    ----------
    df         : DataFrame with ``player_col`` column and feature columns.
    player_col : Name of the athlete-identifier column.
    features   : Feature names to z-score (only existing columns are processed).

    Returns
    -------
    Copy of ``df`` with the specified features replaced by their z-scores.
    """
    df_out = df.copy()
    cols_present = [c for c in features if c in df_out.columns]

    def _zscore_group(group: pd.DataFrame) -> pd.DataFrame:
        for col in cols_present:
            series = group[col]
            mu = series.mean()
            sigma = series.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                group = group.copy()
                group[col] = 0.0
            else:
                group = group.copy()
                group[col] = (series - mu) / sigma
        return group

    df_out = df_out.groupby(player_col, group_keys=False).apply(_zscore_group)
    df_out = df_out.reset_index(drop=True)

    # Log summary stats for verification
    if cols_present:
        z_means = df_out[cols_present].mean().round(4).to_dict()
        z_stds  = df_out[cols_present].std(ddof=1).round(4).to_dict()
        logger.info(
            "Per-athlete z-score applied to %d features.\n"
            "  Global means (should be ≈0): %s\n"
            "  Global stds  (should be ≈1): %s",
            len(cols_present), z_means, z_stds,
        )
    return df_out


def _impute_missing_features(
    df: pd.DataFrame,
    pmdata_medians: Dict[str, float],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Ensure every R5 feature column is present in df.
    - Missing columns are added with the PMData training-set median value.
    - Existing columns with remaining NaNs are filled with the same median.
    The Random Forest treats constant-valued columns as zero-importance features.
    """
    for col in feature_cols:
        fill_val = float(pmdata_medians.get(col, 0.0))
        if col not in df.columns:
            df[col] = fill_val
        elif df[col].isna().any():
            df[col] = df[col].fillna(fill_val)
    return df


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def transform(
    result: ExtractResult,
    cfg: SoccerMonConfig,
    pmdata_medians: Dict[str, float],
    r5_feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the final player-day DataFrames in R5-compatible format.

    Pipeline:
        1. Compute load features (acute_load_7d, chronic_load_28d)
        2. Merge all wellness variables
        3. Merge pre-computed ACWR (overrides computed ACWR for accuracy)
        4. Compute wellness_score
        5. Build is_injured labels (onset + recovery_window days)
        6. Filter to rows with ≥1 wellness observation (monitored days only)
        7. Impute all missing R5 features with PMData training medians
        8. Apply per-athlete z-score to load + wellness features → df_z
        9. Return (df_raw, df_z): participant_id | date | <37 features> | is_injured

    Parameters
    ----------
    result        : ExtractResult from extract.extract_all()
    cfg           : SoccerMonConfig
    pmdata_medians: dict of {feature_name: median_value} from PMData training set
    r5_feature_cols: ordered list of the 37 R5 feature column names

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        df_raw         : raw (unscaled) R5-compatible player-day dataset.
        df_z           : same dataset with per-athlete z-score applied to load+wellness features.
        df_z_prospective : df_z plus ``injury_next{window}d`` column (tail rows dropped).
    """

    # Step 1 — Load features
    df = _compute_load_features(result.load)

    # Step 2 — Merge wellness variables
    for var, wdf in result.wellness.items():
        df = df.merge(wdf[["date", "player_id", var]], on=["date", "player_id"], how="left")

    # Step 3 — Merge pre-computed ACWR (SoccerMon provides this directly)
    acwr_df = result.acwr.copy()
    df = df.merge(acwr_df[["date", "player_id", "acwr"]], on=["date", "player_id"], how="left")
    df["acwr"] = df["acwr"].fillna(0.0)

    # Step 4 — Wellness score
    df["wellness_score"] = _compute_wellness_score(df)

    # Step 5 — Injury labels
    df["is_injured"] = _build_injury_labels(
        df, result.injuries, window=cfg.recovery_window_days,
    )

    # Step 6 — Filter to monitored days (at least one wellness value present)
    wellness_cols = [c for c in cfg.wellness_vars if c in df.columns]
    has_wellness = df[wellness_cols].notna().any(axis=1)
    n_before = len(df)
    df = df[has_wellness].reset_index(drop=True)
    n_after = len(df)
    logger.info(
        "Filtered %d → %d rows (kept days with ≥1 wellness observation)",
        n_before, n_after,
    )

    # Step 7 — Impute missing R5 features
    n_real = sum(1 for c in r5_feature_cols if c in df.columns)
    df = _impute_missing_features(df, pmdata_medians, r5_feature_cols)
    logger.info(
        "Feature coverage: %d/%d real/computable, %d imputed with PMData medians",
        n_real, len(r5_feature_cols), len(r5_feature_cols) - n_real,
    )

    # Step 8 — Finalise output schema
    df = df.rename(columns={"player_id": "participant_id"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    meta_cols = ["participant_id", "date"]
    final_cols = meta_cols + r5_feature_cols + ["is_injured"]
    df_raw = df[final_cols].reset_index(drop=True)

    # Step 9 — Per-athlete z-score for cross-domain evaluation
    logger.info("Applying per-athlete z-score to %d features", len(cfg.zscore_features))
    df_z = _apply_per_athlete_zscore(df_raw, "participant_id", cfg.zscore_features)

    # Step 10 — Prospective target for combined PMData+SoccerMon training
    df_z_prospective = _create_prospective_target(df_z.copy(), window=cfg.prospective_window)

    return df_raw, df_z, df_z_prospective
