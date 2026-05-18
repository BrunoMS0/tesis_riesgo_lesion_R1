"""
dataset.py – Data preparation for R5 Injury Risk Prediction.

Responsibilities
----------------
1. Load the un-normalised feature CSV produced by the ETL.
2. Merge R4 DFI predictions (left-join on participant_id + date).
3. Select feature columns for injury prediction.
4. Split by participant (train / val / test) — same seed as R4.

Public API
----------
load_and_merge(cfg) -> pd.DataFrame
prepare_features(df, cfg, target_col=None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
split_participants(df, cfg) -> Tuple[List[str], List[str], List[str]]
build_injury_datasets(cfg) -> InjuryDatasetBundle
apply_per_athlete_zscore(X, meta, feature_list) -> pd.DataFrame
create_prospective_target(df, window) -> pd.DataFrame
build_combined_dataset(cfg) -> InjuryDatasetBundle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import InjuryConfig
from .normalize import NormalizerResult, apply_normalizer, fit_normalizer

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 0a. Prospective target creation
# ────────────────────────────────────────────────────────────

def create_prospective_target(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Create a prospective injury target column ``injury_next{window}d``.

    For each day T, the label is 1 if there is **any** injury event in the
    next ``window`` days (T+1 to T+window) for that participant.  The raw
    ``is_injured`` column is preserved unchanged.

    Formula (per-athlete):
        rolling(window, min_periods=1).max().shift(-window)

    At position T this resolves to max(is_injured[T+1 : T+window+1]), i.e.
    a positive window of exactly ``window`` days beginning tomorrow.

    No data leakage: features at day T use only data from day T and
    earlier (all backward-looking rolling windows in engineering stage),
    while this label looks strictly ahead.

    Rows where the future window is incomplete (last ``window`` rows per
    athlete) are dropped to avoid undefined labels.

    Parameters
    ----------
    df     : DataFrame with columns [``participant_id``, ``is_injured``, ``date``].
             Must be sorted by (participant_id, date) or will be sorted here.
    window : Look-ahead window in days (default 7).

    Returns
    -------
    df_out : Copy of ``df`` with new integer column ``injury_next{window}d``;
             rows with NaN labels (incomplete future window) are dropped.
    """
    df_out = df.copy().sort_values(["participant_id", "date"]).reset_index(drop=True)
    col_name = f"injury_next{window}d"

    df_out[col_name] = df_out.groupby("participant_id")["is_injured"].transform(
        lambda x: x.rolling(window, min_periods=1).max().shift(-window)
    )

    # Drop rows where the prospective window extends beyond the athlete's data
    n_before = len(df_out)
    df_out = df_out.dropna(subset=[col_name]).reset_index(drop=True)
    df_out[col_name] = df_out[col_name].astype(int)
    n_dropped = n_before - len(df_out)

    logger.info(
        "Prospective target '%s' (window=%d): %d rows (dropped %d tail rows), "
        "prevalence=%.1f%% vs is_injured prevalence=%.1f%%",
        col_name, window, len(df_out), n_dropped,
        100.0 * df_out[col_name].mean(),
        100.0 * df_out["is_injured"].mean(),
    )
    return df_out


# ────────────────────────────────────────────────────────────
# 0b. Per-athlete z-score normalisation (cross-domain feature alignment)
# ────────────────────────────────────────────────────────────

def apply_per_athlete_zscore(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Apply per-athlete z-score normalisation to the specified features.

    For each participant p and feature f:
        z_{p,f} = (x_{p,f} - mean_{p,f}) / std_{p,f}

    Eliminates absolute-load shift (~21× between sports) and wellness-scale
    mismatch (PMData 1-6 vs SoccerMon 1-10 Likert) so that cross-domain AUC
    reflects pattern similarity rather than distributional shift.

    Edge cases:
        - std == 0 (constant feature for a participant) → z = 0.0, no exception.
        - NaN values are preserved as-is (imputed upstream).

    Parameters
    ----------
    X            : Feature DataFrame (n_rows × n_features).
    meta         : DataFrame with ``participant_id`` column, same index as X.
    feature_list : Features to z-score; columns absent in X are silently skipped.

    Returns
    -------
    Copy of X with z-scored values for the specified features.
    """
    cols = [c for c in feature_list if c in X.columns]
    if not cols:
        logger.warning("apply_per_athlete_zscore: no matching columns found in X")
        return X.copy()

    X_out = X.copy()
    X_out["_pid"] = meta["participant_id"].values

    def _zscore_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in cols:
            mu = g[col].mean()
            sigma = g[col].std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                g[col] = 0.0
            else:
                g[col] = (g[col] - mu) / sigma
        return g

    X_out = X_out.groupby("_pid", group_keys=False).apply(_zscore_group)
    X_out = X_out.drop(columns=["_pid"]).reset_index(drop=True)

    logger.info(
        "Per-athlete z-score applied to %d/%d features (participants=%d)",
        len(cols), len(X.columns),
        meta["participant_id"].nunique(),
    )
    return X_out


# ────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────

@dataclass
class InjuryDatasetBundle:
    """Train / val / test splits ready for modelling."""

    X_train: pd.DataFrame
    y_train: pd.Series
    meta_train: pd.DataFrame

    X_val: pd.DataFrame
    y_val: pd.Series
    meta_val: pd.DataFrame

    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame

    train_pids: List[str]
    val_pids: List[str]
    test_pids: List[str]

    feature_columns: List[str]
    normalizer: Optional[NormalizerResult] = None

    # Raw (pre-normalization) copies for LOSO per-fold normalisation
    X_train_raw: Optional[pd.DataFrame] = None
    X_val_raw: Optional[pd.DataFrame] = None
    X_test_raw: Optional[pd.DataFrame] = None


# ────────────────────────────────────────────────────────────
# 1. Load & merge DFI predictions
# ────────────────────────────────────────────────────────────

def load_and_merge(cfg: InjuryConfig) -> pd.DataFrame:
    """
    Load the ETL feature CSV and left-join R4 DFI predictions.

    Rows without DFI (cold-start) are filled with the per-participant
    median DFI value.
    """
    df = pd.read_csv(cfg.input_csv, parse_dates=["date"])
    df.sort_values(["participant_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d rows × %d cols from %s",
                len(df), len(df.columns), cfg.input_csv)

    # Impute ACWR per participant using forward-fill then backward-fill.
    # This reduces NaN rate from ~22% (missing history at period start) to ~4%.
    if "acwr" in df.columns:
        n_acwr_nan_before = int(df["acwr"].isna().sum())
        df["acwr"] = (
            df.groupby("participant_id")["acwr"]
            .transform(lambda s: s.ffill().bfill())
        )
        n_acwr_nan_after = int(df["acwr"].isna().sum())
        logger.info(
            "ACWR imputation: NaN reduced from %d to %d (%.1f%% → %.1f%%)",
            n_acwr_nan_before, n_acwr_nan_after,
            100 * n_acwr_nan_before / len(df),
            100 * n_acwr_nan_after / len(df),
        )

    # Merge DFI predictions from R4
    dfi = pd.read_csv(cfg.dfi_csv, parse_dates=["date"])
    dfi_col = cfg.dfi_col
    dfi = dfi[["participant_id", "date", dfi_col]].copy()
    logger.info("Loaded %d DFI predictions from %s", len(dfi), cfg.dfi_csv)

    df = df.merge(dfi, on=["participant_id", "date"], how="left")

    # Fill cold-start NaNs with per-participant median
    n_missing = df[dfi_col].isna().sum()
    if n_missing > 0:
        medians = df.groupby("participant_id")[dfi_col].transform("median")
        df[dfi_col] = df[dfi_col].fillna(medians)
        # If an entire participant is missing, use global median
        global_median = df[dfi_col].median()
        df[dfi_col] = df[dfi_col].fillna(global_median)
        logger.info("Filled %d cold-start DFI values with participant/global median",
                     n_missing)

    logger.info("Merged dataset: %d rows × %d cols", len(df), len(df.columns))
    return df


# ────────────────────────────────────────────────────────────
# 2. Feature selection
# ────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    cfg: InjuryConfig,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract feature matrix X, target y, and metadata from the merged df.

    Parameters
    ----------
    target_col : Override ``cfg.target_col`` for this call.  Used internally
                 when a prospective target column has been created.

    Returns
    -------
    X : DataFrame  (n_rows × n_features)
    y : Series     (n_rows,) — binary 0/1
    meta : DataFrame with [participant_id, date]
    """
    # Only keep features that actually exist in the DataFrame
    available = [c for c in cfg.feature_columns if c in df.columns]
    missing = set(cfg.feature_columns) - set(available)
    if missing:
        logger.warning("Features not found in data (skipped): %s", missing)

    actual_target = target_col if target_col is not None else cfg.target_col
    X = df[available].copy()
    y = df[actual_target].astype(int)
    meta = df[["participant_id", "date"]].copy()

    logger.info("Features: %d cols, target '%s' (%.1f%% positive)",
                len(available), actual_target, 100 * y.mean())
    return X, y, meta


# ────────────────────────────────────────────────────────────
# 3. Participant-level split (same logic as R4)
# ────────────────────────────────────────────────────────────

def split_participants(
    df: pd.DataFrame, cfg: InjuryConfig,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_pids, val_pids, test_pids) with deterministic shuffle.

    Uses explicit participant lists from config when non-empty;
    otherwise falls back to ratio-based split.
    """
    pids = sorted(df["participant_id"].unique())

    # Use explicit lists if configured
    if cfg.train_participants:
        train_pids = [p for p in cfg.train_participants if p in pids]
        val_pids = [p for p in cfg.val_participants if p in pids]
        test_pids = [p for p in cfg.test_participants if p in pids]
        logger.info("Participant split — train: %s, val: %s, test: %s",
                    train_pids, val_pids, test_pids)
        return train_pids, val_pids, test_pids

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(pids)

    n = len(pids)
    n_train = max(1, int(n * cfg.train_split))
    n_val = max(1, int(n * cfg.val_split))

    train_pids = list(pids[:n_train])
    val_pids = list(pids[n_train:n_train + n_val])
    test_pids = list(pids[n_train + n_val:])

    logger.info("Participant split — train: %s, val: %s, test: %s",
                train_pids, val_pids, test_pids)
    return train_pids, val_pids, test_pids


# ────────────────────────────────────────────────────────────
# 4. Build complete dataset bundle
# ────────────────────────────────────────────────────────────

def build_injury_datasets(cfg: InjuryConfig) -> InjuryDatasetBundle:
    """
    End-to-end data preparation: load → merge DFI → feature select → split.

    Returns
    -------
    InjuryDatasetBundle
    """
    df = load_and_merge(cfg)

    # Create prospective target when requested
    effective_target_col = cfg.target_col
    if cfg.use_prospective_target:
        df = create_prospective_target(df, cfg.prospective_window)
        effective_target_col = f"injury_next{cfg.prospective_window}d"

    X, y, meta = prepare_features(df, cfg, target_col=effective_target_col)
    train_pids, val_pids, test_pids = split_participants(df, cfg)

    # Available feature columns (may differ from config if some were missing)
    feat_cols = list(X.columns)

    def _subset(pids: List[str]):
        mask = meta["participant_id"].isin(pids)
        return X.loc[mask].reset_index(drop=True), \
               y.loc[mask].reset_index(drop=True), \
               meta.loc[mask].reset_index(drop=True)

    X_train, y_train, meta_train = _subset(train_pids)
    X_val, y_val, meta_val = _subset(val_pids)
    X_test, y_test, meta_test = _subset(test_pids)

    logger.info("Dataset splits — train: %d, val: %d, test: %d",
                len(X_train), len(X_val), len(X_test))
    # ── Per-athlete z-score (optional, for cross-domain evaluation) ────
    # Applied BEFORE Yeo-Johnson so the power transform works on
    # z-scored distributions (RF does not require it, but it ensures
    # the same scale between PMData training and SoccerMon test data).
    if cfg.use_per_athlete_zscore:
        meta_full = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
        X_all_z = apply_per_athlete_zscore(
            pd.concat([X_train, X_val, X_test], ignore_index=True),
            meta_full,
            cfg.zscore_features,
        )
        n_tr, n_v, n_te = len(X_train), len(X_val), len(X_test)
        X_train = X_all_z.iloc[:n_tr].reset_index(drop=True)
        X_val   = X_all_z.iloc[n_tr:n_tr + n_v].reset_index(drop=True)
        X_test  = X_all_z.iloc[n_tr + n_v:].reset_index(drop=True)
        logger.info("Per-athlete z-score applied to PMData splits")
    # ── Normalisation (KS + Yeo-Johnson + z-score) ────────
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    normalizer = fit_normalizer(X_train)
    X_train = apply_normalizer(X_train, normalizer)
    X_val = apply_normalizer(X_val, normalizer)
    if len(X_test) > 0:
        X_test = apply_normalizer(X_test, normalizer)
    logger.info("Normalisation applied (fit on train only)")

    return InjuryDatasetBundle(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val, y_val=y_val, meta_val=meta_val,
        X_test=X_test, y_test=y_test, meta_test=meta_test,
        train_pids=train_pids, val_pids=val_pids, test_pids=test_pids,
        feature_columns=feat_cols,
        normalizer=normalizer,
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
    )


# ────────────────────────────────────────────────────────────
# 5. Combined PMData + SoccerMon dataset (Fase 4)
# ────────────────────────────────────────────────────────────

def build_combined_dataset(cfg: "InjuryConfig") -> "InjuryDatasetBundle":
    """
    Build a combined dataset merging PMData and SoccerMon on the 11 shared
    features, using ``injury_next{window}d`` as the target in both datasets.

    Design decisions
    ----------------
    - Only the 11 ``SOCCERMON_SHARED_FEATURES`` are used (present in both
      datasets), plus a binary ``source_dataset`` feature (0=PMData, 1=SoccerMon)
      so the RF can learn domain-specific patterns.
    - Both datasets are already per-athlete z-scored before this function runs:
      PMData is z-scored inline (cfg.use_per_athlete_zscore=True path in
      build_injury_datasets), SoccerMon is loaded from
      cfg.soccermon_prospective_csv which was saved by run_soccermon.py.
    - SoccerMon participant IDs are prefixed with "sm_" to avoid collision.
    - Split: PMData TRAIN_PARTICIPANTS → train, PMData VAL_PARTICIPANTS → val,
      all SoccerMon → test (external).  For LOAO all 66 athletes are pooled.
    - Yeo-Johnson normalisation is applied, fit on the combined train split only.

    Parameters
    ----------
    cfg : InjuryConfig — must have ``soccermon_prospective_csv`` pointing to
          the SoccerMon prospective CSV produced by run_soccermon.py.

    Returns
    -------
    InjuryDatasetBundle  with feature_columns = 11 shared features + source_dataset.
    """
    from .config import SOCCERMON_SHARED_FEATURES

    target_col = f"injury_next{cfg.prospective_window}d"
    shared_features = list(SOCCERMON_SHARED_FEATURES)
    all_features = shared_features + ["source_dataset"]

    # ── PMData ────────────────────────────────────────────
    df_pm = load_and_merge(cfg)
    df_pm = create_prospective_target(df_pm, cfg.prospective_window)
    # Per-athlete z-score on PMData shared features
    X_pm_full = df_pm[shared_features].copy()
    meta_pm_full = df_pm[["participant_id", "date"]].copy()
    X_pm_full = apply_per_athlete_zscore(X_pm_full, meta_pm_full, shared_features)
    X_pm_full["source_dataset"] = 0
    y_pm_full = df_pm[target_col].astype(int)

    logger.info(
        "PMData combined: %d rows, %d positives (%.1f%%), %d participants",
        len(X_pm_full), int(y_pm_full.sum()), 100.0 * y_pm_full.mean(),
        meta_pm_full["participant_id"].nunique(),
    )

    # ── SoccerMon ─────────────────────────────────────────
    sm_path = cfg.soccermon_prospective_csv
    if not pd.io.common.file_exists(sm_path):
        raise FileNotFoundError(
            f"SoccerMon prospective CSV not found: {sm_path}\n"
            "Run 'python run_soccermon.py' first to generate it."
        )
    df_sm = pd.read_csv(sm_path, parse_dates=["date"])
    # Prefix participant IDs to avoid collision with PMData
    df_sm["participant_id"] = "sm_" + df_sm["participant_id"].astype(str)
    meta_sm_full = df_sm[["participant_id", "date"]].copy()

    # Keep only shared features; fill any gaps with 0
    for col in shared_features:
        if col not in df_sm.columns:
            df_sm[col] = 0.0
    X_sm_full = df_sm[shared_features].copy()
    X_sm_full["source_dataset"] = 1

    if target_col not in df_sm.columns:
        raise KeyError(
            f"Column '{target_col}' not found in SoccerMon prospective CSV. "
            "Regenerate with run_soccermon.py after the prospective window update."
        )
    y_sm_full = df_sm[target_col].astype(int)

    logger.info(
        "SoccerMon combined: %d rows, %d positives (%.1f%%), %d participants",
        len(X_sm_full), int(y_sm_full.sum()), 100.0 * y_sm_full.mean(),
        meta_sm_full["participant_id"].nunique(),
    )

    # ── Concatenate ───────────────────────────────────────
    X_all = pd.concat([X_pm_full, X_sm_full], ignore_index=True)
    y_all = pd.concat([y_pm_full, y_sm_full], ignore_index=True)
    meta_all = pd.concat([meta_pm_full, meta_sm_full], ignore_index=True)

    logger.info(
        "Combined dataset: %d rows total, %d positives (%.1f%%), %d athletes",
        len(X_all), int(y_all.sum()), 100.0 * y_all.mean(),
        meta_all["participant_id"].nunique(),
    )

    # ── Split: PMData train/val, SoccerMon as external test ──
    pm_pids = df_pm["participant_id"].unique().tolist()
    sm_pids = [f"sm_{p}" for p in pd.read_csv(sm_path)["participant_id"].unique()]

    train_pids = [p for p in cfg.train_participants if p in pm_pids]
    val_pids = [p for p in cfg.val_participants if p in pm_pids]
    test_pids = sm_pids  # all SoccerMon = external test

    def _subset(pids: List[str]):
        mask = meta_all["participant_id"].isin(pids)
        return (X_all.loc[mask].reset_index(drop=True),
                y_all.loc[mask].reset_index(drop=True),
                meta_all.loc[mask].reset_index(drop=True))

    X_train, y_train, meta_train = _subset(train_pids)
    X_val, y_val, meta_val = _subset(val_pids)
    X_test, y_test, meta_test = _subset(test_pids)

    logger.info(
        "Combined split — train(PMData): %d, val(PMData): %d, test(SoccerMon): %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Normalise (fit on train split only) ───────────────
    # Exclude 'source_dataset' from Yeo-Johnson: it's binary (0/1) and
    # would be treated as a constant by fit_normalizer (train is all PMData=0),
    # which would zero out the SoccerMon test rows.
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    feat_to_norm = [c for c in all_features if c != "source_dataset"]
    normalizer = fit_normalizer(X_train[feat_to_norm])

    def _norm_and_restore(X_in: pd.DataFrame) -> pd.DataFrame:
        src = X_in["source_dataset"].values
        X_out = apply_normalizer(X_in[feat_to_norm], normalizer)
        X_out["source_dataset"] = src
        return X_out[all_features]  # preserve column order

    X_train = _norm_and_restore(X_train)
    X_val = _norm_and_restore(X_val)
    if len(X_test) > 0:
        X_test = _norm_and_restore(X_test)

    return InjuryDatasetBundle(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val, y_val=y_val, meta_val=meta_val,
        X_test=X_test, y_test=y_test, meta_test=meta_test,
        train_pids=train_pids, val_pids=val_pids, test_pids=test_pids,
        feature_columns=all_features,
        normalizer=normalizer,
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
    )
