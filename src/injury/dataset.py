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
prepare_features(df, cfg) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
split_participants(df, cfg) -> Tuple[List[str], List[str], List[str]]
build_injury_datasets(cfg) -> InjuryDatasetBundle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import InjuryConfig

logger = logging.getLogger(__name__)


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

    # Merge DFI predictions from R4
    dfi = pd.read_csv(cfg.dfi_csv, parse_dates=["date"])
    dfi = dfi[["participant_id", "date", "dfi_predicted"]].copy()
    logger.info("Loaded %d DFI predictions from %s", len(dfi), cfg.dfi_csv)

    df = df.merge(dfi, on=["participant_id", "date"], how="left")

    # Fill cold-start NaNs with per-participant median
    n_missing = df["dfi_predicted"].isna().sum()
    if n_missing > 0:
        medians = df.groupby("participant_id")["dfi_predicted"].transform("median")
        df["dfi_predicted"] = df["dfi_predicted"].fillna(medians)
        # If an entire participant is missing, use global median
        global_median = df["dfi_predicted"].median()
        df["dfi_predicted"] = df["dfi_predicted"].fillna(global_median)
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
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract feature matrix X, target y, and metadata from the merged df.

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

    X = df[available].copy()
    y = df[cfg.target_col].astype(int)
    meta = df[["participant_id", "date"]].copy()

    logger.info("Features: %d cols, target '%s' (%.1f%% positive)",
                len(available), cfg.target_col, 100 * y.mean())
    return X, y, meta


# ────────────────────────────────────────────────────────────
# 3. Participant-level split (same logic as R4)
# ────────────────────────────────────────────────────────────

def split_participants(
    df: pd.DataFrame, cfg: InjuryConfig,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_pids, val_pids, test_pids) with deterministic shuffle."""
    pids = sorted(df["participant_id"].unique())
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
    X, y, meta = prepare_features(df, cfg)
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

    return InjuryDatasetBundle(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val, y_val=y_val, meta_val=meta_val,
        X_test=X_test, y_test=y_test, meta_test=meta_test,
        train_pids=train_pids, val_pids=val_pids, test_pids=test_pids,
        feature_columns=feat_cols,
    )
