"""
dataset.py – Data preparation for the R4 Fatigue model.

Responsibilities
----------------
1. Load the un-normalised feature CSV produced by the ETL.
2. Compute the Dynamic Fatigue Index (DFI) target.
3. MinMax-scale objective features (fitted on training participants).
4. Create sliding-window sequences (14-day lookback → next-day DFI).
5. Split by participant (train / val / test).
6. Build ``tf.data.Dataset`` pipelines ready for Keras.

Public API
----------
load_dataframe(cfg) -> pd.DataFrame
compute_dfi(series) -> pd.Series
build_fatigue_datasets(cfg) -> FatigueDatasetBundle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import FatigueConfig

logger = logging.getLogger(__name__)

# Lazy TF import
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# ────────────────────────────────────────────────────────────
# Result containers
# ────────────────────────────────────────────────────────────

@dataclass
class FatigueDatasetBundle:
    """Train / val / test ``tf.data.Dataset`` objects + metadata."""

    train: object          # tf.data.Dataset
    val: object
    test: object
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    window_size: int
    scaler: MinMaxScaler
    # Per-sample metadata for evaluation
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame


# ────────────────────────────────────────────────────────────
# 1. Load & prepare
# ────────────────────────────────────────────────────────────

def load_dataframe(cfg: FatigueConfig) -> pd.DataFrame:
    """Read the un-normalised feature CSV and sort by participant + date."""
    df = pd.read_csv(cfg.input_csv, parse_dates=["date"])
    df.sort_values(["participant_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d rows × %d cols from %s",
                len(df), len(df.columns), cfg.input_csv)
    return df


def compute_dfi(fatigue_series: pd.Series) -> pd.Series:
    """
    Dynamic Fatigue Index: ``DFI = (5 − fatigue) / 4``.

    Maps the PMSYS fatigue scale (1=fresh … 5=very fatigued) to
    ``[0, 1]`` where **1 = maximum fatigue**.
    """
    return (5.0 - fatigue_series) / 4.0


# ────────────────────────────────────────────────────────────
# 2. Participant-level train/val/test split
# ────────────────────────────────────────────────────────────

def split_participants(
    df: pd.DataFrame, cfg: FatigueConfig,
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
# 3. Normalisation (MinMax fitted on train only)
# ────────────────────────────────────────────────────────────

def fit_scaler(
    df: pd.DataFrame,
    train_pids: List[str],
    feature_cols: List[str],
) -> MinMaxScaler:
    """Fit a MinMaxScaler on training-participant rows only."""
    mask = df["participant_id"].isin(train_pids)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[mask, feature_cols])
    logger.info("MinMaxScaler fitted on %d training rows, %d features",
                mask.sum(), len(feature_cols))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Apply a fitted MinMaxScaler (in-place on a copy)."""
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


# ────────────────────────────────────────────────────────────
# 4. Sliding-window sequence creation
# ────────────────────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build (X, y, meta) from a per-participant-sorted DataFrame.

    For each participant, a sliding window of *window_size* consecutive
    days forms one input sample;  the **DFI of the next day** is the target.

    Returns
    -------
    X : ndarray, shape (N, window_size, n_features)
    y : ndarray, shape (N,)
    meta : DataFrame with columns [participant_id, date] for the target day.
    """
    X_list: list = []
    y_list: list = []
    meta_rows: list = []

    for pid, grp in df.groupby("participant_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feat_vals = grp[feature_cols].values.astype(np.float32)
        dfi_vals = grp["dfi"].values.astype(np.float32)
        dates = grp["date"].values

        for i in range(window_size, len(grp)):
            X_list.append(feat_vals[i - window_size:i])
            y_list.append(dfi_vals[i])
            meta_rows.append({"participant_id": pid, "date": dates[i]})

    X = np.stack(X_list, axis=0)   # (N, window_size, n_features)
    y = np.array(y_list)            # (N,)
    meta = pd.DataFrame(meta_rows)

    logger.info("Created %d sequences  (window=%d, features=%d)",
                len(X), window_size, X.shape[2])
    return X, y, meta


# ────────────────────────────────────────────────────────────
# 5. tf.data pipeline builder
# ────────────────────────────────────────────────────────────

def _to_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    cfg: FatigueConfig,
    *,
    shuffle: bool = False,
) -> object:
    """Wrap numpy arrays into a batched, prefetched tf.data.Dataset."""
    tf = _get_tf()

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(cfg.shuffle_buffer, len(X)),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    ds = ds.prefetch(cfg.prefetch)
    return ds


# ────────────────────────────────────────────────────────────
# 6. Main entry-point
# ────────────────────────────────────────────────────────────

def build_fatigue_datasets(
    cfg: Optional[FatigueConfig] = None,
) -> FatigueDatasetBundle:
    """
    End-to-end data preparation: load → DFI → split → scale → window → tf.data.

    Parameters
    ----------
    cfg : FatigueConfig, optional
        Uses defaults when *None*.

    Returns
    -------
    FatigueDatasetBundle
    """
    if cfg is None:
        cfg = FatigueConfig()

    # --- Load --------------------------------------------------------
    df = load_dataframe(cfg)

    # Keep only columns we need (objective features + target + meta)
    available = [c for c in cfg.objective_features if c in df.columns]
    missing = set(cfg.objective_features) - set(available)
    if missing:
        logger.warning("Features missing from CSV (will be ignored): %s", missing)
    feature_cols = available

    # Fill residual NaN in features with 0 (safety net)
    df[feature_cols] = df[feature_cols].fillna(0)

    # --- DFI target --------------------------------------------------
    df["dfi"] = compute_dfi(df[cfg.target_raw_col])
    logger.info("DFI range: [%.3f, %.3f], mean=%.3f",
                df["dfi"].min(), df["dfi"].max(), df["dfi"].mean())

    # --- Participant split -------------------------------------------
    train_pids, val_pids, test_pids = split_participants(df, cfg)

    # --- Normalise features (fit on train only) ----------------------
    scaler = fit_scaler(df, train_pids, feature_cols)
    df = apply_scaler(df, scaler, feature_cols)

    # --- Windowed sequences ------------------------------------------
    X_all, y_all, meta_all = create_sequences(df, feature_cols, cfg.window_size)

    # Partition sequences by the participant they belong to
    train_mask = meta_all["participant_id"].isin(train_pids).values
    val_mask = meta_all["participant_id"].isin(val_pids).values
    test_mask = meta_all["participant_id"].isin(test_pids).values

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    meta_train = meta_all[train_mask].reset_index(drop=True)
    meta_val = meta_all[val_mask].reset_index(drop=True)
    meta_test = meta_all[test_mask].reset_index(drop=True)

    logger.info("Sequences — train: %d, val: %d, test: %d",
                len(X_train), len(X_val), len(X_test))

    # --- tf.data Datasets --------------------------------------------
    ds_train = _to_tf_dataset(X_train, y_train, cfg, shuffle=True)
    ds_val = _to_tf_dataset(X_val, y_val, cfg)
    ds_test = _to_tf_dataset(X_test, y_test, cfg)

    return FatigueDatasetBundle(
        train=ds_train,
        val=ds_val,
        test=ds_test,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=len(feature_cols),
        window_size=cfg.window_size,
        scaler=scaler,
        meta_train=meta_train,
        meta_val=meta_val,
        meta_test=meta_test,
    )
