"""
dataset.py — Runner Dataset stratified split and InjuryDatasetBundle construction.

Produces the same InjuryDatasetBundle format as src/injury/dataset.py,
enabling direct reuse of loso_cross_validation, train_injury_model,
evaluate_model, etc.

Split strategy: stratified 70/10/20 by athlete-level injury presence
(seed=42) so both injured and uninjured athletes appear in all splits.

Public API
----------
build_runner_datasets(csv_path, feature_cols, save_processed) -> InjuryDatasetBundle
make_runner_injury_config(feature_cols) -> InjuryConfig
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.injury.config import InjuryConfig
from src.injury.dataset import InjuryDatasetBundle
from src.injury.normalize import apply_normalizer, fit_normalizer

from .config import (
    AUGMENTATION_METHOD,
    PMDATA_COMMON_FEATURES,
    RF_CLASS_WEIGHT,
    RF_MAX_FEATURES,
    RF_N_ESTIMATORS,
    RUNNER_COMMON_FEATURES,
    RUNNER_CSV,
    RUNNER_FEATURE_COLUMNS,
    RUNNER_OUTPUT_CSV,
    SEED,
    SMOTE_K_NEIGHBORS,
    TARGET_COL,
    TARGET_RATIO,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from .extract import load_runner_csv
from .transform import compute_features

logger = logging.getLogger(__name__)


# ─── Stratified split ─────────────────────────────────────────────────────────

def _stratified_split(
    pids: List[str],
    injury_flags: List[int],
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified 70 / 10 / 20 split at the athlete level.

    Stratification is by injury_flag (0 = no injury, 1 = ≥1 injury)
    to ensure both groups are proportionally represented in all splits.
    """
    train_ids, temp_ids, _, temp_flags = train_test_split(
        pids, injury_flags,
        test_size=1.0 - TRAIN_SPLIT,
        random_state=seed,
        stratify=injury_flags,
    )
    # val = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT) of temp
    val_ratio = VAL_SPLIT / (1.0 - TRAIN_SPLIT)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - val_ratio,
        random_state=seed,
        stratify=temp_flags,
    )
    return list(train_ids), list(val_ids), list(test_ids)


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_runner_datasets(
    csv_path: str = RUNNER_CSV,
    feature_cols: Optional[List[str]] = None,
    save_processed: bool = True,
) -> InjuryDatasetBundle:
    """
    End-to-end Runner Dataset preparation pipeline:
      extract → feature engineering → stratified split → Yeo-Johnson normalise.

    Parameters
    ----------
    csv_path      : Path to day_approach_maskedID_timeseries.csv.
    feature_cols  : Override feature column list (default: RUNNER_FEATURE_COLUMNS).
    save_processed: If True, saves the processed CSV to RUNNER_OUTPUT_CSV.

    Returns
    -------
    InjuryDatasetBundle  compatible with src/injury/ pipeline components
    (loso_cross_validation, train_injury_model, evaluate_model, etc.).

    Notes
    -----
    - target_col = 'injury'  (already prospective — no create_prospective_target)
    - dfi_predicted is NOT present (runner dataset has no Fitbit sensor)
    - participant_id values are strings: 'runner_0' … 'runner_73'
    """
    if feature_cols is None:
        feature_cols = RUNNER_FEATURE_COLUMNS

    # ── 1. Extract ─────────────────────────────────────────────────────────────
    raw_df = load_runner_csv(csv_path)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    df = compute_features(raw_df)

    # ── 3. (Optional) Save processed CSV ──────────────────────────────────────
    if save_processed:
        Path(RUNNER_OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RUNNER_OUTPUT_CSV, index=False)
        logger.info("Saved processed dataset -> %s (%d rows)", RUNNER_OUTPUT_CSV, len(df))

    # ── 4. Stratified athlete split ────────────────────────────────────────────
    pids = sorted(df["participant_id"].unique())
    injury_per_athlete = df.groupby("participant_id")["injury"].max()
    flags = [int(injury_per_athlete[p]) for p in pids]

    train_pids, val_pids, test_pids = _stratified_split(pids, flags, seed=SEED)

    n_inj = lambda plist: int(sum(injury_per_athlete[p] for p in plist))
    logger.info(
        "Athlete split — train: %d (%d injured) | val: %d (%d) | test: %d (%d)",
        len(train_pids), n_inj(train_pids),
        len(val_pids),   n_inj(val_pids),
        len(test_pids),  n_inj(test_pids),
    )

    # ── 5. Build feature matrix ────────────────────────────────────────────────
    avail = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(avail)
    if missing:
        logger.warning("Feature columns not found in processed data (skipped): %s", missing)

    X    = df[avail]
    y    = df[TARGET_COL].astype(int)
    meta = df[["participant_id", "date"]].copy()

    def _subset(pids_list: List[str]):
        mask = meta["participant_id"].isin(pids_list)
        return (
            X.loc[mask].reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
            meta.loc[mask].reset_index(drop=True),
        )

    X_train, y_train, meta_train = _subset(train_pids)
    X_val,   y_val,   meta_val   = _subset(val_pids)
    X_test,  y_test,  meta_test  = _subset(test_pids)

    logger.info(
        "Row split — train: %d (%d+) | val: %d (%d+) | test: %d (%d+)",
        len(X_train), int(y_train.sum()),
        len(X_val),   int(y_val.sum()),
        len(X_test),  int(y_test.sum()),
    )

    # ── 6. Yeo-Johnson normalisation (fit on train only) ──────────────────────
    X_train_raw = X_train.copy()
    X_val_raw   = X_val.copy()
    X_test_raw  = X_test.copy()

    normalizer = fit_normalizer(X_train)
    X_train    = apply_normalizer(X_train, normalizer)
    X_val      = apply_normalizer(X_val,   normalizer)
    if len(X_test) > 0:
        X_test = apply_normalizer(X_test, normalizer)

    logger.info("Yeo-Johnson normalisation applied (fit on train split only)")

    return InjuryDatasetBundle(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val,     y_val=y_val,     meta_val=meta_val,
        X_test=X_test,   y_test=y_test,   meta_test=meta_test,
        train_pids=train_pids, val_pids=val_pids, test_pids=test_pids,
        feature_columns=avail,
        normalizer=normalizer,
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
    )


# ─── InjuryConfig factory for runner dataset ─────────────────────────────────

def make_runner_injury_config(
    feature_cols: Optional[List[str]] = None,
) -> InjuryConfig:
    """
    Return an InjuryConfig pre-configured for the Runner dataset.

    This config is passed to loso_cross_validation, build_model, and
    augment_training_data — all from src.injury — to reuse them directly.

    Key overrides vs. default InjuryConfig:
      - feature_columns : RUNNER_FEATURE_COLUMNS (no dfi_predicted)
      - target_col      : 'injury'
      - model_type      : 'rf'
      - augmentation    : smote with lower target_ratio (1.36% base prevalence)
      - train/val/test  : empty lists → ratio-based split is bypassed externally
    """
    return InjuryConfig(
        feature_columns=list(feature_cols or RUNNER_FEATURE_COLUMNS),
        target_col=TARGET_COL,
        use_prospective_target=False,       # injury is already prospective
        train_participants=[],              # splits handled by build_runner_datasets
        val_participants=[],
        test_participants=[],
        model_type="rf",
        rf_n_estimators=RF_N_ESTIMATORS,
        rf_max_features=RF_MAX_FEATURES,
        rf_class_weight=RF_CLASS_WEIGHT,
        augmentation_method=AUGMENTATION_METHOD,
        target_ratio=TARGET_RATIO,
        smote_k_neighbors=SMOTE_K_NEIGHBORS,
        seed=SEED,
    )
