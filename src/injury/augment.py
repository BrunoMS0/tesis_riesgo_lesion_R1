"""
augment.py – Data augmentation for R5 Injury Risk Prediction.

Provides two strategies to address class imbalance:
1. **SMOTE** – Synthetic Minority Over-sampling Technique (recommended).
2. **Gaussian Copula** – Full synthetic athlete profiles via SDV.

The ``augment_training_data`` dispatcher selects the strategy based on
``cfg.augmentation_method`` and returns only (X, y) to simplify the
downstream pipeline.

Public API
----------
augment_training_data(X_train, y_train, meta_train, cfg) -> Tuple[X, y]
apply_smote(X_train, y_train, cfg) -> Tuple[X, y]
generate_synthetic_athletes(X_train, y_train, meta_train, cfg) -> Tuple[X, y, meta]
validate_synthetic(real_df, synth_df) -> Dict[str, float]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import InjuryConfig

logger = logging.getLogger(__name__)


def augment_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    cfg: InjuryConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Dispatch to the configured augmentation method.

    Parameters
    ----------
    X_train : feature matrix
    y_train : binary target
    meta_train : metadata (participant_id, date)
    cfg : InjuryConfig with ``augmentation_method`` ('smote' | 'copula')

    Returns
    -------
    X_aug, y_aug : augmented training data
    """
    method = cfg.augmentation_method.lower()

    if method == "smote":
        return apply_smote(X_train, y_train, cfg)
    elif method == "copula":
        X_aug, y_aug, _ = generate_synthetic_athletes(
            X_train, y_train, meta_train, cfg,
        )
        return X_aug, y_aug
    else:
        raise ValueError(
            f"Unknown augmentation method: {method!r}. Use 'smote' or 'copula'."
        )


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: InjuryConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the training set.

    Parameters
    ----------
    X_train : DataFrame of shape (n_train, n_features)
    y_train : Series of shape (n_train,) — binary 0/1
    cfg : InjuryConfig with target_ratio and smote_k_neighbors

    Returns
    -------
    X_resampled : DataFrame (balanced features)
    y_resampled : Series (balanced target)
    """
    from imblearn.over_sampling import SMOTE

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    if n_pos == 0:
        logger.warning("No positive samples — SMOTE cannot be applied. "
                       "Returning original data.")
        return X_train.copy(), y_train.copy()

    # Ensure k_neighbors doesn't exceed minority class size
    k = min(cfg.smote_k_neighbors, n_pos - 1) if n_pos > 1 else 1

    # sampling_strategy = desired ratio of minority to majority
    sampling_strategy = cfg.target_ratio / (1 - cfg.target_ratio)
    # Clamp: can't undersample majority via SMOTE
    sampling_strategy = min(sampling_strategy, 1.0)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
        random_state=cfg.seed,
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series with original column names
    X_resampled = pd.DataFrame(X_res, columns=X_train.columns)
    y_resampled = pd.Series(y_res, name=y_train.name)

    n_synthetic = len(X_resampled) - len(X_train)
    logger.info("SMOTE: generated %d synthetic samples (k=%d, ratio=%.2f). "
                "Total: %d (injury rate %.1f%%)",
                n_synthetic, k, sampling_strategy,
                len(X_resampled), 100 * y_resampled.mean())
    return X_resampled, y_resampled


def generate_synthetic_athletes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    cfg: InjuryConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generate synthetic athletes using Gaussian Copula and combine with real data.

    Parameters
    ----------
    X_train : DataFrame of shape (n_train, n_features)
    y_train : Series of shape (n_train,) — binary 0/1
    meta_train : DataFrame with [participant_id, date]
    cfg : InjuryConfig

    Returns
    -------
    X_aug : DataFrame (real + synthetic features)
    y_aug : Series (real + synthetic targets)
    meta_aug : DataFrame (real + synthetic metadata)
    """
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    # Build a single training table for the copula
    train_df = X_train.copy()
    train_df[cfg.target_col] = y_train.values
    train_df["participant_id"] = meta_train["participant_id"].values

    n_real_participants = train_df["participant_id"].nunique()
    rows_per_participant = len(train_df) // max(n_real_participants, 1)

    # Infer SDV metadata from the training table
    metadata = Metadata.detect_from_dataframes({"training": train_df})
    table_meta = metadata.get_table_metadata("training")
    # Override sdtypes: all numeric features + categorical participant_id
    table_meta.update_column("participant_id", sdtype="categorical")
    for col in train_df.columns:
        if col == "participant_id":
            continue
        table_meta.update_column(col, sdtype="numerical")

    synthesizer = GaussianCopulaSynthesizer(
        metadata=table_meta,
        enforce_min_max_values=True,
        enforce_rounding=True,
    )
    synthesizer.fit(train_df)

    n_synth_rows = cfg.n_synthetic_athletes * rows_per_participant
    synth_df = synthesizer.sample(num_rows=n_synth_rows)
    logger.info("Generated %d synthetic rows (%d virtual athletes × ~%d days)",
                len(synth_df), cfg.n_synthetic_athletes, rows_per_participant)

    # Assign synthetic participant IDs
    synth_ids = []
    for i in range(cfg.n_synthetic_athletes):
        pid = f"synth_{i+1:03d}"
        synth_ids.extend([pid] * rows_per_participant)
    # Handle rounding differences
    synth_df = synth_df.iloc[:len(synth_ids)].copy()
    synth_df["participant_id"] = synth_ids[:len(synth_df)]

    # Clip binary target to {0, 1}
    synth_df[cfg.target_col] = synth_df[cfg.target_col].round().clip(0, 1).astype(int)

    # Separate features, target, metadata
    feature_cols = [c for c in X_train.columns]
    X_synth = synth_df[feature_cols].copy()
    y_synth = synth_df[cfg.target_col].copy()
    meta_synth = pd.DataFrame({
        "participant_id": synth_df["participant_id"].values,
        "date": pd.NaT,  # synthetic rows have no real date
    })

    # Combine real + synthetic
    X_aug = pd.concat([X_train, X_synth], ignore_index=True)
    y_aug = pd.concat([y_train, y_synth], ignore_index=True)
    meta_aug = pd.concat([meta_train, meta_synth], ignore_index=True)

    logger.info("Augmented training set: %d rows (real=%d, synthetic=%d), "
                "injury rate=%.1f%%",
                len(X_aug), len(X_train), len(X_synth), 100 * y_aug.mean())
    return X_aug, y_aug, meta_aug


def validate_synthetic(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, float]:
    """
    Compare real vs synthetic distributions using KS test.

    Returns
    -------
    Dict mapping feature name → KS test p-value.
    A p-value > 0.05 means distributions are not significantly different.
    """
    from scipy.stats import ks_2samp

    results: Dict[str, float] = {}
    for col in feature_columns:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        _, p_value = ks_2samp(real_vals, synth_vals)
        results[col] = round(p_value, 4)

    n_pass = sum(1 for p in results.values() if p > 0.05)
    logger.info("KS validation: %d/%d features pass (p > 0.05)",
                n_pass, len(results))
    return results
