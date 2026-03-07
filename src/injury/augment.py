"""
augment.py – Synthetic athlete generation via Gaussian Copula (SDV).

Generates full synthetic athlete profiles from real training data to
address the extreme class imbalance (~3% injury rate).  Only applied
to the training set — validation and test data remain 100% real.

Public API
----------
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
