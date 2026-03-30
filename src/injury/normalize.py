"""
normalize.py – Formal normality testing and transformation for R5.

Implements Kolmogorov-Smirnov normality diagnostics and Yeo-Johnson
power transformation + z-score standardisation so that features fed
to Logistic Regression satisfy the normality assumption.

The workflow is:
  1. **Pre-test**: KS test on each raw feature → ``pre_report``.
  2. **Transform**: Fit ``PowerTransformer(yeo-johnson)`` on training data.
  3. **Post-test**: KS test on each transformed feature → ``post_report``.
  4. **Apply**: Re-use the fitted transformer on val / test / LOSO folds.

The transformer is fitted on **training data only** to avoid data leakage.

Public API
----------
check_normality(X, alpha) -> pd.DataFrame
fit_normalizer(X_train, alpha) -> NormalizerResult
apply_normalizer(X, normalizer) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kstest, skew, kurtosis
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger(__name__)


@dataclass
class NormalizerResult:
    """Artifacts produced by ``fit_normalizer``."""

    transformer: PowerTransformer
    pre_report: pd.DataFrame
    post_report: pd.DataFrame
    feature_cols: List[str]
    # Columns that were constant (zero-variance) and left untouched
    constant_cols: List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 1. Kolmogorov-Smirnov normality diagnostic
# ────────────────────────────────────────────────────────────

def check_normality(
    X: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run a one-sample Kolmogorov-Smirnov test against the normal
    distribution for every numeric column in *X*.

    Parameters
    ----------
    X : DataFrame  (n_samples × n_features)
    alpha : significance level (default 0.05)

    Returns
    -------
    pd.DataFrame with columns
        [feature, n, ks_statistic, p_value, is_normal, skewness, kurtosis]
    """
    rows = []
    for col in X.columns:
        series = X[col].dropna()
        n = len(series)
        if n < 2:
            rows.append({
                "feature": col, "n": n,
                "ks_statistic": np.nan, "p_value": np.nan,
                "is_normal": False, "skewness": np.nan, "kurtosis": np.nan,
            })
            continue

        mean, std = series.mean(), series.std()
        if std == 0:
            # Constant column — not meaningful to test
            rows.append({
                "feature": col, "n": n,
                "ks_statistic": np.nan, "p_value": np.nan,
                "is_normal": False, "skewness": 0.0, "kurtosis": 0.0,
            })
            continue

        stat, p = kstest(series, "norm", args=(mean, std))
        rows.append({
            "feature": col,
            "n": n,
            "ks_statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "is_normal": p >= alpha,
            "skewness": round(float(skew(series)), 4),
            "kurtosis": round(float(kurtosis(series)), 4),
        })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 2. Fit normalizer (KS pre → Yeo-Johnson → KS post)
# ────────────────────────────────────────────────────────────

def fit_normalizer(
    X_train: pd.DataFrame,
    alpha: float = 0.05,
) -> NormalizerResult:
    """
    Fit a normalization pipeline on training data:

    1. Run KS test on raw features (``pre_report``).
    2. Fit ``PowerTransformer(yeo-johnson, standardize=True)``.
    3. Transform training data.
    4. Run KS test on transformed features (``post_report``).

    Parameters
    ----------
    X_train : Training feature matrix (will NOT be mutated).
    alpha : Significance level for KS tests.

    Returns
    -------
    NormalizerResult
    """
    feature_cols = list(X_train.columns)

    # Identify constant columns (PowerTransformer doesn't handle them)
    constant_cols = [c for c in feature_cols if X_train[c].nunique() <= 1]
    transform_cols = [c for c in feature_cols if c not in constant_cols]

    # --- Pre-transformation normality ---
    pre_report = check_normality(X_train, alpha=alpha)
    n_normal_pre = int(pre_report["is_normal"].sum())
    logger.info("Pre-normalization KS test: %d/%d features are normal (α=%.2f)",
                n_normal_pre, len(feature_cols), alpha)

    # --- Fit PowerTransformer on non-constant columns ---
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    if transform_cols:
        pt.fit(X_train[transform_cols].fillna(0))

    # --- Transform training data for post-test ---
    X_transformed = _apply_transform(X_train, pt, transform_cols, constant_cols)

    # --- Post-transformation normality ---
    post_report = check_normality(X_transformed, alpha=alpha)
    n_normal_post = int(post_report["is_normal"].sum())
    logger.info("Post-normalization KS test: %d/%d features are normal (α=%.2f)",
                n_normal_post, len(feature_cols), alpha)

    if constant_cols:
        logger.info("Constant columns (left as 0): %s", constant_cols)

    return NormalizerResult(
        transformer=pt,
        pre_report=pre_report,
        post_report=post_report,
        feature_cols=feature_cols,
        constant_cols=constant_cols,
    )


# ────────────────────────────────────────────────────────────
# 3. Apply fitted normalizer
# ────────────────────────────────────────────────────────────

def apply_normalizer(
    X: pd.DataFrame,
    normalizer: NormalizerResult,
) -> pd.DataFrame:
    """
    Apply a previously fitted normalizer to new data (val, test, fold).

    Parameters
    ----------
    X : Feature matrix with the same columns as the training data.
    normalizer : Result from ``fit_normalizer``.

    Returns
    -------
    pd.DataFrame — transformed copy of X.
    """
    transform_cols = [c for c in normalizer.feature_cols
                      if c not in normalizer.constant_cols]
    return _apply_transform(
        X, normalizer.transformer, transform_cols, normalizer.constant_cols,
    )


# ────────────────────────────────────────────────────────────
# Internal helper
# ────────────────────────────────────────────────────────────

def _apply_transform(
    X: pd.DataFrame,
    pt: PowerTransformer,
    transform_cols: List[str],
    constant_cols: List[str],
) -> pd.DataFrame:
    """Apply PowerTransformer to ``transform_cols``, zero-fill ``constant_cols``."""
    out = X.copy()
    if transform_cols:
        out[transform_cols] = pt.transform(out[transform_cols].fillna(0))
    if constant_cols:
        out[constant_cols] = 0.0
    return out
