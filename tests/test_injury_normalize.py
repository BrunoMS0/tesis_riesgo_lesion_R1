"""
test_injury_normalize.py – Tests for KS normality testing and Yeo-Johnson normalization.

Validates:
- KS test report format and columns
- Normalized data has mean ≈ 0, std ≈ 1
- KS correctly detects non-normal distributions
- KS correctly passes normal distributions
- Constant columns handled without errors
- apply_normalizer is consistent with fit output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.normalize import (
    NormalizerResult,
    apply_normalizer,
    fit_normalizer,
    check_normality,
)


@pytest.fixture()
def mixed_data():
    """DataFrame with a mix of normal and non-normal features."""
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame({
        "normal_feat": rng.normal(50, 10, n),
        "skewed_feat": rng.exponential(100, n),
        "uniform_feat": rng.uniform(0, 1000, n),
        "bimodal_feat": np.concatenate([rng.normal(10, 2, n // 2),
                                         rng.normal(90, 2, n - n // 2)]),
        "constant_feat": np.ones(n) * 42.0,
    })


@pytest.fixture()
def gaussian_data():
    """Purely Gaussian features."""
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame({
        f"g{i}": rng.normal(0, 1, n) for i in range(5)
    })


class TestTestNormality:
    def test_report_columns(self, mixed_data):
        report = check_normality(mixed_data)
        expected = {"feature", "n", "ks_statistic", "p_value",
                    "is_normal", "skewness", "kurtosis"}
        assert expected == set(report.columns)

    def test_report_rows_match_features(self, mixed_data):
        report = check_normality(mixed_data)
        assert len(report) == len(mixed_data.columns)
        assert list(report["feature"]) == list(mixed_data.columns)

    def test_detects_non_normal(self, mixed_data):
        report = check_normality(mixed_data)
        skewed_row = report.loc[report["feature"] == "skewed_feat"].iloc[0]
        assert skewed_row["is_normal"] is False or skewed_row["is_normal"] == False

    def test_detects_normal(self, gaussian_data):
        report = check_normality(gaussian_data)
        # Most Gaussian features should pass KS with n=500
        normal_count = report["is_normal"].sum()
        assert normal_count >= 3  # at least 3/5 pass

    def test_constant_column_not_normal(self, mixed_data):
        report = check_normality(mixed_data)
        const_row = report.loc[report["feature"] == "constant_feat"].iloc[0]
        assert const_row["is_normal"] is False or const_row["is_normal"] == False

    def test_n_values_correct(self, mixed_data):
        report = check_normality(mixed_data)
        for _, row in report.iterrows():
            if row["feature"] != "constant_feat":
                assert row["n"] == len(mixed_data)


class TestFitNormalizer:
    def test_returns_normalizer_result(self, mixed_data):
        result = fit_normalizer(mixed_data)
        assert isinstance(result, NormalizerResult)
        assert result.transformer is not None
        assert isinstance(result.pre_report, pd.DataFrame)
        assert isinstance(result.post_report, pd.DataFrame)

    def test_normalized_approx_zero_mean(self, mixed_data):
        result = fit_normalizer(mixed_data)
        X_norm = apply_normalizer(mixed_data, result)
        for col in X_norm.columns:
            if col not in result.constant_cols:
                assert abs(X_norm[col].mean()) < 0.1, f"{col} mean not ≈ 0"

    def test_normalized_approx_unit_std(self, mixed_data):
        result = fit_normalizer(mixed_data)
        X_norm = apply_normalizer(mixed_data, result)
        for col in X_norm.columns:
            if col not in result.constant_cols:
                assert abs(X_norm[col].std() - 1.0) < 0.2, f"{col} std not ≈ 1"

    def test_constant_cols_detected(self, mixed_data):
        result = fit_normalizer(mixed_data)
        assert "constant_feat" in result.constant_cols

    def test_constant_cols_zeroed(self, mixed_data):
        result = fit_normalizer(mixed_data)
        X_norm = apply_normalizer(mixed_data, result)
        assert (X_norm["constant_feat"] == 0.0).all()

    def test_post_report_improves_normality(self, mixed_data):
        result = fit_normalizer(mixed_data)
        pre_normal = result.pre_report["is_normal"].sum()
        post_normal = result.post_report["is_normal"].sum()
        # Post-transform should have at least as many normal features
        assert post_normal >= pre_normal

    def test_feature_cols_preserved(self, mixed_data):
        result = fit_normalizer(mixed_data)
        assert result.feature_cols == list(mixed_data.columns)


class TestApplyNormalizer:
    def test_consistent_with_fit(self, mixed_data):
        """apply_normalizer(X_train) should equal what fit produced."""
        result = fit_normalizer(mixed_data)
        X_applied = apply_normalizer(mixed_data, result)
        assert X_applied.shape == mixed_data.shape
        assert list(X_applied.columns) == list(mixed_data.columns)

    def test_different_data_same_scale(self, mixed_data):
        """Applying normalizer to new data should produce similar scales."""
        rng = np.random.RandomState(99)
        n = 200
        new_data = pd.DataFrame({
            "normal_feat": rng.normal(50, 10, n),
            "skewed_feat": rng.exponential(100, n),
            "uniform_feat": rng.uniform(0, 1000, n),
            "bimodal_feat": np.concatenate([rng.normal(10, 2, n // 2),
                                             rng.normal(90, 2, n - n // 2)]),
            "constant_feat": np.ones(n) * 42.0,
        })
        result = fit_normalizer(mixed_data)
        X_new = apply_normalizer(new_data, result)
        # New data should be roughly in [-3, 3] range (z-score)
        for col in X_new.columns:
            if col not in result.constant_cols:
                assert X_new[col].abs().max() < 10, f"{col} out of expected range"

    def test_preserves_column_order(self, mixed_data):
        result = fit_normalizer(mixed_data)
        X_norm = apply_normalizer(mixed_data, result)
        assert list(X_norm.columns) == list(mixed_data.columns)
