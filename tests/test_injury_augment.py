"""
test_injury_augment.py – Tests for R5 synthetic data generation.

Validates:
- Augmented data has more rows than original
- Synthetic participant IDs are distinct from real ones
- Target column remains binary {0, 1}
- KS validation returns p-values for each feature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.augment import generate_synthetic_athletes, validate_synthetic


@pytest.fixture()
def cfg():
    return InjuryConfig(n_synthetic_athletes=2)


@pytest.fixture()
def training_data():
    """Small synthetic training data mimicking R5 input."""
    rng = np.random.RandomState(42)
    n = 100
    pids = ["p01"] * 50 + ["p02"] * 50
    X = pd.DataFrame({
        "dfi_predicted": rng.uniform(0, 1, n),
        "session_load": rng.uniform(100, 800, n),
        "acwr": rng.uniform(0.5, 2.0, n),
        "fatigue": rng.randint(1, 6, n).astype(float),
        "steps": rng.uniform(2000, 15000, n),
    })
    y = pd.Series((rng.rand(n) > 0.95).astype(int))
    meta = pd.DataFrame({
        "participant_id": pids,
        "date": pd.date_range("2020-01-01", periods=n),
    })
    return X, y, meta


class TestGenerateSyntheticAthletes:
    def test_augmented_has_more_rows(self, training_data, cfg):
        X, y, meta = training_data
        X_aug, y_aug, meta_aug = generate_synthetic_athletes(X, y, meta, cfg)
        assert len(X_aug) > len(X)

    def test_real_data_preserved(self, training_data, cfg):
        X, y, meta = training_data
        X_aug, y_aug, _ = generate_synthetic_athletes(X, y, meta, cfg)
        # First n rows should be identical to original
        pd.testing.assert_frame_equal(
            X_aug.iloc[:len(X)].reset_index(drop=True),
            X.reset_index(drop=True),
        )

    def test_synthetic_pids_distinct(self, training_data, cfg):
        X, y, meta = training_data
        _, _, meta_aug = generate_synthetic_athletes(X, y, meta, cfg)
        real_pids = set(meta["participant_id"].unique())
        synth_pids = set(meta_aug["participant_id"].unique()) - real_pids
        assert len(synth_pids) > 0
        for pid in synth_pids:
            assert pid.startswith("synth_")

    def test_target_remains_binary(self, training_data, cfg):
        X, y, meta = training_data
        _, y_aug, _ = generate_synthetic_athletes(X, y, meta, cfg)
        assert set(y_aug.unique()).issubset({0, 1})

    def test_feature_columns_preserved(self, training_data, cfg):
        X, y, meta = training_data
        X_aug, _, _ = generate_synthetic_athletes(X, y, meta, cfg)
        assert list(X_aug.columns) == list(X.columns)


class TestValidateSynthetic:
    def test_returns_pvalues(self, training_data, cfg):
        X, y, meta = training_data
        X_aug, _, _ = generate_synthetic_athletes(X, y, meta, cfg)
        # Split out synthetic portion
        X_synth = X_aug.iloc[len(X):]
        results = validate_synthetic(X, X_synth, list(X.columns))
        assert len(results) == len(X.columns)
        for p in results.values():
            assert 0.0 <= p <= 1.0
