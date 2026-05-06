"""
test_injury_augment.py – Tests for R5 data augmentation (SMOTE + Copula).

Validates:
- SMOTE augmented data has more rows than original
- SMOTE preserves feature columns and binary target
- Copula augmented data has more rows than original
- Synthetic participant IDs are distinct from real ones
- Target column remains binary {0, 1}
- KS validation returns p-values for each feature
- augment_training_data dispatcher works for both methods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig
from src.injury.augment import (
    apply_smote,
    augment_training_data,
    generate_synthetic_athletes,
    validate_synthetic,
)


@pytest.fixture()
def cfg_smote():
    return InjuryConfig(augmentation_method="smote", target_ratio=0.3, smote_k_neighbors=3)


@pytest.fixture()
def cfg_copula():
    return InjuryConfig(augmentation_method="copula")


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
    # ~5% positive so SMOTE has minority to oversample
    y = pd.Series(([1] * 5 + [0] * 45) * 2)
    meta = pd.DataFrame({
        "participant_id": pids,
        "date": pd.date_range("2020-01-01", periods=n),
    })
    return X, y, meta


class TestApplySmote:
    def test_augmented_has_more_rows(self, training_data, cfg_smote):
        X, y, meta = training_data
        X_aug, y_aug = apply_smote(X, y, meta, cfg_smote)
        assert len(X_aug) >= len(X)

    def test_target_remains_binary(self, training_data, cfg_smote):
        X, y, meta = training_data
        _, y_aug = apply_smote(X, y, meta, cfg_smote)
        assert set(y_aug.unique()).issubset({0, 1})

    def test_feature_columns_preserved(self, training_data, cfg_smote):
        X, y, meta = training_data
        X_aug, _ = apply_smote(X, y, meta, cfg_smote)
        assert list(X_aug.columns) == list(X.columns)

    def test_minority_class_increased(self, training_data, cfg_smote):
        X, y, meta = training_data
        original_minority = (y == 1).sum()
        _, y_aug = apply_smote(X, y, meta, cfg_smote)
        new_minority = (y_aug == 1).sum()
        assert new_minority > original_minority


class TestAugmentDispatcher:
    def test_smote_method(self, training_data, cfg_smote):
        X, y, meta = training_data
        X_aug, y_aug = augment_training_data(X, y, meta, cfg_smote)
        assert len(X_aug) >= len(X)

    def test_copula_method(self, training_data, cfg_copula):
        X, y, meta = training_data
        X_aug, y_aug = augment_training_data(X, y, meta, cfg_copula)
        assert len(X_aug) >= len(X)


class TestGenerateSyntheticAthletes:
    def test_augmented_has_more_rows(self, training_data, cfg_copula):
        X, y, meta = training_data
        X_aug, y_aug, meta_aug = generate_synthetic_athletes(X, y, meta, cfg_copula)
        assert len(X_aug) > len(X)

    def test_real_data_preserved(self, training_data, cfg_copula):
        X, y, meta = training_data
        X_aug, y_aug, _ = generate_synthetic_athletes(X, y, meta, cfg_copula)
        # First n rows should be identical to original
        pd.testing.assert_frame_equal(
            X_aug.iloc[:len(X)].reset_index(drop=True),
            X.reset_index(drop=True),
        )

    def test_synthetic_pids_distinct(self, training_data, cfg_copula):
        X, y, meta = training_data
        _, _, meta_aug = generate_synthetic_athletes(X, y, meta, cfg_copula)
        real_pids = set(meta["participant_id"].unique())
        synth_pids = set(meta_aug["participant_id"].unique()) - real_pids
        assert len(synth_pids) > 0
        for pid in synth_pids:
            assert pid.startswith("synth_")

    def test_target_remains_binary(self, training_data, cfg_copula):
        X, y, meta = training_data
        _, y_aug, _ = generate_synthetic_athletes(X, y, meta, cfg_copula)
        assert set(y_aug.unique()).issubset({0, 1})

    def test_feature_columns_preserved(self, training_data, cfg_copula):
        X, y, meta = training_data
        X_aug, _, _ = generate_synthetic_athletes(X, y, meta, cfg_copula)
        assert list(X_aug.columns) == list(X.columns)


class TestValidateSynthetic:
    def test_returns_pvalues(self, training_data, cfg_copula):
        X, y, meta = training_data
        X_aug, _, _ = generate_synthetic_athletes(X, y, meta, cfg_copula)
        # Split out synthetic portion
        X_synth = X_aug.iloc[len(X):]
        results = validate_synthetic(X, X_synth, list(X.columns))
        assert len(results) == len(X.columns)
        for p in results.values():
            assert 0.0 <= p <= 1.0
