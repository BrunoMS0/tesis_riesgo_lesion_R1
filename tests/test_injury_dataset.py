"""
test_injury_dataset.py – Tests for R5 data preparation.

Validates:
- DFI merge logic (left join + cold-start fill)
- Feature selection (correct columns)
- Participant split (no leakage, all assigned)
- InjuryDatasetBundle construction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig, FEATURE_COLUMNS
from src.injury.dataset import (
    build_injury_datasets,
    load_and_merge,
    prepare_features,
    split_participants,
)

# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

N_DAYS = 30
PIDS = ["p01", "p02", "p03", "p04", "p05"]
RNG = np.random.RandomState(42)

# All features that R5 expects (minus dfi_predicted, which comes from DFI CSV)
_BASE_FEATURES = [c for c in FEATURE_COLUMNS if c != "dfi_predicted"]


@pytest.fixture()
def synthetic_csvs(tmp_path):
    """Create a minimal feature CSV and DFI predictions CSV."""
    rows = []
    dfi_rows = []
    for pid in PIDS:
        dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
        for i, d in enumerate(dates):
            row = {
                "participant_id": pid, "date": d,
                "is_injured": 1 if i == 10 else 0,
            }
            for feat in _BASE_FEATURES:
                row[feat] = float(RNG.uniform(0, 100))
            # Add extra columns that exist in real data but aren't R5 features
            row["distance"] = float(RNG.uniform(0, 10))
            row["perceived_exertion"] = float(RNG.randint(1, 10))
            row["duration_min"] = float(RNG.randint(20, 120))
            row["sleep_duration_h"] = float(RNG.uniform(5, 9))
            rows.append(row)

            # DFI predictions (skip first 14 days to simulate cold start)
            if i >= 14:
                dfi_rows.append({
                    "participant_id": pid,
                    "date": d,
                    "dfi_predicted": float(RNG.uniform(0, 1)),
                    "dfi_actual": float(RNG.uniform(0, 1)),
                })

    feature_csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(feature_csv, index=False)

    dfi_csv = tmp_path / "dfi_predictions.csv"
    pd.DataFrame(dfi_rows).to_csv(dfi_csv, index=False)

    return feature_csv, dfi_csv


@pytest.fixture()
def injury_cfg(synthetic_csvs, tmp_path):
    """InjuryConfig pointing to synthetic data."""
    feature_csv, dfi_csv = synthetic_csvs
    return InjuryConfig(
        input_csv=str(feature_csv),
        dfi_csv=str(dfi_csv),
        output_path=str(tmp_path / "injury_out"),
        n_synthetic_athletes=0,  # disable augmentation for dataset tests
    )


# ────────────────────────────────────────────────────────────
# Tests — Load & Merge
# ────────────────────────────────────────────────────────────

class TestLoadAndMerge:
    def test_merge_adds_dfi_column(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        assert "dfi_predicted" in df.columns

    def test_cold_start_filled(self, injury_cfg):
        """First 14 days per participant have no DFI — should be filled."""
        df = load_and_merge(injury_cfg)
        assert df["dfi_predicted"].isna().sum() == 0

    def test_row_count_preserved(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        assert len(df) == N_DAYS * len(PIDS)


# ────────────────────────────────────────────────────────────
# Tests — Feature Selection
# ────────────────────────────────────────────────────────────

class TestPrepareFeatures:
    def test_no_metadata_in_features(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        X, y, meta = prepare_features(df, injury_cfg)
        assert "participant_id" not in X.columns
        assert "date" not in X.columns
        assert "is_injured" not in X.columns

    def test_target_is_binary(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        _, y, _ = prepare_features(df, injury_cfg)
        assert set(y.unique()).issubset({0, 1})

    def test_dfi_in_features(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        X, _, _ = prepare_features(df, injury_cfg)
        assert "dfi_predicted" in X.columns


# ────────────────────────────────────────────────────────────
# Tests — Participant Split
# ────────────────────────────────────────────────────────────

class TestSplitParticipants:
    def test_all_assigned(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        train, val, test = split_participants(df, injury_cfg)
        assert set(train + val + test) == set(PIDS)

    def test_no_overlap(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        train, val, test = split_participants(df, injury_cfg)
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)

    def test_deterministic(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        split1 = split_participants(df, injury_cfg)
        split2 = split_participants(df, injury_cfg)
        assert split1 == split2


# ────────────────────────────────────────────────────────────
# Tests — Dataset Bundle
# ────────────────────────────────────────────────────────────

class TestBuildInjuryDatasets:
    def test_bundle_shapes(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert len(bundle.X_train) == len(bundle.y_train)
        assert len(bundle.X_val) == len(bundle.y_val)
        assert len(bundle.X_test) == len(bundle.y_test)

    def test_no_participant_leakage(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        train_pids = set(bundle.meta_train["participant_id"].unique())
        val_pids = set(bundle.meta_val["participant_id"].unique())
        test_pids = set(bundle.meta_test["participant_id"].unique())
        assert not train_pids & test_pids
        assert not train_pids & val_pids

    def test_total_rows(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
        assert total == N_DAYS * len(PIDS)


# ────────────────────────────────────────────────────────────
# Tests — Normalization
# ────────────────────────────────────────────────────────────

class TestBundleNormalization:
    def test_normalizer_present(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert bundle.normalizer is not None

    def test_train_mean_near_zero(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        means = bundle.X_train.mean()
        assert means.abs().max() < 0.5, f"Max train mean {means.abs().max():.3f}"

    def test_train_std_near_one(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        stds = bundle.X_train.std()
        assert (stds - 1).abs().max() < 0.5, f"Max std deviation {stds.max():.3f}"

    def test_val_test_scaled(self, injury_cfg):
        """Val/test should be scaled using train normalizer (not raw)."""
        bundle = build_injury_datasets(injury_cfg)
        # Raw data was uniform(0, 100) — normalized means must be far from 50
        assert bundle.X_val.mean().abs().mean() < 10
        assert bundle.X_test.mean().abs().mean() < 10

    def test_normalizer_has_reports(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert len(bundle.normalizer.pre_report) > 0
        assert len(bundle.normalizer.post_report) > 0
        assert "ks_statistic" in bundle.normalizer.pre_report.columns
        assert "is_normal" in bundle.normalizer.post_report.columns
