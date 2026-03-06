"""
test_fatigue_dataset.py – Tests for the R4 data preparation pipeline.

Uses lightweight synthetic data to validate:
- DFI computation
- Participant-level splitting (no leakage)
- Sliding-window shape correctness
- Target range [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fatigue.config import FatigueConfig
from src.fatigue.dataset import (
    compute_dfi,
    split_participants,
    fit_scaler,
    apply_scaler,
    create_sequences,
    build_fatigue_datasets,
)


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

N_DAYS = 30
PIDS = ["p01", "p02", "p03", "p04", "p05"]
RNG = np.random.RandomState(42)

# Features that will exist in our synthetic CSV
_FEATURES = [
    "steps", "distance", "calories", "resting_hr",
    "hr_zone_below", "hr_zone_1", "hr_zone_2", "hr_zone_3",
    "exercise_duration_min", "exercise_calories", "exercise_steps",
    "exercise_avg_hr", "exercise_sessions",
    "lightly_active_minutes", "moderately_active_minutes",
    "very_active_minutes", "sedentary_minutes",
    "overall_score", "composition_score", "revitalization_score",
    "duration_score", "deep_sleep_in_minutes", "restlessness",
    "sleep_rhr", "minutesAsleep", "efficiency", "minutesAwake",
    "timeInBed",
    "trimp", "trimp_7d_sum", "steps_7d_sum", "distance_7d_sum",
    "calories_7d_sum", "acute_load_7d", "chronic_load_28d", "acwr",
    "sleep_7d_avg", "sleep_debt", "rhr_baseline_7d", "rhr_drift",
    "rhr_variability_7d", "total_active_min", "active_ratio",
]


@pytest.fixture()
def synthetic_csv(tmp_path):
    """Write a small synthetic CSV that mimics dataset_features_sin_normalizar.csv."""
    rows = []
    for pid in PIDS:
        dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
        for d in dates:
            row = {"participant_id": pid, "date": d,
                   "is_injured": 0, "fatigue": float(RNG.randint(1, 6))}
            # Subjective cols (should be ignored by the model)
            for col in ["mood", "readiness", "sleep_quality",
                        "sleep_duration_h", "soreness", "stress",
                        "perceived_exertion", "session_load",
                        "duration_min", "wellness_score"]:
                row[col] = float(RNG.uniform(0, 5))
            for feat in _FEATURES:
                row[feat] = float(RNG.uniform(0, 1000))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def fatigue_cfg(synthetic_csv, tmp_path):
    """FatigueConfig pointing to synthetic data."""
    return FatigueConfig(
        input_csv=str(synthetic_csv),
        output_path=str(tmp_path / "fatigue_out"),
        window_size=7,          # smaller window for fast tests
        objective_features=_FEATURES,
    )


# ────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────

class TestComputeDFI:
    def test_range(self):
        series = pd.Series([1, 2, 3, 4, 5])
        dfi = compute_dfi(series)
        assert dfi.min() >= 0.0
        assert dfi.max() <= 1.0

    def test_inversion(self):
        """fatigue=5 (max fatigue) should map to DFI=0;
        fatigue=1 (fresh) should map to DFI=1."""
        assert compute_dfi(pd.Series([5]))[0] == pytest.approx(0.0)
        assert compute_dfi(pd.Series([1]))[0] == pytest.approx(1.0)


class TestSplitParticipants:
    def test_all_assigned(self, fatigue_cfg):
        df = pd.read_csv(fatigue_cfg.input_csv, parse_dates=["date"])
        train, val, test = split_participants(df, fatigue_cfg)
        all_pids = set(train + val + test)
        assert all_pids == set(PIDS)

    def test_no_overlap(self, fatigue_cfg):
        df = pd.read_csv(fatigue_cfg.input_csv, parse_dates=["date"])
        train, val, test = split_participants(df, fatigue_cfg)
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)


class TestCreateSequences:
    def test_shape(self, fatigue_cfg):
        df = pd.read_csv(fatigue_cfg.input_csv, parse_dates=["date"])
        df["dfi"] = compute_dfi(df["fatigue"])
        feature_cols = [c for c in _FEATURES if c in df.columns]
        X, y, meta = create_sequences(df, feature_cols, fatigue_cfg.window_size)

        assert X.ndim == 3
        assert X.shape[1] == fatigue_cfg.window_size
        assert X.shape[2] == len(feature_cols)
        assert len(y) == len(X)
        assert len(meta) == len(X)

    def test_target_range(self, fatigue_cfg):
        df = pd.read_csv(fatigue_cfg.input_csv, parse_dates=["date"])
        df["dfi"] = compute_dfi(df["fatigue"])
        feature_cols = [c for c in _FEATURES if c in df.columns]
        _, y, _ = create_sequences(df, feature_cols, fatigue_cfg.window_size)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_expected_count(self, fatigue_cfg):
        """Each participant contributes (N_DAYS - window_size) sequences."""
        df = pd.read_csv(fatigue_cfg.input_csv, parse_dates=["date"])
        df["dfi"] = compute_dfi(df["fatigue"])
        feature_cols = [c for c in _FEATURES if c in df.columns]
        X, _, _ = create_sequences(df, feature_cols, fatigue_cfg.window_size)
        expected = len(PIDS) * (N_DAYS - fatigue_cfg.window_size)
        assert len(X) == expected


class TestBuildFatigueDatasets:
    def test_returns_bundle(self, fatigue_cfg):
        bundle = build_fatigue_datasets(fatigue_cfg)
        assert bundle.n_train > 0
        assert bundle.n_val > 0
        assert bundle.n_test > 0
        assert bundle.n_features == len(_FEATURES)

    def test_no_participant_leakage(self, fatigue_cfg):
        bundle = build_fatigue_datasets(fatigue_cfg)
        train_pids = set(bundle.meta_train["participant_id"])
        val_pids = set(bundle.meta_val["participant_id"])
        test_pids = set(bundle.meta_test["participant_id"])
        assert not train_pids & test_pids
        assert not train_pids & val_pids

    def test_tf_dataset_shapes(self, fatigue_cfg):
        bundle = build_fatigue_datasets(fatigue_cfg)
        for batch_x, batch_y in bundle.train.take(1):
            assert batch_x.shape[1] == fatigue_cfg.window_size
            assert batch_x.shape[2] == bundle.n_features
            assert batch_y.ndim == 1
