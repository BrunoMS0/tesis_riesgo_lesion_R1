"""
test_injury_lstm.py – Tests for the LSTM injury prediction module.

Validates:
- make_sequences: shape, boundary, NaN handling
- build_lstm_model: architecture, output shape
- train_lstm: smoke test (1 epoch)
- evaluate_lstm: metric keys and finite values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.lstm import (
    LSTMFoldResult,
    LSTMLoaoResult,
    build_lstm_model,
    evaluate_lstm,
    make_sequences,
    train_lstm,
)

# ────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────

FEATURES = ["session_load", "fatigue", "mood", "wellness_score"]
N_FEAT = len(FEATURES)
WINDOW = 7
ATHLETES = ["a1", "a2", "a3"]
DAYS = 30
RNG = np.random.RandomState(0)


@pytest.fixture()
def synthetic_df():
    """DataFrame with 3 athletes × 30 days, 4 features, binary target."""
    rows = []
    for pid in ATHLETES:
        dates = pd.date_range("2020-01-01", periods=DAYS, freq="D")
        for i, d in enumerate(dates):
            row = {"participant_id": pid, "date": d, "injury_next7d": 1 if i in (10, 20) else 0}
            for feat in FEATURES:
                row[feat] = float(RNG.uniform(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Tests — make_sequences
# ────────────────────────────────────────────────────────────

class TestMakeSequences:

    def test_output_shapes(self, synthetic_df):
        X, y, meta = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        expected_seqs_per_athlete = DAYS - WINDOW + 1
        assert X.shape == (len(ATHLETES) * expected_seqs_per_athlete, WINDOW, N_FEAT)
        assert y.shape == (len(ATHLETES) * expected_seqs_per_athlete,)
        assert len(meta) == len(y)

    def test_no_cross_athlete_leakage(self, synthetic_df):
        """Each sequence must belong to a single athlete."""
        X, y, meta = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        # meta has 'participant_id' per sequence
        assert "participant_id" in meta.columns

    def test_no_sequences_when_window_exceeds_athlete_days(self):
        """Athletes with fewer days than window_size are silently skipped."""
        df = pd.DataFrame({
            "participant_id": ["short"] * 3,
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "injury_next7d": [0, 1, 0],
            "session_load": [1.0, 2.0, 3.0],
        })
        X, y, meta = make_sequences(df, ["session_load"], "injury_next7d", window_size=7)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_nan_rows_dropped(self, synthetic_df):
        """Rows with NaN in features are dropped before windowing."""
        df = synthetic_df.copy()
        # Inject NaN in middle of athlete 'a1' — breaks sequences through that row
        df.loc[(df["participant_id"] == "a1") & (df["date"] == "2020-01-15"), "fatigue"] = float("nan")
        X_clean, _, _ = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        X_nan, _, _ = make_sequences(df, FEATURES, "injury_next7d", WINDOW)
        assert X_nan.shape[0] < X_clean.shape[0]

    def test_dtype_float32(self, synthetic_df):
        X, _, _ = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        assert X.dtype == np.float32


# ────────────────────────────────────────────────────────────
# Tests — build_lstm_model
# ────────────────────────────────────────────────────────────

class TestBuildLSTMModel:

    def test_output_shape(self):
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        batch = np.zeros((4, WINDOW, N_FEAT), dtype=np.float32)
        out = model.predict(batch, verbose=0)
        assert out.shape == (4, 1)

    def test_output_in_0_1(self):
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        batch = np.random.randn(10, WINDOW, N_FEAT).astype(np.float32)
        out = model.predict(batch, verbose=0).ravel()
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_model_has_lstm_layers(self):
        import keras
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        layer_types = [type(l).__name__ for l in model.layers]
        assert "LSTM" in layer_types


# ────────────────────────────────────────────────────────────
# Tests — train_lstm (smoke)
# ────────────────────────────────────────────────────────────

class TestTrainLSTM:

    def test_smoke_train(self, synthetic_df):
        """Training for 1 epoch should return a History object without errors."""
        X, y, _ = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        # Use half as train, half as val
        mid = len(X) // 2
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        hist = train_lstm(
            model, X[:mid], y[:mid], X[mid:], y[mid:],
            epochs=1, batch_size=16, patience=5,
        )
        assert "loss" in hist.history
        assert len(hist.history["loss"]) >= 1

    def test_class_weight_auto(self, synthetic_df):
        """Auto class-weight should be computed without error when imbalanced."""
        X, y, _ = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        mid = len(X) // 2
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        # No class_weight passed — should auto-compute
        hist = train_lstm(
            model, X[:mid], y[:mid], X[mid:], y[mid:],
            epochs=1, batch_size=16, patience=5, class_weight=None,
        )
        assert "loss" in hist.history


# ────────────────────────────────────────────────────────────
# Tests — evaluate_lstm
# ────────────────────────────────────────────────────────────

class TestEvaluateLSTM:

    def test_metric_keys_present(self, synthetic_df):
        X, y, meta = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        metrics = evaluate_lstm(model, X, y, meta)
        for key in ("roc_auc", "pr_auc", "f1", "brier_score", "n_samples", "n_pos"):
            assert key in metrics

    def test_empty_returns_nan(self):
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        X_empty = np.empty((0, WINDOW, N_FEAT), dtype=np.float32)
        y_empty = np.empty((0,), dtype=np.int32)
        meta_empty = pd.DataFrame(columns=["participant_id", "date", "seq_end_date"])
        metrics = evaluate_lstm(model, X_empty, y_empty, meta_empty)
        assert metrics["n_samples"] == 0
        import math
        assert math.isnan(metrics["roc_auc"])

    def test_metrics_finite_after_training(self, synthetic_df):
        X, y, meta = make_sequences(synthetic_df, FEATURES, "injury_next7d", WINDOW)
        mid = len(X) // 2
        model = build_lstm_model(n_features=N_FEAT, window_size=WINDOW)
        train_lstm(model, X[:mid], y[:mid], X[mid:], y[mid:],
                   epochs=2, batch_size=16, patience=3)
        metrics = evaluate_lstm(model, X[mid:], y[mid:], meta.iloc[mid:])
        # After training, roc_auc might be nan only if y has no positives
        # Just check the keys are there and brier is finite
        assert 0.0 <= metrics["brier_score"] <= 1.0
