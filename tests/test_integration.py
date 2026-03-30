"""
test_integration.py – Integration tests for R6 Two-Stage Pipeline.

Test plan
---------
T-INT-1  Model loading (R4 Keras + R5 joblib)
T-INT-2  End-to-end smoke test (synthetic 5-participant data)
T-INT-3  DFI handoff integrity (in-memory vs. CSV)
T-INT-4  Cold-start imputation (participant with no DFI)
T-INT-5  Error handling (missing model files / bad input)
T-INT-6  Report completeness (stages, durations, metrics)
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from src.fatigue.config import FatigueConfig
from src.fatigue.model import TemporalAttention, build_fatigue_model
from src.injury.config import InjuryConfig, FEATURE_COLUMNS
from src.injury.model import build_logistic_regression
from src.integration.config import IntegrationConfig
from src.integration.pipeline import (
    IntegrationReport,
    generate_dfi,
    load_models,
    merge_dfi_features,
    predict_injury,
    run,
)

# ────────────────────────────────────────────────────────────
# Constants for synthetic data
# ────────────────────────────────────────────────────────────
N_DAYS = 30
PIDS = ["p01", "p02", "p03", "p04", "p05"]
RNG = np.random.RandomState(99)

# Minimal feature set that covers both R4 objective features and
# the R5 FEATURE_COLUMNS + target.
_R4_FEATS = [
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
    "trimp", "trimp_7d_sum",
    "steps_7d_sum", "distance_7d_sum", "calories_7d_sum",
    "acute_load_7d", "chronic_load_28d", "acwr",
    "sleep_7d_avg", "sleep_debt",
    "rhr_baseline_7d", "rhr_drift", "rhr_variability_7d",
    "total_active_min", "active_ratio",
]

# Additional subjective / engineered columns needed by R5
_R5_EXTRA = [
    "session_load", "fatigue", "mood", "readiness", "sleep_quality",
    "soreness", "stress", "wellness_score",
    "steps_7d_sum", "calories_7d_sum",
]


# ────────────────────────────────────────────────────────────
# Synthetic data fixtures
# ────────────────────────────────────────────────────────────

def _build_synthetic_df(pids=None, n_days=N_DAYS):
    """Create a synthetic DataFrame with all columns needed by R4 and R5."""
    if pids is None:
        pids = PIDS
    rng = np.random.RandomState(99)
    rows = []
    all_cols = set(_R4_FEATS) | set(_R5_EXTRA)
    for pid in pids:
        for d in pd.date_range("2020-01-01", periods=n_days, freq="D"):
            row = {
                "participant_id": pid,
                "date": d,
                "is_injured": int(rng.rand() > 0.92),
                "fatigue": float(rng.randint(1, 6)),
            }
            # Subjective extras
            for col in ["mood", "readiness", "sleep_quality",
                        "sleep_duration_h", "soreness", "stress",
                        "perceived_exertion", "session_load",
                        "duration_min", "wellness_score"]:
                row[col] = float(rng.uniform(0, 5))
            # Objective / engineered features
            for f in _R4_FEATS:
                if f not in row:
                    row[f] = float(rng.uniform(0, 1000))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_csv(tmp_path):
    """Write synthetic feature CSV and return path."""
    df = _build_synthetic_df()
    csv_path = tmp_path / "synth_features.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture()
def fatigue_model_path(tmp_path, synthetic_csv):
    """Build, train-for-2-epochs, and save a tiny R4 Keras model."""
    import tensorflow as tf

    cfg = FatigueConfig(
        input_csv=synthetic_csv,
        output_path=str(tmp_path / "fatigue_out"),
        window_size=7,
        batch_size=16,
        max_epochs=2,
        early_stop_patience=2,
        lr_patience=1,
        objective_features=list(_R4_FEATS),
        # Tiny architecture for speed
        lstm1_units=4,
        lstm2_units=2,
        dense_units=4,
    )
    from src.fatigue.dataset import build_fatigue_datasets
    from src.fatigue.train import train_fatigue_model

    bundle = build_fatigue_datasets(cfg)
    model = build_fatigue_model(
        n_features=bundle.n_features,
        window_size=bundle.window_size,
        cfg=cfg,
    )
    train_fatigue_model(model, bundle, cfg)

    model_path = tmp_path / "test_fatigue.keras"
    model.save(str(model_path))
    return str(model_path)


@pytest.fixture()
def injury_model_path(tmp_path):
    """Train a tiny Logistic Regression model on random data and save as .joblib."""
    rng = np.random.RandomState(42)

    # Use the actual R5 feature columns so predict_proba works
    feature_cols = list(FEATURE_COLUMNS)
    n_train = 200

    X_train = pd.DataFrame(
        {f: rng.randn(n_train) for f in feature_cols})
    y_train = pd.Series((rng.rand(n_train) > 0.9).astype(int))

    cfg = InjuryConfig()
    model = build_logistic_regression(cfg)
    model.fit(X_train, y_train)

    model_path = tmp_path / "test_lr.joblib"
    joblib.dump(model, str(model_path))
    return str(model_path)


@pytest.fixture()
def normalizer_path(tmp_path):
    """Fit a PowerTransformer on random data and save as .joblib."""
    rng = np.random.RandomState(42)
    feature_cols = list(FEATURE_COLUMNS)
    n_train = 200
    X_train = pd.DataFrame({f: rng.randn(n_train) for f in feature_cols})
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(X_train)
    path = tmp_path / "test_normalizer.joblib"
    joblib.dump(pt, str(path))
    return str(path)


@pytest.fixture()
def integration_cfg(synthetic_csv, fatigue_model_path, injury_model_path,
                    normalizer_path, tmp_path):
    """Full IntegrationConfig wired to synthetic artefacts."""
    return IntegrationConfig(
        fatigue_model_path=fatigue_model_path,
        injury_model_path=injury_model_path,
        normalizer_path=normalizer_path,
        output_path=str(tmp_path / "integration_out"),
        fatigue_cfg=FatigueConfig(
            input_csv=synthetic_csv,
            output_path=str(tmp_path / "fatigue_out"),
            window_size=7,
            batch_size=16,
            objective_features=list(_R4_FEATS),
            lstm1_units=4,
            lstm2_units=2,
            dense_units=4,
        ),
        injury_cfg=InjuryConfig(
            input_csv=synthetic_csv,
        ),
    )


# ════════════════════════════════════════════════════════════
# T-INT-1: Model Loading
# ════════════════════════════════════════════════════════════

class TestModelLoading:
    def test_loads_both_models(self, integration_cfg):
        """R4 (Keras + TemporalAttention) and R5 (joblib) load OK."""
        fatigue_model, injury_model, normalizer = load_models(integration_cfg)
        assert fatigue_model is not None
        assert fatigue_model.count_params() > 0
        assert hasattr(injury_model, "predict_proba")
        assert hasattr(normalizer, "transform")

    def test_fatigue_model_has_attention(self, integration_cfg):
        """Loaded model contains the TemporalAttention custom layer."""
        fatigue_model, _, _ = load_models(integration_cfg)
        layer_types = [type(l).__name__ for l in fatigue_model.layers]
        assert "TemporalAttention" in layer_types


# ════════════════════════════════════════════════════════════
# T-INT-2: End-to-End Smoke Test
# ════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_runs(self, integration_cfg):
        """Pipeline completes all 4 stages without error."""
        report = run(integration_cfg)
        assert isinstance(report, IntegrationReport)
        assert len(report.stages) == 4

    def test_dfi_in_range(self, integration_cfg):
        """DFI predictions are within [0, 1]."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)
        assert dfi_df["dfi_predicted"].between(0, 1).all()

    def test_injury_probs_in_range(self, integration_cfg):
        """Injury probabilities are within [0, 1]."""
        report = run(integration_cfg)
        results = pd.read_csv(report.output_csv)
        assert results["injury_probability"].between(0, 1).all()

    def test_output_columns(self, integration_cfg):
        """Output CSV has expected columns."""
        report = run(integration_cfg)
        results = pd.read_csv(report.output_csv)
        expected = {"participant_id", "date", "dfi_predicted",
                    "injury_probability", "injury_predicted",
                    "injury_actual"}
        assert expected.issubset(set(results.columns))

    def test_no_nan_in_predictions(self, integration_cfg):
        """No NaN in prediction columns."""
        report = run(integration_cfg)
        results = pd.read_csv(report.output_csv)
        assert not results["dfi_predicted"].isna().any()
        assert not results["injury_probability"].isna().any()
        assert not results["injury_predicted"].isna().any()


# ════════════════════════════════════════════════════════════
# T-INT-3: DFI Handoff Integrity
# ════════════════════════════════════════════════════════════

class TestHandoffIntegrity:
    def test_merge_produces_dfi_column(self, integration_cfg):
        """In-memory merge injects dfi_predicted column."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)
        merged = merge_dfi_features(dfi_df, integration_cfg)
        assert "dfi_predicted" in merged.columns
        assert not merged["dfi_predicted"].isna().all()

    def test_merge_preserves_row_count(self, integration_cfg):
        """LEFT JOIN does not inflate or drop rows."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)
        icfg = integration_cfg.injury_cfg
        original = pd.read_csv(icfg.input_csv)
        merged = merge_dfi_features(dfi_df, integration_cfg)
        assert len(merged) == len(original)

    def test_feature_count_matches_config(self, integration_cfg):
        """R5 receives the expected number of feature columns."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)
        merged = merge_dfi_features(dfi_df, integration_cfg)
        icfg = integration_cfg.injury_cfg
        available = [c for c in icfg.feature_columns if c in merged.columns]
        # At least dfi_predicted + core features should be present
        assert len(available) >= 10
        assert "dfi_predicted" in available


# ════════════════════════════════════════════════════════════
# T-INT-4: Cold-Start Imputation
# ════════════════════════════════════════════════════════════

class TestColdStartImputation:
    def test_missing_participant_filled(self, integration_cfg):
        """Participant with NO DFI predictions gets imputed values."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)

        # Remove all predictions for one participant
        dfi_df = dfi_df[dfi_df["participant_id"] != "p01"].copy()

        merged = merge_dfi_features(dfi_df, integration_cfg)
        p01_dfi = merged.loc[
            merged["participant_id"] == "p01", "dfi_predicted"
        ]
        # Should be filled, not NaN
        assert not p01_dfi.isna().any(), "Cold-start imputation failed for p01"

    def test_partial_missing_filled(self, integration_cfg):
        """Rows without DFI (e.g., first 13 days) get filled."""
        fatigue_model, _, _ = load_models(integration_cfg)
        dfi_df = generate_dfi(fatigue_model, integration_cfg)
        merged = merge_dfi_features(dfi_df, integration_cfg)
        assert not merged["dfi_predicted"].isna().any()


# ════════════════════════════════════════════════════════════
# T-INT-5: Error Handling
# ════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_missing_fatigue_model(self, integration_cfg):
        """RuntimeError when fatigue model file missing."""
        bad_cfg = IntegrationConfig(
            fatigue_model_path="/nonexistent/model.keras",
            injury_model_path=integration_cfg.injury_model_path,
            normalizer_path=integration_cfg.normalizer_path,
            fatigue_cfg=integration_cfg.fatigue_cfg,
            injury_cfg=integration_cfg.injury_cfg,
        )
        with pytest.raises(RuntimeError, match="R4 fatigue model not found"):
            load_models(bad_cfg)

    def test_missing_injury_model(self, integration_cfg):
        """RuntimeError when injury model file missing."""
        bad_cfg = IntegrationConfig(
            fatigue_model_path=integration_cfg.fatigue_model_path,
            injury_model_path="/nonexistent/model.joblib",
            normalizer_path=integration_cfg.normalizer_path,
            fatigue_cfg=integration_cfg.fatigue_cfg,
            injury_cfg=integration_cfg.injury_cfg,
        )
        with pytest.raises(RuntimeError, match="R5 injury model not found"):
            load_models(bad_cfg)


# ════════════════════════════════════════════════════════════
# T-INT-6: Report Completeness
# ════════════════════════════════════════════════════════════

class TestReportCompleteness:
    def test_four_stages(self, integration_cfg):
        """Report contains exactly 4 stages."""
        report = run(integration_cfg)
        assert len(report.stages) == 4

    def test_stage_names(self, integration_cfg):
        """Stages have expected names."""
        report = run(integration_cfg)
        names = [s.name for s in report.stages]
        assert names == [
            "LoadModels", "FatiguePrediction",
            "FeatureHandoff", "InjuryPrediction",
        ]

    def test_durations_positive(self, integration_cfg):
        """All stage durations are non-negative."""
        report = run(integration_cfg)
        for s in report.stages:
            assert s.duration_s >= 0

    def test_total_duration(self, integration_cfg):
        """Total duration is positive."""
        report = run(integration_cfg)
        assert report.total_duration_s > 0

    def test_metrics_finite(self, integration_cfg):
        """Injury metrics are finite numbers."""
        report = run(integration_cfg)
        for key, val in report.injury_metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_output_csv_exists(self, integration_cfg):
        """Output CSV is written to disk."""
        report = run(integration_cfg)
        assert report.output_csv is not None
        assert os.path.isfile(report.output_csv)

    def test_model_paths_recorded(self, integration_cfg):
        """Report records the model file paths used."""
        report = run(integration_cfg)
        assert report.fatigue_model_path == integration_cfg.fatigue_model_path
        assert report.injury_model_path == integration_cfg.injury_model_path
