"""
test_fatigue_train.py – Smoke tests for training and evaluation.

Runs a single-epoch training pass on tiny synthetic data to verify
the training loop and evaluation pipeline don't crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.fatigue.config import FatigueConfig
from src.fatigue.dataset import build_fatigue_datasets, FatigueDatasetBundle
from src.fatigue.model import build_fatigue_model
from src.fatigue.train import train_fatigue_model
from src.fatigue.evaluate import evaluate_model


# ────────────────────────────────────────────────────────────
# Fixtures  (tiny synthetic dataset)
# ────────────────────────────────────────────────────────────

N_DAYS = 30
PIDS = ["p01", "p02", "p03", "p04", "p05"]
RNG = np.random.RandomState(99)
_FEATS = [
    "steps", "distance", "calories", "resting_hr",
    "hr_zone_below", "hr_zone_1", "hr_zone_2", "hr_zone_3",
    "overall_score", "minutesAsleep", "efficiency",
    "trimp", "acwr", "active_ratio",
]


@pytest.fixture()
def synthetic_csv(tmp_path):
    rows = []
    for pid in PIDS:
        for d in pd.date_range("2020-01-01", periods=N_DAYS, freq="D"):
            row = {"participant_id": pid, "date": d,
                   "is_injured": 0, "fatigue": float(RNG.randint(1, 6))}
            for col in ["mood", "readiness", "sleep_quality",
                        "sleep_duration_h", "soreness", "stress",
                        "perceived_exertion", "session_load",
                        "duration_min", "wellness_score"]:
                row[col] = float(RNG.uniform(0, 5))
            for f in _FEATS:
                row[f] = float(RNG.uniform(0, 1000))
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synth.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def smoke_cfg(synthetic_csv, tmp_path):
    return FatigueConfig(
        input_csv=str(synthetic_csv),
        output_path=str(tmp_path / "fatigue_out"),
        window_size=7,
        batch_size=16,
        max_epochs=2,
        early_stop_patience=2,
        lr_patience=1,
        objective_features=_FEATS,
    )


@pytest.fixture()
def bundle(smoke_cfg):
    return build_fatigue_datasets(smoke_cfg)


@pytest.fixture()
def model(bundle, smoke_cfg):
    return build_fatigue_model(
        n_features=bundle.n_features,
        window_size=bundle.window_size,
        cfg=smoke_cfg,
    )


# ────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────

class TestTrainSmokeTest:
    def test_training_runs(self, model, bundle, smoke_cfg):
        """Train for 2 epochs without error."""
        history = train_fatigue_model(model, bundle, smoke_cfg)
        assert len(history.history["loss"]) <= smoke_cfg.max_epochs
        assert "val_loss" in history.history

    def test_loss_is_finite(self, model, bundle, smoke_cfg):
        history = train_fatigue_model(model, bundle, smoke_cfg)
        for loss in history.history["loss"]:
            assert np.isfinite(loss)


class TestEvaluateSmokeTest:
    def test_evaluation_runs(self, model, bundle, smoke_cfg):
        """Evaluate after minimal training without error."""
        train_fatigue_model(model, bundle, smoke_cfg)
        result = evaluate_model(model, bundle, smoke_cfg)
        assert "mse" in result.metrics
        assert "r2" in result.metrics
        assert result.metrics["n_samples"] == bundle.n_test

    def test_per_participant_breakdown(self, model, bundle, smoke_cfg):
        train_fatigue_model(model, bundle, smoke_cfg)
        result = evaluate_model(model, bundle, smoke_cfg)
        assert len(result.per_participant) > 0
        assert "participant_id" in result.per_participant.columns
