"""
test_extract.py – Tests for the EXTRACT stage.

Test plan
---------
E‑1  extract_participant returns DataFrame with required columns.
E‑2  extract_participant returns None for missing participant.
E‑3  extract_all consolidates all participants.
E‑4  Date column is timezone‑naïve datetime64.
E‑5  JSON daily‑sum files produce valid numeric columns.
E‑6  resting_heart_rate nested dict is unwrapped correctly.
E‑7  exercise entries are aggregated to one row per day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl.config import PipelineConfig
from src.etl.extract import extract_all, extract_participant


# ────────────────────────────────────────────────────────────
# E‑1  Single participant produces DataFrame with key columns
# ────────────────────────────────────────────────────────────
class TestExtractParticipant:

    def test_returns_dataframe(self, cfg: PipelineConfig):
        df = extract_participant("p01", cfg)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, cfg: PipelineConfig):
        df = extract_participant("p01", cfg)
        for col in ["participant_id", "date", "is_injured"]:
            assert col in df.columns, f"Missing column: {col}"

    # E‑2
    def test_missing_participant_returns_none(self, cfg: PipelineConfig):
        result = extract_participant("p99", cfg)
        assert result is None

    # E‑4
    def test_date_is_naive_datetime(self, cfg: PipelineConfig):
        df = extract_participant("p01", cfg)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].dt.tz is None  # naïve

    # E‑5
    def test_daily_sum_columns_numeric(self, cfg: PipelineConfig):
        df = extract_participant("p01", cfg)
        for col in ["steps", "distance", "calories"]:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])

    # E‑6
    def test_resting_hr_extracted(self, cfg: PipelineConfig):
        df = extract_participant("p01", cfg)
        assert "resting_hr" in df.columns
        assert df["resting_hr"].between(30, 120).all()


# ────────────────────────────────────────────────────────────
# E‑3  extract_all consolidation
# ────────────────────────────────────────────────────────────
class TestExtractAll:

    def test_all_participants_present(self, cfg: PipelineConfig):
        df = extract_all(cfg)
        pids = sorted(df["participant_id"].unique())
        assert pids == sorted(cfg.participants)

    def test_no_duplicate_pid_date(self, cfg: PipelineConfig):
        df = extract_all(cfg)
        dups = df.duplicated(subset=["participant_id", "date"]).sum()
        assert dups == 0, f"Found {dups} duplicate (pid, date) rows"

    # E‑7
    def test_exercise_aggregated_per_day(self, cfg: PipelineConfig):
        df = extract_all(cfg)
        # After aggregation each (pid, date) must have at most 1 row
        cnt = df.groupby(["participant_id", "date"]).size()
        assert cnt.max() == 1
