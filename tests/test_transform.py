"""
test_transform.py – Tests for the TRANSFORM stage.

Test plan
---------
T‑1  clean: event variables have no remaining NaN.
T‑2  clean: columns with >60 % nulls are dropped.
T‑3  clean: no nulls remain in numeric columns after imputation.
T‑4  clean: duplicates (participant_id, date) are removed.
T‑5  engineer_features: ACWR column exists and is finite where chronic > 0.
T‑6  engineer_features: TRIMP column exists.
T‑7  engineer_features: Sleep Debt column exists.
T‑8  engineer_features: RHR Drift column exists.
T‑9  engineer_features: Wellness Score ∈ [1, 7] (valid range).
T‑10 select_features: dropped features are fewer than total features.
T‑13 transform: full pipeline returns TransformResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl.config import PipelineConfig
from src.etl.transform import (
    TransformResult,
    clean,
    engineer_features,
    select_features,
    transform,
)


# ────────────────────────────────────────────────────────────
# Clean sub‑stage
# ────────────────────────────────────────────────────────────
class TestClean:

    def test_event_vars_no_nulls(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        df = clean(raw_df, cfg)
        for col in cfg.event_vars:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"NaN in event col: {col}"

    def test_high_null_columns_dropped(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        """Inject a mostly‑null column → verify it is dropped."""
        df = raw_df.copy()
        df["_fake_null_col"] = np.nan
        result = clean(df, cfg)
        assert "_fake_null_col" not in result.columns

    def test_no_numeric_nulls(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        df = clean(raw_df, cfg)
        num_nulls = df.select_dtypes(include=[np.number]).isnull().sum().sum()
        assert num_nulls == 0, f"Remaining numeric nulls: {num_nulls}"

    def test_no_duplicates(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        df = clean(raw_df, cfg)
        dups = df.duplicated(subset=["participant_id", "date"]).sum()
        assert dups == 0


# ────────────────────────────────────────────────────────────
# Feature engineering sub‑stage
# ────────────────────────────────────────────────────────────
class TestEngineerFeatures:

    @pytest.fixture()
    def df_clean(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        return clean(raw_df, cfg)

    def test_acwr_exists(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        assert "acwr" in df.columns

    def test_acwr_finite(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        mask = df["chronic_load_28d"] > 0
        assert np.isfinite(df.loc[mask, "acwr"]).all()

    def test_trimp_exists(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        assert "trimp" in df.columns

    def test_sleep_debt_exists(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        assert "sleep_debt" in df.columns

    def test_rhr_drift_exists(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        assert "rhr_drift" in df.columns

    def test_wellness_score_range(self, df_clean: pd.DataFrame, cfg: PipelineConfig):
        df = engineer_features(df_clean, cfg)
        if "wellness_score" in df.columns:
            ws = df["wellness_score"].dropna()
            assert ws.min() >= 1.0
            assert ws.max() <= 7.0


# ────────────────────────────────────────────────────────────
# Variable selection sub‑stage
# ────────────────────────────────────────────────────────────
class TestSelectFeatures:

    @pytest.fixture()
    def prepared(self, raw_df: pd.DataFrame, cfg: PipelineConfig):
        df_c = clean(raw_df, cfg)
        df_f = engineer_features(df_c, cfg)
        exclude = {"is_injured", "participant_id", "date"}
        feat_cols = [
            c for c in df_f.columns
            if c not in exclude and df_f[c].dtype in ("float64", "int64")
        ]
        return df_f, feat_cols

    def test_features_reduced_or_equal(
        self, prepared, cfg: PipelineConfig
    ):
        df_f, feat_cols = prepared
        _, final, report = select_features(df_f, feat_cols, cfg)
        assert len(final) <= len(feat_cols)

    def test_target_preserved(self, prepared, cfg: PipelineConfig):
        df_f, feat_cols = prepared
        df_sel, _, _ = select_features(df_f, feat_cols, cfg)
        assert "is_injured" in df_sel.columns


# ────────────────────────────────────────────────────────────
# Full transform pipeline
# ────────────────────────────────────────────────────────────
class TestTransformFull:

    def test_returns_transform_result(
        self, raw_df: pd.DataFrame, cfg: PipelineConfig
    ):
        result = transform(raw_df, cfg)
        assert isinstance(result, TransformResult)
        assert result.df_cleaned is not None
        assert result.df_features is not None
        assert result.df_selected is not None
        assert len(result.feature_cols) > 0
