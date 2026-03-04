"""
test_load.py – Tests for the LOAD stage.

Test plan
---------
L‑1  save_csv writes file that exists on disk.
L‑2  save_csv output is readable and row‑count matches.
L‑3  build_tf_datasets returns DatasetBundle with 3 splits.
L‑4  train + val + test rows match original row count.
L‑5  Each batch yields (features, label) with correct shapes.
L‑6  load convenience function returns LoadResult.L‑7  save_tfrecord writes 3 files to disk.
L‑8  TFRecord roundtrip: read back matches row count.
L‑9  TFRecord parsed features have correct dimensionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl.config import PipelineConfig
from src.etl.load import (
    DatasetBundle, LoadResult, build_tf_datasets, load, save_csv,
    save_tfrecord, read_tfrecord,
)
from src.etl.transform import clean, engineer_features, standardise


@pytest.fixture()
def df_processed(raw_df: pd.DataFrame, cfg: PipelineConfig):
    """Fully processed DataFrame (clean → engineer → standardise)."""
    df_c = clean(raw_df, cfg)
    df_f = engineer_features(df_c, cfg)
    df_s, _, feat_cols = standardise(df_f, cfg)
    return df_s, feat_cols


# ────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────
class TestSaveCsv:

    def test_file_exists(self, df_processed, cfg: PipelineConfig):
        df, _ = df_processed
        path = save_csv(df, cfg, "test_out.csv")
        assert path.exists()

    def test_row_count_matches(self, df_processed, cfg: PipelineConfig):
        df, _ = df_processed
        path = save_csv(df, cfg, "test_out2.csv")
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)


# ────────────────────────────────────────────────────────────
# tf.data.Dataset
# ────────────────────────────────────────────────────────────
class TestBuildTfDatasets:

    def test_returns_bundle(self, df_processed, cfg: PipelineConfig):
        df, feat_cols = df_processed
        bundle = build_tf_datasets(df, cfg, feature_cols=feat_cols)
        assert isinstance(bundle, DatasetBundle)

    def test_splits_cover_all_rows(self, df_processed, cfg: PipelineConfig):
        df, feat_cols = df_processed
        bundle = build_tf_datasets(df, cfg, feature_cols=feat_cols)
        total = bundle.n_train + bundle.n_val + bundle.n_test
        assert total == len(df), (
            f"Split mismatch: {total} vs {len(df)}"
        )

    def test_batch_shapes(self, df_processed, cfg: PipelineConfig):
        tf = pytest.importorskip("tensorflow")
        df, feat_cols = df_processed
        bundle = build_tf_datasets(df, cfg, feature_cols=feat_cols)
        for X_batch, y_batch in bundle.train.take(1):
            assert X_batch.shape[1] == len(feat_cols)
            assert len(y_batch.shape) == 1  # 1‑D label


# ────────────────────────────────────────────────────────────
# TFRecord export / import
# ────────────────────────────────────────────────────────────
class TestTFRecord:

    def test_files_exist(self, df_processed, cfg: PipelineConfig):
        df, feat_cols = df_processed
        paths = save_tfrecord(df, cfg, feat_cols)
        for split in ("train", "val", "test"):
            assert split in paths
            assert paths[split].exists()
        # At least train must be non-empty (test may be empty with few participants)
        assert paths["train"].stat().st_size > 0

    def test_roundtrip_row_count(self, df_processed, cfg: PipelineConfig):
        tf = pytest.importorskip("tensorflow")
        df, feat_cols = df_processed
        paths = save_tfrecord(df, cfg, feat_cols)
        total = 0
        for split_path in paths.values():
            ds = read_tfrecord(split_path, n_features=len(feat_cols))
            for X_b, y_b in ds:
                total += X_b.shape[0]
        assert total == len(df)

    def test_feature_dimensionality(self, df_processed, cfg: PipelineConfig):
        tf = pytest.importorskip("tensorflow")
        df, feat_cols = df_processed
        paths = save_tfrecord(df, cfg, feat_cols)
        ds = read_tfrecord(paths["train"], n_features=len(feat_cols))
        for X_b, y_b in ds.take(1):
            assert X_b.shape[1] == len(feat_cols)
            assert len(y_b.shape) == 1


# ────────────────────────────────────────────────────────────
# Full load convenience
# ────────────────────────────────────────────────────────────
class TestLoadFull:

    def test_returns_load_result(self, df_processed, cfg: PipelineConfig):
        df, feat_cols = df_processed
        result = load(df, cfg, feature_cols=feat_cols)
        assert isinstance(result, LoadResult)
        assert result.csv_path.exists()
        assert result.tfrecord_paths is not None
