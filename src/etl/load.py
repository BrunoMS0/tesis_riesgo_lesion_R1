"""
load.py – LOAD stage of the ETL pipeline.

Responsibilities
----------------
1. Export the final processed DataFrame to CSV.
2. Build ``tf.data.Dataset`` objects (train / val / test) ready for
   TensorFlow / Keras model consumption.

All tf.data parameters (batch size, shuffle buffer, prefetch, splits)
are read from :class:`~config.PipelineConfig`.

Public API
----------
save_csv(df, cfg, filename) -> Path
save_tfrecord(df, cfg, feature_cols, target) -> Dict[str, Path]
read_tfrecord(path, feature_cols) -> tf.data.Dataset
build_tf_datasets(df, cfg, target, feature_cols) -> DatasetBundle
load(df, cfg, feature_cols, target) -> LoadResult
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Lazy TensorFlow import so that tests can run without TF if needed
_tf = None


def _get_tf():
    """Lazily import TensorFlow to avoid heavy startup cost."""
    global _tf
    if _tf is None:
        import tensorflow as tf  # noqa: F811

        _tf = tf
        logger.info("TensorFlow %s loaded", tf.__version__)
    return _tf


# ────────────────────────────────────────────────────────────
# Result containers
# ────────────────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    """Holds train / val / test ``tf.data.Dataset`` objects."""

    train: object  # tf.data.Dataset
    val: object
    test: object
    feature_cols: List[str]
    target_col: str
    n_train: int
    n_val: int
    n_test: int


@dataclass
class LoadResult:
    """Everything produced by the Load stage."""

    csv_path: Path
    datasets: Optional[DatasetBundle]
    tfrecord_paths: Optional[Dict[str, Path]] = None
    metadata: Dict = None  # type: ignore[assignment]


# ────────────────────────────────────────────────────────────
# 1. CSV export
# ────────────────────────────────────────────────────────────

def save_csv(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    filename: str = "dataset_etl_output.csv",
) -> Path:
    """Write *df* to ``cfg.output_path / filename`` and return the path."""
    os.makedirs(cfg.output_path, exist_ok=True)
    csv_path = Path(cfg.output_path) / filename
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved: %s (%d rows, %d cols)",
                csv_path, len(df), len(df.columns))
    return csv_path


# ────────────────────────────────────────────────────────────
# 2. TFRecord export / import
# ────────────────────────────────────────────────────────────

def _make_example(features: np.ndarray, label: float) -> bytes:
    """Serialise one row as a ``tf.train.Example``."""
    tf = _get_tf()
    feature_dict = {
        "features": tf.train.Feature(
            float_list=tf.train.FloatList(value=features.tolist())
        ),
        "label": tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(label)])
        ),
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )
    return example.SerializeToString()


def save_tfrecord(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: List[str],
    target: str = "is_injured",
) -> Dict[str, Path]:
    """
    Serialise train / val / test splits to ``.tfrecord`` files.

    Returns
    -------
    dict[str, Path]
        Mapping ``{"train": Path, "val": Path, "test": Path}``.
    """
    tf = _get_tf()
    os.makedirs(cfg.output_path, exist_ok=True)

    # Fill residual NaN (safety net)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0)
    df[target] = df[target].fillna(0)

    df_train, df_val, df_test = _split_by_participant(df, cfg)
    paths: Dict[str, Path] = {}

    for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        fpath = Path(cfg.output_path) / f"{split_name}.tfrecord"
        with tf.io.TFRecordWriter(str(fpath)) as writer:
            X = df_split[feature_cols].values.astype(np.float32)
            y = df_split[target].values.astype(np.float32)
            for i in range(len(df_split)):
                writer.write(_make_example(X[i], y[i]))
        paths[split_name] = fpath
        logger.info("TFRecord saved: %s (%d examples)", fpath, len(df_split))

    return paths


def read_tfrecord(
    path: str | Path,
    n_features: int,
    batch_size: int = 64,
) -> object:
    """
    Read a ``.tfrecord`` file back into a ``tf.data.Dataset``.

    Each element is a ``(features, label)`` tuple of float32 tensors.
    """
    tf = _get_tf()

    feature_spec = {
        "features": tf.io.FixedLenFeature([n_features], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.float32),
    }

    def _parse(serialised):
        parsed = tf.io.parse_single_example(serialised, feature_spec)
        return parsed["features"], tf.squeeze(parsed["label"])

    ds = tf.data.TFRecordDataset(str(path))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ────────────────────────────────────────────────────────────
# 3. tf.data.Dataset construction (in‑memory)
# ────────────────────────────────────────────────────────────

def _split_by_participant(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by *participant* (not row) to avoid data leakage.

    Each participant's full history goes into exactly one fold.
    """
    pids = list(df["participant_id"].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(pids)

    n = len(pids)
    n_train = max(1, int(n * cfg.train_split))
    n_val = max(1, int(n * cfg.val_split))

    train_pids = pids[:n_train]
    val_pids = pids[n_train:n_train + n_val]
    test_pids = pids[n_train + n_val:]

    df_train = df[df["participant_id"].isin(train_pids)]
    df_val = df[df["participant_id"].isin(val_pids)]
    df_test = df[df["participant_id"].isin(test_pids)]

    logger.info(
        "Split — train: %d pids (%d rows), val: %d pids (%d rows), "
        "test: %d pids (%d rows)",
        len(train_pids), len(df_train),
        len(val_pids), len(df_val),
        len(test_pids), len(df_test),
    )
    return df_train, df_val, df_test


def _df_to_tf_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    cfg: PipelineConfig,
    *,
    shuffle: bool = False,
) -> object:
    """Convert a pandas DataFrame into a batched tf.data.Dataset."""
    tf = _get_tf()

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(cfg.shuffle_buffer, len(df)),
            seed=42,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    ds = ds.prefetch(cfg.prefetch)

    return ds


def build_tf_datasets(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    target: str = "is_injured",
    feature_cols: Optional[List[str]] = None,
) -> DatasetBundle:
    """
    Build ``tf.data.Dataset`` objects for train / val / test.

    The dataset yields ``(features_tensor, label_scalar)`` pairs.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target and c not in ("participant_id",)
        ]

    # Fill any residual NaN (safety net)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0)
    df[target] = df[target].fillna(0)

    df_train, df_val, df_test = _split_by_participant(df, cfg)

    ds_train = _df_to_tf_dataset(df_train, feature_cols, target, cfg, shuffle=True)
    ds_val = _df_to_tf_dataset(df_val, feature_cols, target, cfg)
    ds_test = _df_to_tf_dataset(df_test, feature_cols, target, cfg)

    return DatasetBundle(
        train=ds_train,
        val=ds_val,
        test=ds_test,
        feature_cols=feature_cols,
        target_col=target,
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
    )


# ────────────────────────────────────────────────────────────
# 4. Convenience: full load stage
# ────────────────────────────────────────────────────────────

def load(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: Optional[List[str]] = None,
    target: str = "is_injured",
) -> LoadResult:
    """
    Execute the complete Load stage.

    1. Persist as CSV.
    2. Export TFRecord files (train / val / test).
    3. Build ``tf.data.Dataset`` objects (in‑memory).
    """
    csv_path = save_csv(df, cfg)

    # Resolve feature_cols once so both TFRecord and tf.data use the same set
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target and c not in ("participant_id",)
        ]

    bundle: Optional[DatasetBundle] = None
    tfr_paths: Optional[Dict[str, Path]] = None

    try:
        tfr_paths = save_tfrecord(df, cfg, feature_cols, target=target)
    except ImportError:
        logger.warning("TensorFlow not available – skipping TFRecord export")

    try:
        bundle = build_tf_datasets(df, cfg, target=target, feature_cols=feature_cols)
    except ImportError:
        logger.warning("TensorFlow not available – skipping tf.data creation")

    meta: Dict = {
        "csv_path": str(csv_path),
        "csv_rows": len(df),
        "csv_cols": len(df.columns),
    }
    if bundle is not None:
        meta.update({
            "n_train": bundle.n_train,
            "n_val": bundle.n_val,
            "n_test": bundle.n_test,
            "n_features": len(bundle.feature_cols),
        })
    if tfr_paths is not None:
        meta["tfrecord_paths"] = {k: str(v) for k, v in tfr_paths.items()}

    return LoadResult(
        csv_path=csv_path,
        datasets=bundle,
        tfrecord_paths=tfr_paths,
        metadata=meta,
    )
