"""
pipeline.py – Orchestrator for the ETL pipeline  (Extract → Transform → Load).

This module wires together the three stages and provides a single
entry‑point function: :func:`run`.  It also emits structured logging
so every stage is traceable in CI / production runs.

Public API
----------
run(cfg) -> PipelineReport
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .config import PipelineConfig
from .extract import extract_all
from .transform import transform, TransformResult
from .load import load, LoadResult

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Report container
# ────────────────────────────────────────────────────────────

@dataclass
class StageReport:
    name: str
    duration_s: float
    rows_out: int
    cols_out: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    """Summary of an end‑to‑end pipeline execution."""

    stages: list  # List[StageReport]
    total_duration_s: float
    final_csv: Optional[str] = None
    final_features: int = 0
    final_rows: int = 0
    tf_datasets_created: bool = False
    tfrecord_paths: Optional[dict] = None


# ────────────────────────────────────────────────────────────
# Pipeline execution
# ────────────────────────────────────────────────────────────

def run(cfg: Optional[PipelineConfig] = None) -> PipelineReport:
    """
    Execute the full **Extract → Transform → Load** pipeline.

    Parameters
    ----------
    cfg : PipelineConfig, optional
        If *None*, a default config is created.

    Returns
    -------
    PipelineReport
        Metadata about each stage and the overall execution.
    """
    if cfg is None:
        cfg = PipelineConfig()

    stages: list = []
    t_global = time.perf_counter()

    # ── EXTRACT ───────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 1 / 3 — EXTRACT")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    df_raw: pd.DataFrame = extract_all(cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Extract",
        duration_s=round(dt, 2),
        rows_out=len(df_raw),
        cols_out=len(df_raw.columns),
        details={"participants": df_raw["participant_id"].nunique()},
    ))
    logger.info("Extract finished in %.2fs  → %d rows, %d cols",
                dt, len(df_raw), len(df_raw.columns))

    # ── TRANSFORM ─────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 2 / 3 — TRANSFORM")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    tr: TransformResult = transform(df_raw, cfg)
    dt = time.perf_counter() - t0
    df_final = tr.df_standardised   # the fully processed frame
    stages.append(StageReport(
        name="Transform",
        duration_s=round(dt, 2),
        rows_out=len(df_final),
        cols_out=len(df_final.columns),
        details=tr.metadata,
    ))
    logger.info("Transform finished in %.2fs  → %d rows, %d features",
                dt, len(df_final), tr.metadata.get("n_features_final", "?"))

    # Save feature selection report if available
    if tr.selection_report is not None:
        import os
        report_path = os.path.join(cfg.output_path, "feature_selection_report.csv")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        tr.selection_report.to_csv(report_path, index=False)
        logger.info("Feature selection report saved to %s", report_path)

    # ── LOAD ──────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 3 / 3 — LOAD")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    lr: LoadResult = load(
        df_final, cfg,
        feature_cols=tr.feature_cols,
        target="is_injured",
    )
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Load",
        duration_s=round(dt, 2),
        rows_out=lr.metadata.get("csv_rows", len(df_final)),
        cols_out=lr.metadata.get("csv_cols", len(df_final.columns)),
        details=lr.metadata,
    ))
    logger.info("Load finished in %.2fs", dt)

    # ── Report ────────────────────────────────────────────
    total = round(time.perf_counter() - t_global, 2)
    report = PipelineReport(
        stages=stages,
        total_duration_s=total,
        final_csv=str(lr.csv_path),
        final_features=len(tr.feature_cols),
        final_rows=len(df_final),
        tf_datasets_created=lr.datasets is not None,
        tfrecord_paths=(
            {k: str(v) for k, v in lr.tfrecord_paths.items()}
            if lr.tfrecord_paths else None
        ),
    )

    logger.info("═" * 60)
    logger.info("PIPELINE COMPLETED in %.2f s", total)
    logger.info("  CSV  : %s", report.final_csv)
    logger.info("  Rows : %d   Features : %d", report.final_rows,
                report.final_features)
    logger.info("  tf.data created : %s", report.tf_datasets_created)
    logger.info("  TFRecord files  : %s",
                list(report.tfrecord_paths.values()) if report.tfrecord_paths else "N/A")
    logger.info("═" * 60)

    return report
