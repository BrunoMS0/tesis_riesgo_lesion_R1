"""
pipeline.py – Orchestrator for the R4 Fatigue model pipeline.

Wires together data preparation, model construction, training,
evaluation, and full-dataset prediction into one ``run()`` call.

Public API
----------
run(cfg) -> FatigueReport
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import FatigueConfig
from .dataset import build_fatigue_datasets, FatigueDatasetBundle
from .model import build_fatigue_model
from .train import train_fatigue_model
from .evaluate import evaluate_model, EvaluationResult
from .predict import predict_all

logger = logging.getLogger(__name__)


@dataclass
class StageReport:
    name: str
    duration_s: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FatigueReport:
    """Summary of a full R4 pipeline execution."""

    stages: list                     # List[StageReport]
    total_duration_s: float
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_features: int = 0
    test_metrics: Dict[str, float] = field(default_factory=dict)
    predictions_csv: Optional[str] = None


def run(cfg: Optional[FatigueConfig] = None) -> FatigueReport:
    """
    Execute the full **Data → Train → Evaluate → Predict** pipeline.

    Parameters
    ----------
    cfg : FatigueConfig, optional
        If *None* a default config is used.

    Returns
    -------
    FatigueReport
    """
    if cfg is None:
        cfg = FatigueConfig()

    stages: list = []
    t_global = time.perf_counter()

    # ── DATA PREPARATION ─────────────────────────────────
    logger.info("═" * 60)
    logger.info("R4 STAGE 1 / 4 — DATA PREPARATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    bundle: FatigueDatasetBundle = build_fatigue_datasets(cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="DataPreparation",
        duration_s=round(dt, 2),
        details={
            "n_train": bundle.n_train,
            "n_val": bundle.n_val,
            "n_test": bundle.n_test,
            "n_features": bundle.n_features,
            "window_size": bundle.window_size,
        },
    ))
    logger.info("Data ready in %.2fs – train=%d, val=%d, test=%d, "
                "features=%d, window=%d",
                dt, bundle.n_train, bundle.n_val, bundle.n_test,
                bundle.n_features, bundle.window_size)

    # ── MODEL BUILD ──────────────────────────────────────
    logger.info("═" * 60)
    logger.info("R4 STAGE 2 / 4 — MODEL BUILD & TRAIN")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    model = build_fatigue_model(
        n_features=bundle.n_features,
        window_size=bundle.window_size,
        cfg=cfg,
    )
    history = train_fatigue_model(model, bundle, cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Training",
        duration_s=round(dt, 2),
        details={
            "epochs_run": len(history.history["loss"]),
            "best_val_loss": round(min(history.history["val_loss"]), 6),
        },
    ))
    logger.info("Training finished in %.2fs", dt)

    # ── EVALUATION ───────────────────────────────────────
    logger.info("═" * 60)
    logger.info("R4 STAGE 3 / 4 — EVALUATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    eval_result: EvaluationResult = evaluate_model(model, bundle, cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Evaluation",
        duration_s=round(dt, 2),
        details=eval_result.metrics,
    ))
    logger.info("Evaluation done in %.2fs – %s", dt, eval_result.metrics)

    # ── FULL PREDICTION ──────────────────────────────────
    logger.info("═" * 60)
    logger.info("R4 STAGE 4 / 4 — FULL PREDICTION")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    pred_df = predict_all(model, cfg)
    dt = time.perf_counter() - t0
    from pathlib import Path as _Path
    pred_csv = str(_Path(cfg.output_path) / "fatigue_index_predictions.csv")
    stages.append(StageReport(
        name="Prediction",
        duration_s=round(dt, 2),
        details={"rows": len(pred_df)},
    ))
    logger.info("Prediction done in %.2fs – %d rows", dt, len(pred_df))

    total = time.perf_counter() - t_global

    return FatigueReport(
        stages=stages,
        total_duration_s=round(total, 2),
        n_train=bundle.n_train,
        n_val=bundle.n_val,
        n_test=bundle.n_test,
        n_features=bundle.n_features,
        test_metrics=eval_result.metrics,
        predictions_csv=pred_csv,
    )
