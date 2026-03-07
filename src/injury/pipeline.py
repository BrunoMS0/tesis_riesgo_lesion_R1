"""
pipeline.py – Orchestrator for the R5 Injury Risk Prediction pipeline.

Wires together data integration, synthetic augmentation, model training,
evaluation, LOSO cross-validation, and SHAP interpretability.

Public API
----------
run(cfg) -> InjuryReport
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import InjuryConfig
from .dataset import build_injury_datasets, InjuryDatasetBundle
from .augment import generate_synthetic_athletes, validate_synthetic
from .model import build_xgboost, build_random_forest
from .train import compute_scale_pos_weight, train_injury_model, save_model
from .evaluate import (
    EvaluationResult,
    ShapResult,
    compare_models,
    compute_shap_values,
    evaluate_model,
    save_evaluation_report,
    save_shap_plot,
)
from .validate import LOSOResult, loso_cross_validation

logger = logging.getLogger(__name__)


@dataclass
class StageReport:
    name: str
    duration_s: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InjuryReport:
    """Summary of a full R5 pipeline execution."""

    stages: list                     # List[StageReport]
    total_duration_s: float
    n_train: int = 0
    n_train_augmented: int = 0
    n_val: int = 0
    n_test: int = 0
    n_features: int = 0
    xgb_metrics: Dict[str, float] = field(default_factory=dict)
    rf_metrics: Dict[str, float] = field(default_factory=dict)
    loso_pr_auc: float = 0.0
    comparison_table: Optional[pd.DataFrame] = None


def run(cfg: Optional[InjuryConfig] = None) -> InjuryReport:
    """
    Execute the full R5 pipeline:
    Data → Augment → Train (XGB + RF) → Evaluate → LOSO → SHAP → Compare.

    Parameters
    ----------
    cfg : InjuryConfig, optional
        If *None* a default config is used.

    Returns
    -------
    InjuryReport
    """
    if cfg is None:
        cfg = InjuryConfig()

    stages: list = []
    t_global = time.perf_counter()

    # ── STAGE 1: DATA INTEGRATION ────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 1 / 6 — DATA INTEGRATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    bundle: InjuryDatasetBundle = build_injury_datasets(cfg)
    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="DataIntegration",
        duration_s=round(dt, 2),
        details={
            "n_train": len(bundle.X_train),
            "n_val": len(bundle.X_val),
            "n_test": len(bundle.X_test),
            "n_features": len(bundle.feature_columns),
            "train_pids": bundle.train_pids,
            "val_pids": bundle.val_pids,
            "test_pids": bundle.test_pids,
        },
    ))
    logger.info("Data ready in %.2fs — train=%d, val=%d, test=%d, features=%d",
                dt, len(bundle.X_train), len(bundle.X_val),
                len(bundle.X_test), len(bundle.feature_columns))

    # ── STAGE 2: SYNTHETIC DATA AUGMENTATION ─────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 2 / 6 — SYNTHETIC AUGMENTATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    X_aug, y_aug, meta_aug = generate_synthetic_athletes(
        bundle.X_train, bundle.y_train, bundle.meta_train, cfg,
    )

    # Validate synthetic distributions
    n_real = len(bundle.X_train)
    X_synth_only = X_aug.iloc[n_real:]
    ks_results = validate_synthetic(
        bundle.X_train, X_synth_only, bundle.feature_columns,
    )

    dt = time.perf_counter() - t0
    n_ks_pass = sum(1 for p in ks_results.values() if p > 0.05)
    stages.append(StageReport(
        name="SyntheticAugmentation",
        duration_s=round(dt, 2),
        details={
            "n_real_train": n_real,
            "n_augmented_train": len(X_aug),
            "n_synthetic": len(X_aug) - n_real,
            "ks_pass_rate": f"{n_ks_pass}/{len(ks_results)}",
        },
    ))
    logger.info("Augmentation done in %.2fs — %d real + %d synthetic = %d total",
                dt, n_real, len(X_aug) - n_real, len(X_aug))

    # ── STAGE 3: MODEL TRAINING ──────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 3 / 6 — MODEL TRAINING")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    # XGBoost (on augmented data)
    spw = compute_scale_pos_weight(y_aug)
    xgb_model = build_xgboost(cfg, scale_pos_weight=spw)
    xgb_model = train_injury_model(
        xgb_model, X_aug, y_aug, bundle.X_val, bundle.y_val, cfg,
    )
    save_model(xgb_model, cfg.output_path, "xgboost_injury")

    # Random Forest (on augmented data)
    rf_model = build_random_forest(cfg)
    rf_model = train_injury_model(
        rf_model, X_aug, y_aug, bundle.X_val, bundle.y_val, cfg,
    )
    save_model(rf_model, cfg.output_path, "rf_injury")

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Training",
        duration_s=round(dt, 2),
        details={"models_trained": ["XGBoost", "RandomForest"]},
    ))
    logger.info("Training done in %.2fs", dt)

    # ── STAGE 4: EVALUATION ──────────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 4 / 6 — EVALUATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    xgb_eval = evaluate_model(
        xgb_model, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
        model_name="XGBoost",
    )
    save_evaluation_report(xgb_eval, cfg.output_path)

    rf_eval = evaluate_model(
        rf_model, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
        model_name="RandomForest",
    )
    save_evaluation_report(rf_eval, cfg.output_path)

    # XGBoost ablation (without DFI)
    ablation_eval = None
    if "dfi_predicted" in bundle.feature_columns:
        cols_no_dfi = [c for c in bundle.feature_columns if c != "dfi_predicted"]
        X_train_no_dfi = X_aug[cols_no_dfi]
        X_val_no_dfi = bundle.X_val[cols_no_dfi]
        X_test_no_dfi = bundle.X_test[cols_no_dfi]

        xgb_ablation = build_xgboost(cfg, scale_pos_weight=spw)
        xgb_ablation = train_injury_model(
            xgb_ablation, X_train_no_dfi, y_aug,
            X_val_no_dfi, bundle.y_val, cfg,
        )
        ablation_eval = evaluate_model(
            xgb_ablation, X_test_no_dfi, bundle.y_test, bundle.meta_test, cfg,
            model_name="XGBoost_no_DFI",
        )
        save_evaluation_report(ablation_eval, cfg.output_path)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Evaluation",
        duration_s=round(dt, 2),
        details={
            "xgb": xgb_eval.metrics,
            "rf": rf_eval.metrics,
            "ablation": ablation_eval.metrics if ablation_eval else None,
        },
    ))
    logger.info("Evaluation done in %.2fs", dt)

    # ── STAGE 5: LOSO CROSS-VALIDATION ───────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 5 / 6 — LOSO CROSS-VALIDATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    # Use all real data (train + val + test) for LOSO
    X_all = pd.concat([bundle.X_train, bundle.X_val, bundle.X_test],
                       ignore_index=True)
    y_all = pd.concat([bundle.y_train, bundle.y_val, bundle.y_test],
                       ignore_index=True)
    meta_all = pd.concat([bundle.meta_train, bundle.meta_val, bundle.meta_test],
                          ignore_index=True)

    loso_result: LOSOResult = loso_cross_validation(
        X_all, y_all, meta_all, cfg,
    )

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="LOSO_CrossValidation",
        duration_s=round(dt, 2),
        details={
            "n_folds": len(loso_result.folds),
            "mean_pr_auc": loso_result.mean_pr_auc,
            "std_pr_auc": loso_result.std_pr_auc,
            "mean_roc_auc": loso_result.mean_roc_auc,
            "mean_f1": loso_result.mean_f1,
        },
    ))
    logger.info("LOSO done in %.2fs", dt)

    # ── STAGE 6: INTERPRETABILITY (SHAP) ─────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 6 / 6 — SHAP INTERPRETABILITY")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    try:
        shap_result = compute_shap_values(xgb_model, bundle.X_test, cfg)
        save_shap_plot(shap_result, bundle.X_test, cfg.output_path)

        # Save SHAP feature importance
        if shap_result.feature_importance is not None:
            shap_path = Path(cfg.output_path) / "shap_feature_importance.csv"
            shap_result.feature_importance.to_csv(shap_path, index=False)
    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="SHAP_Interpretability",
        duration_s=round(dt, 2),
    ))
    logger.info("SHAP done in %.2fs", dt)

    # ── COMPARISON TABLE ─────────────────────────────────
    comparison_dict = {"XGBoost": xgb_eval, "RandomForest": rf_eval}
    if ablation_eval:
        comparison_dict["XGBoost_no_DFI"] = ablation_eval
    comparison_df = compare_models(comparison_dict)

    # Save comparison
    comp_path = Path(cfg.output_path) / "model_comparison.csv"
    comparison_df.to_csv(comp_path)
    logger.info("Model comparison saved to %s", comp_path)

    total = time.perf_counter() - t_global

    return InjuryReport(
        stages=stages,
        total_duration_s=round(total, 2),
        n_train=len(bundle.X_train),
        n_train_augmented=len(X_aug),
        n_val=len(bundle.X_val),
        n_test=len(bundle.X_test),
        n_features=len(bundle.feature_columns),
        xgb_metrics=xgb_eval.metrics,
        rf_metrics=rf_eval.metrics,
        loso_pr_auc=loso_result.mean_pr_auc,
        comparison_table=comparison_df,
    )
