"""
pipeline.py – Orchestrator for the R5 Injury Risk Prediction pipeline.

Wires together data integration, synthetic augmentation (SMOTE / Copula),
Logistic Regression training, evaluation (AUC-ROC), LOSO cross-validation,
and coefficient-based interpretability.

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

import joblib
import pandas as pd

from .config import InjuryConfig
from .dataset import build_injury_datasets, InjuryDatasetBundle
from .augment import augment_training_data, validate_synthetic
from .model import build_logistic_regression, build_baseline_model
from .train import train_injury_model, save_model, grid_search_C
from .evaluate import (
    EvaluationResult,
    compare_models,
    compute_coefficient_importance,
    evaluate_model,
    save_coefficient_plot,
    save_evaluation_report,
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
    lr_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    loso_roc_auc: float = 0.0
    comparison_table: Optional[pd.DataFrame] = None


def run(cfg: Optional[InjuryConfig] = None) -> InjuryReport:
    """
    Execute the full R5 pipeline:
    Data → Augment → Train (LR + Baseline) → Evaluate → LOSO → Coefficients.

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

    # Save normality reports and fitted normalizer
    if bundle.normalizer is not None:
        out = Path(cfg.output_path)
        bundle.normalizer.pre_report.to_csv(
            out / "normality_pre_ks.csv", index=False,
        )
        bundle.normalizer.post_report.to_csv(
            out / "normality_post_ks.csv", index=False,
        )
        joblib.dump(bundle.normalizer.transformer, out / "normalizer.joblib")

        n_normal_pre = bundle.normalizer.pre_report["is_normal"].sum()
        n_normal_post = bundle.normalizer.post_report["is_normal"].sum()
        n_feats = len(bundle.normalizer.pre_report)
        logger.info(
            "KS normality — pre: %d/%d normal, post: %d/%d normal",
            n_normal_pre, n_feats, n_normal_post, n_feats,
        )

    # ── STAGE 2: SYNTHETIC DATA AUGMENTATION ─────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 2 / 6 — SYNTHETIC AUGMENTATION (%s)",
                cfg.augmentation_method.upper())
    logger.info("═" * 60)
    t0 = time.perf_counter()

    X_aug, y_aug = augment_training_data(
        bundle.X_train, bundle.y_train, bundle.meta_train, cfg,
    )

    dt = time.perf_counter() - t0
    n_real = len(bundle.X_train)
    stages.append(StageReport(
        name="SyntheticAugmentation",
        duration_s=round(dt, 2),
        details={
            "method": cfg.augmentation_method,
            "n_real_train": n_real,
            "n_augmented_train": len(X_aug),
            "n_synthetic": len(X_aug) - n_real,
            "target_ratio": cfg.target_ratio,
            "injury_rate": f"{100 * y_aug.mean():.1f}%",
        },
    ))
    logger.info("Augmentation done in %.2fs — %d real + %d synthetic = %d total "
                "(injury rate %.1f%%)",
                dt, n_real, len(X_aug) - n_real, len(X_aug), 100 * y_aug.mean())

    # ── STAGE 3: MODEL TRAINING ──────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 3 / 6 — MODEL TRAINING (Logistic Regression)")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    # Grid search over C using validation set
    best_C, grid_results = grid_search_C(
        X_aug, y_aug, bundle.X_val, bundle.y_val, cfg,
    )
    grid_results.to_csv(
        Path(cfg.output_path) / "grid_search_C_results.csv", index=False,
    )

    # Build final LR with best C
    from dataclasses import replace as dc_replace
    cfg_best = dc_replace(cfg, lr_C=best_C)
    lr_model = build_logistic_regression(cfg_best)
    lr_model = train_injury_model(lr_model, X_aug, y_aug, cfg_best)
    save_model(lr_model, cfg.output_path, "logistic_injury")

    # Baseline (DummyClassifier)
    baseline = build_baseline_model(cfg)
    baseline = train_injury_model(baseline, X_aug, y_aug, cfg)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Training",
        duration_s=round(dt, 2),
        details={
            "models_trained": ["LogisticRegression", "Baseline"],
            "best_C": best_C,
            "grid_search": grid_results.to_dict(orient="records"),
        },
    ))
    logger.info("Training done in %.2fs — best C=%.4f", dt, best_C)

    # ── STAGE 4: EVALUATION ──────────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 4 / 6 — EVALUATION (AUC-ROC)")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    lr_eval = evaluate_model(
        lr_model, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
        model_name="LogisticRegression",
        X_val=bundle.X_val, y_val=bundle.y_val,
    )
    save_evaluation_report(lr_eval, cfg.output_path)

    baseline_eval = evaluate_model(
        baseline, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
        model_name="Baseline",
    )
    save_evaluation_report(baseline_eval, cfg.output_path)

    # Ablation: LR without DFI
    ablation_eval = None
    if "dfi_predicted" in bundle.feature_columns:
        cols_no_dfi = [c for c in bundle.feature_columns if c != "dfi_predicted"]
        X_aug_no_dfi = X_aug[cols_no_dfi]
        X_test_no_dfi = bundle.X_test[cols_no_dfi]
        X_val_no_dfi = bundle.X_val[cols_no_dfi]

        lr_ablation = build_logistic_regression(cfg)
        lr_ablation = train_injury_model(lr_ablation, X_aug_no_dfi, y_aug, cfg)
        ablation_eval = evaluate_model(
            lr_ablation, X_test_no_dfi, bundle.y_test, bundle.meta_test, cfg,
            model_name="LR_no_DFI",
            X_val=X_val_no_dfi, y_val=bundle.y_val,
        )
        save_evaluation_report(ablation_eval, cfg.output_path)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Evaluation",
        duration_s=round(dt, 2),
        details={
            "lr": lr_eval.metrics,
            "baseline": baseline_eval.metrics,
            "ablation": ablation_eval.metrics if ablation_eval else None,
        },
    ))
    logger.info("Evaluation done in %.2fs", dt)

    # ── STAGE 5: LOSO CROSS-VALIDATION ───────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 5 / 6 — LOSO CROSS-VALIDATION")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    # Use all real data (train + val + test) for LOSO — raw (unnormalized)
    # so each fold can fit its own normalizer independently.
    X_all = pd.concat([bundle.X_train_raw, bundle.X_val_raw, bundle.X_test_raw],
                       ignore_index=True)
    y_all = pd.concat([bundle.y_train, bundle.y_val, bundle.y_test],
                       ignore_index=True)
    meta_all = pd.concat([bundle.meta_train, bundle.meta_val, bundle.meta_test],
                          ignore_index=True)

    loso_result: LOSOResult = loso_cross_validation(
        X_all, y_all, meta_all, cfg_best,
    )

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="LOSO_CrossValidation",
        duration_s=round(dt, 2),
        details={
            "n_folds": len(loso_result.folds),
            "n_skipped_folds": loso_result.n_skipped_folds,
            "mean_roc_auc": loso_result.mean_roc_auc,
            "std_roc_auc": loso_result.std_roc_auc,
            "mean_f1": loso_result.mean_f1,
        },
    ))
    logger.info("LOSO done in %.2fs", dt)

    # ── STAGE 6: INTERPRETABILITY (COEFFICIENTS) ─────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 6 / 6 — COEFFICIENT INTERPRETABILITY")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    try:
        coef_importance = compute_coefficient_importance(
            lr_model, list(bundle.X_test.columns),
        )
        save_coefficient_plot(coef_importance, cfg.output_path)

        coef_path = Path(cfg.output_path) / "coefficient_importance.csv"
        coef_importance.to_csv(coef_path, index=False)
    except Exception as exc:
        logger.warning("Coefficient analysis failed: %s", exc)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="CoefficientInterpretability",
        duration_s=round(dt, 2),
    ))
    logger.info("Coefficient analysis done in %.2fs", dt)

    # ── COMPARISON TABLE ─────────────────────────────────
    comparison_dict = {
        "LogisticRegression": lr_eval,
        "Baseline": baseline_eval,
    }
    if ablation_eval:
        comparison_dict["LR_no_DFI"] = ablation_eval
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
        lr_metrics=lr_eval.metrics,
        baseline_metrics=baseline_eval.metrics,
        loso_roc_auc=loso_result.mean_roc_auc,
        comparison_table=comparison_df,
    )
