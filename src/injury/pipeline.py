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

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .config import InjuryConfig, SOCCERMON_SHARED_FEATURES
from .dataset import build_injury_datasets, InjuryDatasetBundle, build_combined_dataset
from .augment import augment_training_data, validate_synthetic
from .model import build_logistic_regression, build_random_forest, build_model, build_baseline_model
from .train import train_injury_model, save_model, grid_search_C, grid_search_RF
from .evaluate import (
    EvaluationResult,
    compare_models,
    compute_coefficient_importance,
    compute_feature_importance,
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
    soccermon_metrics: Dict[str, float] = field(default_factory=dict)
    soccermon_metrics_37z: Dict[str, float] = field(default_factory=dict)
    soccermon_metrics_11: Dict[str, float] = field(default_factory=dict)
    ablation_wellness_auc: float = 0.0
    delta_auc_wearables: float = 0.0


def _evaluate_on_soccermon(
    model,
    bundle: "InjuryDatasetBundle",
    cfg: InjuryConfig,
    model_name: str = "Model",
    *,
    use_zscore_csv: bool = False,
    feature_cols: Optional[list] = None,
    apply_yj: bool = True,
    target_col: Optional[str] = None,
) -> "EvaluationResult":
    """
    Load a SoccerMon CSV and evaluate ``model`` on it.

    Parameters
    ----------
    model         : Trained sklearn estimator.
    bundle        : InjuryDatasetBundle from the matching training run (supplies
                    the fitted Yeo-Johnson transformer and training medians).
    cfg           : InjuryConfig.
    model_name    : Label used in log messages and saved reports.
    use_zscore_csv: If True, load ``cfg.soccermon_zscore_csv`` (per-athlete
                    z-scored); otherwise load ``cfg.soccermon_csv`` (raw).
    feature_cols  : Subset of features to use.  Defaults to ``cfg.feature_columns``.
    apply_yj      : Whether to apply the PMData-trained Yeo-Johnson transformer.
                    Should be False when the data is already z-scored (RF-37-z,
                    RF-11) because the distribution no longer matches the
                    PMData distribution the transformer was fitted on.
    target_col    : Label column to evaluate against.  If None, uses
                    ``injury_next{cfg.prospective_window}d`` when the
                    prospective CSV is available, else falls back to
                    ``is_injured``.  Pass ``"is_injured"`` explicitly to
                    force the legacy label regardless.
    """
    # -- Choose CSV path ---------------------------------------------------
    # When evaluating with a prospective target, prefer the prospective CSV
    # (which contains injury_next{n}d) over the raw/zscore CSVs.
    prospective_col = f"injury_next{cfg.prospective_window}d"
    if target_col is None:
        # Auto-detect: use prospective target when model was trained on it
        target_col = prospective_col if cfg.use_prospective_target else "is_injured"

    use_prospective_csv = (
        target_col == prospective_col
        and not use_zscore_csv          # zscore CSV has the same columns
        and Path(cfg.soccermon_prospective_csv).exists()
    )
    if use_prospective_csv:
        csv_path = cfg.soccermon_prospective_csv
    else:
        csv_path = cfg.soccermon_zscore_csv if use_zscore_csv else cfg.soccermon_csv
    sm_df = pd.read_csv(csv_path)

    cols = feature_cols if feature_cols is not None else list(cfg.feature_columns)
    # Keep only columns present in the CSV; fill residual NaN with training medians
    cols_available = [c for c in cols if c in sm_df.columns]
    X_sm = sm_df[cols_available].copy()
    for col in X_sm.columns:
        if X_sm[col].isna().any():
            fill_val = bundle.X_train[col].median() if col in bundle.X_train.columns else 0.0
            X_sm[col] = X_sm[col].fillna(fill_val)

    # Use the chosen target; fall back to is_injured if column is absent
    if target_col not in sm_df.columns:
        logger.warning(
            "_evaluate_on_soccermon: column '%s' not in %s — falling back to is_injured",
            target_col, Path(csv_path).name,
        )
        target_col = "is_injured"
    y_sm = sm_df[target_col].astype(int)
    meta_sm = sm_df[["participant_id", "date"]].copy()

    # Apply the PMData-trained Yeo-Johnson normalizer (only for raw CSV / RF-37 baseline)
    if apply_yj and bundle.normalizer is not None:
        transformer = bundle.normalizer.transformer
        feature_names = list(bundle.X_train.columns)
        cols_to_transform = [c for c in X_sm.columns if c in feature_names]
        if cols_to_transform:
            X_arr = transformer.transform(X_sm[cols_to_transform])
            X_sm[cols_to_transform] = X_arr

    result = evaluate_model(
        model, X_sm, y_sm, meta_sm, cfg,
        model_name=f"{model_name}_SoccerMon",
    )
    logger.info(
        "[SoccerMon %s] ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  "
        "(n=%d, injuries=%d, prevalence=%.1f%%, csv=%s, target=%s)",
        model_name,
        result.metrics.get("roc_auc", 0),
        result.metrics.get("pr_auc", 0),
        result.metrics.get("f1", 0),
        len(y_sm),
        int(y_sm.sum()),
        100.0 * y_sm.mean(),
        "zscore" if use_zscore_csv else ("prospective" if use_prospective_csv else "raw"),
        target_col,
    )
    return result


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
    model_label = "Random Forest" if cfg.model_type == "rf" else "Logistic Regression"
    logger.info("═" * 60)
    logger.info("R5 STAGE 3 / 6 — MODEL TRAINING (%s)", model_label)
    logger.info("═" * 60)
    t0 = time.perf_counter()

    from dataclasses import replace as dc_replace

    if cfg.model_type == "rf":
        # Grid search over RF hyper-parameter grid
        best_params, grid_results = grid_search_RF(
            X_aug, y_aug, bundle.X_val, bundle.y_val, cfg,
        )
        grid_results.to_csv(
            Path(cfg.output_path) / "grid_search_RF_results.csv", index=False,
        )
        main_model = build_random_forest(cfg, **best_params)
        main_model = train_injury_model(main_model, X_aug, y_aug, cfg)
        save_model(main_model, cfg.output_path, "rf_injury")
        best_param_info = best_params
        model_name_eval = "RandomForest"
    else:
        # Grid search over C using validation set
        best_C, grid_results = grid_search_C(
            X_aug, y_aug, bundle.X_val, bundle.y_val, cfg,
        )
        grid_results.to_csv(
            Path(cfg.output_path) / "grid_search_C_results.csv", index=False,
        )
        cfg_best = dc_replace(cfg, lr_C=best_C)
        main_model = build_logistic_regression(cfg_best)
        main_model = train_injury_model(main_model, X_aug, y_aug, cfg_best)
        save_model(main_model, cfg.output_path, "logistic_injury")
        best_param_info = {"best_C": best_C}
        model_name_eval = "LogisticRegression"

    # Always train LR for interpretability comparison
    lr_model = build_logistic_regression(cfg)
    lr_model = train_injury_model(lr_model, X_aug, y_aug, cfg)

    # Baseline (DummyClassifier)
    baseline = build_baseline_model(cfg)
    baseline = train_injury_model(baseline, X_aug, y_aug, cfg)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Training",
        duration_s=round(dt, 2),
        details={
            "model_type": cfg.model_type,
            "models_trained": [model_name_eval, "LogisticRegression", "Baseline"],
            "best_params": best_param_info,
            "grid_search": grid_results.to_dict(orient="records"),
        },
    ))
    logger.info("Training done in %.2fs — best params=%s", dt, best_param_info)

    # ── STAGE 3b: CROSS-DOMAIN MODELS (RF-37-z and RF-11) ─
    logger.info("═" * 60)
    logger.info("R5 STAGE 3b — CROSS-DOMAIN MODEL TRAINING")
    logger.info("  RF-37-z : 37 features + per-athlete z-score (eliminates load/scale shift)")
    logger.info("  RF-11   : 11 shared features + z-score (wellness/load only, no wearables)")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    rf_37z = None
    rf_11 = None
    bundle_z = None
    bundle_11 = None
    cfg_z = None
    cfg_11 = None

    try:
        # ── RF-37-z ──────────────────────────────────────
        cfg_z = dc_replace(cfg, use_per_athlete_zscore=True)
        bundle_z = build_injury_datasets(cfg_z)
        X_aug_z, y_aug_z = augment_training_data(
            bundle_z.X_train, bundle_z.y_train, bundle_z.meta_train, cfg_z,
        )
        best_params_z, _ = grid_search_RF(
            X_aug_z, y_aug_z, bundle_z.X_val, bundle_z.y_val, cfg_z,
        )
        rf_37z = build_random_forest(cfg_z, **best_params_z)
        rf_37z = train_injury_model(rf_37z, X_aug_z, y_aug_z, cfg_z)
        save_model(rf_37z, cfg.output_path, "rf_37z_injury")
        logger.info("RF-37-z trained (best_params=%s)", best_params_z)

        # ── RF-11 ─────────────────────────────────────────
        cfg_11 = dc_replace(
            cfg_z,
            feature_columns=list(SOCCERMON_SHARED_FEATURES),
        )
        bundle_11 = build_injury_datasets(cfg_11)
        X_aug_11, y_aug_11 = augment_training_data(
            bundle_11.X_train, bundle_11.y_train, bundle_11.meta_train, cfg_11,
        )
        best_params_11, _ = grid_search_RF(
            X_aug_11, y_aug_11, bundle_11.X_val, bundle_11.y_val, cfg_11,
        )
        rf_11 = build_random_forest(cfg_11, **best_params_11)
        rf_11 = train_injury_model(rf_11, X_aug_11, y_aug_11, cfg_11)
        save_model(rf_11, cfg.output_path, "rf_11_injury")
        logger.info("RF-11 trained (best_params=%s)", best_params_11)

    except Exception as exc:
        logger.warning("Stage 3b failed — cross-domain models unavailable: %s", exc)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="CrossDomainTraining",
        duration_s=round(dt, 2),
        details={
            "rf_37z_trained": rf_37z is not None,
            "rf_11_trained": rf_11 is not None,
        },
    ))
    logger.info("Stage 3b done in %.2fs", dt)
    # Release large intermediate objects before the memory-intensive LOAO stage
    if bundle_z is not None:
        del bundle_z
    if bundle_11 is not None:
        del bundle_11
    gc.collect()
    # Windows heap compaction — prevents fragmentation from large RF allocations
    try:
        import ctypes
        ctypes.windll.kernel32.HeapCompact(
            ctypes.windll.kernel32.GetProcessHeap(), 0
        )
    except Exception:
        pass
    bundle_z = None
    bundle_11 = None
    # Note: rf_37z and rf_11 are kept until after Stage 7 (SoccerMon evaluation)

    # ── STAGE 4: EVALUATION ──────────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 4 / 6 — EVALUATION (AUC-ROC)")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    has_pmdata_test = len(bundle.X_test) > 0

    if has_pmdata_test:
        # Evaluate the primary model (RF or LR depending on cfg.model_type)
        main_eval = evaluate_model(
            main_model, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
            model_name=model_name_eval,
            X_val=bundle.X_val, y_val=bundle.y_val,
        )
        save_evaluation_report(main_eval, cfg.output_path)

        # Always compare against LR (may be same model if model_type='lr')
        if cfg.model_type == "rf":
            lr_eval = evaluate_model(
                lr_model, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
                model_name="LogisticRegression",
                X_val=bundle.X_val, y_val=bundle.y_val,
            )
            save_evaluation_report(lr_eval, cfg.output_path)
        else:
            lr_eval = main_eval  # same model, avoid double eval

        baseline_eval = evaluate_model(
            baseline, bundle.X_test, bundle.y_test, bundle.meta_test, cfg,
            model_name="Baseline",
        )
        save_evaluation_report(baseline_eval, cfg.output_path)
    else:
        # No PMData test — evaluate on validation set as proxy for stage 4 logs
        logger.info("PMData test set is empty — evaluating on validation set as proxy")
        main_eval = evaluate_model(
            main_model, bundle.X_val, bundle.y_val, bundle.meta_val, cfg,
            model_name=f"{model_name_eval}_val",
            X_val=bundle.X_val, y_val=bundle.y_val,
        )
        save_evaluation_report(main_eval, cfg.output_path)
        lr_eval = evaluate_model(
            lr_model, bundle.X_val, bundle.y_val, bundle.meta_val, cfg,
            model_name="LogisticRegression_val",
        )
        save_evaluation_report(lr_eval, cfg.output_path)
        baseline_eval = evaluate_model(
            baseline, bundle.X_val, bundle.y_val, bundle.meta_val, cfg,
            model_name="Baseline_val",
        )
        save_evaluation_report(baseline_eval, cfg.output_path)

    # Ablation: main model without DFI
    ablation_eval = None
    if has_pmdata_test and "dfi_predicted" in bundle.feature_columns:
        cols_no_dfi = [c for c in bundle.feature_columns if c != "dfi_predicted"]
        X_aug_no_dfi = X_aug[cols_no_dfi]
        X_test_no_dfi = bundle.X_test[cols_no_dfi]
        X_val_no_dfi = bundle.X_val[cols_no_dfi]

        ablation_model = build_model(cfg, **best_param_info) if cfg.model_type == "rf" \
            else build_logistic_regression(cfg)
        ablation_model = train_injury_model(ablation_model, X_aug_no_dfi, y_aug, cfg)
        ablation_name = f"{model_name_eval}_no_DFI"
        ablation_eval = evaluate_model(
            ablation_model, X_test_no_dfi, bundle.y_test, bundle.meta_test, cfg,
            model_name=ablation_name,
            X_val=X_val_no_dfi, y_val=bundle.y_val,
        )
        save_evaluation_report(ablation_eval, cfg.output_path)

    # Ablation: wearable contribution — RF-37-z (all features) vs RF-11 (wellness only)
    # ∆AUC = RF-37-z_val_AUC − RF-11_val_AUC quantifies what wearables add beyond
    # wellness + load monitoring alone.
    ablation_wellness_auc = 0.0
    delta_auc_wearables = 0.0
    if rf_37z is not None and rf_11 is not None and bundle_z is not None and bundle_11 is not None:
        try:
            # Evaluate both models on their respective validation sets
            eval_37z_val = evaluate_model(
                rf_37z, bundle_z.X_val, bundle_z.y_val, bundle_z.meta_val, cfg_z,
                model_name="RF37z_val",
            )
            eval_11_val = evaluate_model(
                rf_11, bundle_11.X_val, bundle_11.y_val, bundle_11.meta_val, cfg_11,
                model_name="RF11_wellness_val",
            )
            save_evaluation_report(eval_37z_val, cfg.output_path)
            save_evaluation_report(eval_11_val, cfg.output_path)

            auc_37z = eval_37z_val.metrics.get("roc_auc", 0.0)
            ablation_wellness_auc = eval_11_val.metrics.get("roc_auc", 0.0)
            delta_auc_wearables = auc_37z - ablation_wellness_auc

            # Save ablation comparison table
            ablation_df = pd.DataFrame([
                {"model": "RF-37-z (full features + z-score)",
                 "n_features": len(SOCCERMON_SHARED_FEATURES) + (len(bundle_z.feature_columns) - len(SOCCERMON_SHARED_FEATURES)),
                 "val_auc": round(auc_37z, 4),
                 "note": "wellness + load + wearables (z-scored)"},
                {"model": "RF-11 (wellness + load only)",
                 "n_features": len(SOCCERMON_SHARED_FEATURES),
                 "val_auc": round(ablation_wellness_auc, 4),
                 "note": "no wearables — shared features only"},
                {"model": "∆AUC (wearable contribution)",
                 "n_features": len(bundle_z.feature_columns) - len(SOCCERMON_SHARED_FEATURES),
                 "val_auc": round(delta_auc_wearables, 4),
                 "note": "RF-37-z minus RF-11"},
            ])
            ablation_df.to_csv(
                Path(cfg.output_path) / "ablation_wellness_vs_wearables.csv",
                index=False,
            )
            logger.info(
                "Wearable ablation — RF-37-z val AUC=%.4f | RF-11 val AUC=%.4f | "
                "∆AUC (wearables) = %.4f",
                auc_37z, ablation_wellness_auc, delta_auc_wearables,
            )
        except Exception as exc:
            logger.warning("Wearable ablation failed: %s", exc)

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="Evaluation",
        duration_s=round(dt, 2),
        details={
            model_name_eval: main_eval.metrics,
            "LogisticRegression": lr_eval.metrics,
            "baseline": baseline_eval.metrics,
            "ablation": ablation_eval.metrics if ablation_eval else None,
            "ablation_wellness_auc": ablation_wellness_auc,
            "delta_auc_wearables": delta_auc_wearables,
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
        X_all, y_all, meta_all, cfg,
    )

    # Save per-fold and aggregate LOSO results to CSV
    loao_fold_rows = [
        {
            "fold_athlete": f.participant_id,
            "n_samples": f.n_samples,
            "n_pos_test": f.n_injuries,
            "roc_auc": f.roc_auc,
            "pr_auc": f.pr_auc,
            "f1": f.f1,
            "skipped": f.skipped,
            "fpr": f.fpr,
        }
        for f in loso_result.folds
    ]
    # Append summary row
    loao_fold_rows.append({
        "fold_athlete": "MEAN±STD",
        "n_samples": sum(f.n_samples for f in loso_result.folds),
        "n_pos_test": sum(f.n_injuries for f in loso_result.folds),
        "roc_auc": f"{loso_result.mean_roc_auc:.4f} \u00b1 {loso_result.std_roc_auc:.4f}",
        "pr_auc": f"{loso_result.mean_pr_auc:.4f} \u00b1 {loso_result.std_pr_auc:.4f}",
        "f1": f"{loso_result.mean_f1:.4f} \u00b1 {loso_result.std_f1:.4f}",
        "skipped": loso_result.n_skipped_folds,
        "fpr": None,
    })
    loao_df = pd.DataFrame(loao_fold_rows)
    loao_path = Path(cfg.output_path) / "loao_results.csv"
    loao_df.to_csv(loao_path, index=False)
    logger.info(
        "LOAO results saved to %s  (AUC=%.4f\u00b1%.4f, PR-AUC=%.4f\u00b1%.4f, n_folds=%d, n_skipped=%d)",
        loao_path,
        loso_result.mean_roc_auc, loso_result.std_roc_auc,
        loso_result.mean_pr_auc, loso_result.std_pr_auc,
        len(loso_result.folds), loso_result.n_skipped_folds,
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
            "mean_pr_auc": loso_result.mean_pr_auc,
            "std_pr_auc": loso_result.std_pr_auc,
            "mean_f1": loso_result.mean_f1,
            "std_f1": loso_result.std_f1,
            "loao_csv": str(loao_path),
        },
    ))
    logger.info("LOSO done in %.2fs", dt)

    # ── STAGE 6: INTERPRETABILITY ─────────────────────────
    logger.info("═" * 60)
    logger.info("R5 STAGE 6 / 6 — FEATURE IMPORTANCE / INTERPRETABILITY")
    logger.info("═" * 60)
    t0 = time.perf_counter()

    try:
        importance_df = compute_feature_importance(
            main_model, list(bundle.X_test.columns),
        )
        title = ("Random Forest – Feature Importance (Top %d)"
                 if cfg.model_type == "rf"
                 else "Logistic Regression – Feature Coefficients (Top %d)")
        save_coefficient_plot(importance_df, cfg.output_path, title=title)
        importance_path = Path(cfg.output_path) / "coefficient_importance.csv"
        importance_df.to_csv(importance_path, index=False)
    except Exception as exc:
        logger.warning("Feature importance analysis failed: %s", exc)
    finally:
        # Free large model/data objects before memory-intensive subsequent stages
        gc.collect()
        try:
            import ctypes
            ctypes.windll.kernel32.HeapCompact(
                ctypes.windll.kernel32.GetProcessHeap(), 0
            )
        except Exception:
            pass

    dt = time.perf_counter() - t0
    stages.append(StageReport(
        name="CoefficientInterpretability",
        duration_s=round(dt, 2),
    ))
    logger.info("Coefficient analysis done in %.2fs", dt)

    # ── COMPARISON TABLE ─────────────────────────────────
    comparison_dict = {
        model_name_eval: main_eval,
        "Baseline": baseline_eval,
    }
    if cfg.model_type == "rf":
        comparison_dict["LogisticRegression"] = lr_eval
    if ablation_eval:
        comparison_dict[ablation_eval.model_name] = ablation_eval
    comparison_df = compare_models(comparison_dict)

    # Save comparison
    comp_path = Path(cfg.output_path) / "model_comparison.csv"
    comparison_df.to_csv(comp_path)
    logger.info("Model comparison saved to %s", comp_path)

    soccermon_eval = None

    # ── STAGE 4b: COMBINED PMData+SoccerMon RF-11 ──────────
    combined_loao_auc: float = 0.0
    combined_loao_std: float = 0.0
    combined_loao_pr_auc: float = 0.0
    rf_11_combined = None
    bundle_combined = None

    prospective_path = Path(cfg.soccermon_prospective_csv)
    loao_combined_path = Path(cfg.output_path) / "loao_combined_results.csv"
    # Skip Stage 4b if results already exist (it takes ~5 min and is expensive)
    _skip_combined = loao_combined_path.exists() and not cfg.rerun_combined
    if _skip_combined:
        logger.info(
            "Stage 4b skipped — loao_combined_results.csv already exists. "
            "Pass rerun_combined=True to force re-run."
        )
    if cfg.use_combined_training and prospective_path.exists() and not _skip_combined:
        logger.info("="*60)
        logger.info("R5 STAGE 4b — COMBINED PMData+SoccerMon RF-11 TRAINING")
        logger.info("  Train : PMData (%d athletes) + z-score per-athlete", len(cfg.train_participants))
        logger.info("  Test  : SoccerMon (%s)", prospective_path.name)
        logger.info("  Target: injury_next%dd (prospective)", cfg.prospective_window)
        logger.info("  LOAO  : all 66 athletes pooled", )
        logger.info("="*60)
        t0 = time.perf_counter()
        try:
            bundle_combined = build_combined_dataset(cfg)
            # Augment combined training split
            cfg_11c = dc_replace(
                cfg,
                feature_columns=list(SOCCERMON_SHARED_FEATURES) + ["source_dataset"],
                use_per_athlete_zscore=False,  # already z-scored in build_combined_dataset
            )
            X_aug_comb, y_aug_comb = augment_training_data(
                bundle_combined.X_train, bundle_combined.y_train,
                bundle_combined.meta_train, cfg_11c,
            )
            best_params_comb, _ = grid_search_RF(
                X_aug_comb, y_aug_comb,
                bundle_combined.X_val, bundle_combined.y_val, cfg_11c,
            )
            rf_11_combined = build_random_forest(cfg_11c, **best_params_comb)
            rf_11_combined = train_injury_model(rf_11_combined, X_aug_comb, y_aug_comb, cfg_11c)
            save_model(rf_11_combined, cfg.output_path, "rf_11_combined")

            # LOAO over all 66 athletes (PMData + SoccerMon pooled)
            X_loao_comb = pd.concat(
                [bundle_combined.X_train_raw,
                 bundle_combined.X_val_raw,
                 bundle_combined.X_test_raw],
                ignore_index=True,
            )
            y_loao_comb = pd.concat(
                [bundle_combined.y_train,
                 bundle_combined.y_val,
                 bundle_combined.y_test],
                ignore_index=True,
            )
            meta_loao_comb = pd.concat(
                [bundle_combined.meta_train,
                 bundle_combined.meta_val,
                 bundle_combined.meta_test],
                ignore_index=True,
            )

            loao_combined_result = loso_cross_validation(
                X_loao_comb, y_loao_comb, meta_loao_comb, cfg_11c,
                use_augmentation=True,
            )
            combined_loao_auc = loao_combined_result.mean_roc_auc
            combined_loao_std = loao_combined_result.std_roc_auc
            combined_loao_pr_auc = loao_combined_result.mean_pr_auc

            # Save per-fold results
            loao_comb_rows = [
                {
                    "fold_athlete": f.participant_id,
                    "dataset": "SoccerMon" if str(f.participant_id).startswith("sm_") else "PMData",
                    "n_samples": f.n_samples,
                    "n_pos_test": f.n_injuries,
                    "roc_auc": f.roc_auc,
                    "pr_auc": f.pr_auc,
                    "f1": f.f1,
                    "skipped": f.skipped,
                }
                for f in loao_combined_result.folds
            ]
            loao_comb_rows.append({
                "fold_athlete": "MEAN±STD",
                "dataset": "ALL",
                "n_samples": sum(f.n_samples for f in loao_combined_result.folds),
                "n_pos_test": sum(f.n_injuries for f in loao_combined_result.folds),
                "roc_auc": f"{loao_combined_result.mean_roc_auc:.4f} ± {loao_combined_result.std_roc_auc:.4f}",
                "pr_auc": f"{loao_combined_result.mean_pr_auc:.4f} ± {loao_combined_result.std_pr_auc:.4f}",
                "f1": f"{loao_combined_result.mean_f1:.4f} ± {loao_combined_result.std_f1:.4f}",
                "skipped": loao_combined_result.n_skipped_folds,
            })
            pd.DataFrame(loao_comb_rows).to_csv(
                Path(cfg.output_path) / "loao_combined_results.csv", index=False,
            )
            logger.info(
                "Combined LOAO (%d athletes, target=%s): AUC=%.4f±%.4f, "
                "PR-AUC=%.4f±%.4f, n_valid=%d, n_skipped=%d",
                meta_loao_comb["participant_id"].nunique(),
                f"injury_next{cfg.prospective_window}d",
                combined_loao_auc, combined_loao_std,
                loao_combined_result.mean_pr_auc, loao_combined_result.std_pr_auc,
                len([f for f in loao_combined_result.folds if not f.skipped]),
                loao_combined_result.n_skipped_folds,
            )

            dt = time.perf_counter() - t0
            stages.append(StageReport(
                name="CombinedTraining",
                duration_s=round(dt, 2),
                details={
                    "n_train": len(bundle_combined.X_train),
                    "n_test_soccermon": len(bundle_combined.X_test),
                    "n_features": len(bundle_combined.feature_columns),
                    "loao_athletes": meta_loao_comb["participant_id"].nunique(),
                    "loao_roc_auc": combined_loao_auc,
                    "loao_std_roc_auc": combined_loao_std,
                    "loao_pr_auc": combined_loao_pr_auc,
                },
            ))
        except Exception as exc:
            logger.warning("Stage 4b (combined training) failed: %s", exc, exc_info=True)
        finally:
            # Explicitly free combined bundle memory before LSTM stage
            bundle_combined = None
            rf_11_combined = None
            gc.collect()
    else:
        if not cfg.use_combined_training:
            logger.info("Stage 4b skipped (use_combined_training=False)")
        elif _skip_combined:
            pass  # already logged above (loao_combined_results.csv exists)
        else:
            logger.info(
                "Stage 4b skipped — SoccerMon prospective CSV not found at %s. "
                "Run python run_soccermon.py first.", prospective_path,
            )

    # ── STAGE 5: LSTM TEMPORAL MODEL ─────────────────────────
    lstm_loao_result = None
    if cfg.use_lstm:
        logger.info("=" * 60)
        logger.info("R5 STAGE 5 — LSTM TEMPORAL MODEL (window=%dd)", cfg.lstm_window_size)
        logger.info("  Features  : %d (%s)", len(cfg.lstm_feature_cols),
                    "shared" if cfg.lstm_use_shared_features else "all")
        logger.info("  Target    : injury_next%dd", cfg.prospective_window)
        logger.info("  LOAO      : all athletes in PMData train+val")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        try:
            from .lstm import loao_lstm, make_sequences, build_lstm_model, train_lstm

            # Build the full PMData DataFrame (pre-normalisation) for LOAO
            from .dataset import load_and_merge, create_prospective_target, apply_per_athlete_zscore

            df_lstm_all = load_and_merge(cfg)
            df_lstm_all = create_prospective_target(df_lstm_all, cfg.prospective_window)
            lstm_target = f"injury_next{cfg.prospective_window}d"

            # Per-athlete z-score on LSTM features (same as combined RF-11)
            meta_lstm = df_lstm_all[["participant_id", "date"]].copy()
            df_lstm_all[cfg.lstm_feature_cols] = apply_per_athlete_zscore(
                df_lstm_all[cfg.lstm_feature_cols].copy(),
                meta_lstm,
                cfg.lstm_feature_cols,
            )

            # Only use PMData train+val participants for LOAO
            all_lstm_pids = cfg.train_participants + cfg.val_participants
            df_lstm_all = df_lstm_all[
                df_lstm_all["participant_id"].isin(all_lstm_pids)
            ].copy()

            lstm_loao_result = loao_lstm(
                df_all=df_lstm_all,
                feature_cols=cfg.lstm_feature_cols,
                target_col=lstm_target,
                window_size=cfg.lstm_window_size,
                athlete_col="participant_id",
                lstm_units=cfg.lstm_units,
                lstm_units_2=cfg.lstm_units_2,
                dense_units=cfg.lstm_dense_units,
                dropout=cfg.lstm_dropout,
                epochs=cfg.lstm_epochs,
                batch_size=cfg.lstm_batch_size,
                patience=cfg.lstm_patience,
                seed=cfg.seed,
            )

            # Save LOAO results CSV
            lstm_rows = [
                {
                    "fold_athlete": f.participant_id,
                    "n_sequences": f.n_samples,
                    "n_pos_sequences": f.n_injuries,
                    "roc_auc": f.roc_auc,
                    "pr_auc": f.pr_auc,
                    "f1": f.f1,
                    "skipped": f.skipped,
                }
                for f in lstm_loao_result.folds
            ]
            lstm_rows.append({
                "fold_athlete": "MEAN±STD",
                "n_sequences": sum(f.n_samples for f in lstm_loao_result.folds),
                "n_pos_sequences": sum(f.n_injuries for f in lstm_loao_result.folds),
                "roc_auc": f"{lstm_loao_result.mean_roc_auc:.4f} ± {lstm_loao_result.std_roc_auc:.4f}",
                "pr_auc": f"{lstm_loao_result.mean_pr_auc:.4f} ± {lstm_loao_result.std_pr_auc:.4f}",
                "f1": f"{lstm_loao_result.mean_f1:.4f} ± {lstm_loao_result.std_f1:.4f}",
                "skipped": lstm_loao_result.n_skipped_folds,
            })
            pd.DataFrame(lstm_rows).to_csv(
                Path(cfg.output_path) / "loao_lstm_results.csv", index=False,
            )

            dt = time.perf_counter() - t0
            logger.info(
                "LSTM LOAO (%d athletes, window=%d, target=%s): "
                "AUC=%.4f±%.4f, PR-AUC=%.4f±%.4f, n_valid=%d, n_skipped=%d",
                len(all_lstm_pids),
                cfg.lstm_window_size,
                lstm_target,
                lstm_loao_result.mean_roc_auc,
                lstm_loao_result.std_roc_auc,
                lstm_loao_result.mean_pr_auc,
                lstm_loao_result.std_pr_auc,
                len([f for f in lstm_loao_result.folds if not f.skipped]),
                lstm_loao_result.n_skipped_folds,
            )
            stages.append(StageReport(
                name="LSTM_LOAO",
                duration_s=round(dt, 2),
                details={
                    "window_size": cfg.lstm_window_size,
                    "n_features": len(cfg.lstm_feature_cols),
                    "loao_roc_auc": lstm_loao_result.mean_roc_auc,
                    "loao_std_roc_auc": lstm_loao_result.std_roc_auc,
                    "loao_pr_auc": lstm_loao_result.mean_pr_auc,
                    "n_valid_folds": len([f for f in lstm_loao_result.folds if not f.skipped]),
                    "n_skipped_folds": lstm_loao_result.n_skipped_folds,
                },
            ))
        except Exception as exc:
            logger.warning("Stage 5 (LSTM) failed: %s", exc, exc_info=True)
    else:
        logger.info("Stage 5 (LSTM) skipped (use_lstm=False)")

    # ── STAGE 7: SOCCERMON EXTERNAL TEST EVALUATION ──────────
    soccermon_path = Path(cfg.soccermon_csv)
    if soccermon_path.exists():
        logger.info("═" * 60)
        logger.info("R5 STAGE 7 / 7 — SOCCERMON EXTERNAL TEST EVALUATION")
        logger.info("═" * 60)
        t0 = time.perf_counter()
        soccermon_eval_37z = None
        soccermon_eval_11 = None
        try:
            # Model 1: RF-37 original (raw CSV + Yeo-Johnson) — backward-compatible baseline
            soccermon_eval = _evaluate_on_soccermon(
                main_model, bundle, cfg,
                model_name=f"{model_name_eval}_RF37_original",
                use_zscore_csv=False,
                apply_yj=True,
            )
            save_evaluation_report(soccermon_eval, cfg.output_path)

            # Model 2: RF-37-z (z-scored CSV, no Yeo-Johnson)
            zscore_path = Path(cfg.soccermon_zscore_csv)
            # Lazily rebuild bundle_z if it was freed during LOSO (memory already available here)
            if rf_37z is not None and bundle_z is None and cfg_z is not None:
                logger.info("Stage 7: rebuilding bundle_z for RF-37z evaluation...")
                bundle_z = build_injury_datasets(cfg_z)
            if rf_37z is not None and bundle_z is not None and zscore_path.exists():
                soccermon_eval_37z = _evaluate_on_soccermon(
                    rf_37z, bundle_z, cfg_z,
                    model_name="RF37z",
                    use_zscore_csv=True,
                    apply_yj=False,
                )
                save_evaluation_report(soccermon_eval_37z, cfg.output_path)

            # Model 3: RF-11 (z-scored CSV, 11 shared features only, no wearables)
            # Lazily rebuild bundle_11 if it was freed during LOSO
            if rf_11 is not None and bundle_11 is None and cfg_11 is not None:
                logger.info("Stage 7: rebuilding bundle_11 for RF-11 evaluation...")
                bundle_11 = build_injury_datasets(cfg_11)
            if rf_11 is not None and bundle_11 is not None and zscore_path.exists():
                soccermon_eval_11 = _evaluate_on_soccermon(
                    rf_11, bundle_11, cfg_11,
                    model_name="RF11_wellness",
                    use_zscore_csv=True,
                    apply_yj=False,
                    feature_cols=list(SOCCERMON_SHARED_FEATURES),
                )
                save_evaluation_report(soccermon_eval_11, cfg.output_path)

            dt = time.perf_counter() - t0

            auc_37  = soccermon_eval.metrics.get("roc_auc", float("nan"))
            auc_37z = soccermon_eval_37z.metrics.get("roc_auc", float("nan")) if soccermon_eval_37z else float("nan")
            auc_11  = soccermon_eval_11.metrics.get("roc_auc", float("nan")) if soccermon_eval_11 else float("nan")

            logger.info("═" * 60)
            logger.info("SOCCERMON CROSS-DOMAIN AUC COMPARISON")
            logger.info("  RF-37 original (baseline)  : %.4f", auc_37)
            logger.info("  RF-37-z (z-score fix)      : %.4f", auc_37z)
            logger.info("  RF-11  (wellness/load only): %.4f", auc_11)
            logger.info("  ∆(37z − 37orig)            : %+.4f  [effect of z-score]", auc_37z - auc_37)
            logger.info("  ∆(11 − 37orig)             : %+.4f  [effect of feature reduction]", auc_11 - auc_37)
            logger.info("═" * 60)

            # Save cross-domain comparison table
            cross_domain_df = pd.DataFrame([
                {"model": "RF-37 original",
                 "n_features": len(cfg.feature_columns),
                 "preprocessing": "PMData medians + Yeo-Johnson",
                 "soccermon_auc": round(auc_37, 4)},
                {"model": "RF-37-z",
                 "n_features": len(cfg.feature_columns),
                 "preprocessing": "Per-athlete z-score (no Yeo-Johnson)",
                 "soccermon_auc": round(auc_37z, 4)},
                {"model": "RF-11 wellness+load",
                 "n_features": len(SOCCERMON_SHARED_FEATURES),
                 "preprocessing": "Per-athlete z-score (no wearables)",
                 "soccermon_auc": round(auc_11, 4)},
            ])
            cross_domain_df.to_csv(
                Path(cfg.output_path) / "soccermon_cross_domain_comparison.csv",
                index=False,
            )

            stages.append(StageReport(
                name="SoccerMon_Evaluation",
                duration_s=round(dt, 2),
                details={
                    "RF37_original": soccermon_eval.metrics,
                    "RF37z": soccermon_eval_37z.metrics if soccermon_eval_37z else {},
                    "RF11": soccermon_eval_11.metrics if soccermon_eval_11 else {},
                },
            ))
        except Exception as exc:
            logger.warning("SoccerMon evaluation failed: %s", exc)
    else:
        logger.info(
            "SoccerMon dataset not found at %s — skipping Stage 7. "
            "Run python run_soccermon.py first.", cfg.soccermon_csv,
        )

    total = time.perf_counter() - t_global

    return InjuryReport(
        stages=stages,
        total_duration_s=round(total, 2),
        n_train=len(bundle.X_train),
        n_train_augmented=len(X_aug),
        n_val=len(bundle.X_val),
        n_test=len(bundle.X_test),
        n_features=len(bundle.feature_columns),
        lr_metrics=main_eval.metrics,
        baseline_metrics=baseline_eval.metrics,
        loso_roc_auc=loso_result.mean_roc_auc,
        comparison_table=comparison_df,
        soccermon_metrics=soccermon_eval.metrics if soccermon_eval else {},
        soccermon_metrics_37z=soccermon_eval_37z.metrics if soccermon_eval_37z else {},
        soccermon_metrics_11=soccermon_eval_11.metrics if soccermon_eval_11 else {},
        ablation_wellness_auc=ablation_wellness_auc,
        delta_auc_wearables=delta_auc_wearables,
    )
