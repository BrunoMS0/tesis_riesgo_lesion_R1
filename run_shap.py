#!/usr/bin/env python
"""
run_shap.py — R10: Análisis SHAP del modelo de lesión (Condición A, Runner Dataset).

Retrains M2 Condición A (RF Classifier, 10 GPS features = FATIGUE_FEATURE_COLUMNS)
on the exact same train split (seed=42, 51/7/16 athletes), applies
shap.TreeExplainer on the test split (16 athletes), generates 4 plot types,
and exports a ranked feature importance CSV.

Features (Cond A — GPS objetivas puras, sin dato subjetivo):
    acute_load_7d, chronic_load_28d, acwr, high_intensity_km_7d,
    nr_sessions_7d, nr_rest_days_7d, km_sprint_7d, strength_days_7d,
    alt_hours_7d, recent_km

Rationale for Cond A (not full 18-feat M2):
  - Isolates objective GPS signal without subjective-GPS interaction
    features (session_load_proxy = km × sRPE) that complicate attribution
  - Ablation study (Cap 6) confirmed Cond A ≈ full M2 (ΔAUC = 0.27%)
  - Cleaner SHAP attribution for thesis interpretability chapter

Outputs
-------
  src/outputs/shap_feature_importance.csv        — top-10 ranking by mean |SHAP|
  src/outputs/shap_values.csv                    — full SHAP matrix (n_samples × n_features + y_actual)
  src/outputs/plots/shap_beeswarm.png            — global importance + effect direction
  src/outputs/plots/shap_bar.png                 — top-10 mean |SHAP| bar chart
  src/outputs/plots/shap_dependence_{feat}.png   — dependence for top 3 features
  src/outputs/plots/shap_waterfall_case{n}.png   — waterfall for 3 high-risk cases

Usage
-----
  python run_shap.py          # full analysis
  python run_shap.py -v       # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ── Constants ─────────────────────────────────────────────────────────────────

SHAP_CSV_PATH = "src/outputs/shap_feature_importance.csv"
SHAP_VALUES_CSV_PATH = "src/outputs/shap_values.csv"
PLOTS_DIR = Path("src/outputs/plots")

# M2 Cond A hyperparameters (confirmed immutable facts)
# min_samples_leaf=1, max_depth=None — same as ablation Cond A
RF_OVERRIDES = dict(min_samples_leaf=1, max_depth=None)

# Top N features for dependence plots
N_DEPENDENCE = 3

# Number of waterfall cases (highest-probability true positives)
N_WATERFALL = 3

# SHAP sample size: stratified sample from test set (standard practice for
# large test sets with deep unconstrained trees). None = use full test set.
# TreeExplainer with max_depth=None can produce very deep trees (~thousands
# of nodes), making full-test-set computation prohibitively slow.
SHAP_SAMPLE_SIZE = 2000


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    from src.runner.config import RUNNER_CSV
    from src.runner.dataset import build_runner_datasets, make_runner_injury_config
    from src.runner.fatigue import FATIGUE_FEATURE_COLUMNS
    from src.injury.augment import augment_training_data
    from src.injury.model import build_random_forest

    logger = logging.getLogger("run_shap")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Stage 1: Load data (exact same split as training, seed=42) ────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — Load Runner Dataset (Cond A, %d GPS features)", len(FATIGUE_FEATURE_COLUMNS))
    logger.info("Features: %s", FATIGUE_FEATURE_COLUMNS)
    logger.info("=" * 60)

    bundle = build_runner_datasets(
        csv_path=RUNNER_CSV,
        feature_cols=FATIGUE_FEATURE_COLUMNS,
        save_processed=False,
    )
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test,  y_test  = bundle.X_test,  bundle.y_test
    # Raw (pre-Yeo-Johnson) test values for display in beeswarm colorbar
    X_test_raw = bundle.X_test_raw

    logger.info(
        "Data ready — train: %d rows (%d+) | test: %d rows (%d+) | features: %d",
        len(X_train), int(y_train.sum()),
        len(X_test),  int(y_test.sum()),
        X_train.shape[1],
    )

    # ── Stage 2: SMOTE + Train RF Cond A ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — SMOTE augmentation + RF Classifier training (Cond A)")
    logger.info("=" * 60)

    cfg = make_runner_injury_config(feature_cols=FATIGUE_FEATURE_COLUMNS)
    X_aug, y_aug = augment_training_data(X_train, y_train, bundle.meta_train, cfg)

    model = build_random_forest(cfg, **RF_OVERRIDES)
    model.fit(X_aug, y_aug)

    logger.info(
        "Model trained: %d trees | train rows=%d (%d+ after SMOTE)",
        model.n_estimators, len(X_aug), int(y_aug.sum()),
    )

    # ── Stage 3: SHAP TreeExplainer ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — SHAP TreeExplainer (test split, %d observations)", len(X_test))
    logger.info("=" * 60)

    # Stratified sample for SHAP (preserve injury class ratio, seed=42)
    from src.runner.config import SEED
    rng = np.random.RandomState(SEED)

    if SHAP_SAMPLE_SIZE is not None and len(X_test) > SHAP_SAMPLE_SIZE:
        pos_idx  = np.where(y_test.values == 1)[0]
        neg_idx  = np.where(y_test.values == 0)[0]
        n_pos    = min(len(pos_idx),  int(SHAP_SAMPLE_SIZE * y_test.mean()) + 1)
        n_neg    = SHAP_SAMPLE_SIZE - n_pos
        sampled  = np.concatenate([
            rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ])
        sampled  = np.sort(sampled)
        X_shap   = X_test.iloc[sampled].reset_index(drop=True)
        y_shap   = y_test.iloc[sampled].reset_index(drop=True)
        X_shap_raw = X_test_raw.iloc[sampled].reset_index(drop=True) if X_test_raw is not None else X_shap
        logger.info(
            "SHAP sample: %d / %d observations (stratified, seed=%d) — "
            "%d positive, %d negative",
            len(sampled), len(X_test), SEED,
            int(y_shap.sum()), int((y_shap == 0).sum()),
        )
    else:
        X_shap, y_shap, X_shap_raw = X_test, y_test, X_test_raw
        logger.info("SHAP: using full test set (%d observations)", len(X_test))

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_shap)

    # Handle all SHAP output shapes for binary RF:
    #   - New API: ndarray shape (n_samples, n_features, n_classes)  → slice [:, :, 1]
    #   - Old API: list [class_0_array, class_1_array]               → [1]
    #   - Single array (some versions)                               → use as-is
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_pos = shap_values_raw[:, :, 1]
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    elif isinstance(shap_values_raw, list):
        shap_pos = shap_values_raw[1]
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    else:
        shap_pos = shap_values_raw
        ev = explainer.expected_value
        base_value = float(ev) if not hasattr(ev, "__len__") else float(ev[1])

    logger.info("SHAP values computed: shape=%s, base_value=%.4f", shap_pos.shape, base_value)

    # Build Explanation object using raw feature values for display (interpretable scale)
    # base_values must be a per-sample array for shap.plots.bar/beeswarm compatibility
    base_values_arr = np.full(len(shap_pos), base_value)
    explanation = shap.Explanation(
        values=shap_pos,
        base_values=base_values_arr,
        data=X_shap_raw.values,          # original unscaled GPS values in colorbars
        feature_names=list(FATIGUE_FEATURE_COLUMNS),
    )

    # ── Stage 4: Feature ranking CSV ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — Export feature importance ranking CSV")
    logger.info("=" * 60)

    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    importance_df = (
        pd.DataFrame({
            "feature": FATIGUE_FEATURE_COLUMNS,
            "mean_abs_shap": mean_abs_shap,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))
    Path(SHAP_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(SHAP_CSV_PATH, index=False)

    logger.info("Feature ranking saved → %s", SHAP_CSV_PATH)
    logger.info("\nTop 10 features by mean |SHAP|:\n%s",
                importance_df[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    # Export full SHAP values matrix (one row per sample, one column per feature)
    shap_cols = {f"shap_{f}": shap_pos[:, i] for i, f in enumerate(FATIGUE_FEATURE_COLUMNS)}
    shap_values_df = pd.DataFrame({"y_actual": y_shap.values, **shap_cols})
    shap_values_df.index.name = "observation_idx"
    shap_values_df.to_csv(SHAP_VALUES_CSV_PATH)
    logger.info(
        "SHAP values matrix saved → %s  (shape: %d × %d)",
        SHAP_VALUES_CSV_PATH, shap_values_df.shape[0], shap_values_df.shape[1],
    )

    top_features = importance_df["feature"].tolist()

    # ── Stage 5: Generate plots ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 — Generate SHAP plots (beeswarm, bar, dependence, waterfall)")
    logger.info("=" * 60)

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    # 5a. Beeswarm plot (global importance + direction of effect)
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.savefig(PLOTS_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Beeswarm plot saved → %s", PLOTS_DIR / "shap_beeswarm.png")

    # 5b. Bar plot (top 10 by mean |SHAP|)
    shap.plots.bar(explanation, max_display=10, show=False)
    plt.savefig(PLOTS_DIR / "shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Bar plot saved → %s", PLOTS_DIR / "shap_bar.png")

    # 5c. Dependence plots for top N_DEPENDENCE features
    for feat in top_features[:N_DEPENDENCE]:
        feat_idx = list(FATIGUE_FEATURE_COLUMNS).index(feat)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx,
            shap_pos,
            X_shap,
            feature_names=list(FATIGUE_FEATURE_COLUMNS),
            ax=ax,
            show=False,
        )
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel(f"SHAP value — {feat}", fontsize=11)
        plt.tight_layout()
        dep_path = PLOTS_DIR / f"shap_dependence_{feat}.png"
        plt.savefig(dep_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        logger.info("Dependence plot saved → %s", dep_path)

    # 5d. Waterfall plots for N_WATERFALL high-risk actual positive cases
    injury_proba = model.predict_proba(X_shap)[:, 1]
    pos_mask = y_shap.values == 1

    if pos_mask.sum() >= N_WATERFALL:
        pos_idx = np.where(pos_mask)[0]
        case_indices = pos_idx[np.argsort(injury_proba[pos_idx])[-N_WATERFALL:]]
    else:
        # Fallback: highest predicted probability regardless of actual label
        case_indices = np.argsort(injury_proba)[-N_WATERFALL:]

    for i, idx in enumerate(case_indices):
        shap.plots.waterfall(explanation[int(idx)], max_display=10, show=False)
        waterfall_path = PLOTS_DIR / f"shap_waterfall_case{i + 1}.png"
        plt.savefig(waterfall_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        logger.info(
            "Waterfall plot saved → %s  (shap_idx=%d, proba=%.4f, actual=%d)",
            waterfall_path, idx, injury_proba[idx], int(y_shap.iloc[idx]),
        )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("R10 SHAP analysis complete in %.1fs", elapsed)
    logger.info("Outputs:")
    logger.info("  Ranking CSV : %s", SHAP_CSV_PATH)
    logger.info("  Plots dir   : %s/", PLOTS_DIR)
    logger.info("  Plots       : shap_beeswarm.png, shap_bar.png, "
                "shap_dependence_*.png, shap_waterfall_*.png")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R10 SHAP analysis — Cond A (GPS-only RF, Runner Dataset)"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    args = parser.parse_args()
    _setup_logging(args.verbose)
    _run(args)


if __name__ == "__main__":
    main()
