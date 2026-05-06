"""
run_runner.py — Runner Dataset injury prediction pipeline (Fase 3).

Stages
------
1. ETL + Feature Engineering   → src/outputs/runner_dataset_processed.csv
2. RF Training (train split)    → val AUC reported
3. RF Grid Search               → best (max_depth, min_samples_leaf)
4. LOAO Cross-Validation        → 74-fold AUC; fallback to injury_next14d if < 0.60
5. Save model + results         → src/outputs/rf_runner_model.pkl
                                   src/outputs/loao_runner_results.csv

Usage
-----
  python run_runner.py              # full pipeline
  python run_runner.py --no-loao   # skip LOAO (faster, just train + val AUC)
  python run_runner.py -v          # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Imports (after logging) ────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    from src.runner.config import (
        RUNNER_LOAO_RESULTS,
        RUNNER_MODEL_PATH,
        RUNNER_FEATURE_COLUMNS,
    )
    from src.runner.dataset import build_runner_datasets, make_runner_injury_config
    from src.injury.model import build_random_forest, build_model
    from src.injury.evaluate import evaluate_model
    from src.injury.train import grid_search_RF
    from src.injury.validate import loso_cross_validation
    import joblib

    logger = logging.getLogger("run_runner")
    t0 = time.time()

    # ── Stage 1: ETL + Feature Engineering ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — ETL + Feature Engineering")
    logger.info("=" * 60)

    bundle = build_runner_datasets(save_processed=True)
    cfg = make_runner_injury_config()

    logger.info(
        "Dataset ready: %d train rows (%d+), %d val rows (%d+), %d test rows (%d+)",
        len(bundle.X_train), int(bundle.y_train.sum()),
        len(bundle.X_val),   int(bundle.y_val.sum()),
        len(bundle.X_test),  int(bundle.y_test.sum()),
    )

    # ── Stage 2: RF Grid Search on val set ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — RF Grid Search (val AUC)")
    logger.info("=" * 60)

    best_params, _gs_results = grid_search_RF(
        bundle.X_train, bundle.y_train,
        bundle.X_val,   bundle.y_val,
        cfg,
    )
    best_val_auc = float(_gs_results["roc_auc"].max())
    logger.info("Best params: %s | val AUC = %.4f", best_params, best_val_auc)

    # Retrain best model on full train split with best params
    best_model = build_random_forest(cfg, **best_params)
    best_model.fit(bundle.X_train, bundle.y_train)

    # Full val + test evaluation
    val_metrics  = evaluate_model(best_model, bundle.X_val,  bundle.y_val,
                                  bundle.meta_val,  cfg)
    test_metrics = evaluate_model(best_model, bundle.X_test, bundle.y_test,
                                  bundle.meta_test, cfg)

    logger.info("Val  - AUC=%.4f | PR-AUC=%.4f | F1=%.4f",
                val_metrics.metrics.get("roc_auc", 0),
                val_metrics.metrics.get("pr_auc", 0),
                val_metrics.metrics.get("f1", 0))
    logger.info("Test - AUC=%.4f | PR-AUC=%.4f | F1=%.4f",
                test_metrics.metrics.get("roc_auc", 0),
                test_metrics.metrics.get("pr_auc", 0),
                test_metrics.metrics.get("f1", 0))

    # ── Stage 3: Save model ────────────────────────────────────────────────────
    Path(RUNNER_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, RUNNER_MODEL_PATH)
    logger.info("Model saved → %s", RUNNER_MODEL_PATH)

    # ── Stage 4: LOAO Cross-Validation ────────────────────────────────────────
    if args.no_loao:
        logger.info("Skipping LOAO (--no-loao flag set)")
        _print_summary(best_val_auc, None, time.time() - t0)
        return

    logger.info("=" * 60)
    logger.info("STAGE 3 — LOAO Cross-Validation (74 athletes)")
    logger.info("=" * 60)

    # Build the full dataset (all athletes combined) for LOAO
    raw_df_all = _load_full_runner_df()
    loao_result = loso_cross_validation(
        raw_df_all["X_all"],
        raw_df_all["y_all"],
        raw_df_all["meta_all"],
        cfg,
        use_augmentation=True,
    )

    mean_auc = loao_result.mean_roc_auc
    std_auc  = loao_result.std_roc_auc
    n_valid  = sum(1 for f in loao_result.folds if not f.skipped)
    n_skip   = loao_result.n_skipped_folds

    logger.info(
        "LOAO result: AUC = %.4f ± %.4f | valid folds: %d / %d (skipped: %d)",
        mean_auc, std_auc,
        n_valid, len(loao_result.folds), n_skip,
    )

    # ── Fallback: injury_next14d if LOAO AUC < 0.60 ───────────────────────────
    if mean_auc < 0.60:
        logger.warning(
            "LOAO AUC %.4f < 0.60 — running fallback with injury_next14d target",
            mean_auc,
        )
        loao_result = _run_loao_next14d(raw_df_all, cfg, logger)
        mean_auc = loao_result.mean_roc_auc
        std_auc  = loao_result.std_roc_auc
        n_valid  = sum(1 for f in loao_result.folds if not f.skipped)
        logger.info(
            "LOAO (14d target): AUC = %.4f ± %.4f | valid folds: %d",
            mean_auc, std_auc, n_valid,
        )

    # ── Save LOAO results ─────────────────────────────────────────────────────
    _save_loao_results(loao_result, RUNNER_LOAO_RESULTS)
    logger.info("LOAO results saved → %s", RUNNER_LOAO_RESULTS)

    _print_summary(best_val_auc, loao_result, time.time() - t0)


def _load_full_runner_df() -> dict:
    """
    Load the full processed runner dataset (all 74 athletes) as raw
    (unnormalized) X, y, meta for use in LOAO.

    LOAO normalises per fold internally (fit on fold train only).
    """
    from src.runner.config import RUNNER_OUTPUT_CSV, RUNNER_FEATURE_COLUMNS, TARGET_COL

    df = pd.read_csv(RUNNER_OUTPUT_CSV)
    avail = [c for c in RUNNER_FEATURE_COLUMNS if c in df.columns]
    X_all   = df[avail].reset_index(drop=True)
    y_all   = df[TARGET_COL].astype(int).reset_index(drop=True)
    meta_all = df[["participant_id", "date"]].reset_index(drop=True)
    return {"X_all": X_all, "y_all": y_all, "meta_all": meta_all}


def _run_loao_next14d(raw_df_all: dict, cfg, logger) -> object:
    """
    Fallback: add injury_next14d target via forward-max over per-athlete
    event rows (T3.5).  Uses calendar-distance via the 'date' column.
    """
    from src.runner.config import RUNNER_OUTPUT_CSV, RUNNER_FEATURE_COLUMNS
    from src.injury.validate import loso_cross_validation
    from src.injury.config import InjuryConfig

    df = pd.read_csv(RUNNER_OUTPUT_CSV)
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)

    # Create injury_next14d: for each row at date D, label = 1 if any
    # injury occurs within the next 14 calendar days (D+1 to D+14).
    def _add_next14d(group: pd.DataFrame) -> pd.Series:
        dates  = group["date"].values
        injury = group["injury"].values
        result = np.zeros(len(group), dtype=int)
        for i, d in enumerate(dates):
            future_mask = (dates > d) & (dates <= d + 14)
            if future_mask.any() and injury[future_mask].max() > 0:
                result[i] = 1
        return pd.Series(result, index=group.index)

    df["injury_next14d"] = df.groupby("participant_id", group_keys=False).apply(
        _add_next14d
    )

    avail = [c for c in RUNNER_FEATURE_COLUMNS if c in df.columns]
    cfg14 = make_runner_injury_config_14d(avail)

    X_all    = df[avail].reset_index(drop=True)
    y_all    = df["injury_next14d"].astype(int).reset_index(drop=True)
    meta_all = df[["participant_id", "date"]].reset_index(drop=True)

    logger.info(
        "injury_next14d prevalence: %.2f%% (%d events)",
        100.0 * y_all.mean(), int(y_all.sum()),
    )
    return loso_cross_validation(X_all, y_all, meta_all, cfg14, use_augmentation=True)


def make_runner_injury_config_14d(feature_cols):
    """InjuryConfig for the 14-day fallback target."""
    from src.runner.dataset import make_runner_injury_config
    cfg = make_runner_injury_config(feature_cols)
    cfg.target_col = "injury_next14d"
    return cfg


def _save_loao_results(loao_result, output_path: str) -> None:
    """Save per-fold LOAO results to CSV."""
    rows = []
    for f in loao_result.folds:
        rows.append({
            "participant_id": f.participant_id,
            "n_samples":      f.n_samples,
            "n_injuries":     f.n_injuries,
            "roc_auc":        f.roc_auc,
            "pr_auc":         f.pr_auc,
            "f1":             f.f1,
            "skipped":        f.skipped,
        })
    rows.append({
        "participant_id": "MEAN",
        "n_samples":      None,
        "n_injuries":     None,
        "roc_auc":        loao_result.mean_roc_auc,
        "pr_auc":         loao_result.mean_pr_auc,
        "f1":             loao_result.mean_f1,
        "skipped":        False,
    })
    rows.append({
        "participant_id": "STD",
        "n_samples":      None,
        "n_injuries":     None,
        "roc_auc":        loao_result.std_roc_auc,
        "pr_auc":         loao_result.std_pr_auc,
        "f1":             loao_result.std_f1,
        "skipped":        False,
    })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _print_summary(val_auc: float, loao_result, elapsed_s: float) -> None:
    print()
    print("=" * 60)
    print("  RUNNER PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Val AUC (RF, 70/10 split)  : {val_auc:.4f}")
    if loao_result is not None:
        n_valid = sum(1 for f in loao_result.folds if not f.skipped)
        print(f"  LOAO AUC (Runner, 74 ath.) : {loao_result.mean_roc_auc:.4f} ± {loao_result.std_roc_auc:.4f}")
        print(f"  Valid folds                : {n_valid} / {len(loao_result.folds)}")
        met = "✅" if loao_result.mean_roc_auc >= 0.65 else "❌"
        print(f"  Meta AUC >= 0.65           : {met}")
    print(f"  Total runtime              : {elapsed_s:.1f}s")
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner Dataset injury prediction pipeline"
    )
    parser.add_argument(
        "--no-loao", action="store_true",
        help="Skip LOAO cross-validation (faster dev mode)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)
    _run(args)


if __name__ == "__main__":
    main()
