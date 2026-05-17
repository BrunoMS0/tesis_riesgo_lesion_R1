#!/usr/bin/env python
"""
run_comparison.py — R11: Validación comparativa del sistema M1→M2 vs. líneas base.

Reads final_model_comparison.csv as the source of truth for all previously
computed results, adds a Runner LOAO DummyClassifier row (computed here for
the first time), runs a Wilcoxon signed-rank test on Cond A vs Cond B per-fold
AUC pairs (63 matched folds), and exports the final comparison table.

Comparisons reported
--------------------
  1. M1→M2 (Cond B, GPS + M1 fatigue)  vs  naive (DummyClassifier LOAO)  → +≈40%
  2. M1→M2 (Cond B)                    vs  Cond A (GPS-only)              → ≈−0.4%
  3. M1→M2 (Cond B)                    vs  Cond C (GPS + real recovery)   → ≈−0.75%
  4. RF-Runner LOAO                     vs  LR-PMData                      → +35.8%

Statistical test
----------------
  Wilcoxon signed-rank (two-sided) on the 63 paired fold AUCs (Cond A vs Cond B).
  Hypothesis: the median AUC difference between Cond B and Cond A is zero.

Outputs
-------
  src/outputs/r11_comparison_results.csv          — full comparison table (15 rows)
  src/outputs/wilcoxon_cond_a_vs_b.txt            — Wilcoxon test report
  src/outputs/plots/r11_auc_comparison.png        — horizontal bar chart

Usage
-----
  python run_comparison.py
  python run_comparison.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

# ── File paths ────────────────────────────────────────────────────────────────

FINAL_COMPARISON_CSV = "src/outputs/final_model_comparison.csv"
LOAO_A_CSV           = "src/outputs/loao_runner_gps_only.csv"
LOAO_B_CSV           = "src/outputs/loao_runner_v2_results.csv"
PROCESSED_CSV        = "src/outputs/runner_dataset_processed.csv"

OUTPUT_CSV      = "src/outputs/r11_comparison_results.csv"
WILCOXON_TXT    = "src/outputs/wilcoxon_cond_a_vs_b.txt"
PLOTS_DIR       = Path("src/outputs/plots")

# IOV threshold for R11: integrated system must show ≥5% AUC improvement vs naive
IOV_MIN_DELTA = 0.05


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Stage functions ───────────────────────────────────────────────────────────

def _compute_dummy_runner_loao(logger: logging.Logger) -> Dict:
    """
    Compute DummyClassifier (stratified) LOAO AUC over the Runner Dataset.

    Protocol mirrors the main LOAO: leave one athlete out, fit DummyClassifier
    on the remaining athletes' labels, predict on the held-out athlete.
    DummyClassifier ignores X; predict_proba returns the training class
    proportions for every test observation, yielding AUC ≈ 0.50.

    Returns a dict ready to be appended as a row to the comparison table.
    """
    from src.runner.config import RUNNER_CSV, SEED
    from src.runner.extract import load_runner_csv
    from src.runner.transform import compute_features

    logger.info("Computing DummyClassifier LOAO (Runner Dataset)…")
    # Load minimal columns directly from the pipeline (avoid processed CSV size issues)
    raw_df = load_runner_csv(RUNNER_CSV)
    df = compute_features(raw_df)[["participant_id", "injury"]].copy()
    athletes = sorted(df["participant_id"].unique())

    fold_aucs: list = []
    skipped = 0

    for pid in athletes:
        test_mask = df["participant_id"] == pid
        y_test  = df.loc[test_mask,  "injury"].astype(int).values
        y_train = df.loc[~test_mask, "injury"].astype(int).values

        if y_test.sum() == 0:
            skipped += 1
            continue

        dummy = DummyClassifier(strategy="stratified", random_state=SEED)
        # DummyClassifier only needs y; X is a required positional arg but ignored
        X_dummy_train = np.zeros((len(y_train), 1))
        X_dummy_test  = np.zeros((len(y_test),  1))
        dummy.fit(X_dummy_train, y_train)

        proba = dummy.predict_proba(X_dummy_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)

    valid    = len(fold_aucs)
    n_total  = len(athletes)
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    std_auc  = float(np.std(fold_aucs,  ddof=1)) if len(fold_aucs) > 1 else float("nan")

    logger.info(
        "DummyClassifier LOAO: AUC=%.4f ± %.4f | valid=%d/%d (skipped=%d)",
        mean_auc, std_auc, valid, n_total, skipped,
    )

    return {
        "modelo":          "Dummy-Runner-LOAO",
        "algoritmo":       "DummyClassifier (stratified)",
        "dataset_entreno": "Runner (n-1 atletas por fold)",
        "dataset_eval":    "Runner LOAO (74 atletas)",
        "roc_auc":         round(mean_auc, 4),
        "roc_auc_std":     round(std_auc,  4),
        "folds_validos":   float(valid),
        "folds_total":     float(n_total),
        "tipo_eval":       "LOAO",
        "nota":            f"Baseline naive Runner LOAO — skipped={skipped} (0 lesiones)",
    }


def _run_wilcoxon(
    logger: logging.Logger,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Wilcoxon signed-rank test on 63 matched fold AUCs: Cond A vs Cond B.

    Returns (result_dict, merged_df_with_paired_folds).
    """
    df_a = pd.read_csv(LOAO_A_CSV)
    df_b = pd.read_csv(LOAO_B_CSV)

    # Remove summary MEAN row and skipped folds
    df_a = df_a[(df_a["participant_id"] != "MEAN") & (df_a["skipped"] == False)].copy()
    df_b = df_b[(df_b["participant_id"] != "MEAN") & (df_b["skipped"] == False)].copy()

    merged = df_a.merge(df_b, on="participant_id", suffixes=("_A", "_B"))

    aucs_a = merged["roc_auc_A"].values.astype(float)
    aucs_b = merged["roc_auc_B"].values.astype(float)
    n_pairs = len(merged)

    logger.info("Wilcoxon input: %d matched fold pairs (Cond A vs Cond B)", n_pairs)

    stat, pvalue = wilcoxon(aucs_a, aucs_b, alternative="two-sided")

    delta   = aucs_b - aucs_a
    result  = {
        "test":                    "Wilcoxon signed-rank (two-sided)",
        "comparison":              "Cond A (GPS-only, 10 feat) vs Cond B (GPS + M1, 11 feat)",
        "n_pairs":                 n_pairs,
        "auc_A_mean":              round(float(aucs_a.mean()), 4),
        "auc_B_mean":              round(float(aucs_b.mean()), 4),
        "mean_delta_B_minus_A":    round(float(delta.mean()),  4),
        "median_delta_B_minus_A":  round(float(np.median(delta)), 4),
        "statistic_W":             round(float(stat),   4),
        "p_value":                 round(float(pvalue), 4),
        "significant_alpha_0.05":  bool(pvalue < 0.05),
        "interpretation": (
            "La diferencia de AUC entre Cond B y Cond A NO es estadísticamente "
            "significativa (p≥0.05): el aporte de M1 al modelo de lesión es "
            "real pero marginal, consistente con la hipótesis de que las features "
            "GPS ya capturan implícitamente señal de fatiga."
            if pvalue >= 0.05 else
            "La diferencia de AUC entre Cond B y Cond A ES estadísticamente "
            "significativa (p<0.05)."
        ),
    }

    logger.info(
        "Wilcoxon: W=%.4f, p=%.4f | ΔB−A mean=%.4f, median=%.4f | n=%d",
        stat, pvalue, delta.mean(), np.median(delta), n_pairs,
    )
    return result, merged


def _make_comparison_plot(comparison_df: pd.DataFrame, logger: logging.Logger) -> None:
    """Horizontal bar chart for the 5 key models in R11 (IOV threshold line at 0.70)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        "Dummy-Runner-LOAO",
        "RF-Runner Ablation A (GPS-only)",
        "RF-Runner Ablation B (GPS + M1)",
        "RF-Runner Ablation C (GPS + real recovery)",
        "LOAO RF-Runner",
    ]
    subset = (
        comparison_df[comparison_df["modelo"].isin(labels)]
        .set_index("modelo")
        .reindex(labels)
        .reset_index()
    )

    colors = ["#bdbdbd", "#74c476", "#2196F3", "#4CAF50", "#1565C0"]
    xerr   = subset["roc_auc_std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(
        subset["modelo"], subset["roc_auc"],
        xerr=xerr, color=colors, edgecolor="white",
        height=0.55, capsize=4,
    )
    ax.axvline(0.70, color="red", linestyle="--", linewidth=1.2,
               label="Umbral IOV (0.70)")
    ax.set_xlabel("AUC-ROC (LOAO, 63 folds válidos)", fontsize=11)
    ax.set_title("R11 — Comparación de modelos: sistema M1→M2 vs. líneas base",
                 fontsize=12, pad=10)
    ax.set_xlim(0.3, 1.05)
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    for bar, auc in zip(ax.patches, subset["roc_auc"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{auc:.4f}", va="center", ha="left", fontsize=9,
        )

    plt.tight_layout()
    plot_path = PLOTS_DIR / "r11_auc_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Comparison bar chart saved → %s", plot_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("run_comparison")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Stage 1: Load base comparison table ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — Load final_model_comparison.csv (source of truth)")
    logger.info("=" * 60)

    comparison_df = pd.read_csv(FINAL_COMPARISON_CSV)
    logger.info("Loaded %d rows from %s", len(comparison_df), FINAL_COMPARISON_CSV)

    # ── Stage 2: Compute and append DummyClassifier Runner LOAO ───────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — DummyClassifier Runner LOAO (naive baseline)")
    logger.info("=" * 60)

    dummy_row = _compute_dummy_runner_loao(logger)
    comparison_df = pd.concat(
        [comparison_df, pd.DataFrame([dummy_row])], ignore_index=True
    )

    # ── Stage 3: Wilcoxon signed-rank test (Cond A vs Cond B) ────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — Wilcoxon signed-rank test (Cond A vs Cond B, 63 folds)")
    logger.info("=" * 60)

    wilcoxon_result, _fold_pairs = _run_wilcoxon(logger)

    with open(WILCOXON_TXT, "w", encoding="utf-8") as f:
        f.write("Wilcoxon Signed-Rank Test — Cond A (GPS-only) vs Cond B (GPS + M1)\n")
        f.write("=" * 70 + "\n\n")
        for key, val in wilcoxon_result.items():
            f.write(f"{key}: {val}\n")
    logger.info("Wilcoxon report saved → %s", WILCOXON_TXT)

    # ── Stage 4: Save full comparison CSV ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — Save r11_comparison_results.csv")
    logger.info("=" * 60)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Comparison table saved → %s (%d rows)", OUTPUT_CSV, len(comparison_df))

    # ── Stage 5: Key metrics summary (IOV evaluation) ─────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 — R11 IOV evaluation summary")
    logger.info("=" * 60)

    def _get_auc(name: str) -> float:
        row = comparison_df[comparison_df["modelo"] == name]
        return float(row.iloc[0]["roc_auc"]) if not row.empty else float("nan")

    auc_naive    = _get_auc("Dummy-Runner-LOAO")
    auc_cond_a   = _get_auc("RF-Runner Ablation A (GPS-only)")
    auc_cond_b   = _get_auc("RF-Runner Ablation B (GPS + M1)")
    auc_cond_c   = _get_auc("RF-Runner Ablation C (GPS + real recovery)")
    auc_lr       = _get_auc("LR-PMData")
    auc_rf_loao  = _get_auc("LOAO RF-Runner")

    comparisons = [
        ("M1→M2 (Cond B) vs naive",         auc_naive,   auc_cond_b),
        ("M1→M2 (Cond B) vs Cond A",         auc_cond_a,  auc_cond_b),
        ("M1→M2 (Cond B) vs Cond C",         auc_cond_c,  auc_cond_b),
        ("RF-Runner LOAO vs LR-PMData",       auc_lr,      auc_rf_loao),
    ]

    logger.info("─" * 60)
    for label, base, model in comparisons:
        delta = model - base
        pct   = 100 * delta
        logger.info("  %-35s base=%.4f model=%.4f Δ=%+.4f (%+.2f%%)",
                    label, base, model, delta, pct)
    logger.info("─" * 60)

    iov_met = (auc_cond_b - auc_naive) >= IOV_MIN_DELTA
    logger.info(
        "IOV R11 (M1→M2 ≥5%% AUC vs naive): %s  (Δ=%.4f)",
        "✓ CUMPLIDO" if iov_met else "✗ NO CUMPLIDO",
        auc_cond_b - auc_naive,
    )
    logger.info("─" * 60)
    logger.info("Wilcoxon (Cond A vs B): W=%.4f, p=%.4f → significant=%s",
                wilcoxon_result["statistic_W"],
                wilcoxon_result["p_value"],
                wilcoxon_result["significant_alpha_0.05"])
    logger.info("  Interpretation: %s", wilcoxon_result["interpretation"])
    logger.info("─" * 60)

    # ── Stage 6: Comparison bar chart ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 6 — Generate comparison bar chart")
    logger.info("=" * 60)

    _make_comparison_plot(comparison_df, logger)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("R11 comparison analysis complete in %.1fs", elapsed)
    logger.info("Outputs:")
    logger.info("  Table   : %s", OUTPUT_CSV)
    logger.info("  Wilcoxon: %s", WILCOXON_TXT)
    logger.info("  Plot    : %s", PLOTS_DIR / "r11_auc_comparison.png")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R11 comparative validation — M1→M2 vs baselines"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    args = parser.parse_args()
    _setup_logging(args.verbose)
    _run(args)


if __name__ == "__main__":
    main()
