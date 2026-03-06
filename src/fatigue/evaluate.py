"""
evaluate.py – Evaluation utilities for the R4 Fatigue model.

Computes metrics on the test set and generates thesis-ready artefacts
(CSV tables, scatter plot, residual plot, attention heatmap).

Public API
----------
evaluate_model(model, bundle, cfg) -> EvaluationResult
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import FatigueConfig
from .dataset import FatigueDatasetBundle

logger = logging.getLogger(__name__)

# Lazy imports for plotting (not required at import time)
_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


@dataclass
class EvaluationResult:
    """Holds all evaluation artefacts."""

    metrics: Dict[str, float]
    per_participant: pd.DataFrame
    predictions_df: pd.DataFrame
    attention_weights: Optional[np.ndarray] = None


# ────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MSE, RMSE, MAE, R² and Pearson r."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "pearson_r": round(pearson_r, 6),
        "n_samples": len(y_true),
    }


# ────────────────────────────────────────────────────────────
# Attention extraction
# ────────────────────────────────────────────────────────────

def _extract_attention_weights(model, dataset) -> np.ndarray:
    """
    Build a sub-model that outputs attention weights and run it on
    the supplied dataset.

    Returns shape ``(N_samples, window_size)``.
    """
    import tensorflow as tf

    att_layer = None
    for layer in model.layers:
        if layer.name == "attention":
            att_layer = layer
            break

    if att_layer is None:
        logger.warning("Attention layer not found – skipping extraction")
        return np.array([])

    # Build sub-model up to attention with dual output
    att_input = model.input
    # Walk through layers to reconstruct outputs
    x = att_input
    for layer in model.layers[1:]:
        if layer.name == "attention":
            context, alpha = layer(x, return_attention=True)
            break
        x = layer(x)

    att_model = tf.keras.Model(inputs=att_input, outputs=alpha,
                               name="attention_extractor")

    weights_list = []
    for batch_x, _ in dataset:
        w = att_model(batch_x, training=False).numpy()
        weights_list.append(w)

    return np.concatenate(weights_list, axis=0)


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────

def _plot_scatter(y_true, y_pred, save_path):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none")
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="grey", linewidth=0.8)
    ax.set_xlabel("DFI Actual")
    ax.set_ylabel("DFI Predicted")
    ax.set_title("Predicted vs Actual – Dynamic Fatigue Index")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved: %s", save_path)


def _plot_residuals(y_true, y_pred, save_path):
    plt = _get_plt()
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, edgecolor="white", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual (Predicted − Actual)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Residual plot saved: %s", save_path)


def _plot_attention_heatmap(att_weights, window_size, save_path):
    plt = _get_plt()
    mean_att = att_weights.mean(axis=0)  # (window_size,)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.bar(range(1, window_size + 1), mean_att, color="steelblue")
    ax.set_xlabel("Day in Window (1 = oldest)")
    ax.set_ylabel("Avg Attention Weight")
    ax.set_title("Temporal Attention – Average Weight per Day")
    ax.set_xticks(range(1, window_size + 1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Attention heatmap saved: %s", save_path)


# ────────────────────────────────────────────────────────────
# Main evaluation
# ────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    bundle: FatigueDatasetBundle,
    cfg: Optional[FatigueConfig] = None,
) -> EvaluationResult:
    """
    Evaluate the trained model on the **test** split and save artefacts.

    Produces
    --------
    - ``fatigue_evaluation.csv`` — overall + per-participant metrics.
    - ``scatter_dfi.png`` — predicted vs actual scatter.
    - ``residuals_dfi.png`` — residual histogram.
    - ``attention_heatmap.png`` — average attention weights.
    """
    if cfg is None:
        cfg = FatigueConfig()

    out_dir = Path(cfg.output_path) / "fatigue_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Predictions on test set ------------------------------------
    y_pred_list, y_true_list = [], []
    for batch_x, batch_y in bundle.test:
        preds = model(batch_x, training=False).numpy().squeeze()
        y_pred_list.append(preds)
        y_true_list.append(batch_y.numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    # --- Global metrics ---------------------------------------------
    metrics = _compute_metrics(y_true, y_pred)
    logger.info("Test metrics: %s", metrics)

    # --- Per-participant breakdown -----------------------------------
    pred_df = bundle.meta_test.copy()
    pred_df["dfi_actual"] = y_true
    pred_df["dfi_predicted"] = y_pred

    per_part_rows = []
    for pid, grp in pred_df.groupby("participant_id"):
        m = _compute_metrics(grp["dfi_actual"].values,
                             grp["dfi_predicted"].values)
        m["participant_id"] = pid
        per_part_rows.append(m)
    per_participant = pd.DataFrame(per_part_rows)

    # --- Save CSV ----------------------------------------------------
    eval_csv = out_dir / "fatigue_evaluation.csv"
    summary = pd.DataFrame([{**metrics, "participant_id": "ALL"}])
    pd.concat([summary, per_participant], ignore_index=True).to_csv(
        eval_csv, index=False)
    logger.info("Evaluation CSV saved: %s", eval_csv)

    # --- Plots -------------------------------------------------------
    _plot_scatter(y_true, y_pred, out_dir / "scatter_dfi.png")
    _plot_residuals(y_true, y_pred, out_dir / "residuals_dfi.png")

    # --- Attention weights -------------------------------------------
    att_w = _extract_attention_weights(model, bundle.test)
    if att_w.size > 0:
        _plot_attention_heatmap(att_w, cfg.window_size,
                                out_dir / "attention_heatmap.png")

    return EvaluationResult(
        metrics=metrics,
        per_participant=per_participant,
        predictions_df=pred_df,
        attention_weights=att_w if att_w.size > 0 else None,
    )
