"""
evaluate.py – Evaluation suite for R5 Injury Risk Prediction.

Computes metrics, generates plots, SHAP feature importance,
per-participant breakdown, and model comparison (ablation).

Public API
----------
evaluate_model(model, X_test, y_test, meta_test, cfg) -> EvaluationResult
compute_shap_values(model, X_test, cfg) -> ShapResult
compare_models(results_dict) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import InjuryConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation output for a single model."""

    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    per_participant: Optional[pd.DataFrame] = None
    optimal_threshold: float = 0.5


@dataclass
class ShapResult:
    """SHAP analysis output."""

    shap_values: Any = None          # np.ndarray
    feature_importance: Optional[pd.DataFrame] = None


def _find_optimal_threshold(y_true, y_prob) -> float:
    """Find threshold that maximises F1 from the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    # thresholds has len = len(precision) - 1
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    cfg: InjuryConfig,
    model_name: str = "XGBoost",
) -> EvaluationResult:
    """
    Run the full metrics suite on a trained model.

    Returns
    -------
    EvaluationResult with metrics dict, confusion matrix, and per-participant
    breakdown.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    # Optimal threshold from PR curve
    if y_test.sum() > 0:
        threshold = _find_optimal_threshold(y_test, y_prob)
    else:
        threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    # Primary metric: PR-AUC
    try:
        pr_auc = average_precision_score(y_test, y_prob)
    except ValueError:
        pr_auc = 0.0

    # Secondary metrics
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0.0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0.0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0.0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "optimal_threshold": round(threshold, 4),
    }

    cm = confusion_matrix(y_test, y_pred)

    # Per-participant breakdown
    per_part = _per_participant_breakdown(y_test, y_prob, y_pred, meta_test, threshold)

    logger.info("[%s] PR-AUC=%.4f, ROC-AUC=%.4f, F1=%.4f, threshold=%.4f",
                model_name, metrics["pr_auc"], metrics["roc_auc"],
                metrics["f1"], threshold)

    return EvaluationResult(
        model_name=model_name,
        metrics=metrics,
        confusion=cm,
        per_participant=per_part,
        optimal_threshold=threshold,
    )


def _per_participant_breakdown(
    y_test: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    meta_test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Compute per-participant detection metrics."""
    rows = []
    for pid, grp in meta_test.groupby("participant_id"):
        idx = grp.index
        yt = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        yp = y_pred[idx] if isinstance(y_pred, np.ndarray) else y_pred.iloc[idx]

        n_injured = int(yt.sum())
        n_detected = int((yt & yp).sum()) if n_injured > 0 else 0

        rows.append({
            "participant_id": pid,
            "n_samples": len(idx),
            "n_injuries": n_injured,
            "n_detected": n_detected,
            "detection_rate": round(n_detected / max(n_injured, 1), 4),
        })
    return pd.DataFrame(rows)


def compute_shap_values(
    model,
    X_test: pd.DataFrame,
    cfg: InjuryConfig,
) -> ShapResult:
    """
    Compute SHAP values for model interpretability.

    Returns
    -------
    ShapResult with SHAP values array and feature importance DataFrame.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature importance: mean absolute SHAP value per feature
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("SHAP analysis complete — top feature: %s (%.4f)",
                importance.iloc[0]["feature"],
                importance.iloc[0]["mean_abs_shap"])

    return ShapResult(shap_values=shap_values, feature_importance=importance)


def save_shap_plot(
    shap_result: ShapResult,
    X_test: pd.DataFrame,
    output_path: str,
) -> str:
    """Save a SHAP summary bar plot to disk."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(output_path, "shap_summary.png")

    shap.summary_plot(shap_result.shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP summary plot saved to %s", fig_path)
    return fig_path


def save_evaluation_report(
    result: EvaluationResult,
    output_path: str,
) -> str:
    """Save evaluation metrics and per-participant breakdown to CSV."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Metrics CSV
    metrics_path = os.path.join(output_path,
                                f"metrics_{result.model_name.lower()}.csv")
    pd.DataFrame([result.metrics]).to_csv(metrics_path, index=False)

    # Per-participant CSV
    if result.per_participant is not None:
        pp_path = os.path.join(output_path,
                               f"per_participant_{result.model_name.lower()}.csv")
        result.per_participant.to_csv(pp_path, index=False)

    logger.info("Evaluation report saved to %s", output_path)
    return metrics_path


def compare_models(results: Dict[str, EvaluationResult]) -> pd.DataFrame:
    """
    Build a comparison table across multiple model evaluation results.

    Parameters
    ----------
    results : dict mapping model_name -> EvaluationResult

    Returns
    -------
    DataFrame with one row per model and metric columns.
    """
    rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update(res.metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    logger.info("Model comparison:\n%s", df.to_string())
    return df
