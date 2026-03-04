"""
transform.py – TRANSFORM stage of the ETL pipeline.

Takes the raw consolidated DataFrame produced by :mod:`extract` and
applies three sequential sub‑stages:

1. **Clean** – null treatment, column pruning, deduplication.
2. **Engineer** – derived features (ACWR, TRIMP, Sleep Debt, …).
3. **Standardise** – Yeo‑Johnson power transform + z‑score.

Public API
----------
transform(df_raw, cfg) -> TransformResult
clean(df, cfg) -> pd.DataFrame
engineer_features(df, cfg) -> pd.DataFrame
standardise(df, cfg) -> Tuple[pd.DataFrame, PowerTransformer]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import PowerTransformer

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────

@dataclass
class TransformResult:
    """Bundle every artefact produced by the Transform stage."""

    df_cleaned: pd.DataFrame
    df_features: pd.DataFrame
    df_standardised: pd.DataFrame
    transformer: PowerTransformer
    feature_cols: List[str]
    metadata: Dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 1. CLEAN
# ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules to the raw global DataFrame.

    Steps
    -----
    1. Fill event variables with 0 (absence ≡ event did not happen).
    2. Drop columns with > ``cfg.null_drop_threshold`` % nulls
       (protected columns are never dropped).
    3. Forward‑fill → backward‑fill → global‑median imputation
       per participant for remaining numeric nulls.
    4. Remove duplicates by ``(participant_id, date)``.
    """
    out = df.copy()

    # 1 – Event variables → 0
    for col in cfg.event_vars:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    # 2 – Drop high‑null columns
    null_pct = out.isnull().sum() / len(out) * 100
    to_drop = null_pct[null_pct > cfg.null_drop_threshold].index.tolist()
    safe_keep = set(cfg.protected_cols) | set(cfg.event_vars)
    to_drop = [c for c in to_drop if c not in safe_keep]
    if to_drop:
        logger.info("Dropping columns with >%d%% nulls: %s",
                     cfg.null_drop_threshold, to_drop)
        out = out.drop(columns=to_drop)

    # 3 – Imputation (per participant: ffill → bfill → median)
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if out[c].isnull().any() and c not in cfg.protected_cols
    ]
    for col in numeric_cols:
        out[col] = out.groupby("participant_id")[col].transform(
            lambda s: s.ffill().bfill()
        )
        remaining = out[col].isnull().sum()
        if remaining > 0:
            out[col] = out[col].fillna(out[col].median())

    # 4 – Deduplicate
    before = len(out)
    out = out.drop_duplicates(subset=["participant_id", "date"], keep="last")
    after = len(out)
    if before != after:
        logger.info("Removed %d duplicate rows", before - after)

    logger.info("Clean: %d rows, %d cols, nulls=%d",
                len(out), len(out.columns), out.isnull().sum().sum())
    return out.reset_index(drop=True)


# ────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Derive scientifically‑justified features from the cleaned data.

    Features created
    ----------------
    * **ACWR** – Acute : Chronic Workload Ratio *(Gabbett, 2016)*
    * **TRIMP** – Training Impulse from HR zones *(Edwards, 1993)*
    * **Sleep Debt** – 7‑day rolling avg vs daily *(Milewski et al., 2014)*
    * **RHR Drift** – resting HR deviation from baseline *(Buchheit, 2013)*
    * **Rolling 7d sums** – steps, distance, calories
    * **Wellness Score** – mean of subjective components
    * **Active Ratio** – active / (active + sedentary) minutes
    """
    out = df.copy().sort_values(["participant_id", "date"]).reset_index(drop=True)
    created: List[str] = []

    def _grp_roll(series_name: str, window: int, fn: str = "mean", min_p: int = 1):
        return out.groupby("participant_id")[series_name].transform(
            lambda x: getattr(x.rolling(window, min_periods=min_p), fn)()
        )

    # 2.1 ACWR ─────────────────────────────────────────────
    if "session_load" in out.columns:
        out["acute_load_7d"] = _grp_roll("session_load", cfg.window_acute, "mean")
        out["chronic_load_28d"] = _grp_roll(
            "session_load", cfg.window_chronic, "mean", min_p=7
        )
        out["acwr"] = np.where(
            out["chronic_load_28d"] > 0,
            out["acute_load_7d"] / out["chronic_load_28d"],
            np.nan,
        )
        created.extend(["acute_load_7d", "chronic_load_28d", "acwr"])

    # 2.2 TRIMP ────────────────────────────────────────────
    avail_zones = [z for z in cfg.zone_weights if z in out.columns]
    if len(avail_zones) >= 3:
        out["trimp"] = sum(
            out[z].fillna(0) * cfg.zone_weights[z] for z in avail_zones
        )
        out["trimp_7d_sum"] = _grp_roll("trimp", cfg.window_acute, "sum")
        created.extend(["trimp", "trimp_7d_sum"])

    # 2.3 Sleep Debt ───────────────────────────────────────
    if "minutesAsleep" in out.columns:
        out["sleep_7d_avg"] = _grp_roll("minutesAsleep", 7, "mean", min_p=3)
        out["sleep_debt"] = out["sleep_7d_avg"] - out["minutesAsleep"]
        created.extend(["sleep_7d_avg", "sleep_debt"])

    # 2.4 RHR Drift ────────────────────────────────────────
    if "resting_hr" in out.columns:
        out["rhr_baseline_7d"] = _grp_roll("resting_hr", 7, "mean", min_p=3)
        out["rhr_drift"] = out["resting_hr"] - out["rhr_baseline_7d"]
        out["rhr_variability_7d"] = _grp_roll("resting_hr", 7, "std", min_p=3)
        created.extend(["rhr_baseline_7d", "rhr_drift", "rhr_variability_7d"])

    # 2.5 Rolling 7d sums ─────────────────────────────────
    for col_name in ["steps", "distance", "calories"]:
        if col_name in out.columns:
            feat = f"{col_name}_7d_sum"
            out[feat] = _grp_roll(col_name, 7, "sum")
            created.append(feat)

    # 2.6 Wellness Score ──────────────────────────────────
    avail_well = [c for c in cfg.wellness_components if c in out.columns]
    if len(avail_well) >= 3:
        out["wellness_score"] = out[avail_well].mean(axis=1)
        created.append("wellness_score")

    # 2.7 Active Ratio ────────────────────────────────────
    active_cols = [
        c for c in out.columns
        if "active_minutes" in c and "sedentary" not in c
    ]
    if active_cols and "sedentary_minutes" in out.columns:
        out["total_active_min"] = out[active_cols].sum(axis=1)
        total = out["total_active_min"] + out["sedentary_minutes"]
        out["active_ratio"] = np.where(total > 0, out["total_active_min"] / total, 0)
        created.extend(["total_active_min", "active_ratio"])

    logger.info("Feature engineering: %d new features → %d total columns",
                len(created), len(out.columns))
    return out


# ────────────────────────────────────────────────────────────
# 3. STANDARDISATION  (Yeo‑Johnson)
# ────────────────────────────────────────────────────────────

def standardise(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, PowerTransformer, List[str]]:
    """
    Apply Yeo‑Johnson power transformation + z‑score.

    Returns
    -------
    df_std : pd.DataFrame
        Transformed DataFrame (non‑numeric and excluded columns untouched).
    pt : PowerTransformer
        Fitted transformer (serialisable for inference).
    feat_cols : list[str]
        Names of the columns that were transformed.
    """
    if exclude is None:
        exclude = ["is_injured", "participant_id", "date"]

    feat_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "int64")
    ]

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    out = df.copy()

    # Only fit on non‑constant columns
    yj_cols = [c for c in feat_cols if df[c].nunique() > 1]
    out[yj_cols] = pt.fit_transform(df[yj_cols].fillna(0))

    # Constant columns → leave as 0 (already zero‑variance)
    const_cols = [c for c in feat_cols if c not in yj_cols]
    if const_cols:
        out[const_cols] = 0.0
        logger.info("Constant columns (left as 0): %s", const_cols)

    logger.info("Standardisation: %d columns transformed (Yeo-Johnson)",
                len(yj_cols))
    return out, pt, feat_cols


# ────────────────────────────────────────────────────────────
# 4. VARIABLE SELECTION
# ────────────────────────────────────────────────────────────

def select_features(
    df: pd.DataFrame,
    feat_cols: List[str],
    cfg: PipelineConfig,
    target: str = "is_injured",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features by multicollinearity (|ρ| > threshold) and
    low correlation with the target (|ρ| < low_corr_threshold).

    Returns
    -------
    df_selected : pd.DataFrame
    final_features : list[str]
    """
    to_drop: set = set()

    # ── Spearman correlations with target ─────────────────
    corr_with_target: Dict[str, float] = {}
    for col in feat_cols:
        valid = df[[col, target]].dropna()
        if len(valid) < 20:
            continue
        rho, _ = spearmanr(valid[col], valid[target])
        corr_with_target[col] = abs(rho)

    # ── Multicollinearity ─────────────────────────────────
    corr_matrix = df[feat_cols].corr(method="spearman")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = abs(corr_matrix.iloc[i, j])
            if r > cfg.multicol_threshold:
                v1, v2 = corr_matrix.columns[i], corr_matrix.columns[j]
                c1 = corr_with_target.get(v1, 0)
                c2 = corr_with_target.get(v2, 0)
                drop = v2 if c1 >= c2 else v1
                to_drop.add(drop)
                logger.debug("Multicollinearity: drop %s (ρ=%.3f)", drop, r)

    # ── Low‑correlation filter ────────────────────────────
    for col, rho in corr_with_target.items():
        if rho < cfg.low_corr_threshold:
            to_drop.add(col)

    # Protect essentials
    safe = set(cfg.protected_cols) | {target}
    to_drop -= safe
    to_drop = {c for c in to_drop if c in df.columns}

    df_selected = df.drop(columns=list(to_drop), errors="ignore")
    final_features = [
        c for c in df_selected.select_dtypes(include=[np.number]).columns
        if c != target
    ]
    logger.info(
        "Variable selection: dropped %d → %d features remain",
        len(to_drop), len(final_features),
    )
    return df_selected, final_features


# ────────────────────────────────────────────────────────────
# Convenience: full transform pipeline
# ────────────────────────────────────────────────────────────

def transform(df_raw: pd.DataFrame, cfg: PipelineConfig) -> TransformResult:
    """Run *Clean → Engineer → Standardise → Select* in sequence."""
    df_clean = clean(df_raw, cfg)
    df_feat  = engineer_features(df_clean, cfg)
    df_std, pt, feat_cols = standardise(df_feat, cfg)
    df_sel, final_feats   = select_features(df_std, feat_cols, cfg)

    return TransformResult(
        df_cleaned=df_clean,
        df_features=df_feat,
        df_standardised=df_std,
        transformer=pt,
        feature_cols=final_feats,
        metadata={
            "rows_after_clean": len(df_clean),
            "n_features_derived": len(df_feat.columns) - len(df_clean.columns),
            "n_features_final": len(final_feats),
            "standardisation": "Yeo-Johnson",
        },
    )
