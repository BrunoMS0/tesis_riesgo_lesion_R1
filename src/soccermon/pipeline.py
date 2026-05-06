"""
pipeline.py – Orchestrator for the SoccerMon ETL pipeline.

Reads raw SoccerMon files, applies feature engineering, creates binary injury
labels (onset + 7-day recovery window), imputes the 26 missing R5 features
using PMData training-set medians, and saves a player-day CSV compatible
with the R5 injury model.

Public API
----------
run(cfg) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import SoccerMonConfig
from .extract import extract_all
from .transform import transform

# R5 feature columns and PMData training participants
from ..injury.config import (
    FEATURE_COLUMNS as R5_FEATURE_COLUMNS,
    TRAIN_PARTICIPANTS as PMDATA_TRAIN_PARTICIPANTS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _compute_pmdata_medians(
    pmdata_csv: str,
    train_participants: List[str],
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Compute per-feature medians from the PMData training split.

    These values are used to impute the 26 R5 features not available in
    SoccerMon (HR, sleep-device, activity, exercise, TRIMP, DFI).
    Using only training participants avoids any leakage from validation data.
    """
    pmdata = pd.read_csv(pmdata_csv)
    if "participant_id" in pmdata.columns and train_participants:
        pmdata = pmdata[pmdata["participant_id"].isin(train_participants)]

    medians: Dict[str, float] = {}
    for col in feature_cols:
        if col in pmdata.columns:
            medians[col] = float(pmdata[col].median())
        else:
            medians[col] = 0.0

    n_available = sum(1 for c in feature_cols if c in pmdata.columns)
    logger.info(
        "PMData medians computed from %d rows (%d participants), "
        "%d/%d features found in PMData",
        len(pmdata), pmdata["participant_id"].nunique() if "participant_id" in pmdata.columns else 0,
        n_available, len(feature_cols),
    )
    return medians


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run(cfg: Optional[SoccerMonConfig] = None) -> pd.DataFrame:
    """
    Execute the full SoccerMon ETL pipeline.

    Stages
    ------
    1. Extract  — read all raw CSVs into long-format DataFrames
    2. Medians  — compute PMData training-set medians for imputation
    3. Transform — engineer features, build injury labels, impute, save CSV

    Returns
    -------
    pd.DataFrame  — R5-compatible player-day dataset with real injury labels.
    """
    if cfg is None:
        cfg = SoccerMonConfig()

    t0 = time.perf_counter()
    logger.info("═" * 60)
    logger.info("SOCCERMON ETL — START")
    logger.info("  Base path  : %s", cfg.base_path)
    logger.info("  Output CSV : %s", cfg.output_csv)
    logger.info("  Inj window : onset + %d days", cfg.recovery_window_days)
    logger.info("═" * 60)

    # ── Stage 1: Extract ─────────────────────────────────
    logger.info("Stage 1/3 — Extract")
    result = extract_all(cfg)

    # ── Stage 2: PMData medians ───────────────────────────
    logger.info("Stage 2/3 — PMData medians for imputation")
    pmdata_medians = _compute_pmdata_medians(
        cfg.pmdata_csv,
        cfg.pmdata_train_participants,
        R5_FEATURE_COLUMNS,
    )

    # ── Stage 3: Transform ───────────────────────────────
    logger.info("Stage 3/3 — Transform")
    df_raw, df_z, df_z_prospective = transform(result, cfg, pmdata_medians, R5_FEATURE_COLUMNS)

    dt = time.perf_counter() - t0

    # ── Save ──────────────────────────────────────────
    # Raw CSV (backward-compatible — used by RF-37 original baseline)
    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(out_path, index=False)
    logger.info("Raw dataset saved to: %s", out_path)

    # Z-scored CSV (used by RF-37-z and RF-11 cross-domain models)
    zscore_path = Path(cfg.zscore_output_csv)
    zscore_path.parent.mkdir(parents=True, exist_ok=True)
    df_z.to_csv(zscore_path, index=False)
    logger.info("Z-scored dataset saved to: %s", zscore_path)

    # Prospective CSV (z-scored + injury_next7d; used for combined PMData+SoccerMon training)
    prospective_path = Path(cfg.prospective_output_csv)
    prospective_path.parent.mkdir(parents=True, exist_ok=True)
    df_z_prospective.to_csv(prospective_path, index=False)
    logger.info("Prospective dataset saved to: %s", prospective_path)

    # Use raw df for summary stats (prevalence, players, etc.)
    df = df_raw

    n_players = df["participant_id"].nunique()
    n_injured = int(df["is_injured"].sum())
    prevalence = 100.0 * df["is_injured"].mean()

    logger.info("═" * 60)
    logger.info("SoccerMon ETL complete in %.2fs", dt)
    logger.info("  Shape      : %d rows × %d cols", *df.shape)
    logger.info("  Players    : %d", n_players)
    logger.info("  Positives  : %d (%.1f%% prevalence)", n_injured, prevalence)
    logger.info("  Raw CSV    : %s", out_path)
    logger.info("  ZScore CSV : %s", zscore_path)
    logger.info("═" * 60)

    return df
