"""
config.py — Configuration for the Runner Dataset pipeline (Löwdal 2021).

Dataset: day_approach_maskedID_timeseries.csv
  - 74 athletes, 583 injuries (63 athletes with ≥1 injury)
  - Wide format: 10 features × 7 days per row (suffixes '' to '.6')
  - suffix '' = D-7 (oldest), suffix '.6' = D-1 (yesterday)
  - 'injury' column is already prospective (no create_prospective_target needed)
  - 'Date' is a per-athlete integer day index (not calendar date)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ─── Paths ───────────────────────────────────────────────────────────────────

_WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

RUNNER_CSV: str = os.path.join(
    _WORKSPACE_ROOT, "Runner dataset", "day_approach_maskedID_timeseries.csv"
)
RUNNER_OUTPUT_CSV: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "runner_dataset_processed.csv"
)
RUNNER_MODEL_PATH: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "rf_runner_model.pkl"
)
RUNNER_LOAO_RESULTS: str = os.path.join(
    _WORKSPACE_ROOT, "src", "outputs", "loao_runner_results.csv"
)

# ─── Raw column definitions ───────────────────────────────────────────────────

# Base names (without day suffix) present in the wide CSV
RAW_FEATURE_BASE_NAMES: List[str] = [
    "nr. sessions",
    "total km",
    "km Z3-4",
    "km Z5-T1-T2",
    "km sprinting",
    "strength training",
    "hours alternative",
    "perceived exertion",
    "perceived trainingSuccess",
    "perceived recovery",
]

# Day suffixes: '' = D-7 (oldest), '.6' = D-1 (yesterday, most recent)
DAY_SUFFIXES: List[str] = ["", ".1", ".2", ".3", ".4", ".5", ".6"]

# Maps suffix → offset from event day (e.g., '' → 7 means D-7)
SUFFIX_TO_OFFSET: Dict[str, int] = {
    "": 7, ".1": 6, ".2": 5, ".3": 4, ".4": 3, ".5": 2, ".6": 1
}

# Value used by the dataset authors to mark rest days (no training)
REST_DAY_VALUE: float = -0.01

# ─── Derived feature columns for the Runner model ────────────────────────────

RUNNER_FEATURE_COLUMNS: List[str] = [
    # Training load
    "acute_load_7d",         # total km summed across D-7 to D-1
    "chronic_load_28d",      # rolling 28-day km (reconstructed from per-athlete history)
    "acwr",                  # acute_load_7d / (chronic_load_28d / 4), clipped at 5.0
    "high_intensity_km_7d",  # sum of km Z3-4 + km Z5-T1-T2 across 7 days
    "session_load_proxy",    # acute_load_7d × mean_perceived_exertion (sRPE proxy)
    "nr_sessions_7d",        # total training sessions in 7 days
    "nr_rest_days_7d",       # days with no training (perceived exertion == -0.01)
    "km_sprint_7d",          # total sprinting km in 7 days
    "strength_days_7d",      # strength training days in 7 days
    "alt_hours_7d",          # alternative (cross) training hours in 7 days
    # Subjective wellness (excluding rest days where value == -0.01)
    "mean_perceived_exertion",   # mean sRPE across training days
    "mean_perceived_recovery",   # mean pre-session recovery across training days
    "mean_perceived_success",    # mean training success across training days
    "wellness_score",            # mean(recovery, success) — composite wellness
    # Most recent day (D-1, suffix '.6') — highest recency weight
    "recent_exertion",    # perceived exertion on D-1
    "recent_recovery",    # perceived recovery on D-1
    "recent_success",     # perceived trainingSuccess on D-1
    "recent_km",          # total km on D-1
]

# ─── Phase 4 cross-domain feature mapping (Runner → PMData) ─────────────────
# Semantic equivalence mapping for the RF-Common cross-domain model.
# Runner feature name → corresponding PMData feature name.
RUNNER_PMDATA_FEATURE_MAP: Dict[str, str] = {
    "acwr":                    "acwr",
    "session_load_proxy":      "session_load",
    "mean_perceived_exertion": "fatigue",
    "mean_perceived_recovery": "readiness",
    "mean_perceived_success":  "mood",
    "high_intensity_km_7d":    "trimp_7d_sum",
}

# Ordered lists for easy DataFrame column renaming
RUNNER_COMMON_FEATURES: List[str] = list(RUNNER_PMDATA_FEATURE_MAP.keys())
PMDATA_COMMON_FEATURES: List[str] = list(RUNNER_PMDATA_FEATURE_MAP.values())

# ─── Target ───────────────────────────────────────────────────────────────────

# 'injury' is already a prospective label (event-day format by dataset authors).
# DO NOT apply create_prospective_target — it would create a double-windowed label.
TARGET_COL: str = "injury"

# ─── Split settings ───────────────────────────────────────────────────────────

SEED: int = 42
TRAIN_SPLIT: float = 0.70   # ≈ 52 / 74 athletes
VAL_SPLIT: float = 0.10     # ≈  7 / 74 athletes  (test = 0.20 ≈ 15 athletes)

# ─── Model / augmentation settings ───────────────────────────────────────────

RF_N_ESTIMATORS: int = 200
RF_MAX_FEATURES: str = "sqrt"
RF_CLASS_WEIGHT: str = "balanced"

# SMOTE augmentation — lower target ratio than PMData (base prevalence 1.36%)
AUGMENTATION_METHOD: str = "smote"
TARGET_RATIO: float = 0.15
SMOTE_K_NEIGHBORS: int = 5
