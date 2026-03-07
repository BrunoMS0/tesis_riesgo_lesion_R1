"""
config.py – Centralised configuration for the R5 Injury Risk Prediction model.

All feature lists, hyper-parameters, and paths live here so that
every other module in ``src.injury`` imports from one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────
_DEFAULT_INPUT_CSV = os.environ.get(
    "INJURY_INPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "notebooks", "outputs", "dataset_features_sin_normalizar.csv",
    ),
)
_DEFAULT_DFI_CSV = os.environ.get(
    "INJURY_DFI_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "fatigue_index_predictions.csv",
    ),
)
_DEFAULT_OUTPUT = os.environ.get(
    "INJURY_OUTPUT_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    ),
)

# ──────────────────────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────────────────────

# All input features for injury prediction (objective + subjective + DFI)
FEATURE_COLUMNS: List[str] = [
    # R4 Dynamic Fatigue Index (primary novel feature)
    "dfi_predicted",
    # Workload
    "session_load", "acute_load_7d", "chronic_load_28d", "acwr",
    # TRIMP
    "trimp", "trimp_7d_sum",
    # Wellness (subjective)
    "fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
    "wellness_score",
    # Sleep
    "minutesAsleep", "efficiency", "minutesAwake", "timeInBed",
    "overall_score", "deep_sleep_in_minutes", "sleep_debt", "sleep_7d_avg",
    # Activity
    "steps", "calories", "steps_7d_sum", "calories_7d_sum",
    "active_ratio", "sedentary_minutes",
    # Heart rate
    "resting_hr", "rhr_drift", "rhr_variability_7d", "rhr_baseline_7d",
    "hr_zone_below", "hr_zone_1",
    # Exercise
    "exercise_duration_min", "exercise_calories", "exercise_sessions",
]

# Target column
TARGET_COL: str = "is_injured"

# Metadata columns (preserved but not used as features)
META_COLUMNS: List[str] = ["participant_id", "date"]

# ──────────────────────────────────────────────────────────────
# Splits (by participant, consistent with R4)
# ──────────────────────────────────────────────────────────────
SEED: int = 42
TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15    # test = 1 − train − val

# ──────────────────────────────────────────────────────────────
# Synthetic data generation (SDV Gaussian Copula)
# ──────────────────────────────────────────────────────────────
N_SYNTHETIC_ATHLETES: int = 32

# ──────────────────────────────────────────────────────────────
# XGBoost hyper-parameters
# ──────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.05
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE_BYTREE: float = 0.8
XGB_REG_ALPHA: float = 0.1
XGB_REG_LAMBDA: float = 1.0
XGB_MIN_CHILD_WEIGHT: int = 5
XGB_EARLY_STOPPING: int = 30

# ──────────────────────────────────────────────────────────────
# Random Forest baseline hyper-parameters
# ──────────────────────────────────────────────────────────────
RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_LEAF: int = 5


@dataclass
class InjuryConfig:
    """Immutable snapshot of all R5 settings."""

    # Paths
    input_csv: str = _DEFAULT_INPUT_CSV
    dfi_csv: str = _DEFAULT_DFI_CSV
    output_path: str = _DEFAULT_OUTPUT

    # Features
    feature_columns: List[str] = field(
        default_factory=lambda: list(FEATURE_COLUMNS))
    target_col: str = TARGET_COL

    # Splits
    seed: int = SEED
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT

    # Synthetic data
    n_synthetic_athletes: int = N_SYNTHETIC_ATHLETES

    # XGBoost
    xgb_n_estimators: int = XGB_N_ESTIMATORS
    xgb_max_depth: int = XGB_MAX_DEPTH
    xgb_learning_rate: float = XGB_LEARNING_RATE
    xgb_subsample: float = XGB_SUBSAMPLE
    xgb_colsample_bytree: float = XGB_COLSAMPLE_BYTREE
    xgb_reg_alpha: float = XGB_REG_ALPHA
    xgb_reg_lambda: float = XGB_REG_LAMBDA
    xgb_min_child_weight: int = XGB_MIN_CHILD_WEIGHT
    xgb_early_stopping: int = XGB_EARLY_STOPPING

    # Random Forest
    rf_n_estimators: int = RF_N_ESTIMATORS
    rf_max_depth: int = RF_MAX_DEPTH
    rf_min_samples_leaf: int = RF_MIN_SAMPLES_LEAF
