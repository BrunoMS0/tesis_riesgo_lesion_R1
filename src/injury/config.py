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
# Synthetic data / augmentation
# ──────────────────────────────────────────────────────────────
N_SYNTHETIC_ATHLETES: int = 32
AUGMENTATION_METHOD: str = "smote"   # 'smote' | 'copula'
TARGET_RATIO: float = 0.3           # minority class proportion after SMOTE
SMOTE_K_NEIGHBORS: int = 5

# ──────────────────────────────────────────────────────────────
# Logistic Regression hyper-parameters
# ──────────────────────────────────────────────────────────────
LR_C: float = 1.0                   # inverse regularisation strength
LR_PENALTY: str = "l2"              # 'l1', 'l2', or 'elasticnet'
LR_SOLVER: str = "lbfgs"            # 'lbfgs' (L2), 'saga' (elasticnet/L1)
LR_MAX_ITER: int = 1000
LR_CLASS_WEIGHT: str = "balanced"   # handle residual class imbalance
C_GRID: List[float] = [0.01, 0.1, 1.0, 10.0]

# Primary evaluation metric
PRIMARY_METRIC: str = "roc_auc"


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

    # Augmentation
    n_synthetic_athletes: int = N_SYNTHETIC_ATHLETES
    augmentation_method: str = AUGMENTATION_METHOD
    target_ratio: float = TARGET_RATIO
    smote_k_neighbors: int = SMOTE_K_NEIGHBORS

    # Logistic Regression
    lr_C: float = LR_C
    lr_penalty: str = LR_PENALTY
    lr_solver: str = LR_SOLVER
    lr_max_iter: int = LR_MAX_ITER
    lr_class_weight: str = LR_CLASS_WEIGHT
    c_grid: List[float] = field(default_factory=lambda: list(C_GRID))

    # Evaluation
    primary_metric: str = PRIMARY_METRIC
