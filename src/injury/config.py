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
_DEFAULT_SOCCERMON_CSV = os.environ.get(
    "SOCCERMON_DATASET_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_final.csv",
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

# Target column (raw injury label — always present in source data)
TARGET_COL: str = "is_injured"

# Prospective target: "will there be an injury in the next N days?"
# Creates column 'injury_next{N}d'; rows with undefined future window are dropped.
PROSPECTIVE_TARGET_COL: str = "injury_next7d"
PROSPECTIVE_WINDOW: int = 7
USE_PROSPECTIVE_TARGET: bool = True

# Metadata columns (preserved but not used as features)
META_COLUMNS: List[str] = ["participant_id", "date"]

# Features shared between PMData and SoccerMon (directly available in both).
# Used to train the RF-11 cross-domain model and for cross-domain evaluation.
SOCCERMON_SHARED_FEATURES: List[str] = [
    # Training load
    "session_load", "acute_load_7d", "chronic_load_28d", "acwr",
    # Wellness (subjective self-report)
    "fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
    # Derived wellness aggregate
    "wellness_score",
]

# Features to normalise with per-athlete z-score for cross-domain evaluation.
# Eliminates absolute-load shift (~21×) and wellness-scale mismatch (1-6 vs 1-10).
ZSCORE_FEATURES: List[str] = list(SOCCERMON_SHARED_FEATURES)

# ──────────────────────────────────────────────────────────────
# Splits (by participant, consistent with R4)
# ──────────────────────────────────────────────────────────────
SEED: int = 42
TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15    # test = 1 − train − val

# Explicit participant lists — take priority over ratio-based split.
# All 16 PMData participants used for training (14) and validation (2).
# External test set: SoccerMon (real injury labels).
# VAL (2): p02=1 injury, p16=0 — used only for grid-search of hyperparameters.
# TRAIN (14): all remaining PMData participants — maximises positive signal
#   (p05=10, p12 bias-corrected, p04=3, p07=2, p11=2, p14=3, p15=6 → 40 events)
TRAIN_PARTICIPANTS: List[str] = [
    "p01", "p03", "p04", "p05", "p06", "p07", "p08",
    "p09", "p10", "p11", "p12", "p13", "p14", "p15",
]
VAL_PARTICIPANTS: List[str] = ["p02", "p16"]
TEST_PARTICIPANTS: List[str] = []  # PMData test not used; SoccerMon is the external test set

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

# ──────────────────────────────────────────────────────────────
# Random Forest hyper-parameters
# ──────────────────────────────────────────────────────────────
RF_N_ESTIMATORS: int = 200
RF_MAX_FEATURES: str = "sqrt"
RF_CLASS_WEIGHT: str = "balanced"
# Grid of (max_depth, min_samples_leaf) pairs evaluated on the val set
RF_PARAM_GRID: List[dict] = [
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": 10,   "min_samples_leaf": 1},
    {"max_depth": 10,   "min_samples_leaf": 5},
]

# ──────────────────────────────────────────────────────────────
# Model type selector
# ──────────────────────────────────────────────────────────────
MODEL_TYPE: str = "rf"   # 'lr' | 'rf'


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
    train_participants: List[str] = field(
        default_factory=lambda: list(TRAIN_PARTICIPANTS))
    val_participants: List[str] = field(
        default_factory=lambda: list(VAL_PARTICIPANTS))
    test_participants: List[str] = field(
        default_factory=lambda: list(TEST_PARTICIPANTS))

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

    # Random Forest
    rf_n_estimators: int = RF_N_ESTIMATORS
    rf_max_features: str = RF_MAX_FEATURES
    rf_class_weight: str = RF_CLASS_WEIGHT
    rf_param_grid: List[dict] = field(
        default_factory=lambda: [dict(d) for d in RF_PARAM_GRID])

    # Model type selector
    model_type: str = MODEL_TYPE

    # Evaluation
    primary_metric: str = PRIMARY_METRIC

    # SoccerMon external test set path (real injury labels)
    soccermon_csv: str = _DEFAULT_SOCCERMON_CSV

    # Per-athlete z-score settings for cross-domain evaluation
    # When True, ZSCORE_FEATURES are z-scored per participant before Yeo-Johnson.
    use_per_athlete_zscore: bool = False
    zscore_features: List[str] = field(default_factory=lambda: list(ZSCORE_FEATURES))

    # Prospective injury target settings
    # When True, 'injury_next{prospective_window}d' is created and used as target.
    # Increases positives per lesion event from 1 → window rows, improving PR-AUC.
    use_prospective_target: bool = USE_PROSPECTIVE_TARGET
    prospective_window: int = PROSPECTIVE_WINDOW

    # Z-scored SoccerMon CSV (produced by run_soccermon.py ETL)
    soccermon_zscore_csv: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_zscore.csv",
    )

    # Prospective SoccerMon CSV (z-scored + injury_next7d; produced by run_soccermon.py ETL)
    # Used for combined PMData+SoccerMon RF-11 training (Fase 4).
    soccermon_prospective_csv: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_prospective.csv",
    )

    # Combined PMData+SoccerMon training flag
    # When True, pipeline Stage 4 trains RF-11 on both datasets combined and
    # runs LOAO over all 66 athletes.
    use_combined_training: bool = True
    # Set rerun_combined=True to force Stage 4b even if loao_combined_results.csv exists.
    rerun_combined: bool = False

    # ── LSTM temporal model (Stage 5) ────────────────────────
    # Set use_lstm=True to enable the LSTM LOAO stage.
    # Default is False to keep the main pipeline fast unless requested.
    use_lstm: bool = False

    # Features for LSTM (default: 11 shared features, same as RF-11)
    # Override with list(FEATURE_COLUMNS) to use all 37 PMData features.
    lstm_use_shared_features: bool = True
    lstm_feature_cols: List[str] = field(
        default_factory=lambda: list(SOCCERMON_SHARED_FEATURES))

    # Sequence window (days of history per sample)
    lstm_window_size: int = 14

    # Architecture
    lstm_units: int = 64
    lstm_units_2: int = 32
    lstm_dense_units: int = 16
    lstm_dropout: float = 0.3

    # Training
    lstm_epochs: int = 50
    lstm_batch_size: int = 64
    lstm_patience: int = 10

