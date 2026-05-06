"""
config.py – Centralised configuration for the R4 Fatigue model.

All feature lists, hyper-parameters, and paths live here so that
every other module in ``src.fatigue`` imports from one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────
_DEFAULT_INPUT_CSV = os.environ.get(
    "FATIGUE_INPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "notebooks", "outputs", "dataset_features_sin_normalizar.csv",
    ),
)
_DEFAULT_OUTPUT = os.environ.get(
    "FATIGUE_OUTPUT_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    ),
)

# ──────────────────────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────────────────────

# Objective Fitbit-derived input features (model inputs)
OBJECTIVE_FEATURES: List[str] = [
    # Raw Fitbit daily aggregates
    "steps", "distance", "calories", "resting_hr",
    # HR zones
    "hr_zone_below", "hr_zone_1", "hr_zone_2", "hr_zone_3",
    # Exercise
    "exercise_duration_min", "exercise_calories", "exercise_steps",
    "exercise_avg_hr", "exercise_sessions",
    # Activity intensity
    "lightly_active_minutes", "moderately_active_minutes",
    "very_active_minutes", "sedentary_minutes",
    # Fitbit sleep
    "overall_score", "composition_score", "revitalization_score",
    "duration_score", "deep_sleep_in_minutes", "restlessness",
    "sleep_rhr", "minutesAsleep", "efficiency", "minutesAwake",
    "timeInBed",
    # Engineered – Fitbit derived
    "trimp", "trimp_7d_sum",
    "steps_7d_sum", "distance_7d_sum", "calories_7d_sum",
    # Engineered – workload (ACWR uses session_load, already computed)
    "acute_load_7d", "chronic_load_28d", "acwr",
    # Engineered – sleep / HR
    "sleep_7d_avg", "sleep_debt",
    "rhr_baseline_7d", "rhr_drift", "rhr_variability_7d",
    # Engineered – activity
    "total_active_min", "active_ratio",
]

# Subjective PMSYS columns (excluded from model inputs)
SUBJECTIVE_COLUMNS: List[str] = [
    "fatigue", "mood", "readiness", "sleep_quality",
    "sleep_duration_h", "soreness", "stress",
    "perceived_exertion", "session_load", "duration_min",
    "wellness_score",
]

# Target
TARGET_RAW_COL: str = "fatigue"  # raw column from PMSYS (1–5 scale)

# Metadata (preserved but not used as features)
META_COLUMNS: List[str] = ["participant_id", "date", "is_injured"]

# ──────────────────────────────────────────────────────────────
# Sequence / windowing
# ──────────────────────────────────────────────────────────────
WINDOW_SIZE: int = 14       # days of lookback
SEED: int = 42

# ──────────────────────────────────────────────────────────────
# Splits (by participant, same as ETL convention)
# ──────────────────────────────────────────────────────────────
TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15     # test = 1 − train − val

# Explicit participant lists — must match R5 (injury) split exactly
# so that DFI predictions for test participants are never contaminated
# by R4 training on those same participants.
TRAIN_PARTICIPANTS: List[str] = [
    "p01", "p03", "p05", "p06", "p08", "p09", "p10", "p12", "p13",
]
VAL_PARTICIPANTS: List[str] = ["p02", "p16"]
TEST_PARTICIPANTS: List[str] = ["p04", "p07", "p11", "p14", "p15"]

# ──────────────────────────────────────────────────────────────
# tf.data parameters
# ──────────────────────────────────────────────────────────────
BATCH_SIZE: int = 32
SHUFFLE_BUFFER: int = 1024
PREFETCH: int = 2

# ──────────────────────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────────────────────
LSTM1_UNITS: int = 64
LSTM2_UNITS: int = 32
DENSE_UNITS: int = 32
DROPOUT_LSTM: float = 0.3
DROPOUT_DENSE: float = 0.2
L2_REG: float = 1e-4

# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
LEARNING_RATE: float = 1e-3
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 20
LR_PATIENCE: int = 10
LR_FACTOR: float = 0.5
LR_MIN: float = 1e-6


@dataclass
class FatigueConfig:
    """Immutable snapshot of all R4 settings."""

    # Paths
    input_csv: str = _DEFAULT_INPUT_CSV
    output_path: str = _DEFAULT_OUTPUT

    # Features
    objective_features: List[str] = field(
        default_factory=lambda: list(OBJECTIVE_FEATURES))
    target_raw_col: str = TARGET_RAW_COL

    # Sequence
    window_size: int = WINDOW_SIZE
    seed: int = SEED

    # Splits
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    train_participants: List[str] = field(
        default_factory=lambda: list(TRAIN_PARTICIPANTS))
    val_participants: List[str] = field(
        default_factory=lambda: list(VAL_PARTICIPANTS))
    test_participants: List[str] = field(
        default_factory=lambda: list(TEST_PARTICIPANTS))

    # tf.data
    batch_size: int = BATCH_SIZE
    shuffle_buffer: int = SHUFFLE_BUFFER
    prefetch: int = PREFETCH

    # Architecture
    lstm1_units: int = LSTM1_UNITS
    lstm2_units: int = LSTM2_UNITS
    dense_units: int = DENSE_UNITS
    dropout_lstm: float = DROPOUT_LSTM
    dropout_dense: float = DROPOUT_DENSE
    l2_reg: float = L2_REG

    # Training
    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    lr_patience: int = LR_PATIENCE
    lr_factor: float = LR_FACTOR
    lr_min: float = LR_MIN
