"""
config.py – Centralised configuration for the ETL pipeline.

All paths, thresholds and hyper‑parameters live here so that every
other module imports them from a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ──────────────────────────────────────────────────────────────
# Default paths  (override via environment variables if needed)
# ──────────────────────────────────────────────────────────────
_DEFAULT_RAW_DATA = os.environ.get(
    "PMDATA_RAW_PATH",
    r"C:\Users\brunoabc\Desktop\tesis\tesis_riesgo_lesion_R1\pmdata\pmdata",
)
_DEFAULT_OUTPUT = os.environ.get(
    "ETL_OUTPUT_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs"),
)

# ──────────────────────────────────────────────────────────────
# Participant IDs
# ──────────────────────────────────────────────────────────────
PARTICIPANT_IDS: List[str] = [f"p{str(i).zfill(2)}" for i in range(1, 17)]

# ──────────────────────────────────────────────────────────────
# PMSYS file paths (relative inside each participant folder)
# ──────────────────────────────────────────────────────────────
PMSYS_FILES = {
    "injury":   "pmsys/injury.csv",
    "srpe":     "pmsys/srpe.csv",
    "wellness": "pmsys/wellness.csv",
}

# ──────────────────────────────────────────────────────────────
# Fitbit file paths
# ──────────────────────────────────────────────────────────────
FITBIT_JSON_DAILY_SUM = ["steps", "distance", "calories"]
FITBIT_ACTIVE_MINUTES = [
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
    "sedentary_minutes",
]
FITBIT_OTHER_JSON = [
    "resting_heart_rate",
    "time_in_heart_rate_zones",
    "exercise",
    "sleep",
]

# ──────────────────────────────────────────────────────────────
# Cleaning thresholds
# ──────────────────────────────────────────────────────────────
NULL_DROP_THRESHOLD_PCT: int = 60          # drop column if >60 % null
EVENT_VARIABLES: List[str] = [
    "is_injured", "session_load", "perceived_exertion", "duration_min",
]
PROTECTED_COLUMNS: List[str] = ["participant_id", "date"]

# ──────────────────────────────────────────────────────────────
# Feature‑engineering parameters
# ──────────────────────────────────────────────────────────────
ZONE_WEIGHTS: Dict[str, float] = {
    "hr_zone_below": 0.5,
    "hr_zone_1":     1.0,
    "hr_zone_2":     2.0,
    "hr_zone_3":     3.0,
}
WELLNESS_COMPONENTS: List[str] = ["fatigue", "mood", "readiness", "sleep_quality"]
ROLLING_WINDOW_ACUTE: int  = 7
ROLLING_WINDOW_CHRONIC: int = 28

# ──────────────────────────────────────────────────────────────
# Variable‑selection thresholds
# ──────────────────────────────────────────────────────────────
MULTICOLLINEARITY_THRESHOLD: float = 0.90
LOW_CORR_THRESHOLD: float = 0.02

# ──────────────────────────────────────────────────────────────
# tf.data parameters
# ──────────────────────────────────────────────────────────────
TF_BATCH_SIZE: int   = 64
TF_SHUFFLE_BUFFER: int = 1024
TF_PREFETCH: int     = 2           # tf.data.AUTOTUNE could also be used
TRAIN_SPLIT: float   = 0.7
VAL_SPLIT: float     = 0.15        # remainder = test


@dataclass
class PipelineConfig:
    """Immutable snapshot of all pipeline settings."""

    raw_data_path: str  = _DEFAULT_RAW_DATA
    output_path: str    = _DEFAULT_OUTPUT
    participants: List[str] = field(default_factory=lambda: list(PARTICIPANT_IDS))
    null_drop_threshold: int  = NULL_DROP_THRESHOLD_PCT
    event_vars: List[str]     = field(default_factory=lambda: list(EVENT_VARIABLES))
    protected_cols: List[str] = field(default_factory=lambda: list(PROTECTED_COLUMNS))
    zone_weights: Dict[str, float] = field(default_factory=lambda: dict(ZONE_WEIGHTS))
    wellness_components: List[str] = field(default_factory=lambda: list(WELLNESS_COMPONENTS))
    window_acute: int   = ROLLING_WINDOW_ACUTE
    window_chronic: int = ROLLING_WINDOW_CHRONIC
    multicol_threshold: float = MULTICOLLINEARITY_THRESHOLD
    low_corr_threshold: float = LOW_CORR_THRESHOLD
    batch_size: int     = TF_BATCH_SIZE
    shuffle_buffer: int = TF_SHUFFLE_BUFFER
    prefetch: int       = TF_PREFETCH
    train_split: float  = TRAIN_SPLIT
    val_split: float    = VAL_SPLIT
