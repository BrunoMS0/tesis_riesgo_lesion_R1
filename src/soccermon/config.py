"""
config.py – Centralised configuration for the SoccerMon ETL pipeline.

SoccerMon dataset: Midoglu et al., 2024, Scientific Data.
50 soccer players (TeamA: 27, TeamB: 23), 731 days (01.01.2020–31.07.2021).
162 real injury events across 15 players.

Injury label strategy:
    is_injured = 1 for the onset day and the following recovery_window_days days.
    With window=7 this yields ~3% prevalence — consistent with PMData.

Feature coverage:
    11 features directly available/computable from SoccerMon raw files.
    26 features imputed with PMData training-set medians (constant per column;
    the Random Forest ignores zero-variance columns at prediction time).

Z-score normalisation:
    Per-athlete z-score is applied to ZSCORE_FEATURES (load + wellness) to
    eliminate the ~21× absolute-load shift between sports and the 1-6 vs 1-10
    wellness-scale mismatch.  The result is saved as a separate CSV so the
    original raw CSV remains untouched for backward compatibility.

Public API
----------
SoccerMonConfig — dataclass with all settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────
_DEFAULT_BASE_PATH = os.environ.get(
    "SOCCERMON_BASE_PATH",
    r"C:\Users\brunoabc\Downloads\subjective\subjective",
)

_DEFAULT_OUTPUT_CSV = os.environ.get(
    "SOCCERMON_OUTPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_final.csv",
    ),
)

_DEFAULT_PMDATA_CSV = os.environ.get(
    "INJURY_INPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "notebooks", "outputs", "dataset_features_sin_normalizar.csv",
    ),
)

_DEFAULT_ZSCORE_OUTPUT_CSV = os.environ.get(
    "SOCCERMON_ZSCORE_OUTPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_zscore.csv",
    ),
)

_DEFAULT_PROSPECTIVE_OUTPUT_CSV = os.environ.get(
    "SOCCERMON_PROSPECTIVE_OUTPUT_CSV",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "soccermon_dataset_prospective.csv",
    ),
)

# ──────────────────────────────────────────────────────────────
# Wellness variable names (SoccerMon file names = R5 column names)
# ──────────────────────────────────────────────────────────────
WELLNESS_VARS: List[str] = [
    "fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
]

# R5 features that are directly available or computable from SoccerMon
# (the remaining 26 R5 features are imputed from PMData medians)
SOCCERMON_FEATURES: List[str] = [
    # Training load
    "session_load", "acute_load_7d", "chronic_load_28d", "acwr",
    # Wellness (subjective self-report)
    "fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
    # Derived wellness aggregate
    "wellness_score",
]

# Features subject to per-athlete z-score normalisation for cross-domain evaluation.
# Includes training-load features (eliminates 21× magnitude shift between sports)
# and wellness features (eliminates 1-6 vs 1-10 Likert scale mismatch).
ZSCORE_FEATURES: List[str] = [
    # Training load
    "session_load", "acute_load_7d", "chronic_load_28d", "acwr",
    # Wellness (subjective self-report)
    "fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
    # Derived wellness aggregate
    "wellness_score",
]

# PMData training participants used to compute imputation medians
PMDATA_TRAIN_PARTICIPANTS: List[str] = [
    "p01", "p03", "p04", "p05", "p06", "p07", "p08",
    "p09", "p10", "p11", "p12", "p13", "p14", "p15",
]


@dataclass
class SoccerMonConfig:
    """Immutable snapshot of all SoccerMon ETL settings."""

    # Path to the SoccerMon 'subjective' folder (contains training-load/, wellness/, injury/)
    base_path: str = _DEFAULT_BASE_PATH

    # Output CSV path — R5-compatible format
    output_csv: str = _DEFAULT_OUTPUT_CSV

    # PMData CSV used to compute imputation medians (training participants only)
    pmdata_csv: str = _DEFAULT_PMDATA_CSV

    # Injury label: onset day + recovery_window_days are marked is_injured=1
    recovery_window_days: int = 7

    # Wellness variable names (must match SoccerMon wellness/*.csv filenames)
    wellness_vars: List[str] = field(default_factory=lambda: list(WELLNESS_VARS))

    # Features directly available / computable from SoccerMon raw data
    soccermon_features: List[str] = field(
        default_factory=lambda: list(SOCCERMON_FEATURES))

    # Per-athlete z-score output CSV path
    zscore_output_csv: str = _DEFAULT_ZSCORE_OUTPUT_CSV

    # Prospective target output CSV (z-scored + injury_next{window}d)
    # Used for combined PMData+SoccerMon training (Fase 4).
    prospective_output_csv: str = _DEFAULT_PROSPECTIVE_OUTPUT_CSV

    # Look-ahead window in days for the prospective target (must match InjuryConfig)
    prospective_window: int = 7

    # Features to z-score per athlete for cross-domain evaluation
    zscore_features: List[str] = field(default_factory=lambda: list(ZSCORE_FEATURES))

    # PMData training participants for imputation medians
    pmdata_train_participants: List[str] = field(
        default_factory=lambda: list(PMDATA_TRAIN_PARTICIPANTS))
