"""
config.py – Centralised configuration for R6 Integration Pipeline.

Embeds ``FatigueConfig`` and ``InjuryConfig`` plus paths to pre-trained
model artefacts so that the full two-stage prediction can be reproduced
from a single configuration object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.fatigue.config import FatigueConfig
from src.injury.config import InjuryConfig

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DEFAULT_FATIGUE_MODEL = os.environ.get(
    "INTEGRATION_FATIGUE_MODEL",
    os.path.join(_PROJECT_ROOT, "src", "outputs",
                 "fatigue_model", "best_weights.keras"),
)

_DEFAULT_INJURY_MODEL = os.environ.get(
    "INTEGRATION_INJURY_MODEL",
    os.path.join(_PROJECT_ROOT, "src", "outputs", "xgboost_injury.joblib"),
)

_DEFAULT_OUTPUT = os.environ.get(
    "INTEGRATION_OUTPUT_PATH",
    os.path.join(_PROJECT_ROOT, "src", "outputs", "integration"),
)


@dataclass
class IntegrationConfig:
    """Immutable snapshot of all R6 settings."""

    # Pre-trained model artefact paths
    fatigue_model_path: str = _DEFAULT_FATIGUE_MODEL
    injury_model_path: str = _DEFAULT_INJURY_MODEL

    # Output directory for integration results
    output_path: str = _DEFAULT_OUTPUT

    # Embedded sub-configs (carry feature lists, seeds, etc.)
    fatigue_cfg: FatigueConfig = field(default_factory=FatigueConfig)
    injury_cfg: InjuryConfig = field(default_factory=InjuryConfig)
