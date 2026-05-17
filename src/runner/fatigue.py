"""
fatigue.py — Redirect de compatibilidad.

El código fuente está en src/runner/models/fatigue.py.
Este archivo se mantiene para que los imports existentes sigan funcionando sin cambios.

Para nuevos desarrollos, importar directamente desde:
    from src.runner.models.fatigue import run_fatigue_pipeline, FATIGUE_FEATURE_COLUMNS
"""
from .models.fatigue import (  # noqa: F401
    run_fatigue_pipeline,
    prepare_fatigue_dataset,
    run_loao_fatigue,
    train_final_fatigue_model,
    FATIGUE_FEATURE_COLUMNS,
    FATIGUE_TARGET_COL,
    FATIGUE_LOAO_RESULTS,
    FATIGUE_MODEL_PATH,
    FATIGUE_IMPORTANCE_PATH,
    FATIGUE_PREDICTIONS_PATH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RMSE_TARGET,
    RMSE_WARN,
)
from .config import RUNNER_CSV  # noqa: F401  (re-exportado para compatibilidad con run_runner_fatigue.py)

__all__ = [
    "run_fatigue_pipeline", "prepare_fatigue_dataset", "run_loao_fatigue",
    "train_final_fatigue_model", "FATIGUE_FEATURE_COLUMNS", "FATIGUE_TARGET_COL",
    "RUNNER_CSV",
]
