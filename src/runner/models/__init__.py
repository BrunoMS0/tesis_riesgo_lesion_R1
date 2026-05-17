"""
src/runner/models — Modelos de machine learning sobre el Runner Dataset.

Contiene los dos modelos que forman el pipeline predictivo M1→M2:

Módulos
-------
fatigue.py   M1 — RF Regressor: estima la fatiga/recuperación percibida del atleta
             Entrada : 10 features objetivas de carga GPS (sin dato subjetivo)
             Salida  : fatigue_score_predicted ∈ [0, 1]  (proxy de perceived_recovery)
             Validado: LOAO 75 atletas | RMSE=0.1652 | MAE=0.1383

injury.py    M2 — RF Classifier: predice el riesgo de lesión en el horizonte prospectivo
             Entrada : 19 features GPS + fatigue_score_predicted de M1
             Salida  : probabilidad de lesión ∈ [0, 1]
             Validado: LOAO 64 atletas con ≥1 lesión | ROC-AUC=0.9034

Dependencias entre modelos
--------------------------
  M1 (fatigue.py) → genera fatigue_score_predicted
  M2 (injury.py)  → consume fatigue_score_predicted como feature adicional

Uso rápido
----------
    from src.runner.models import run_fatigue_pipeline, build_runner_datasets
    results = run_fatigue_pipeline()   # M1
    bundle  = build_runner_datasets()  # M2 — dataset para el clasificador
"""

from .fatigue import (
    run_fatigue_pipeline,
    prepare_fatigue_dataset,
    run_loao_fatigue,
    train_final_fatigue_model,
    FATIGUE_FEATURE_COLUMNS,
    FATIGUE_TARGET_COL,
)
from .injury import build_runner_datasets, make_runner_injury_config

__all__ = [
    # M1 — Regressor de fatiga
    "run_fatigue_pipeline",
    "prepare_fatigue_dataset",
    "run_loao_fatigue",
    "train_final_fatigue_model",
    "FATIGUE_FEATURE_COLUMNS",
    "FATIGUE_TARGET_COL",
    # M2 — Dataset y configuración del clasificador de lesión
    "build_runner_datasets",
    "make_runner_injury_config",
]
