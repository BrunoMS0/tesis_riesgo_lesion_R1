"""
src/runner — Pipeline completo de predicción de lesión sobre el Runner Dataset (Löwdal 2021).

Estructura del módulo
---------------------
  config.py            Parámetros, rutas y listas de features compartidas por todo el pipeline

  etl/                 Pipeline de datos: Extract → Transform → Load (Orquestado)
    extract.py           Paso 1: carga el CSV fuente, estandariza columnas de identidad
    transform.py         Paso 2: ingeniería de features (ACWR, cargas, wellness, D-1)
    week_transform.py    Variante semanal: mapea columnas del CSV semanal a nombres diarios
    pipeline.py          Orquestador ETL: ejecuta extract → transform → guarda CSV procesado

  models/              Modelos de machine learning
    fatigue.py           M1 — RF Regressor: estima fatiga/recuperación percibida (10 features GPS)
    injury.py            M2 — RF Classifier: predice riesgo de lesión (19 features + M1 score)

Archivos de compatibilidad en src/runner/
------------------------------------------
Los archivos a nivel de src/runner/ (extract.py, transform.py, fatigue.py,
dataset.py, week_transform.py) son redirects que re-exportan desde los
submódulos etl/ y models/. Permiten que todos los imports existentes en los
scripts raíz sigan funcionando sin modificaciones.

Scripts que orquestan el pipeline completo (en la raíz del repositorio)
------------------------------------------------------------------------
  run_runner_fatigue.py   Fase M1: LOAO + entrenamiento del regresor de fatiga
  run_runner.py           Fase M2: LOAO + entrenamiento del clasificador de lesión
  run_runner_ablation.py  Estudio de ablación (con/sin M1)
  run_runner_week_validation.py  Validación cruzada de granularidad (diario vs. semanal)
  run_comparison.py       Comparación final de configuraciones
  run_shap.py             Análisis de importancia SHAP

Dependencias externas
---------------------
  src/injury/   Infraestructura compartida: InjuryConfig, InjuryDatasetBundle,
                normalizers, augmentation (SMOTE). Usada por models/injury.py.
"""
