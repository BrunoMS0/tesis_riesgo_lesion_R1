"""
src/runner/etl — Pipeline ETL del Runner Dataset (Löwdal 2021).

Este submódulo implementa las tres etapas del pipeline de datos,
desde la carga del CSV fuente hasta el dataset listo para modelar.

Módulos
-------
extract.py        Paso 1 — Carga y estandarización del CSV fuente
                  (renombra columnas de identidad, elimina columnas no necesarias)

transform.py      Paso 2 — Ingeniería de features
                  (ACWR, cargas aguda/crónica, métricas de wellness, features D-1)

week_transform.py Paso 2b — Variante semanal para validación cruzada de granularidad
                  (convierte el CSV semanal a los mismos nombres de feature que daily)

pipeline.py       Orquestador ETL — ejecuta extract → transform → [guarda CSV procesado]
                  (punto de entrada único para reproducir el dataset procesado)

Uso rápido
----------
    from src.runner.etl import run_etl_pipeline
    df = run_etl_pipeline(save_output=True)
"""

from .extract import load_runner_csv
from .transform import compute_features
from .week_transform import load_week_approach
from .pipeline import run_etl_pipeline

__all__ = [
    "load_runner_csv",
    "compute_features",
    "load_week_approach",
    "run_etl_pipeline",
]
