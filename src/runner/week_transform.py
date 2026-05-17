"""
week_transform.py — Redirect de compatibilidad.

El código fuente está en src/runner/etl/week_transform.py.
Este archivo se mantiene para que los imports existentes sigan funcionando sin cambios.

Para nuevos desarrollos, importar directamente desde:
    from src.runner.etl.week_transform import load_week_approach, WEEK_COMMON_FEATURES
"""
from .etl.week_transform import load_week_approach, WEEK_COMMON_FEATURES, WEEK_TO_DAILY_MAP, WEEK_CSV  # noqa: F401

__all__ = ["load_week_approach", "WEEK_COMMON_FEATURES", "WEEK_TO_DAILY_MAP", "WEEK_CSV"]
