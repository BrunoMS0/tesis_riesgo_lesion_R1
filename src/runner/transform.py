"""
transform.py — Redirect de compatibilidad.

El código fuente está en src/runner/etl/transform.py.
Este archivo se mantiene para que los imports existentes sigan funcionando sin cambios.

Para nuevos desarrollos, importar directamente desde:
    from src.runner.etl.transform import compute_features
"""
from .etl.transform import compute_features  # noqa: F401

__all__ = ["compute_features"]
