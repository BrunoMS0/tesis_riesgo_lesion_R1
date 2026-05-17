"""
extract.py — Redirect de compatibilidad.

El código fuente está en src/runner/etl/extract.py.
Este archivo se mantiene para que los imports existentes sigan funcionando sin cambios.

Para nuevos desarrollos, importar directamente desde:
    from src.runner.etl.extract import load_runner_csv
"""
from .etl.extract import load_runner_csv  # noqa: F401

__all__ = ["load_runner_csv"]
