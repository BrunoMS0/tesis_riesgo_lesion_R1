"""
dataset.py — Redirect de compatibilidad.

El código fuente está en src/runner/models/injury.py.
Este archivo se mantiene para que los imports existentes sigan funcionando sin cambios.

Para nuevos desarrollos, importar directamente desde:
    from src.runner.models.injury import build_runner_datasets, make_runner_injury_config
"""
from .models.injury import build_runner_datasets, make_runner_injury_config  # noqa: F401

__all__ = ["build_runner_datasets", "make_runner_injury_config"]
