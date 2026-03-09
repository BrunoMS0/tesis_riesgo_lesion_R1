"""
integration – R6 Two-Stage Predictive Pipeline.

Orchestrates R4 (fatigue estimation) and R5 (injury risk prediction)
in sequence, passing the Dynamic Fatigue Index in-memory.
"""

from .pipeline import run  # noqa: F401
