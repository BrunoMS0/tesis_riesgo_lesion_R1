"""
soccermon — SoccerMon dataset ETL for R5 external validation.

Converts the SoccerMon dataset (Midoglu et al., 2024, Scientific Data) to the
R5-compatible player-day format with real injury labels for external evaluation
of the injury-risk model trained on PMData.
"""
from .config import SoccerMonConfig
from .pipeline import run

__all__ = ["SoccerMonConfig", "run"]
