"""
src.etl – ETL pipeline for the PMData injury-risk prediction project.

Stages
------
1. **Extract** – read raw PMSYS + Fitbit files → consolidated DataFrame.
2. **Transform** – clean, engineer features, standardise.
3. **Load** – persist CSV + build tf.data.Dataset objects.

Quick start::

    from src.etl.pipeline import run
    report = run()   # uses default PipelineConfig
"""

from .config import PipelineConfig
from .pipeline import run

__all__ = ["PipelineConfig", "run"]
