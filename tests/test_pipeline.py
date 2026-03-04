"""
test_pipeline.py – Tests for the end‑to‑end pipeline orchestrator.

Test plan
---------
P‑1  run() returns PipelineReport with 3 stages.
P‑2  Final CSV exists on disk.
P‑3  tf.data datasets are created successfully.
P‑4  Total duration is positive.
P‑5  Report final_features > 0.
"""

from __future__ import annotations

import pytest

from src.etl.config import PipelineConfig
from src.etl.pipeline import PipelineReport, run


class TestPipelineRun:

    def test_returns_report(self, cfg: PipelineConfig):
        report = run(cfg)
        assert isinstance(report, PipelineReport)

    def test_three_stages(self, cfg: PipelineConfig):
        report = run(cfg)
        assert len(report.stages) == 3
        names = [s.name for s in report.stages]
        assert names == ["Extract", "Transform", "Load"]

    def test_csv_exists(self, cfg: PipelineConfig):
        from pathlib import Path

        report = run(cfg)
        assert report.final_csv is not None
        assert Path(report.final_csv).exists()

    def test_tf_datasets_created(self, cfg: PipelineConfig):
        pytest.importorskip("tensorflow")
        report = run(cfg)
        assert report.tf_datasets_created is True

    def test_positive_duration(self, cfg: PipelineConfig):
        report = run(cfg)
        assert report.total_duration_s > 0

    def test_features_positive(self, cfg: PipelineConfig):
        report = run(cfg)
        assert report.final_features > 0
