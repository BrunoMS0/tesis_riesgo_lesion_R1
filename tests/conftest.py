"""
conftest.py – Shared pytest fixtures for ETL pipeline tests.

Provides lightweight synthetic data so that tests execute in seconds
without requiring the real PMData on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.etl.config import PipelineConfig

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
RNG = np.random.RandomState(42)
N_DAYS = 60
PIDs = ["p01", "p02", "p03"]
DATES = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")


def _make_dates_str():
    return [d.strftime("%Y-%m-%d") for d in DATES]


# ────────────────────────────────────────────────────────────
# Fixture: tiny PMData directory tree
# ────────────────────────────────────────────────────────────

@pytest.fixture()
def pmdata_dir(tmp_path: Path) -> Path:
    """
    Create a minimal PMData folder with 3 synthetic participants.

    Returns the root ``pmdata`` directory path.
    """
    dates_str = _make_dates_str()

    for pid in PIDs:
        p = tmp_path / pid
        pmsys = p / "pmsys"
        fitbit = p / "fitbit"
        pmsys.mkdir(parents=True)
        fitbit.mkdir(parents=True)

        # ── PMSYS ──────────────────────────────────────
        # injury.csv  (real PMData uses effective_time_frame + injuries)
        injuries_col = [
            '{"knee": "mild"}' if i == 10 else "{}" for i in range(N_DAYS)
        ]
        inj = pd.DataFrame({
            "effective_time_frame": dates_str,
            "injuries": injuries_col,
        })
        inj.to_csv(pmsys / "injury.csv", index=False)

        # srpe.csv  (real PMData uses end_date_time)
        srpe = pd.DataFrame({
            "end_date_time": dates_str,
            "perceived_exertion": RNG.randint(1, 10, N_DAYS).tolist(),
            "duration_min": RNG.randint(20, 120, N_DAYS).tolist(),
        })
        srpe.to_csv(pmsys / "srpe.csv", index=False)

        # wellness.csv  (real PMData uses effective_time_frame)
        well = pd.DataFrame({
            "effective_time_frame": dates_str,
            "fatigue": RNG.randint(1, 7, N_DAYS).tolist(),
            "mood": RNG.randint(1, 7, N_DAYS).tolist(),
            "readiness": RNG.randint(1, 7, N_DAYS).tolist(),
            "sleep_quality": RNG.randint(1, 7, N_DAYS).tolist(),
            "soreness": RNG.randint(1, 7, N_DAYS).tolist(),
            "stress": RNG.randint(1, 7, N_DAYS).tolist(),
        })
        well.to_csv(pmsys / "wellness.csv", index=False)

        # ── FITBIT (JSON daily‑sum) ───────────────────
        for name in ["steps", "distance", "calories"]:
            data = [
                {"dateTime": d, "value": str(RNG.randint(100, 10000))}
                for d in dates_str
            ]
            (fitbit / f"{name}.json").write_text(json.dumps(data))

        # ── FITBIT active‑minutes ─────────────────────
        for name in [
            "lightly_active_minutes", "moderately_active_minutes",
            "very_active_minutes", "sedentary_minutes",
        ]:
            data = [
                {"dateTime": d, "value": str(RNG.randint(0, 300))}
                for d in dates_str
            ]
            (fitbit / f"{name}.json").write_text(json.dumps(data))

        # ── resting_heart_rate.json ───────────────────
        rhr = [
            {"dateTime": d, "value": {"value": int(RNG.randint(50, 75)),
                                       "error": 1}}
            for d in dates_str
        ]
        (fitbit / "resting_heart_rate.json").write_text(json.dumps(rhr))

        # ── sleep_score.csv ───────────────────────────
        ss = pd.DataFrame({
            "timestamp": dates_str,
            "overall_score": RNG.randint(50, 95, N_DAYS).tolist(),
            "revitalization_score": RNG.randint(10, 30, N_DAYS).tolist(),
            "deep_sleep_in_minutes": RNG.randint(20, 90, N_DAYS).tolist(),
            "resting_heart_rate": RNG.randint(50, 70, N_DAYS).tolist(),
            "restlessness": RNG.uniform(0.01, 0.3, N_DAYS).tolist(),
        })
        ss.to_csv(fitbit / "sleep_score.csv", index=False)

        # ── sleep.json ────────────────────────────────
        sleep = [
            {
                "dateOfSleep": d,
                "minutesAsleep": int(RNG.randint(300, 540)),
                "minutesAwake": int(RNG.randint(10, 60)),
                "efficiency": int(RNG.randint(80, 99)),
                "timeInBed": int(RNG.randint(360, 600)),
            }
            for d in dates_str
        ]
        (fitbit / "sleep.json").write_text(json.dumps(sleep))

        # ── time_in_heart_rate_zones.json ─────────────
        zones = [
            {
                "dateTime": d,
                "value": {
                    "valuesInZones": {
                        "BELOW_DEFAULT_ZONE_1": int(RNG.randint(500, 1400)),
                        "IN_DEFAULT_ZONE_1": int(RNG.randint(0, 60)),
                        "IN_DEFAULT_ZONE_2": int(RNG.randint(0, 30)),
                        "IN_DEFAULT_ZONE_3": int(RNG.randint(0, 15)),
                    }
                },
            }
            for d in dates_str
        ]
        (fitbit / "time_in_heart_rate_zones.json").write_text(
            json.dumps(zones)
        )

        # ── exercise.json ─────────────────────────────
        # Multiple entries per day (to test aggregation)
        exercise = []
        for d in dates_str:
            for _ in range(RNG.randint(0, 3)):
                exercise.append({
                    "startTime": f"{d}T08:00:00.000",
                    "activeDuration": int(RNG.randint(600_000, 3_600_000)),
                    "averageHeartRate": int(RNG.randint(100, 170)),
                    "calories": int(RNG.randint(50, 500)),
                    "duration": int(RNG.randint(600_000, 3_600_000)),
                    "steps": int(RNG.randint(500, 10000)),
                })
        (fitbit / "exercise.json").write_text(json.dumps(exercise))

    return tmp_path


@pytest.fixture()
def cfg(pmdata_dir: Path, tmp_path: Path) -> PipelineConfig:
    """Return a PipelineConfig pointing to the synthetic data."""
    return PipelineConfig(
        raw_data_path=str(pmdata_dir),
        output_path=str(tmp_path / "outputs"),
        participants=PIDs,
    )


@pytest.fixture()
def raw_df(cfg: PipelineConfig) -> pd.DataFrame:
    """Pre‑built raw DataFrame from Extract (for Transform tests)."""
    from src.etl.extract import extract_all
    return extract_all(cfg)
