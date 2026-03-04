"""
extract.py – EXTRACT stage of the ETL pipeline.

Reads raw PMData files (PMSYS CSVs + Fitbit JSONs) for every
participant and returns a single consolidated DataFrame at a
*daily* granularity.

Public API
----------
extract_all(cfg) -> pd.DataFrame
    Run extraction for every participant in ``cfg.participants``.
extract_participant(pid, cfg) -> pd.DataFrame | None
    Extract data for a single participant.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _load_json_safe(filepath: str) -> list:
    """Load a JSON file, returning ``[]`` on any error."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _to_date_naive(series: pd.Series) -> pd.Series:
    """Convert any datetime series to tz‑naive, date‑normalised."""
    dt = pd.to_datetime(series, utc=True)
    return dt.dt.tz_localize(None).dt.normalize()


# ────────────────────────────────────────────────────────────
# Single‑participant extraction
# ────────────────────────────────────────────────────────────

def extract_participant(pid: str, cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    """
    Extract and merge all data sources for *one* participant into a
    daily‑level DataFrame.

    Parameters
    ----------
    pid : str
        Participant ID, e.g. ``"p01"``.
    cfg : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    pd.DataFrame or None
        Merged daily DataFrame, or ``None`` when no data is available.
    """
    pmsys_dir = os.path.join(cfg.raw_data_path, pid, "pmsys")
    fitbit_dir = os.path.join(cfg.raw_data_path, pid, "fitbit")
    frames: List[pd.DataFrame] = []

    # ── 1. PMSYS: injury.csv ──────────────────────────────
    fpath = os.path.join(pmsys_dir, "injury.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["date"] = _to_date_naive(df["effective_time_frame"])
        df["is_injured"] = df["injuries"].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() in ("{}", "") else 1
        )
        frames.append(
            df.groupby("date").agg(is_injured=("is_injured", "max")).reset_index()
        )

    # ── 2. PMSYS: srpe.csv ───────────────────────────────
    fpath = os.path.join(pmsys_dir, "srpe.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["date"] = _to_date_naive(df["end_date_time"])
        df["session_load"] = df["perceived_exertion"] * df["duration_min"]
        agg = (
            df.groupby("date")
            .agg(
                session_load=("session_load", "sum"),
                perceived_exertion=("perceived_exertion", "mean"),
                duration_min=("duration_min", "sum"),
            )
            .reset_index()
        )
        frames.append(agg)

    # ── 3. PMSYS: wellness.csv ────────────────────────────
    fpath = os.path.join(pmsys_dir, "wellness.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["date"] = _to_date_naive(df["effective_time_frame"])
        desired = [
            "date", "fatigue", "mood", "readiness",
            "sleep_duration_h", "sleep_quality", "soreness", "stress",
        ]
        available = [c for c in desired if c in df.columns]
        frames.append(
            df[available].drop_duplicates(subset="date", keep="last")
        )

    # ── 4. Fitbit: steps / distance / calories (minute→daily sum) ─
    for metric in cfg.__class__.__dataclass_fields__ and ["steps", "distance", "calories"]:
        data = _load_json_safe(os.path.join(fitbit_dir, f"{metric}.json"))
        if data:
            df = pd.DataFrame(data)
            df["date"] = _to_date_naive(df["dateTime"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            agg = df.groupby("date").agg(val=("value", "sum")).reset_index()
            agg.columns = ["date", metric]
            frames.append(agg)

    # ── 5. Fitbit: resting_heart_rate.json ────────────────
    data = _load_json_safe(os.path.join(fitbit_dir, "resting_heart_rate.json"))
    if data:
        df = pd.DataFrame(data)
        df["date"] = _to_date_naive(df["dateTime"])
        df["resting_hr"] = df["value"].apply(
            lambda x: x.get("value", np.nan) if isinstance(x, dict) else np.nan
        )
        frames.append(df[["date", "resting_hr"]].copy())

    # ── 6. Fitbit: sleep_score.csv ────────────────────────
    fpath = os.path.join(fitbit_dir, "sleep_score.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["date"] = _to_date_naive(df["timestamp"])
        rename_map = {}
        if "resting_heart_rate" in df.columns:
            rename_map["resting_heart_rate"] = "sleep_rhr"
        keep = [
            "date", "overall_score", "composition_score",
            "revitalization_score", "duration_score",
            "deep_sleep_in_minutes", "restlessness",
        ]
        if rename_map:
            df = df.rename(columns=rename_map)
            keep.append("sleep_rhr")
        available = [c for c in keep if c in df.columns]
        frames.append(
            df[available].drop_duplicates(subset="date", keep="last")
        )

    # ── 7. Fitbit: sleep.json ─────────────────────────────
    data = _load_json_safe(os.path.join(fitbit_dir, "sleep.json"))
    if data:
        df = pd.DataFrame(data)
        if "dateOfSleep" in df.columns:
            df["date"] = _to_date_naive(df["dateOfSleep"])
            sleep_cols = ["date"]
            for c in ["minutesAsleep", "efficiency", "minutesAwake", "timeInBed"]:
                if c in df.columns:
                    sleep_cols.append(c)
            frames.append(
                df[sleep_cols].drop_duplicates(subset="date", keep="last")
            )

    # ── 8. Fitbit: time_in_heart_rate_zones.json ──────────
    data = _load_json_safe(
        os.path.join(fitbit_dir, "time_in_heart_rate_zones.json")
    )
    if data:
        df = pd.DataFrame(data)
        df["date"] = _to_date_naive(df["dateTime"])

        def _extract_zones(val):
            if isinstance(val, dict) and "valuesInZones" in val:
                z = val["valuesInZones"]
                return pd.Series(
                    {
                        "hr_zone_below": z.get("BELOW_DEFAULT_ZONE_1", 0),
                        "hr_zone_1": z.get("IN_DEFAULT_ZONE_1", 0),
                        "hr_zone_2": z.get("IN_DEFAULT_ZONE_2", 0),
                        "hr_zone_3": z.get("IN_DEFAULT_ZONE_3", 0),
                    }
                )
            return pd.Series(
                {"hr_zone_below": 0, "hr_zone_1": 0, "hr_zone_2": 0, "hr_zone_3": 0}
            )

        zones = df["value"].apply(_extract_zones)
        zones["date"] = df["date"].values
        frames.append(zones)

    # ── 9. Fitbit: exercise.json ──────────────────────────
    data = _load_json_safe(os.path.join(fitbit_dir, "exercise.json"))
    if data:
        df = pd.DataFrame(data)
        if "startTime" in df.columns:
            df["date"] = _to_date_naive(df["startTime"])
            df["duration_ms"] = pd.to_numeric(
                df.get("activeDuration", df.get("duration", 0)),
                errors="coerce",
            ).fillna(0)
            df["exercise_cal"] = pd.to_numeric(
                df.get("calories", 0), errors="coerce"
            ).fillna(0)
            df["exercise_steps"] = pd.to_numeric(
                df.get("steps", 0), errors="coerce"
            ).fillna(0)
            df["exercise_avg_hr"] = pd.to_numeric(
                df.get("averageHeartRate", np.nan), errors="coerce"
            )
            agg = (
                df.groupby("date")
                .agg(
                    exercise_duration_min=("duration_ms", lambda x: x.sum() / 60_000),
                    exercise_calories=("exercise_cal", "sum"),
                    exercise_steps=("exercise_steps", "sum"),
                    exercise_avg_hr=("exercise_avg_hr", "mean"),
                    exercise_sessions=("date", "count"),
                )
                .reset_index()
            )
            frames.append(agg)

    # ── 10. Fitbit: *_active_minutes.json ─────────────────
    for intensity in [
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
        "sedentary_minutes",
    ]:
        data = _load_json_safe(os.path.join(fitbit_dir, f"{intensity}.json"))
        if data:
            df = pd.DataFrame(data)
            df["date"] = _to_date_naive(df["dateTime"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            agg = df.groupby("date").agg(val=("value", "sum")).reset_index()
            agg.columns = ["date", intensity]
            frames.append(agg)

    # ── Merge all sources ─────────────────────────────────
    if not frames:
        logger.warning("No data found for participant %s", pid)
        return None

    result = frames[0]
    for f in frames[1:]:
        result = pd.merge(result, f, on="date", how="outer")

    result["participant_id"] = pid
    result = result.sort_values("date").reset_index(drop=True)
    return result


# ────────────────────────────────────────────────────────────
# Batch extraction
# ────────────────────────────────────────────────────────────

def extract_all(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Run :func:`extract_participant` for every participant and
    return a single consolidated DataFrame.

    Raises
    ------
    ValueError
        If no participant yielded any data.
    """
    all_data: List[pd.DataFrame] = []
    for pid in cfg.participants:
        try:
            df_p = extract_participant(pid, cfg)
            if df_p is not None:
                all_data.append(df_p)
                logger.info(
                    "%s: %d days, %d columns", pid, len(df_p), len(df_p.columns)
                )
        except Exception as exc:
            logger.error("%s: extraction error – %s", pid, exc)

    if not all_data:
        raise ValueError("No data extracted from any participant.")

    df_global = pd.concat(all_data, ignore_index=True)
    df_global["date"] = pd.to_datetime(df_global["date"])
    logger.info(
        "Extraction complete: %d rows, %d cols, participants=%d",
        len(df_global),
        len(df_global.columns),
        df_global["participant_id"].nunique(),
    )
    return df_global
