"""
extract.py – EXTRACT stage of the SoccerMon ETL pipeline.

Reads the raw SoccerMon folder structure:
    training-load/daily_load.csv  — session RPE×duration load  (wide: date × player)
    training-load/acwr.csv        — acute:chronic workload ratio (wide: date × player)
    wellness/fatigue.csv          — fatigue self-report 1–10    (wide: date × player)
    wellness/soreness.csv         — soreness self-report 1–10   (wide)
    wellness/mood.csv             — mood self-report 1–10       (wide)
    wellness/readiness.csv        — readiness self-report 1–10  (wide)
    wellness/sleep_quality.csv    — sleep quality 1–10          (wide)
    wellness/stress.csv           — stress self-report 1–10     (wide)
    injury/injury.csv             — injury events (player_id, JSON type, timestamp)

All wide-format files have the date as the first column (label varies per file)
and player UUIDs (e.g. "TeamA-d7299614-...") as the remaining columns.

Public API
----------
extract_all(cfg) -> ExtractResult
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from .config import SoccerMonConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    """Raw long-format DataFrames extracted from SoccerMon."""

    load: pd.DataFrame                    # columns: date, player_id, session_load
    acwr: pd.DataFrame                    # columns: date, player_id, acwr
    wellness: Dict[str, pd.DataFrame]     # var_name → (date, player_id, var_name)
    injuries: pd.DataFrame                # columns: player_id, injury_date, body_part, severity


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _load_wide_csv(path: str) -> pd.DataFrame:
    """
    Load a wide-format CSV where the first column is the date (any label)
    and remaining columns are player UUIDs.

    Returns a DataFrame with a DatetimeIndex named 'date'.
    """
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    return df


def _melt_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide (date index × player columns) to long format:
        date | player_id | value_name
    """
    long = (
        df.reset_index()
        .melt(id_vars="date", var_name="player_id", value_name=value_name)
        .reset_index(drop=True)
    )
    return long


def _parse_injury_events(path: str) -> pd.DataFrame:
    """
    Parse injury.csv into a tidy DataFrame:
        player_id | injury_date | body_part | severity

    The 'type' column contains JSON strings like '{"right_thigh": "minor"}'.
    The 'timestamp' column uses DD.MM.YYYY format.
    """
    df = pd.read_csv(path)
    df["injury_date"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["injury_date"])

    body_parts, severities = [], []
    for t in df["type"]:
        try:
            d = json.loads(t)
            bp = next(iter(d.keys()), "unknown")
            sev = next(iter(d.values()), "unknown")
        except Exception:
            bp, sev = "unknown", str(t)
        body_parts.append(bp)
        severities.append(sev)

    df["body_part"] = body_parts
    df["severity"] = severities
    df = df.rename(columns={"player_name": "player_id"})
    return df[["player_id", "injury_date", "body_part", "severity"]].copy()


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def extract_all(cfg: SoccerMonConfig) -> ExtractResult:
    """Read all SoccerMon raw files and return long-format DataFrames."""
    base = cfg.base_path

    # ── Training load ────────────────────────────────────
    load_path = os.path.join(base, "training-load", "daily_load.csv")
    load_long = _melt_to_long(_load_wide_csv(load_path), "session_load")
    logger.info(
        "Loaded daily_load: %d rows, %d players",
        len(load_long), load_long["player_id"].nunique(),
    )

    # ── ACWR ─────────────────────────────────────────────
    acwr_path = os.path.join(base, "training-load", "acwr.csv")
    acwr_long = _melt_to_long(_load_wide_csv(acwr_path), "acwr")
    logger.info("Loaded acwr: %d rows", len(acwr_long))

    # ── Wellness variables ────────────────────────────────
    wellness: Dict[str, pd.DataFrame] = {}
    for var in cfg.wellness_vars:
        var_path = os.path.join(base, "wellness", f"{var}.csv")
        if os.path.exists(var_path):
            wide = _load_wide_csv(var_path)
            wlong = _melt_to_long(wide, var)
            n_obs = int(wlong[var].notna().sum())
            wellness[var] = wlong
            logger.info("Loaded wellness/%s.csv: %d non-NaN entries", var, n_obs)
        else:
            logger.warning("Wellness file not found, skipping: %s", var_path)

    # ── Injury events ─────────────────────────────────────
    inj_path = os.path.join(base, "injury", "injury.csv")
    injuries = _parse_injury_events(inj_path)
    logger.info(
        "Loaded injury events: %d events across %d players",
        len(injuries), injuries["player_id"].nunique(),
    )

    return ExtractResult(
        load=load_long,
        acwr=acwr_long,
        wellness=wellness,
        injuries=injuries,
    )
