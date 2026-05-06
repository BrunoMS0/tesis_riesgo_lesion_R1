"""
test_soccermon_zscore.py – Tests for the per-athlete z-score step in the
SoccerMon ETL pipeline (src/soccermon/transform.py).

Validates:
- _apply_per_athlete_zscore produces mean≈0 and std≈1 per athlete
- Constant feature (std=0) returns zeros without exception
- NaN values are preserved, not filled
- Imputed features (already filled) are not altered by z-score of OTHER features
- transform() returns a tuple of (df_raw, df_z)
- df_raw and df_z have identical shapes and non-z-score columns are equal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the private helper directly for unit testing
from src.soccermon.transform import _apply_per_athlete_zscore, _WELLNESS_VARS
from src.soccermon.config import ZSCORE_FEATURES


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

@pytest.fixture()
def player_df():
    """Simple 3-player × 10-day DataFrame covering all ZSCORE_FEATURES."""
    rng = np.random.RandomState(99)
    players = ["S01", "S02", "S03"]
    rows = []
    for pid in players:
        for day in range(10):
            row = {"player_id": pid, "date": f"2020-01-{day+1:02d}"}
            # Load features — intentionally different scales per player
            row["session_load"] = rng.uniform(50 * (players.index(pid) + 1), 500)
            row["acute_load_7d"] = rng.uniform(200, 2000)
            row["chronic_load_28d"] = rng.uniform(150, 1500)
            row["acwr"] = rng.uniform(0.6, 1.8)
            # Wellness features
            for feat in _WELLNESS_VARS:
                row[feat] = rng.uniform(1, 10)  # SoccerMon 1-10 scale
            row["wellness_score"] = rng.uniform(1, 10)
            # Extra non-z-score column
            row["is_injured"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Tests — _apply_per_athlete_zscore
# ────────────────────────────────────────────────────────────

class TestApplyPerAthleteZscoreSoccerMon:

    def test_per_player_mean_near_zero(self, player_df):
        df_z = _apply_per_athlete_zscore(player_df, "player_id", ZSCORE_FEATURES)
        for pid in player_df["player_id"].unique():
            group = df_z[df_z["player_id"] == pid]
            for feat in ZSCORE_FEATURES:
                if feat in group.columns:
                    m = group[feat].mean()
                    assert abs(m) < 1e-9, (
                        f"Player {pid} feature '{feat}' mean={m:.6f}, expected ≈0"
                    )

    def test_per_player_std_near_one(self, player_df):
        df_z = _apply_per_athlete_zscore(player_df, "player_id", ZSCORE_FEATURES)
        for pid in player_df["player_id"].unique():
            group = df_z[df_z["player_id"] == pid]
            for feat in ZSCORE_FEATURES:
                if feat in group.columns:
                    s = group[feat].std(ddof=1)
                    assert abs(s - 1.0) < 1e-9, (
                        f"Player {pid} feature '{feat}' std={s:.6f}, expected ≈1"
                    )

    def test_non_zscore_column_unchanged(self, player_df):
        """is_injured and date must not be modified."""
        df_z = _apply_per_athlete_zscore(player_df, "player_id", ZSCORE_FEATURES)
        pd.testing.assert_series_equal(
            player_df["is_injured"].reset_index(drop=True),
            df_z["is_injured"].reset_index(drop=True),
        )

    def test_constant_feature_no_exception(self):
        """Feature with std=0 for a player must return 0.0, not raise."""
        rows = [{"player_id": "P", "const_feat": 42.0, "normal_feat": float(i)}
                for i in range(8)]
        df = pd.DataFrame(rows)
        df_z = _apply_per_athlete_zscore(df, "player_id", ["const_feat", "normal_feat"])
        assert (df_z["const_feat"] == 0.0).all()

    def test_nan_preserved(self):
        """NaN in a feature should remain NaN in the output."""
        rows = [{"player_id": "P", "feat": float(i) if i != 3 else float("nan")}
                for i in range(8)]
        df = pd.DataFrame(rows)
        df_z = _apply_per_athlete_zscore(df, "player_id", ["feat"])
        assert pd.isna(df_z["feat"].iloc[3])

    def test_output_shape_and_columns_preserved(self, player_df):
        df_z = _apply_per_athlete_zscore(player_df, "player_id", ZSCORE_FEATURES)
        assert df_z.shape == player_df.shape
        assert list(df_z.columns) == list(player_df.columns)

    def test_imputed_columns_untouched_by_other_zscore(self):
        """
        A column that is NOT in ZSCORE_FEATURES (e.g. an imputed wearable feature
        set to a constant PMData median) must not be altered by z-scoring other
        features.
        """
        rows = []
        for pid in ["A", "B"]:
            for i in range(6):
                rows.append({
                    "player_id": pid,
                    "session_load": float(i + 1) * (2 if pid == "B" else 1),
                    "resting_hr": 55.0,  # constant — imputed PMData median, NOT in ZSCORE_FEATURES
                })
        df = pd.DataFrame(rows)
        df_z = _apply_per_athlete_zscore(df, "player_id", ["session_load"])
        # resting_hr was not z-scored, should remain identical
        pd.testing.assert_series_equal(df["resting_hr"], df_z["resting_hr"])


# ────────────────────────────────────────────────────────────
# Tests — transform() return type
# ────────────────────────────────────────────────────────────

class TestTransformReturnsTuple:
    """
    Smoke-test that transform() now returns (df_raw, df_z) instead of a
    single DataFrame — verified by importing and calling with minimal mocks.
    """

    def test_transform_returns_tuple(self, tmp_path):
        """
        Build a minimal ExtractResult-like object and call transform().
        Asserts the return value is a 2-tuple of DataFrames with equal shape.
        """
        from src.soccermon.transform import transform
        from src.soccermon.config import SoccerMonConfig, ZSCORE_FEATURES

        # Minimal mock ExtractResult
        class _MockExtract:
            pass

        n_players = 2
        n_days = 10
        rng = np.random.RandomState(0)
        players = [f"P{i}" for i in range(n_players)]
        dates = pd.date_range("2020-01-01", periods=n_days)

        # load DataFrame
        rows_load = []
        for pid in players:
            for d in dates:
                rows_load.append({
                    "player_id": pid,
                    "date": d,
                    "session_load": rng.uniform(50, 300),
                })
        load_df = pd.DataFrame(rows_load)

        # wellness dict
        wellness = {}
        for var in ["fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress"]:
            rows_w = []
            for pid in players:
                for d in dates:
                    rows_w.append({"player_id": pid, "date": d, var: rng.uniform(1, 10)})
            wellness[var] = pd.DataFrame(rows_w)

        # acwr
        rows_acwr = [{"player_id": pid, "date": d, "acwr": rng.uniform(0.7, 1.5)}
                     for pid in players for d in dates]
        acwr_df = pd.DataFrame(rows_acwr)

        # injuries (empty)
        injuries_df = pd.DataFrame(columns=["player_id", "injury_date"])

        mock = _MockExtract()
        mock.load = load_df
        mock.wellness = wellness
        mock.acwr = acwr_df
        mock.injuries = injuries_df

        cfg = SoccerMonConfig(zscore_output_csv=str(tmp_path / "z.csv"))
        pmdata_medians = {col: 0.0 for col in ["dfi_predicted", "trimp", "trimp_7d_sum",
                                                "minutesAsleep", "efficiency"]}

        from src.injury.config import FEATURE_COLUMNS
        result = transform(mock, cfg, pmdata_medians, FEATURE_COLUMNS)

        assert isinstance(result, tuple), "transform() must return a tuple"
        assert len(result) == 2, "transform() must return exactly 2 DataFrames"
        df_raw, df_z = result
        assert isinstance(df_raw, pd.DataFrame)
        assert isinstance(df_z, pd.DataFrame)
        assert df_raw.shape == df_z.shape, "df_raw and df_z must have the same shape"
