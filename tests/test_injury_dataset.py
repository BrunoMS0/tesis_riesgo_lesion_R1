"""
test_injury_dataset.py – Tests for R5 data preparation.

Validates:
- DFI merge logic (left join + cold-start fill)
- Feature selection (correct columns)
- Participant split (no leakage, all assigned)
- InjuryDatasetBundle construction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.injury.config import InjuryConfig, FEATURE_COLUMNS
from src.injury.dataset import (
    apply_per_athlete_zscore,
    build_injury_datasets,
    create_prospective_target,
    load_and_merge,
    prepare_features,
    split_participants,
)

# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

N_DAYS = 30
PIDS = ["p01", "p02", "p03", "p04", "p05"]
RNG = np.random.RandomState(42)

# All features that R5 expects (minus dfi_predicted, which comes from DFI CSV)
_BASE_FEATURES = [c for c in FEATURE_COLUMNS if c != "dfi_predicted"]


@pytest.fixture()
def synthetic_csvs(tmp_path):
    """Create a minimal feature CSV and DFI predictions CSV."""
    rows = []
    dfi_rows = []
    for pid in PIDS:
        dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
        for i, d in enumerate(dates):
            row = {
                "participant_id": pid, "date": d,
                "is_injured": 1 if i == 10 else 0,
            }
            for feat in _BASE_FEATURES:
                row[feat] = float(RNG.uniform(0, 100))
            # Add extra columns that exist in real data but aren't R5 features
            row["distance"] = float(RNG.uniform(0, 10))
            row["perceived_exertion"] = float(RNG.randint(1, 10))
            row["duration_min"] = float(RNG.randint(20, 120))
            row["sleep_duration_h"] = float(RNG.uniform(5, 9))
            rows.append(row)

            # DFI predictions (skip first 14 days to simulate cold start)
            if i >= 14:
                dfi_rows.append({
                    "participant_id": pid,
                    "date": d,
                    "dfi_predicted": float(RNG.uniform(0, 1)),
                    "dfi_actual": float(RNG.uniform(0, 1)),
                })

    feature_csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(feature_csv, index=False)

    dfi_csv = tmp_path / "dfi_predictions.csv"
    pd.DataFrame(dfi_rows).to_csv(dfi_csv, index=False)

    return feature_csv, dfi_csv


@pytest.fixture()
def injury_cfg(synthetic_csvs, tmp_path):
    """InjuryConfig pointing to synthetic data.

    Prospective target is disabled here so that legacy tests remain
    deterministic (row count = N_DAYS × N_PIDS, no tail dropped).
    Use ``cfg_prospective`` fixture for prospective-target tests.
    """
    feature_csv, dfi_csv = synthetic_csvs
    return InjuryConfig(
        input_csv=str(feature_csv),
        dfi_csv=str(dfi_csv),
        output_path=str(tmp_path / "injury_out"),
        n_synthetic_athletes=0,  # disable augmentation for dataset tests
        use_prospective_target=False,  # keep legacy is_injured target for baseline tests
    )


# ────────────────────────────────────────────────────────────
# Tests — Load & Merge
# ────────────────────────────────────────────────────────────

class TestLoadAndMerge:
    def test_merge_adds_dfi_column(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        assert "dfi_predicted" in df.columns

    def test_cold_start_filled(self, injury_cfg):
        """First 14 days per participant have no DFI — should be filled."""
        df = load_and_merge(injury_cfg)
        assert df["dfi_predicted"].isna().sum() == 0

    def test_row_count_preserved(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        assert len(df) == N_DAYS * len(PIDS)


# ────────────────────────────────────────────────────────────
# Tests — Feature Selection
# ────────────────────────────────────────────────────────────

class TestPrepareFeatures:
    def test_no_metadata_in_features(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        X, y, meta = prepare_features(df, injury_cfg)
        assert "participant_id" not in X.columns
        assert "date" not in X.columns
        assert "is_injured" not in X.columns

    def test_target_is_binary(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        _, y, _ = prepare_features(df, injury_cfg)
        assert set(y.unique()).issubset({0, 1})

    def test_dfi_in_features(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        X, _, _ = prepare_features(df, injury_cfg)
        assert "dfi_predicted" in X.columns


# ────────────────────────────────────────────────────────────
# Tests — Participant Split
# ────────────────────────────────────────────────────────────

class TestSplitParticipants:
    def test_all_assigned(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        train, val, test = split_participants(df, injury_cfg)
        assert set(train + val + test) == set(PIDS)

    def test_no_overlap(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        train, val, test = split_participants(df, injury_cfg)
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)

    def test_deterministic(self, injury_cfg):
        df = load_and_merge(injury_cfg)
        split1 = split_participants(df, injury_cfg)
        split2 = split_participants(df, injury_cfg)
        assert split1 == split2


# ────────────────────────────────────────────────────────────
# Tests — Dataset Bundle
# ────────────────────────────────────────────────────────────

class TestBuildInjuryDatasets:
    def test_bundle_shapes(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert len(bundle.X_train) == len(bundle.y_train)
        assert len(bundle.X_val) == len(bundle.y_val)
        assert len(bundle.X_test) == len(bundle.y_test)

    def test_no_participant_leakage(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        train_pids = set(bundle.meta_train["participant_id"].unique())
        val_pids = set(bundle.meta_val["participant_id"].unique())
        test_pids = set(bundle.meta_test["participant_id"].unique())
        assert not train_pids & test_pids
        assert not train_pids & val_pids

    def test_total_rows(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
        assert total == N_DAYS * len(PIDS)


# ────────────────────────────────────────────────────────────
# Tests — Normalization
# ────────────────────────────────────────────────────────────

class TestBundleNormalization:
    def test_normalizer_present(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert bundle.normalizer is not None

    def test_train_mean_near_zero(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        means = bundle.X_train.mean()
        assert means.abs().max() < 0.5, f"Max train mean {means.abs().max():.3f}"

    def test_train_std_near_one(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        stds = bundle.X_train.std()
        assert (stds - 1).abs().max() < 0.5, f"Max std deviation {stds.max():.3f}"

    def test_val_test_scaled(self, injury_cfg):
        """Val should be scaled using train normalizer (not raw).
        X_test may be empty when TEST_PARTICIPANTS=[] — skip assertion in that case."""
        bundle = build_injury_datasets(injury_cfg)
        # Raw data was uniform(0, 100) — normalized val means must be far from 50
        assert bundle.X_val.mean().abs().mean() < 10
        if len(bundle.X_test) > 0:
            assert bundle.X_test.mean().abs().mean() < 10

    def test_normalizer_has_reports(self, injury_cfg):
        bundle = build_injury_datasets(injury_cfg)
        assert len(bundle.normalizer.pre_report) > 0
        assert len(bundle.normalizer.post_report) > 0
        assert "ks_statistic" in bundle.normalizer.pre_report.columns
        assert "is_normal" in bundle.normalizer.post_report.columns


# ────────────────────────────────────────────────────────────
# Tests — Per-Athlete Z-Score (cross-domain alignment)
# ────────────────────────────────────────────────────────────

class TestApplyPerAthleteZscore:
    """Tests for apply_per_athlete_zscore() in dataset.py."""

    @pytest.fixture()
    def simple_data(self):
        """3 athletes × 10 days, 2 features with known distributions."""
        rng = np.random.RandomState(7)
        rows = []
        for pid in ["A", "B", "C"]:
            for day in range(10):
                rows.append({
                    "participant_id": pid,
                    "feat_load": rng.uniform(100, 500),  # different scales per athlete
                    "feat_wellness": rng.uniform(1, 6),
                    "feat_other": rng.uniform(0, 1),
                })
        df = pd.DataFrame(rows)
        X = df[["feat_load", "feat_wellness", "feat_other"]].copy()
        meta = df[["participant_id"]].copy()
        return X, meta

    def test_per_athlete_mean_near_zero(self, simple_data):
        X, meta = simple_data
        X_z = apply_per_athlete_zscore(X, meta, ["feat_load", "feat_wellness"])
        X_z["_pid"] = meta["participant_id"].values
        for pid in ["A", "B", "C"]:
            group = X_z[X_z["_pid"] == pid]
            assert abs(group["feat_load"].mean()) < 1e-9, (
                f"Athlete {pid} feat_load mean not ≈0: {group['feat_load'].mean()}")
            assert abs(group["feat_wellness"].mean()) < 1e-9, (
                f"Athlete {pid} feat_wellness mean not ≈0: {group['feat_wellness'].mean()}")

    def test_per_athlete_std_near_one(self, simple_data):
        X, meta = simple_data
        X_z = apply_per_athlete_zscore(X, meta, ["feat_load", "feat_wellness"])
        X_z["_pid"] = meta["participant_id"].values
        for pid in ["A", "B", "C"]:
            group = X_z[X_z["_pid"] == pid]
            std_load = group["feat_load"].std(ddof=1)
            std_well = group["feat_wellness"].std(ddof=1)
            assert abs(std_load - 1.0) < 1e-9, (
                f"Athlete {pid} feat_load std not ≈1: {std_load}")
            assert abs(std_well - 1.0) < 1e-9, (
                f"Athlete {pid} feat_wellness std not ≈1: {std_well}")

    def test_untouched_feature_unchanged(self, simple_data):
        X, meta = simple_data
        X_z = apply_per_athlete_zscore(X, meta, ["feat_load"])
        # feat_other was NOT in feature_list — should be identical
        pd.testing.assert_series_equal(X["feat_other"], X_z["feat_other"])

    def test_constant_feature_returns_zeros(self):
        """Feature with std=0 for an athlete must not raise and must return 0."""
        X = pd.DataFrame({
            "feat_const": [5.0] * 5,  # all same → std=0
            "feat_normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        meta = pd.DataFrame({"participant_id": ["X"] * 5})
        X_z = apply_per_athlete_zscore(X, meta, ["feat_const", "feat_normal"])
        assert (X_z["feat_const"] == 0.0).all(), "constant feature should be all zeros"

    def test_nan_preserved(self):
        """NaN in input features should remain NaN after z-score."""
        X = pd.DataFrame({
            "feat_a": [1.0, float("nan"), 3.0, 4.0, 5.0],
        })
        meta = pd.DataFrame({"participant_id": ["P"] * 5})
        X_z = apply_per_athlete_zscore(X, meta, ["feat_a"])
        assert pd.isna(X_z["feat_a"].iloc[1]), "NaN should be preserved"

    def test_absent_feature_skipped(self, simple_data):
        """Feature in feature_list but absent in X should be silently skipped."""
        X, meta = simple_data
        X_z = apply_per_athlete_zscore(X, meta, ["feat_load", "nonexistent_col"])
        assert "nonexistent_col" not in X_z.columns

    def test_output_shape_preserved(self, simple_data):
        X, meta = simple_data
        X_z = apply_per_athlete_zscore(X, meta, ["feat_load", "feat_wellness"])
        assert X_z.shape == X.shape


# ────────────────────────────────────────────────────────────
# Tests — Prospective Target (injury_next7d)
# ────────────────────────────────────────────────────────────

class TestCreateProspectiveTarget:
    """Tests for create_prospective_target() in dataset.py."""

    @pytest.fixture()
    def simple_df(self):
        """2 athletes × 20 days; athlete A has injury on day 5 only."""
        rows = []
        for pid, inj_day in [("A", 5), ("B", 15)]:
            dates = pd.date_range("2020-01-01", periods=20, freq="D")
            for i, d in enumerate(dates):
                rows.append({
                    "participant_id": pid,
                    "date": d,
                    "is_injured": 1 if i == inj_day else 0,
                })
        return pd.DataFrame(rows)

    def test_column_created(self, simple_df):
        out = create_prospective_target(simple_df, window=7)
        assert "injury_next7d" in out.columns

    def test_tail_rows_dropped(self, simple_df):
        window = 7
        out = create_prospective_target(simple_df, window=window)
        # Each athlete loses `window` tail rows
        n_athletes = simple_df["participant_id"].nunique()
        expected = len(simple_df) - n_athletes * window
        assert len(out) == expected

    def test_positives_increase(self, simple_df):
        """Each injury event inflates positives by up to window rows."""
        out = create_prospective_target(simple_df, window=7)
        n_original = int(simple_df["is_injured"].sum())      # 2 events
        n_prospective = int(out["injury_next7d"].sum())
        assert n_prospective > n_original, (
            f"Expected more prospective positives ({n_prospective}) than "
            f"original ({n_original})"
        )

    def test_no_data_leakage_shift_correct(self, simple_df):
        """At day T, label is 1 only if injury falls in days T+1..T+window.
        Athlete A has injury at day 5 (index 4 within that athlete's rows).
        Days 0–3 (T+1=5 to T+1+6=11 for T=3) should be positive (3 days before injury).
        Day 4 (injury day itself) should have NaN/dropped or label=0."""
        window = 7
        out = create_prospective_target(simple_df, window=window)
        a = out[out["participant_id"] == "A"].reset_index(drop=True)

        # Day T=0: injury at day 5 is 5 days ahead → within window(7) → label=1
        assert a.loc[0, "injury_next7d"] == 1, "T=0 should be 1 (injury 5 days ahead)"

        # Day T=3: injury at day 5 is 2 days ahead → label=1
        assert a.loc[3, "injury_next7d"] == 1, "T=3 should be 1 (injury 2 days ahead)"

        # Day T=4 would be injury day itself; the label should reflect days T+1..T+7
        # days 5-11 from athlete A: only day 5 is injured → label=1
        assert a.loc[4, "injury_next7d"] == 1, "T=4: day5=injury is in [T+1..T+7]"

        # Day T=5 (injury day): next 7 days (6-12) have no injury → label=0
        assert a.loc[5, "injury_next7d"] == 0, "T=5 (injury day): no future injury"

    def test_target_is_binary_integers(self, simple_df):
        out = create_prospective_target(simple_df, window=7)
        assert set(out["injury_next7d"].unique()).issubset({0, 1})
        assert out["injury_next7d"].dtype in (int, "int64", "int32")

    def test_is_injured_preserved(self, simple_df):
        """Original is_injured column must not be modified."""
        out = create_prospective_target(simple_df, window=7)
        # Only the tail rows are dropped; remaining is_injured values should match
        merged = simple_df.merge(
            out[["participant_id", "date", "is_injured"]].rename(
                columns={"is_injured": "is_injured_out"}
            ),
            on=["participant_id", "date"],
        )
        pd.testing.assert_series_equal(
            merged["is_injured"], merged["is_injured_out"], check_names=False
        )


# ────────────────────────────────────────────────────────────
# Tests — build_injury_datasets with prospective target
# ────────────────────────────────────────────────────────────

class TestBuildDatasetsProspective:
    """Verify that build_injury_datasets uses injury_next7d when configured."""

    @pytest.fixture()
    def cfg_prospective(self, injury_cfg):
        """Config with prospective target enabled (default True)."""
        from dataclasses import replace
        return replace(injury_cfg, use_prospective_target=True, prospective_window=7)

    @pytest.fixture()
    def cfg_no_prospective(self, injury_cfg):
        """Config with prospective target disabled (legacy is_injured)."""
        from dataclasses import replace
        return replace(injury_cfg, use_prospective_target=False)

    def test_prospective_bundle_has_more_positives(self, cfg_prospective, cfg_no_prospective):
        bundle_pro = build_injury_datasets(cfg_prospective)
        bundle_leg = build_injury_datasets(cfg_no_prospective)
        pos_pro = int(bundle_pro.y_train.sum()) + int(bundle_pro.y_val.sum())
        pos_leg = int(bundle_leg.y_train.sum()) + int(bundle_leg.y_val.sum())
        assert pos_pro >= pos_leg, (
            f"Prospective positives ({pos_pro}) should be >= legacy ({pos_leg})"
        )

    def test_legacy_bundle_total_rows(self, cfg_no_prospective):
        """Legacy mode: total rows = N_DAYS * n_pids (no tail dropped)."""
        bundle = build_injury_datasets(cfg_no_prospective)
        total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
        assert total == N_DAYS * len(PIDS)

    def test_prospective_bundle_fewer_total_rows(self, cfg_prospective):
        """Prospective mode drops last 7 days per athlete."""
        bundle = build_injury_datasets(cfg_prospective)
        total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
        expected_max = N_DAYS * len(PIDS)
        assert total < expected_max, (
            f"Expected fewer rows in prospective mode ({total} >= {expected_max})"
        )


# ────────────────────────────────────────────────────────────
# Tests — build_combined_dataset
# ────────────────────────────────────────────────────────────

class TestBuildCombinedDataset:
    """Verify combined PMData+SoccerMon dataset bundle."""

    @pytest.fixture()
    def soccermon_prospective_csv(self, tmp_path):
        """Create a minimal fake SoccerMon prospective CSV."""
        from src.injury.config import SOCCERMON_SHARED_FEATURES
        sm_pids = [f"sm{i:02d}" for i in range(1, 6)]  # 5 fake SoccerMon athletes
        rows = []
        rng = np.random.RandomState(99)
        for pid in sm_pids:
            dates = pd.date_range("2020-01-01", periods=30, freq="D")
            for i, d in enumerate(dates):
                row = {"participant_id": pid, "date": d, "is_injured": 1 if i == 5 else 0}
                row["injury_next7d"] = 1 if i in range(5, 12) else 0
                for feat in SOCCERMON_SHARED_FEATURES:
                    row[feat] = float(rng.uniform(0, 100))
                rows.append(row)
        csv_path = tmp_path / "soccermon_prospective.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture()
    def cfg_combined(self, injury_cfg, soccermon_prospective_csv):
        """InjuryConfig with combined training enabled."""
        from dataclasses import replace
        return replace(
            injury_cfg,
            use_prospective_target=True,
            use_combined_training=True,
            soccermon_prospective_csv=str(soccermon_prospective_csv),
        )

    def test_combined_returns_bundle(self, cfg_combined):
        from src.injury.dataset import build_combined_dataset
        bundle = build_combined_dataset(cfg_combined)
        assert bundle is not None

    def test_combined_feature_columns_are_shared(self, cfg_combined):
        """Feature columns should be SOCCERMON_SHARED_FEATURES + source_dataset."""
        from src.injury.dataset import build_combined_dataset
        from src.injury.config import SOCCERMON_SHARED_FEATURES
        bundle = build_combined_dataset(cfg_combined)
        expected = set(SOCCERMON_SHARED_FEATURES) | {"source_dataset"}
        assert set(bundle.feature_columns) == expected

    def test_combined_train_has_pmdata_only(self, cfg_combined):
        """Training split should have source_dataset == 0 (PMData) only."""
        from src.injury.dataset import build_combined_dataset
        bundle = build_combined_dataset(cfg_combined)
        assert (bundle.X_train["source_dataset"] == 0).all(), (
            "Train split must contain only PMData rows (source_dataset=0)"
        )

    def test_combined_test_has_soccermon_only(self, cfg_combined):
        """Test split should have source_dataset == 1 (SoccerMon) only."""
        from src.injury.dataset import build_combined_dataset
        bundle = build_combined_dataset(cfg_combined)
        assert (bundle.X_test["source_dataset"] == 1).all(), (
            "Test split must contain only SoccerMon rows (source_dataset=1)"
        )

    def test_combined_no_participant_leakage(self, cfg_combined):
        """No athlete should appear in both train and test splits."""
        from src.injury.dataset import build_combined_dataset
        bundle = build_combined_dataset(cfg_combined)
        train_pids = set(bundle.meta_train["participant_id"].unique())
        test_pids = set(bundle.meta_test["participant_id"].unique())
        overlap = train_pids & test_pids
        assert len(overlap) == 0, f"Participant leakage: {overlap}"

    def test_combined_soccermon_pids_prefixed(self, cfg_combined):
        """SoccerMon participant IDs should be prefixed with 'sm_'."""
        from src.injury.dataset import build_combined_dataset
        bundle = build_combined_dataset(cfg_combined)
        sm_pids = bundle.meta_test["participant_id"].unique()
        assert all(str(p).startswith("sm_") for p in sm_pids), (
            f"Expected all test pids to start with 'sm_', got: {sm_pids}"
        )

