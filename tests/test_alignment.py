"""Tests for the sales-price-calendar alignment (Phase P1, task 2).

Builds from cache when available; the first run in a fresh clone parses the
raw M5 CSVs (~1 min) via the session fixture.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config.sku_config import load_sku_payload
from src.demand.alignment import (
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    AlignedData,
    build_aligned_dataset,
    build_feature_matrix,
)


@pytest.fixture(scope="session")
def data() -> AlignedData:
    return build_aligned_dataset()


class TestShapeAndKeys:
    def test_full_grid(self, data):
        n, days = len(data.series_ids), data.n_days
        assert len(data.df) == n * days == 19410
        assert data.df.groupby("series_id").size().eq(days).all()
        assert not data.df.duplicated(["series_id", "day_index"]).any()
        assert sorted(data.df.day_index.unique()) == list(range(1, days + 1))

    def test_sales_nonnegative_integers(self, data):
        y = data.df["y"]
        assert (y >= 0).all()
        assert np.allclose(y, y.round())

    def test_no_missing_values_in_model_columns(self, data):
        cols = FEATURE_COLUMNS + HISTORY_COLUMNS + ["y", "sell_price"]
        assert not data.df[cols].isna().any().any()


class TestTemporalSplit:
    def test_split_boundaries(self, data):
        train, test = data.split()
        payload = load_sku_payload()
        assert data.test_last_n_days == payload["protocol"]["test_last_n_days"] == 28
        assert train.day_index.max() == data.test_start_day - 1
        assert test.day_index.min() == data.test_start_day
        assert test.day_index.max() == data.n_days
        assert len(test) == 10 * 28

    def test_test_dates_match_locked_protocol(self, data):
        """The 28 test days must be exactly the dates recorded in skus.json."""
        payload = load_sku_payload()
        locked_dates = set(payload["calendar"]["test_days"].values())
        _, test = data.split()
        assert set(test["date"].dt.strftime("%Y-%m-%d")) == locked_dates

    def test_no_shuffling(self, data):
        train, _ = data.split()
        assert train.day_index.is_monotonic_increasing is False  # long format
        per_series = train.groupby("series_id")["day_index"].apply(list)
        assert all(days == sorted(days) for days in per_series)


class TestFeatures:
    def test_feature_matrix_shape(self, data):
        X = build_feature_matrix(data.df)
        assert list(X.columns) == FEATURE_COLUMNS
        assert "y" not in X.columns and "hist_7" not in X.columns

    def test_one_hots_sum_to_one(self, data):
        df = data.df
        dow = df[[f"dow_{w}" for w in range(1, 8)]].sum(axis=1)
        month = df[[f"month_{m}" for m in range(1, 13)]].sum(axis=1)
        assert (dow == 1).all() and (month == 1).all()

    def test_weekday_mapping_m5_convention(self, data):
        """M5 wday: 1 = Saturday. 2016-04-25 is a Monday -> wday 3."""
        row = data.df[data.df.date == pd.Timestamp("2016-04-25")].iloc[0]
        assert row["wday"] == 3
        assert row["dow_3"] == 1

    def test_price_rel_consistent_with_locked_base(self, data):
        payload = load_sku_payload()
        base = {s["series_id"]: s["base_price"] for s in payload["skus"]}
        sample = data.df.sample(500, random_state=0)
        expected = sample["sell_price"] / sample["series_id"].map(base)
        assert np.allclose(sample["price_rel"], expected)
        assert (sample["price_rel"] > 0).all()

    def test_snap_only_when_flag_set(self, data):
        assert set(data.df["snap"].unique()) <= {0, 1}
        # SNAP days occur roughly 10 days/month -> a few thousand over 5.3y
        assert 500 < data.df["snap"].sum() < 8000

    def test_event_flags_populated(self, data):
        assert data.df["is_event"].sum() > 100
        assert data.df["event_cultural"].sum() > 0
        assert data.df["event_sporting"].sum() > 0


class TestHistoryFeaturesNoLeakage:
    @staticmethod
    def _day_value(s: pd.DataFrame, d: int, col: str) -> float:
        return s.loc[s.day_index == d, col].iloc[0]

    def test_hist7_is_strictly_lagged(self, data):
        """hist_7 at day t == mean(y[t-7..t-1]) — recomputed by day index."""
        for sid in data.series_ids[:3]:
            s = data.df[data.df.series_id == sid].sort_values("day_index")
            for t in (100, 366, 1000):
                expected = np.mean([self._day_value(s, d, "y") for d in range(t - 7, t)])
                actual = self._day_value(s, t, "hist_7")
                assert np.isclose(actual, expected), (sid, t)

    def test_hist28_is_strictly_lagged(self, data):
        """hist_28 at day t == mean(y[t-28..t-1]) — recomputed by day index."""
        for sid in data.series_ids[:3]:
            s = data.df[data.df.series_id == sid].sort_values("day_index")
            for t in (200, 500, 1200):
                expected = np.mean([self._day_value(s, d, "y") for d in range(t - 28, t)])
                actual = self._day_value(s, t, "hist_28")
                assert np.isclose(actual, expected), (sid, t)

    def test_cold_start_filled_without_future(self, data):
        """Day-1 rows have zero history; no NaN anywhere."""
        first = data.df[data.df.day_index == 1]
        assert (first[["hist_7", "hist_28"]] == 0).all().all()


class TestPriceCoverage:
    def test_price_observed_flag_and_fill(self, data):
        cov = data.price_coverage()
        assert (cov >= 0.5).all(), "a series should not have >50% missing prices"
        # Where the flag is 0, the price is a ffilled/bfilled constant per series:
        filled = data.df[data.df.price_observed == 0]
        for sid, grp in filled.groupby("series_id"):
            assert grp["sell_price"].nunique() <= 2  # ffill value + possible bfill value

    def test_cache_roundtrip(self, data):
        """Second build (cache hit) returns identical table."""
        again = build_aligned_dataset(use_cache=True)
        pd.testing.assert_frame_equal(
            data.df.reset_index(drop=True), again.df.reset_index(drop=True)
        )
