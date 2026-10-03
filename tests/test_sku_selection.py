"""Integrity tests for the locked SKU selection (src/config/skus.json).

These guard the LOCKED selection against drift: any change to skus.json
must come from re-running scripts/select_skus.py, not hand edits.
"""

import pytest

from src.config.sku_config import load_sku_payload, load_skus
from src.config.locked_config import get_locked_config


class TestLockedFile:
    def test_file_exists_and_locked(self):
        payload = load_sku_payload()
        assert payload["locked"] is True

    def test_ten_skus_single_store(self):
        skus = load_skus()
        assert len(skus) == get_locked_config().env.n_skus
        assert len({s.store_id for s in skus}) == 1  # single retail store (T1)


class TestDocSectionFourPointThree:
    def test_category_coverage(self):
        cats = {s.cat_id for s in load_skus()}
        assert len(cats) >= 3  # multiple product categories

    def test_category_cap(self):
        payload = load_sku_payload()
        cap = payload["protocol"]["category_cap"]
        counts: dict[str, int] = {}
        for s in load_skus():
            counts[s.cat_id] = counts.get(s.cat_id, 0) + 1
        assert all(c <= cap for c in counts.values())

    def test_unique_series(self):
        series = [s.series_id for s in load_skus()]
        assert len(series) == len(set(series))

    def test_no_intermittent_items(self):
        for s in load_skus():
            assert s.nonzero_fraction >= 0.85, f"{s.series_id} too intermittent"
            assert s.mean_daily_sales >= 2.0, f"{s.series_id} too low-volume"

    def test_usable_price_variation(self):
        for s in load_skus():
            assert s.price_cv >= 0.02, f"{s.series_id} too little price variation"
            assert s.n_price_points >= 3, f"{s.series_id} too few price points"
            assert s.price_max > s.price_min

    def test_base_prices_positive(self):
        for s in load_skus():
            assert s.base_price > 0
            assert s.price_min <= s.base_price <= s.price_max


class TestProtocolMetadata:
    def test_held_out_test_period_recorded(self):
        payload = load_sku_payload()
        cal = payload["calendar"]
        assert cal["n_days"] == 1941  # M5 evaluation snapshot
        assert len(cal["test_days"]) == payload["protocol"]["test_last_n_days"]

    def test_protocol_thresholds_match_design(self):
        p = load_sku_payload()["protocol"]
        assert p["min_nonzero_fraction"] == 0.85
        assert p["min_price_cv"] == 0.02
        assert p["category_cap"] == 4
