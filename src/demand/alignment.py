"""P1 task 2: align daily sales, weekly prices and calendar for the locked SKUs.

Produces the tidy modelling table the demand calibration (Doc 4.4) consumes:

    one row per (series_id, day) with
        y                 daily unit sales            (target)
        sell_price        weekly price mapped to days  (M5 sell_prices)
        price_rel         sell_price / base_price     (elasticity regressor)
        log_price_rel     log(price_rel)
        wday, month, year, dow_* one-hots, snap,
        event flags (cultural/sporting/religious/national)
        hist_7, hist_28   lagged demand means (H_t) — shift(1), NO leakage

M5 structure: sales are daily, prices are WEEKLY (keyed by wm_yr_wk), so the
calendar is the join key that maps each week's price onto its days.

Caching: the built table is cached in data/processed/ keyed by the locked
series list, so later phases don't re-parse 300 MB of CSV every run.

Temporal protocol (Doc 4.3): the last `test_last_n_days` (28) days of the
1941-day snapshot are the held-out test period — never used for fitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.sku_config import load_sku_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "processed"
CACHE_CSV = CACHE_DIR / "aligned_skus.csv"
CACHE_KEY = CACHE_DIR / "aligned_skus.key.json"

EVENT_TYPES = ("Cultural", "Sporting", "Religious", "National")


@dataclass(frozen=True)
class AlignedData:
    """Tidy aligned table + split metadata."""

    df: pd.DataFrame
    n_days: int
    test_last_n_days: int
    series_ids: tuple[str, ...]

    @property
    def test_start_day(self) -> int:
        return self.n_days - self.test_last_n_days + 1

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(train, test) by day index — chronological, no shuffling."""
        train = self.df[self.df.day_index < self.test_start_day].copy()
        test = self.df[self.df.day_index >= self.test_start_day].copy()
        return train, test

    def price_coverage(self) -> pd.Series:
        """Fraction of days with a directly observed (not filled) price."""
        return self.df.groupby("series_id")["price_observed"].mean()


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def _load_sales_long(series_ids: list[str]) -> pd.DataFrame:
    """Melt the wide M5 sales matrix down to the 10 locked series."""
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    keys = pd.DataFrame(
        [{"store_id": s.split("_")[0] + "_" + s.split("_")[1],
          "item_id": "_".join(s.split("_")[2:]), "series_id": s} for s in series_ids]
    )
    sub = sales.merge(keys, on=["store_id", "item_id"], how="inner")
    assert len(sub) == len(series_ids), "locked series not all found in M5 sales"

    long = sub.melt(
        id_vars=["series_id"], value_vars=day_cols, var_name="d", value_name="y"
    )
    long["day_index"] = long["d"].str.removeprefix("d_").astype(int)
    return long.drop(columns=["d"])


def _load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(RAW_DIR / "calendar.csv")
    cal = cal.rename(columns={"d": "d_key"})
    cal["day_index"] = cal["d_key"].str.removeprefix("d_").astype(int)

    cal["event_type"] = cal["event_type_1"].fillna(cal["event_type_2"])
    for t in EVENT_TYPES:
        cal[f"event_{t.lower()}"] = (cal["event_type"] == t).astype(int)
    cal["is_event"] = cal["event_type"].notna().astype(int)

    # Store is CA_3 -> the relevant SNAP flag is snap_CA.
    cal["snap"] = cal["snap_CA"].astype(int)
    keep = ["day_index", "date", "wm_yr_wk", "wday", "weekday", "month", "year",
            "snap", "is_event"] + [f"event_{t.lower()}" for t in EVENT_TYPES]
    return cal[keep]


def _load_prices_weekly(series_ids: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    keys = pd.DataFrame(
        [{"store_id": s.split("_")[0] + "_" + s.split("_")[1],
          "item_id": "_".join(s.split("_")[2:]), "series_id": s} for s in series_ids]
    )
    sub = prices.merge(keys, on=["store_id", "item_id"], how="inner")
    return sub[["series_id", "wm_yr_wk", "sell_price"]]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hots + lagged history features (H_t, strictly lagged)."""
    df = df.sort_values(["series_id", "day_index"]).reset_index(drop=True)

    # wday 1..7 (M5: 1 = Saturday) and month one-hots
    for w in range(1, 8):
        df[f"dow_{w}"] = (df["wday"] == w).astype(int)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    df["year_idx"] = df["year"] - df["year"].min()

    # H_t: lagged demand history — shift(1) guarantees no leakage of day-t demand.
    grouped = df.groupby("series_id")["y"]
    df["hist_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["hist_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    # Cold start (day 1) has no history: fall back to the series' own global mean
    # is LEAKY; instead use 0 and let the model's intercept absorb it.
    df[["hist_7", "hist_28"]] = df[["hist_7", "hist_28"]].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_aligned_dataset(use_cache: bool = True, force: bool = False) -> AlignedData:
    """Build (or load from cache) the aligned modelling table for locked SKUs."""
    payload = load_sku_payload()
    series_ids = [s["series_id"] for s in payload["skus"]]
    base_prices = {s["series_id"]: s["base_price"] for s in payload["skus"]}
    test_last_n = payload["protocol"]["test_last_n_days"]
    key = {"series_ids": series_ids, "test_last_n_days": test_last_n}

    if use_cache and not force and CACHE_CSV.exists() and CACHE_KEY.exists():
        if json.loads(CACHE_KEY.read_text()) == key:
            df = pd.read_csv(CACHE_CSV, parse_dates=["date"])
            return AlignedData(df, int(df.day_index.max()), test_last_n, tuple(series_ids))

    long = _load_sales_long(series_ids)
    cal = _load_calendar()
    prices = _load_prices_weekly(series_ids)

    df = long.merge(cal, on="day_index", how="left")
    df = df.merge(prices, on=["series_id", "wm_yr_wk"], how="left")

    # Price coverage: M5 omits weeks before an item was introduced at a store.
    df["price_observed"] = df["sell_price"].notna().astype(int)
    df = df.sort_values(["series_id", "day_index"])
    df["sell_price"] = df.groupby("series_id")["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )

    df["base_price"] = df["series_id"].map(base_prices)
    df["price_rel"] = df["sell_price"] / df["base_price"]
    df["log_price_rel"] = np.log(df["price_rel"])

    df = _add_features(df)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    CACHE_KEY.write_text(json.dumps(key, indent=2))
    return AlignedData(df, int(df.day_index.max()), test_last_n, tuple(series_ids))


FEATURE_COLUMNS = (
    ["price_rel", "log_price_rel"]
    + [f"dow_{w}" for w in range(1, 8)]
    + [f"month_{m}" for m in range(1, 13)]
    + ["snap", "is_event", "year_idx"]
    + [f"event_{t.lower()}" for t in EVENT_TYPES]
)
HISTORY_COLUMNS = ["hist_7", "hist_28"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """x_t design matrix (Doc 4.4): price response + calendar/context features.

    The model is  log(mu) = b0 + b_p*log_price_rel + b_x^T x + b_h^T H_t, so
    the returned matrix excludes the intercept, history terms and target.
    """
    return df[FEATURE_COLUMNS].astype(float)


if __name__ == "__main__":  # quick human-readable sanity report
    data = build_aligned_dataset()
    train, test = data.split()
    print(f"rows: {len(data.df)} | days: {data.n_days} | series: {len(data.series_ids)}")
    print(f"train days 1..{data.test_start_day - 1} ({len(train)} rows) | "
          f"test days {data.test_start_day}..{data.n_days} ({len(test)} rows)")
    print(f"test dates: {test.date.min()} .. {test.date.max()}")
    print("\nprice coverage (fraction of days with observed weekly price):")
    print(data.price_coverage().round(3).to_string())
    print("\nfeature matrix:", build_feature_matrix(data.df).shape,
          "| history terms:", HISTORY_COLUMNS)
    print("\nsample row:")
    print(data.df.iloc[500][["series_id", "date", "y", "sell_price", "price_rel",
                             "wday", "snap", "is_event", "hist_7"]].to_string())
