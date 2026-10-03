"""Phase P1, task 1: deterministic SKU selection from M5 -> src/config/skus.json.

Selection follows Doc section 4.3:
  - 10 SKUs (item-store series) for a SINGLE retail store (locked env scope)
  - across multiple product categories (>= 3) for cross-SKU correlation work
  - sufficient sales activity, avoiding extreme intermittency
  - usable price variation (elasticity estimation needs price movement)
  - deterministic: same data -> same selection, no randomness

Criteria (activity measured over the last ACTIVITY_WINDOW_DAYS days):
  - nonzero_fraction >= 0.85        (not intermittent)
  - mean daily sales  >= 2.0
  - price CV          >= 0.02       (relative price variation across weeks)
  - >= 3 distinct weekly price points
  - category cap: <= 4 SKUs per category; guaranteed >= 1 per category

Protocol note (Doc 4.3 + 4.1): M5 contains only packaged grocery/general
merchandise — no fresh/perishable categories — so "non-perishable" is
satisfied by construction; the operative filter is the anti-intermittency
rule above. A held-out temporal test period (last 28 days) is recorded.

Usage:  .venv/Scripts/python scripts/select_skus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
OUT_PATH = REPO_ROOT / "src" / "config" / "skus.json"

# --- locked selection protocol -------------------------------------------------
N_SKUS = 10
ACTIVITY_WINDOW_DAYS = 728          # last ~2 years
MIN_NONZERO_FRACTION = 0.85
MIN_MEAN_DAILY = 2.0
MIN_PRICE_CV = 0.02
MIN_PRICE_POINTS = 3
CATEGORY_CAP = 4
TEST_LAST_N_DAYS = 28               # held-out temporal test period (Doc 4.3)
PRICE_BASE_WEEKS = 52               # base price = median weekly price over last year

PROTOCOL = {
    "n_skus": N_SKUS,
    "store_rule": "single store with the highest total M5 sales (deterministic)",
    "activity_window_days": ACTIVITY_WINDOW_DAYS,
    "min_nonzero_fraction": MIN_NONZERO_FRACTION,
    "min_mean_daily_sales": MIN_MEAN_DAILY,
    "min_price_cv": MIN_PRICE_CV,
    "min_price_points": MIN_PRICE_POINTS,
    "category_cap": CATEGORY_CAP,
    "coverage_rule": ">= 1 SKU per M5 category (HOBBIES/HOUSEHOLD/FOODS)",
    "base_price_rule": f"median weekly sell price over last {PRICE_BASE_WEEKS} weeks",
    "test_last_n_days": TEST_LAST_N_DAYS,
    "perishability_note": "M5 has no fresh categories; non-perishable by construction (Doc 4.3)",
}


def pick_store(sales: pd.DataFrame) -> str:
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    totals = sales.groupby("store_id")[day_cols].sum().sum(axis=1)
    return str(totals.idxmax())


def item_activity(sales_store: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """Per-item activity metrics over the last ACTIVITY_WINDOW_DAYS days."""
    win = day_cols[-ACTIVITY_WINDOW_DAYS:]
    mat = sales_store.set_index("item_id")[win]
    active = (mat > 0).mean(axis=1)
    mean_daily = mat.mean(axis=1)
    return pd.DataFrame({"nonzero_fraction": active, "mean_daily": mean_daily})


def item_price_stats(prices_store: pd.DataFrame) -> pd.DataFrame:
    """Per-item weekly price variation metrics."""
    g = prices_store.groupby("item_id")["sell_price"]
    # Base price: median weekly price over the most recent PRICE_BASE_WEEKS weeks.
    sorted_p = prices_store.sort_values("wm_yr_wk")
    recent_idx = sorted_p.groupby("item_id").tail(PRICE_BASE_WEEKS).index
    base_price = sorted_p.loc[recent_idx].groupby("item_id")["sell_price"].median()

    stats = pd.DataFrame({
        "price_mean": g.mean(),
        "price_std": g.std(),
        "price_min": g.min(),
        "price_max": g.max(),
        "n_price_points": g.nunique(),
        "base_price": base_price,
    })
    stats["price_cv"] = stats["price_std"] / stats["price_mean"]
    return stats


def select(sales: pd.DataFrame, prices: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    store = pick_store(sales)
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    sales_store = sales[sales.store_id == store]
    prices_store = prices[prices.store_id == store]

    act = item_activity(sales_store, day_cols)
    prc = item_price_stats(prices_store)
    meta = sales_store.groupby("item_id")[["cat_id", "dept_id"]].first()

    df = meta.join(act).join(prc).dropna()
    qualify = df[
        (df.nonzero_fraction >= MIN_NONZERO_FRACTION)
        & (df.mean_daily >= MIN_MEAN_DAILY)
        & (df.price_cv >= MIN_PRICE_CV)
        & (df.n_price_points >= MIN_PRICE_POINTS)
    ].copy()

    # Composite score: equal weight percentile ranks of activity and price variation.
    qualify["score"] = (
        0.5 * qualify["mean_daily"].rank(pct=True)
        + 0.5 * qualify["price_cv"].rank(pct=True)
    )
    qualify = qualify.sort_values("score", ascending=False)

    # 1) Guarantee category coverage: best qualifying SKU per category.
    chosen, cats = [], qualify.cat_id.unique()
    for cat in cats:
        top = qualify[qualify.cat_id == cat].iloc[0]
        chosen.append(top.name)
    # 2) Fill remaining slots greedily by score, respecting the category cap.
    per_cat = qualify.loc[chosen].cat_id.value_counts().to_dict()
    for item in qualify.index:
        if len(chosen) >= N_SKUS:
            break
        if item in chosen:
            continue
        cat = qualify.loc[item, "cat_id"]
        if per_cat.get(cat, 0) < CATEGORY_CAP:
            chosen.append(item)
            per_cat[cat] = per_cat.get(cat, 0) + 1

    out = qualify.loc[chosen].copy()
    out["store_id"] = store
    out["item_id_series"] = out.index
    return store, out


def main() -> int:
    if not (RAW / "sales_train_evaluation.csv").exists():
        print("M5 data missing — run scripts/download_m5.py first", file=sys.stderr)
        return 1

    print("Loading M5 (this takes ~1 min) ...")
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")

    store, sel = select(sales, prices)
    print(f"\nSelected store: {store} (highest total sales)")

    # Held-out test window metadata (calendar dates for the last N days)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    test_days = day_cols[-TEST_LAST_N_DAYS:]
    d2date = dict(zip(calendar["d"], calendar["date"]))

    entries = []
    for item, row in sel.iterrows():
        entries.append({
            "series_id": f"{store}_{item}",
            "store_id": store,
            "item_id": item,
            "cat_id": row.cat_id,
            "dept_id": row.dept_id,
            "base_price": round(float(row.base_price), 4),
            "price_min": round(float(row.price_min), 4),
            "price_max": round(float(row.price_max), 4),
            "price_cv": round(float(row.price_cv), 4),
            "n_price_points": int(row.n_price_points),
            "mean_daily_sales": round(float(row.mean_daily), 3),
            "nonzero_fraction": round(float(row.nonzero_fraction), 4),
            "selection_score": round(float(row.score), 4),
        })
    entries.sort(key=lambda e: (e["cat_id"], -e["mean_daily_sales"]))

    payload = {
        "locked": True,
        "source": "M5 sales_train_evaluation + sell_prices (Kaggle m5-forecasting-accuracy)",
        "protocol": PROTOCOL,
        "calendar": {
            "date_min": str(calendar.date.min()),
            "date_max": str(calendar.date.max()),
            "n_days": len(day_cols),
            "test_days": {d: str(d2date[d]) for d in test_days},
        },
        "skus": entries,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Locked {len(entries)} SKUs -> {OUT_PATH}\n")

    print(f"{'series':<24}{'cat':<11}{'base$':>7}{'CV':>7}{'pts':>5}{'mu/day':>8}{'active%':>9}")
    for e in entries:
        print(f"{e['series_id']:<24}{e['cat_id']:<11}{e['base_price']:>7.2f}"
              f"{e['price_cv']:>7.3f}{e['n_price_points']:>5}{e['mean_daily_sales']:>8.2f}"
              f"{e['nonzero_fraction'] * 100:>8.1f}%")

    cats = {e["cat_id"] for e in entries}
    assert len(entries) == N_SKUS and len(cats) >= 3
    print(f"\nOK: {len(entries)} SKUs across {len(cats)} categories, single store {store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
