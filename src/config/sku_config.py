"""Loader for the locked SKU selection (src/config/skus.json).

The selection file is LOCKED (same status as locked_config.py): it pins the
10 SKU-store series, their base prices and the evaluation protocol. Created
by scripts/select_skus.py; do not hand-edit — re-run the script instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKU_FILE = Path(__file__).resolve().parent / "skus.json"


@dataclass(frozen=True)
class SKU:
    """One locked SKU-store series (the env's unit of trade)."""

    series_id: str
    store_id: str
    item_id: str
    cat_id: str
    dept_id: str
    base_price: float
    price_min: float
    price_max: float
    price_cv: float
    n_price_points: int
    mean_daily_sales: float
    nonzero_fraction: float
    selection_score: float


def load_sku_payload() -> dict:
    """Raw locked payload (includes protocol + calendar metadata)."""
    if not SKU_FILE.exists():
        raise FileNotFoundError(
            f"{SKU_FILE} missing — run scripts/select_skus.py to create it"
        )
    return json.loads(SKU_FILE.read_text())


def load_skus() -> list[SKU]:
    """The 10 locked SKUs, in locked order (sorted by category, then volume)."""
    payload = load_sku_payload()
    assert payload.get("locked") is True, "skus.json is not marked locked"
    return [SKU(**entry) for entry in payload["skus"]]
