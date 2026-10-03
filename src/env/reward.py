"""Shared economic reward — the single source of truth for r_t (Doc S2.7).

    r_t = Revenue_t - OrderingCost_t - HoldingCost_t - StockoutCost_t - PriceChangeCost_t

Both agents receive this same scalar (shared-reward cooperative game, T1).
The stub env inlines its own copy for speed; THIS module is the canonical
implementation used by the real environment at integration and by every
baseline. Reward weights are calibrated before the final comparison and
then frozen across algorithms (S2.7 anti-bias rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.locked_config import RewardConfig, get_locked_config


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-term accounting, matching StepInfo fields for metrics reporting."""

    revenue: float
    ordering_cost: float
    holding_cost: float
    stockout_cost: float
    price_change_cost: float

    @property
    def total(self) -> float:
        return (
            self.revenue
            - self.ordering_cost
            - self.holding_cost
            - self.stockout_cost
            - self.price_change_cost
        )


def compute_reward(
    *,
    prices: np.ndarray,            # p_(i,t), shape (n_skus,)
    sales: np.ndarray,             # S_(i,t)
    lost_sales: np.ndarray,        # U_(i,t)
    inventory_end: np.ndarray,     # I_(i,t+1) (holding charged on end-of-day stock)
    orders_placed: np.ndarray,     # q_(i,t)
    unit_costs: np.ndarray,        # c_(o,i) procurement cost per SKU
    prev_prices: np.ndarray,       # p_(i,t-1) for price-change cost
    reward_cfg: RewardConfig | None = None,
) -> RewardBreakdown:
    """Canonical shared-reward computation (S2.7). Pure function: no state."""
    cfg = reward_cfg or get_locked_config().reward

    revenue = float((prices * sales).sum())
    ordering_cost = float((unit_costs * orders_placed).sum())
    holding_cost = float(cfg.holding_cost_per_unit * inventory_end.sum())
    stockout_cost = float(cfg.stockout_cost_per_unit * lost_sales.sum())
    price_change_cost = float(cfg.price_change_cost * np.abs(prices - prev_prices).sum())

    return RewardBreakdown(
        revenue=revenue,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        price_change_cost=price_change_cost,
    )


def fill_rate(sales: np.ndarray, demand: np.ndarray) -> float:
    """Service metric FillRate = sum S / sum D (S2.8, S5.3)."""
    total_demand = float(demand.sum())
    if total_demand == 0.0:
        return 1.0
    return float(sales.sum() / total_demand)
