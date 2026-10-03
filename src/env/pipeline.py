"""Order pipeline with stochastic bounded supplier lead times (Doc S2.6, T1).

Orders placed on day t by the Replenishment Agent enter the pipeline and
arrive at the START of day t + L, where L ~ LeadTimeConfig distribution
(bounded in [min_days, max_days]). Arrivals on day t (A_(i,t)) are added to
on-hand inventory BEFORE sales are computed, matching

    S_(i,t) = min(D_(i,t), I_(i,t) + A_(i,t))            (Doc S2.6)

This module is the single source of truth for Q_t^pipe (Table 2) — the
outstanding-order bookkeeping that feeds the replenishment observation.

Day-cycle contract (whoever orchestrates the env must follow this):
    1. arrivals = pipeline.receive(t)          -> A_(i,t)
    2. inventory.receive(arrivals)             -> capacity-checked on-hand
    3. sales, lost = inventory.sell(demand)
    4. pipeline.place_orders(q_t, day=t)       -> lead times sampled here
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from src.config.locked_config import LeadTimeConfig, get_locked_config


class OrderPipeline:
    """Tracks in-flight orders and their stochastic arrival days.

    Internal state: {arrival_day: qty_vector(n_skus)}.
    """

    def __init__(
        self,
        n_skus: int,
        lead_time_cfg: LeadTimeConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_skus = n_skus
        self.cfg = lead_time_cfg or get_locked_config().lead_time
        self.rng = rng if rng is not None else np.random.default_rng()
        self._in_flight: dict[int, np.ndarray] = {}  # arrival_day -> qty per SKU

    # ------------------------------------------------------------------ lead times

    def sample_lead_time(self) -> int:
        """Draw one lead time L, bounded to [min_days, max_days], integer days."""
        cfg = self.cfg
        lo, hi = cfg.min_days, cfg.max_days
        if hi == lo:
            return int(lo)  # degenerate (fixed lead time, ablation A5)
        if cfg.distribution == "uniform":
            return int(self.rng.integers(lo, hi + 1))
        if cfg.distribution == "truncnormal":
            mean = cfg.mean_days if cfg.mean_days is not None else (lo + hi) / 2.0
            std = cfg.std_days if cfg.std_days is not None else max((hi - lo) / 3.0, 0.5)
            a, b = (lo - mean) / std, (hi - mean) / std
            draw = stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.rng)
            return int(np.clip(round(draw), lo, hi))
        raise ValueError(f"unknown lead-time distribution: {cfg.distribution!r}")

    # ------------------------------------------------------------------ orders

    def place_orders(self, qty: np.ndarray, day: int) -> np.ndarray:
        """Register an order q_(i,t) placed on `day`. Returns sampled lead times.

        Orders with zero quantity do not consume a lead-time draw.
        """
        qty = np.asarray(qty, dtype=float)
        lead_times = np.zeros(self.n_skus, dtype=int)
        for i in np.nonzero(qty > 0)[0]:
            lead_times[i] = self.sample_lead_time()
            arrival_day = day + lead_times[i]
            if arrival_day not in self._in_flight:
                self._in_flight[arrival_day] = np.zeros(self.n_skus)
            self._in_flight[arrival_day][i] += qty[i]
        return lead_times

    def receive(self, day: int) -> np.ndarray:
        """Return A_(i,t): quantities arriving at the start of `day`."""
        return self._in_flight.pop(day, np.zeros(self.n_skus))

    # ------------------------------------------------------------------ bookkeeping (Q_t^pipe)

    def outstanding(self) -> np.ndarray:
        """Total units per SKU currently in flight (all arrival days)."""
        total = np.zeros(self.n_skus)
        for qty in self._in_flight.values():
            total += qty
        return total

    def outstanding_from(self, day: int) -> np.ndarray:
        """Outstanding units per SKU with arrival_day >= `day`."""
        total = np.zeros(self.n_skus)
        for arrival_day, qty in self._in_flight.items():
            if arrival_day >= day:
                total += qty
        return total

    def pending_by_remaining_days(self, current_day: int, horizon: int) -> np.ndarray:
        """Q_t^pipe bucketed by remaining lead time, for observations.

        Returns shape (n_skus, horizon): column r-1 = units arriving on
        current_day + r, for r = 1..horizon. Bucket 0 (same-day) is empty by
        construction since arrivals were already received at step start.
        """
        buckets = np.zeros((self.n_skus, horizon))
        for arrival_day, qty in self._in_flight.items():
            r = arrival_day - current_day
            if 1 <= r <= horizon:
                buckets[:, r - 1] += qty
        return buckets
