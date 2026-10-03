"""Inventory state engine: lost-sales transition + shared on-hand capacity.

Implements the Doc S2.6 transition exactly:

    S_(i,t) = min(D_(i,t), I_(i,t) + A_(i,t))      # sales capped by availability
    U_(i,t) = max(0, D_(i,t) - S_(i,t))            # unmet demand -> LOST (no backlog)
    I_(i,t+1)= I_(i,t) + A_(i,t) - S_(i,t)

Shared warehouse capacity (Doc S4.6, locked convention):

    sum_i I_(i,t) <= C     # applies to ON-HAND inventory only

Feasibility mechanism — THE ONE MECHANISM, fixed across all methods
(Doc S2.4.2 requires heuristics and RL face identical enforcement):

    Arrivals are clipped at RECEIVING time. If total inbound arrivals exceed
    the on-hand headroom (C - sum on-hand), ALL SKUs' arrivals are scaled
    down proportionally (symmetric, deterministic, order-independent).
    The undelivered remainder is cancelled by the supplier at no charge:
    ordering cost is charged only on DELIVERED units.

    Rationale: proportional scaling treats SKUs symmetrically (unlike
    first-come clipping, which would privilege arbitrary orderings) and
    keeps the on-hand invariant sum_i I_(i,t) <= C exact by construction.
"""
from __future__ import annotations

import numpy as np


class InventoryEngine:
    """Per-SKU on-hand inventory with shared capacity and lost sales."""

    def __init__(
        self,
        n_skus: int,
        capacity: float,
        initial_inventory: np.ndarray | None = None,
    ):
        self.n_skus = n_skus
        self.capacity = float(capacity)
        if initial_inventory is None:
            self._on_hand = np.zeros(n_skus)
        else:
            self._on_hand = np.asarray(initial_inventory, dtype=float).copy()
            if self._on_hand.shape != (n_skus,):
                raise ValueError(f"initial_inventory must have shape ({n_skus},)")
            # Never start above capacity.
            if self._on_hand.sum() > self.capacity:
                scale = self.capacity / self._on_hand.sum()
                self._on_hand *= scale
        self.capacity_violation_events = 0   # count of receive() calls that clipped
        self.capacity_clipped_units = 0.0    # total units cancelled by clipping

    # ------------------------------------------------------------------ state

    @property
    def on_hand(self) -> np.ndarray:
        """I_(i,t) — copy, so callers cannot mutate internal state."""
        return self._on_hand.copy()

    @property
    def total_on_hand(self) -> float:
        return float(self._on_hand.sum())

    @property
    def headroom(self) -> float:
        """C - sum_i I_(i,t), floored at 0."""
        return max(self.capacity - self.total_on_hand, 0.0)

    # ------------------------------------------------------------------ day-cycle ops

    def receive(self, arrivals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Add A_(i,t) to on-hand, respecting shared capacity. Returns
        (accepted, clipped). Ordering cost must be charged on `accepted` only.
        """
        arrivals = np.asarray(arrivals, dtype=float)
        if arrivals.shape != (self.n_skus,):
            raise ValueError(f"arrivals must have shape ({self.n_skus},)")
        if np.any(arrivals < 0):
            raise ValueError("arrivals must be non-negative")

        total = float(arrivals.sum())
        if total <= self.headroom + 1e-12:
            accepted, clipped = arrivals, np.zeros_like(arrivals)
        else:
            scale = self.headroom / total
            accepted = arrivals * scale
            clipped = arrivals - accepted
            self.capacity_violation_events += 1
            self.capacity_clipped_units += float(clipped.sum())

        self._on_hand = self._on_hand + accepted
        # Numerical safety: never exceed capacity by float noise.
        if self.total_on_hand > self.capacity:
            self._on_hand *= self.capacity / self.total_on_hand
        return accepted, clipped

    def sell(self, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lost-sales sale step: S = min(D, I); U = D - S; I' = I - S.
        Returns (sales, lost_sales).
        """
        demand = np.asarray(demand, dtype=float)
        if demand.shape != (self.n_skus,):
            raise ValueError(f"demand must have shape ({self.n_skus},)")
        sales = np.minimum(demand, self._on_hand)
        lost = demand - sales
        self._on_hand = self._on_hand - sales
        return sales, lost

    def reset(self, initial_inventory: np.ndarray | None = None) -> None:
        self.__init__(self.n_skus, self.capacity, initial_inventory)  # type: ignore[misc]
