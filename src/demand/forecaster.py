"""Calibrated no-peeking demand forecaster (Phase P2, pricing track).

The PRICING track's real `DemandForecast` implementation, backing D_hat_t in
both agents' observations at integration. Replaces the RecentMeanForecaster
stub (env/forecast.py) with the calibrated NB mean model:

    D_hat(day) = mu_for(day, p_current)

where mu_for is the simulator's deterministic mean (elasticity + calendar +
clamps, WITHOUT stochastic shocks — a forecast must not sample noise).

No-peeking discipline (Doc §2.3 forecast contract): forecast(day) only uses
the calendar mapping for `day` and the CURRENT price — never realized day-`day`
demand, so it is legal to call before __call__(day). The simulator's mean is
exogenous (no self-feedback), so history carries no extra information beyond
the price + calendar already inside the mean.
"""
from __future__ import annotations

import numpy as np

from src.demand.simulator import CalibratedDemandSimulator
from src.env.forecast import DemandForecast


class CalibratedMeanForecaster(DemandForecast):
    """D_hat_t from the calibrated mean model; needs a price pushed in."""

    def __init__(self, simulator: CalibratedDemandSimulator, n_skus: int | None = None):
        if not hasattr(simulator, "mu_for"):
            raise TypeError("simulator must expose mu_for(day, prices)")
        self.sim = simulator
        self.n = n_skus or simulator.n
        self._price: np.ndarray | None = None

    # ------------------------------------------------------------------ contract

    def set_current_price(self, prices: np.ndarray) -> None:
        """Env pushes the current price here too (same seam as the simulator)."""
        self._price = np.asarray(prices, dtype=float).copy()

    def forecast(self, day: int) -> np.ndarray:
        if self._price is None:
            raise ValueError(
                "set_current_price() must be called before forecast() — "
                "D_hat depends on the current price (price-sensitive demand)"
            )
        return self.sim.mu_for(day, self._price)

    def update(self, day: int, demand: np.ndarray) -> None:
        """The calibrated mean is exogenous: realized demand carries no state
        the mean needs. Kept as a no-op for the DemandForecast contract."""
        return None

    def reset(self) -> None:
        self._price = None
