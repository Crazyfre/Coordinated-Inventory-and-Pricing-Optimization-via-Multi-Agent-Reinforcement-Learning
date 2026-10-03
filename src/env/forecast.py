"""Demand-forecast interface — the Phase 0 joint agreement for D_hat_t.

D_hat_t appears in BOTH agents' observations (Doc S2.3.1, S2.3.2), so the
interface is co-owned by both tracks:

    - The PRICING track will provide the real forecaster (a subclass of
      DemandForecast, backed by the calibrated NB mean) at integration.
    - The REPLENISHMENT track consumes this interface and may only use the
      RecentMeanForecaster stub until then (plan R2: "do not build your own").

Contract:
    forecast(day) -> D_hat_t   : predicted demand for `day`, shape (n_skus,).
                                 Must be callable BEFORE day-`day` demand is
                                 realized (no peeking).
    update(day, demand)        : ingest realized demand after it is observed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class DemandForecast(ABC):
    """Interface for D_hat_t (both agents' observations)."""

    @abstractmethod
    def forecast(self, day: int) -> np.ndarray:
        """Predicted demand for `day`, shape (n_skus,). No peeking at day-`day` demand."""

    @abstractmethod
    def update(self, day: int, demand: np.ndarray) -> None:
        """Ingest realized demand for `day` (shape (n_skus,))."""


class RecentMeanForecaster(DemandForecast):
    """Stub: rolling mean of the last `window` realized days (zeros if cold).

    Backs D_hat_t in the R2 env until the pricing track's calibrated
    forecaster is plugged in at integration (P4/R4).
    """

    def __init__(self, n_skus: int, window: int = 7):
        self.n_skus = n_skus
        self.window = window
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._last_day = 0

    def forecast(self, day: int) -> np.ndarray:
        if len(self._history) == 0:
            return np.zeros(self.n_skus)
        return np.mean(np.stack(self._history), axis=0)

    def update(self, day: int, demand: np.ndarray) -> None:
        if day <= self._last_day and len(self._history) > 0:
            raise ValueError(f"forecaster already updated for day {day}")
        self._last_day = day
        self._history.append(np.asarray(demand, dtype=float).copy())

    def reset(self) -> None:
        self._history.clear()
        self._last_day = 0
