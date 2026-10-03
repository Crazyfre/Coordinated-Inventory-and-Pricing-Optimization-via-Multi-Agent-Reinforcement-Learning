"""Synthetic demand source for Phase R1 validation (temporary, env-side).

Lives in env/ (NOT demand/) because the demand/ package belongs to the
pricing track. This stub exists only so the inventory engine, pipeline and
heuristics can be validated WITHOUT RL and WITHOUT the calibrated simulator.
At integration (P4/R4) it is replaced by the pricing track's calibrated NB
simulator behind the same callable interface:

    demand_source(day: int) -> np.ndarray  # shape (n_skus,)
"""
from __future__ import annotations

import numpy as np

from src.config.locked_config import get_locked_config


class SyntheticPoissonDemand:
    """Stationary Poisson demand with optional slow mean drift.

    Scenarios (Doc Table 7 knobs used in R1 validation):
        S1-like: drift_std = 0          -> stable demand
        S3-like: drift_std > 0          -> noisy demand random walk
    The drift is a preview only; the real S1-S8 knobs come from the
    calibrated simulator at integration.
    """

    def __init__(
        self,
        n_skus: int | None = None,
        base_lambda: float = 10.0,
        drift_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.n_skus = n_skus or get_locked_config().env.n_skus
        self.base_lambda = float(base_lambda)
        self.drift_std = float(drift_std)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._lambda = np.full(self.n_skus, self.base_lambda)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self._lambda = np.full(self.n_skus, self.base_lambda)

    def __call__(self, day: int) -> np.ndarray:
        if self.drift_std > 0:
            self._lambda = np.clip(
                self._lambda * self.rng.normal(1.0, self.drift_std, size=self.n_skus),
                1.0, 4.0 * self.base_lambda,
            )
        return self.rng.poisson(self._lambda).astype(float)
