"""Honest forecast-only daily periodic-review reference; no realized demand peek."""
import numpy as np


class ForecastBaseStock:
    """Order to forecast lead+review cover, using staged weekly price.

    The deterministic mean is not a demand sample or known supplier future.
    In-flight units count in inventory position; receiving cancellations are
    observed on subsequent days. Ceiling grid rounding avoids systematic
    under-ordering of small SKUs. No capacity/economic rules are changed.
    """
    def __init__(self, cover_days=6.):
        if not np.isfinite(cover_days) or cover_days < 0:
            raise ValueError('cover_days must be finite and nonnegative')
        self.cover_days = cover_days

    def __call__(self, env, obs=None):
        b = env.observation_components()
        price = env._pending_price if env._pending_price is not None else b['price']
        mean = env.sim.mu_for(env.day, price)
        position = b['inventory'] + b['pipeline'].sum(axis=1)
        qty = np.maximum(self.cover_days*mean-position, 0.)
        return np.ceil(np.clip(qty/env.config.replenishment.order_step,
                              0, len(env.grid)-1)).astype(np.int64)
