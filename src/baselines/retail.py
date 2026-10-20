"""Locked-grid conventional policies. All actions are order-grid INDICES.

EOQ has no fixed setup charge in the canonical simulator. Its K=0 optimum
is the smallest positive feasible batch (5 units), not an invented setup cost.
Nonzero setup_cost is an explicitly labelled sensitivity assumption only.
"""
import numpy as np


class MyopicPricingPolicy:
    """Forecast plug-in one-day economic reward maximizer (not oracle demand).

    Weekly action is held by env. Procurement already due today is constant
    across candidate prices; future arrivals/rewards are deliberately ignored.
    min(E[D], I) is an approximation, NOT E[min(D,I)] under NB uncertainty.
    """
    def __init__(self, env):
        self.env = env

    def __call__(self, obs=None):
        env = self.env
        b = env.observation_components()
        cfg = env.config.reward
        scores = []
        for multiplier in env.config.pricing.multipliers:
            price = env.base_price*multiplier
            mean = env.sim.mu_for(env.day, price)
            sales = np.minimum(mean, b['inventory'])
            scores.append(price*sales
                - cfg.holding_cost_per_unit*(b['inventory']-sales)
                - cfg.stockout_cost_per_unit*(mean-sales)
                - cfg.price_change_cost*np.abs(price-b['price']))
        scores = np.asarray(scores)
        if getattr(env, 'pricing_scope', 'shared') == 'per_sku':
            # Immediate plug-in reward is additive across SKUs in this simulator.
            return np.argmax(scores, axis=0).astype(np.int64)
        return int(np.argmax(scores.sum(axis=1)))


class EOQPolicy:
    def __init__(self, setup_cost=0., safety_days=0.):
        if setup_cost < 0 or safety_days < 0:
            raise ValueError('cost and safety days must be nonnegative')
        self.setup_cost, self.safety_days = setup_cost, safety_days

    def __call__(self, env, obs=None):
        b = env.observation_components()
        mean = b['forecast']
        position = b['inventory'] + b['pipeline'].sum(axis=1)
        cfg = env.config.lead_time
        lead = cfg.mean_days if cfg.mean_days is not None else (cfg.min_days+cfg.max_days)/2
        holding = env.config.reward.holding_cost_per_unit
        if holding <= 0:
            raise ValueError('EOQ requires positive holding cost')
        step = env.config.replenishment.order_step
        batch = np.maximum(step, np.sqrt(2*self.setup_cost*mean/holding))
        qty = np.where((mean > 0) & (position <= mean*(lead+self.safety_days)), batch, 0.)
        return np.rint(np.clip(qty/step, 0, len(env.grid)-1)).astype(np.int64)
