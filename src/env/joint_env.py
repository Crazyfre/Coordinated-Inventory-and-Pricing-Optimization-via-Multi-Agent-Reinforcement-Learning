"""P4 daily joint environment; unchanged R1 receiving/pipeline semantics.

Uses a dictionary observation and JointAction; returns a diagnostics dictionary
rather than Phase-0 StepInfo. select_price stages the weekly action BEFORE the
restocking actor sees its observation, retaining the previous price for costs.
"""
import gymnasium as gym
import numpy as np
from src.env.replenishment_env import ReplenishmentEnv
from src.env.interface import JointAction, pricing_indices


class JointRetailEnv(ReplenishmentEnv):
    def __init__(self, *args, pricing_scope="shared", **kwargs):
        if pricing_scope not in ("shared", "per_sku"):
            raise ValueError("pricing_scope must be shared or per_sku")
        self.pricing_scope = pricing_scope
        super().__init__(*args, **kwargs)
        size = len(self.config.pricing.multipliers)
        self.pricing_nvec = np.full(self.n if pricing_scope == "per_sku" else 1, size, dtype=np.int64)
        self.pricing_action_space = (gym.spaces.MultiDiscrete(self.pricing_nvec)
                                     if pricing_scope == "per_sku" else gym.spaces.Discrete(size))

    @property
    def pricing_due(self):
        return (self.day - 1) % self.config.timing.pricing_interval_days == 0

    def _push_price_to_demand_source(self):
        super()._push_price_to_demand_source()
        if hasattr(self.forecaster, 'set_current_price'):
            self.forecaster.set_current_price(self._current_price)

    def reset(self, *, seed=None, options=None):
        self._pending_price = None
        # R2 warm-up samples before its first price push.
        self._current_price = self.base_price.copy()
        self._push_price_to_demand_source()
        _, info = super().reset(seed=seed, options=options)
        self._finished = False
        return self.observations(), info

    def select_price(self, index):
        if not self.pricing_due:
            return
        indices = pricing_indices(index, self.pricing_scope, self.n,
                                  len(self.config.pricing.multipliers))
        self._pending_price = self.base_price * np.asarray(self.config.pricing.multipliers)[indices]

    def replenishment_observation(self):
        if self._pending_price is None:
            return self._build_obs()
        old = self._current_price
        try:
            self._current_price = self._pending_price
            self._push_price_to_demand_source()
            return self._build_obs()
        finally:
            self._current_price = old
            self._push_price_to_demand_source()

    def context(self):
        sim = getattr(self,'sim',None)
        if sim is not None:
            row = (sim.start_day_index+self.day-2) % len(sim.calendar_feats)
            return sim.calendar_feats[row].astype(np.float32)
        return np.eye(7,dtype=np.float32)[(self.day-1)%7]

    def pricing_observation(self):
        b = self.observation_components()
        context = self.context()
        return np.concatenate([b['inventory'], b['demand_history'].ravel(),
                               b['price'], context, b['forecast'], b['headroom']]).astype(np.float32)

    def observations(self):
        return {'pricing': self.pricing_observation(),
                'replenishment': self.replenishment_observation()}

    def get_state(self):
        # Critic-only global vector, includes pipe/supplier and calendar phase.
        return np.concatenate([self.replenishment_observation(), self.context()]).astype(np.float32)

    def step(self, action: JointAction):
        if self._finished:
            raise RuntimeError('episode finished; reset required')
        if action.pricing_multiplier_index is not None:
            self.select_price(action.pricing_multiplier_index)
        qty = action.replenishment_qty_idx
        if qty is None:
            qty = np.zeros(self.n, dtype=np.int64)
        if not self.action_space.contains(np.asarray(qty)):
            raise ValueError('replenishment requires integer grid indices of shape (n,)')
        self.pricing_hook = lambda day: self._pending_price if self.pricing_due else None
        current_day = self.day
        received = self._arrivals_today.copy()
        accepted = self._accepted_today.copy()
        clipped = self._clipped_today.copy()
        clip_events = self.inventory.capacity_violation_events
        clip_units = self.inventory.capacity_clipped_units
        _, reward, term, trunc, info = super().step(qty)
        info.update(day=current_day, arrivals=received, orders_delivered=accepted,
                    orders_clipped=clipped, capacity_violation_events=clip_events,
                    capacity_clipped_units=clip_units)
        if not (term or trunc):
            info['total_on_hand'] = self.inventory.total_on_hand - self._accepted_today.sum()
        self._pending_price = None
        self._finished = term or trunc
        obs = self.observations() if not self._finished else {
            'pricing': np.zeros(self.n*(self.k+4)+len(self.context()), dtype=np.float32),
            'replenishment': np.zeros(self._obs_dim(), dtype=np.float32)}
        return obs, reward, self._finished, info
