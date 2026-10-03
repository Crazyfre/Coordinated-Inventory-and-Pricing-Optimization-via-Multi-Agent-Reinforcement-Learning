"""Gymnasium-style single-agent ReplenishmentEnv (Phase R2, replenishment track).

Composes the R1 components (InventoryEngine + OrderPipeline + pluggable
demand source) into an env a vanilla PPO can train on BEFORE integration.
The pricing track's calibrated demand simulator and the real forecaster
plug in at P4/R4 without changing this file's public contract:

    - demand source:  any callable  day -> demand(n_skus)   (R1 contract)
    - forecaster:     DemandForecast subclass                   (forecast.py)
    - pricing:        `pricing_hook(day) -> price_vector | None`;
                      at integration a weekly wrapper supplies prices on
                      days 1, 8, 15... and the env HOLDS the price between
                      hook calls (locked cadence, Doc S2.5). Until then a
                      constant stub price is used.

Day cycle inside step(action)  [action = order placed on the CURRENT day]:
    1. arrivals A_t received at day start (already applied when obs was built)
    2. order q_t registered in the pipeline (lead time sampled)
    3. demand D_t realized; lost-sales sale: S = min(D, I+A)
    4. forecaster updated with D_t
    5. shared economic reward r_t (canonical module, env/reward.py)
    6. day advances; next day's arrivals received; obs built for day t+1

Observation (Doc S2.3.2), flattened float32, block order fixed:
    o^R = [ I_t (n) | Q^pipe buckets (n x L_max) | D_hist (n x k)
            | p_t (n) | L_t last realized lead times (n)
            | capacity headroom (n) | D_hat_t (n) ]

Capacity feasibility: R1's receiving-time proportional clipping — identical
to what every heuristic will face. Clipped units + event counts are logged
in info (feeds the "Capacity violations" metric, Doc S5.3).
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import gymnasium as gym
import numpy as np

from src.config.locked_config import LockedConfig, get_locked_config
from src.env.forecast import DemandForecast, RecentMeanForecaster
from src.env.inventory import InventoryEngine
from src.env.pipeline import OrderPipeline
from src.env.reward import compute_reward

# Callable: day -> demand per SKU. Implemented today by env/synthetic_demand.py;
# the pricing track's calibrated simulator implements the same signature.
DemandSource = Callable[[int], np.ndarray]
PricingHook = Callable[[int], np.ndarray | None]


class ReplenishmentEnv(gym.Env):
    """Single-agent daily replenishment environment (replenishment track)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: LockedConfig | None = None,
        episode_length_days: int = 112,
        demand_source: DemandSource | None = None,
        forecaster: DemandForecast | None = None,
        pricing_hook: PricingHook | None = None,
        base_price: float = 10.0,
        unit_cost: float = 6.0,
        initial_inventory: float = 30.0,
        capacity: float = 500.0,
        obs_pipe_horizon: int | None = None,
    ):
        super().__init__()
        self.config = config or get_locked_config()
        self.episode_length_days = episode_length_days
        self.demand_source = demand_source
        self.forecaster = forecaster or RecentMeanForecaster(
            self.config.env.n_skus, window=7
        )
        self.pricing_hook = pricing_hook
        self.base_price = float(base_price)
        self.unit_cost = float(unit_cost)
        self.initial_inventory = float(initial_inventory)
        self.capacity = float(capacity)
        self.obs_pipe_horizon = obs_pipe_horizon or self.config.lead_time.max_days

        self.n = self.config.env.n_skus
        self.k = self.config.demand.history_window_k
        self.grid = np.arange(
            0, self.config.replenishment.q_max + 1, self.config.replenishment.order_step
        )

        # --- spaces ---------------------------------------------------------
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.n, len(self.grid)), dtype=np.int64
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self._obs_dim(),),
            dtype=np.float32,
        )

        # --- state (built in reset) ------------------------------------------
        self.inventory: InventoryEngine | None = None
        self.pipeline: OrderPipeline | None = None
        self.day = 0
        self._demand_history: deque[np.ndarray] = deque(maxlen=self.k)
        self._current_price = np.full(self.n, self.base_price)
        self._last_lead_times = np.zeros(self.n, dtype=int)
        self._current_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ dims

    def _obs_blocks(self) -> dict[str, tuple[int, ...]]:
        n = self.n
        return {
            "inventory": (n,),
            "pipeline": (n, self.obs_pipe_horizon),
            "demand_history": (n, self.k),
            "price": (n,),
            "lead_time": (n,),
            "headroom": (n,),
            "forecast": (n,),
        }

    def _obs_dim(self) -> int:
        return int(sum(int(np.prod(s)) for s in self._obs_blocks().values()))

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random  # gym-seeded generator, shared by all components

        self.inventory = InventoryEngine(
            self.n, self.capacity, initial_inventory=np.full(self.n, self.initial_inventory)
        )
        self.pipeline = OrderPipeline(
            self.n, self.config.lead_time, rng=rng
        )
        if self.demand_source is None:
            from src.env.synthetic_demand import SyntheticPoissonDemand  # lazy: default stub

            self.demand_source = SyntheticPoissonDemand(self.n, rng=rng)
        if hasattr(self.demand_source, "reset"):
            self.demand_source.reset(rng)
        self.forecaster.reset()

        self.day = 1
        self._current_price = np.full(self.n, self.base_price)
        self._last_lead_times = np.zeros(self.n, dtype=int)

        # Warm-up demand history: k days "before day 1" (labeled -k..-1) so
        # D_hist and D_hat are informative from the first decision.
        for offset in range(self.k):
            warmup_day = offset - self.k  # ..., -2, -1
            d = self.demand_source(warmup_day)
            self._demand_history.append(d.astype(float))
            self.forecaster.update(warmup_day, d)

        # Day-1 arrivals: pipeline is empty, this is a no-op kept for symmetry.
        self._receive_arrivals()
        self._push_price_to_demand_source()

        self._current_obs = self._build_obs()
        info = self._info_dict(demand=np.zeros(self.n), sales=np.zeros(self.n),
                               lost=np.zeros(self.n), arrivals=np.zeros(self.n),
                               accepted=np.zeros(self.n), clipped=np.zeros(self.n),
                               reward_breakdown=None)
        return self._current_obs.copy(), info

    def step(self, action):
        assert self._current_obs is not None, "call reset() first"
        cfg = self.config
        day = self.day

        # 2. register the order for today (lead times sampled here)
        qty = self.grid[np.clip(np.asarray(action), 0, len(self.grid) - 1)].astype(float)
        lead_times = self.pipeline.place_orders(qty, day=day)
        self._last_lead_times = np.where(lead_times > 0, lead_times, self._last_lead_times)

        # Day-t price is set at day START (Doc S2.5: day-1 price, held days 2-7)
        # BEFORE day-t demand is sampled — price-sensitive demand (P1).
        prev_price = self._current_price.copy()
        self._apply_pricing_hook(day)
        self._push_price_to_demand_source()

        # 3. demand + lost-sales transition
        demand = np.asarray(self.demand_source(day), dtype=float)
        sales, lost = self.inventory.sell(demand)
        self._demand_history.append(demand)
        self.forecaster.update(day, demand)

        # 5. shared reward — ordering cost on units DELIVERED today (R1 rule)
        breakdown = compute_reward(
            prices=self._current_price,
            sales=sales,
            lost_sales=lost,
            inventory_end=self.inventory.on_hand,
            orders_placed=self._accepted_today,   # delivered units (R1 capacity rule)
            unit_costs=np.full(self.n, self.unit_cost),
            prev_prices=prev_price,
            reward_cfg=cfg.reward,
        )

        # 6. advance to next day; receive its arrivals before building obs
        self.day += 1
        truncated = self.day > self.episode_length_days
        if not truncated:
            self._receive_arrivals()
            self._current_obs = self._build_obs()
        else:
            self._current_obs = None

        info = self._info_dict(demand, sales, lost, self._arrivals_today,
                               self._accepted_today, self._clipped_today, breakdown)
        obs = self._current_obs.copy() if self._current_obs is not None else np.zeros(self._obs_dim(), dtype=np.float32)
        return obs, float(breakdown.total), False, truncated, info

    # ------------------------------------------------------------------ internals

    def _receive_arrivals(self) -> None:
        arrivals = self.pipeline.receive(self.day)
        accepted, clipped = self.inventory.receive(arrivals)
        self._arrivals_today = arrivals
        self._accepted_today = accepted
        self._clipped_today = clipped

    def _apply_pricing_hook(self, day: int) -> None:
        """Integration seam (Doc S2.5): hook returns a new price vector on days
        the Pricing Agent acts; env HOLDS the price otherwise. With no hook,
        price is the constant stub."""
        if self.pricing_hook is not None:
            new_price = self.pricing_hook(day)
            if new_price is not None:
                self._current_price = np.asarray(new_price, dtype=float).copy()

    def _push_price_to_demand_source(self) -> None:
        """If the demand source is price-sensitive (P1 simulator), it receives
        the current price before sampling. Price-agnostic sources ignore this."""
        if hasattr(self.demand_source, "set_current_price"):
            self.demand_source.set_current_price(self._current_price)

    def _build_obs(self) -> np.ndarray:
        n = self.n
        pipe = self.pipeline.pending_by_remaining_days(self.day, self.obs_pipe_horizon)
        hist = np.stack(self._demand_history, axis=1)  # (n, k), oldest..newest
        headroom = self.inventory.headroom
        blocks = [
            self.inventory.on_hand,                      # I_t
            pipe.flatten(),                              # Q_t^pipe
            hist.flatten(),                              # D_(t-k:t-1)
            self._current_price,                         # p_t
            self._last_lead_times.astype(float),         # L_t
            np.full(n, headroom),                        # C headroom
            self.forecaster.forecast(self.day),          # D_hat_t
        ]
        return np.concatenate(blocks).astype(np.float32)

    def _info_dict(self, demand, sales, lost, arrivals, accepted, clipped, reward_breakdown) -> dict:
        return {
            "day": self.day,
            "n_skus": self.n,
            "demand": np.asarray(demand, dtype=float),
            "sales": np.asarray(sales, dtype=float),
            "lost_sales": np.asarray(lost, dtype=float),
            "arrivals": np.asarray(arrivals, dtype=float),
            "orders_delivered": np.asarray(accepted, dtype=float),
            "orders_clipped": np.asarray(clipped, dtype=float),
            "capacity_violation_events": self.inventory.capacity_violation_events,
            "capacity_clipped_units": self.inventory.capacity_clipped_units,
            "price": self._current_price.copy(),
            "total_on_hand": self.inventory.total_on_hand,
            "outstanding_orders": self.pipeline.outstanding(),
            "reward_terms": None if reward_breakdown is None else {
                "revenue": reward_breakdown.revenue,
                "ordering_cost": reward_breakdown.ordering_cost,
                "holding_cost": reward_breakdown.holding_cost,
                "stockout_cost": reward_breakdown.stockout_cost,
                "price_change_cost": reward_breakdown.price_change_cost,
            },
        }

    # ------------------------------------------------------------------ extras

    def observation_components(self) -> dict[str, np.ndarray]:
        """Named obs blocks (for tests/inspection); obs = concat in this order."""
        n = self.n
        pipe = self.pipeline.pending_by_remaining_days(self.day, self.obs_pipe_horizon)
        hist = np.stack(self._demand_history, axis=1)
        return {
            "inventory": self.inventory.on_hand,
            "pipeline": pipe,
            "demand_history": hist,
            "price": self._current_price.copy(),
            "lead_time": self._last_lead_times.astype(float),
            "headroom": np.full(n, self.inventory.headroom),
            "forecast": self.forecaster.forecast(self.day),
        }
