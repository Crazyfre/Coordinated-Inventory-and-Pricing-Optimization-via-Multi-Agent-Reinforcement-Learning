"""Gymnasium-style weekly PricingEnv (Phase P2, pricing track).

Composes the calibrated demand simulator (P1) with the R1 supply machinery
(InventoryEngine + OrderPipeline) into the environment the PRICING agent
trains on BEFORE MAPPO integration. One `step()` advances ONE WEEK (7 env
days): the pricing action is applied at the week's first day, held for the
remaining 6 days (locked cadence, Doc S2.5), while daily replenishment runs
an inner heuristic. This yields a clean weekly MDP whose transition contains
the true aggregate weekly economics: sum over days of revenue, costs and
lost sales, all driven by price-sensitive demand.

Seams (mirrors ReplenishmentEnv, agents.md decision 13):
    - demand source: callable day -> demand(n_skus); P1 simulator plugs in.
    - forecaster:    DemandForecast subclass (env/forecast.py, co-owned).
    - replenishment: `replenishment_hook(day, state) -> qty(n_skus)`; default
      is a demand-cover heuristic standing in for the replenishment agent.

Day cycle (per simulated day inside one weekly step):
    1. arrivals A_t received (start of day, before demand)
    2. price applied (week start only) + pushed to the demand source
    3. replenishment hook may place an order for day t
    4. demand D_t realized; lost-sales sale: S = min(D, I + A)
    5. forecaster updated with D_t
    6. daily economic accounting accumulated into the shared reward
    7. day advances; next day's arrivals received

Week-step return:
    (obs o^P, weekly_reward = sum of 7 daily rewards, terminated, truncated, info)

Observation (Doc S2.3.1), flattened float32, block order fixed:
    o^P = [ I_t (n) | D_hist (n x k) | p_(t-1) (n) | x_t calendar (k_ctx)
            | D_hat_t (n) | inventory headroom (n) ]

The headroom block goes beyond the minimal S2.3.1 list so the pricing agent
can avoid over-stimulating demand when shelves are already full. Order matches
the doc components first, extension last.

Reward: canonical shared reward (env/reward.py) summed over the week's days.
Pricing actions are chosen from the locked multiplier set (S2.4.1); the env
holds the price between weekly decisions by construction.
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

DemandSource = Callable[[int], np.ndarray]
# replenishment_daily_hook(day, view) -> qty (n_skus,) placed on `day`
ReplenishmentHook = Callable[[int, dict], np.ndarray]


def default_replenishment_hook(days_of_cover: float = 3.0):
    """Heuristic daily replenisher standing in for the (untrained) R agent.

    Orders the qty that restores `days_of_cover` of D_hat per SKU, snapped
    to the locked order grid, when the inventory position (on-hand +
    outstanding) falls below that level. Sees only the doc data through the
    view dict — no peeking at env internals.
    """

    class _Hook:
        def __init__(self, cover: float):
            self.cover = float(cover)

        def __call__(self, day: int, view: dict) -> np.ndarray:
            cfg = get_locked_config()
            grid = np.arange(
                0, cfg.replenishment.q_max + 1, cfg.replenishment.order_step
            )
            d_hat = view["forecast"]
            position = view["on_hand"] + view["outstanding"]
            target = self.cover * d_hat + view["lead_time_demand_buffer"]
            target = np.maximum(target - position, 0.0)
            idx = np.searchsorted(grid, target, side="left")
            idx = np.minimum(idx, len(grid) - 1)
            return grid[idx].astype(float)

    return _Hook(days_of_cover)


class PricingEnv(gym.Env):
    """Single-agent WEEKLY pricing environment (pricing track, Phase P2)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: LockedConfig | None = None,
        episode_length_days: int = 112,          # weekly steps = ceil(days/7)
        demand_source: DemandSource | None = None,
        forecaster: DemandForecast | None = None,
        replenishment_hook: ReplenishmentHook | None = None,
        base_price: float | np.ndarray = 10.0,
        unit_cost: float | np.ndarray = 6.0,
        initial_inventory: float | np.ndarray = 30.0,
        capacity: float = 500.0,
        calendar_provider: Callable[[int], np.ndarray] | None = None,
    ):
        super().__init__()
        self.config = config or get_locked_config()
        self.episode_length_days = episode_length_days
        self.demand_source = demand_source
        self.forecaster = forecaster or RecentMeanForecaster(
            self.config.env.n_skus, window=7
        )
        self.replenishment_hook = replenishment_hook or default_replenishment_hook()
        self.calendar_provider = calendar_provider
        self.initial_inventory = (
            float(initial_inventory) if np.isscalar(initial_inventory)
            else np.asarray(initial_inventory, dtype=float).copy()
        )
        self.capacity = float(capacity)

        self.n = self.config.env.n_skus
        self.k = self.config.demand.history_window_k
        self.interval = self.config.timing.pricing_interval_days
        self.base_price = np.broadcast_to(
            np.asarray(base_price, dtype=float), (self.n,)
        ).copy()
        self.unit_cost = np.broadcast_to(
            np.asarray(unit_cost, dtype=float), (self.n,)
        ).copy()
        self.grid = np.arange(
            0, self.config.replenishment.q_max + 1, self.config.replenishment.order_step
        )

        # Context: either a provider (P1 calendar) or a 2-dim weekday/week stub.
        self.ctx_dim = (
            self.calendar_provider(np.array([1])).shape[-1]
            if self.calendar_provider is not None and self._provider_probe_ok()
            else 2
        )

        self.action_space = gym.spaces.Discrete(len(self.config.pricing.multipliers))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32
        )

        # --- state (built in reset) ---
        self.day = 0                    # current env day, 1-based
        self.week = 0                   # completed weekly steps
        self.inventory: InventoryEngine | None = None
        self.pipeline: OrderPipeline | None = None
        self._demand_history: deque[np.ndarray] = deque(maxlen=self.k)
        self._current_price = self.base_price.copy()
        self._prev_price = self.base_price.copy()
        self._last_lead_times = np.zeros(self.n, dtype=int)
        self._current_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ dims

    def _provider_probe_ok(self) -> bool:
        try:
            probe = self.calendar_provider(1)
            return probe is not None and np.asarray(probe).ndim >= 1
        except Exception:
            return False

    def _obs_blocks(self) -> dict[str, tuple[int, ...]]:
        n = self.n
        return {
            "inventory": (n,),
            "demand_history": (n, self.k),
            "price": (n,),                    # p_(t-1) at decision time (held price)
            "context": (self.ctx_dim,),
            "forecast": (n,),
            "headroom": (n,),
        }

    def _obs_dim(self) -> int:
        return int(sum(int(np.prod(s)) for s in self._obs_blocks().values()))

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        self.inventory = InventoryEngine(
            self.n, self.capacity,
            initial_inventory=(
                np.full(self.n, self.initial_inventory)
                if np.isscalar(self.initial_inventory)
                else np.asarray(self.initial_inventory, dtype=float).copy()
            ),
        )
        self.pipeline = OrderPipeline(self.n, self.config.lead_time, rng=rng)
        if self.demand_source is None:
            from src.env.synthetic_demand import SyntheticPoissonDemand

            self.demand_source = SyntheticPoissonDemand(self.n, rng=rng)
        if hasattr(self.demand_source, "reset"):
            self.demand_source.reset(rng)  # reseed: same seed -> same episode
        self.forecaster.reset()

        self.day = 1
        self.week = 0
        self._current_price = self.base_price.copy()
        self._prev_price = self.base_price.copy()
        self._last_lead_times = np.zeros(self.n, dtype=int)

        # Push the initial price BEFORE warm-up sampling: price-sensitive
        # demand sources (P1 simulator) refuse to sample without a price,
        # and warm-up demand must use the same price day 1 will.
        self._push_price_to_demand_source()

        # Warm-up: k days labeled -k..-1 feed D_hist and D_hat.
        for offset in range(self.k):
            warmup_day = offset - self.k
            d = self.demand_source(warmup_day)
            self._demand_history.append(d.astype(float))
            self.forecaster.update(warmup_day, d)

        self._receive_arrivals()
        self._current_obs = self._build_obs()
        info = self._empty_info()
        return self._current_obs.copy(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """One WEEKLY step: apply the multiplier on day 1 of the week, hold
        it for the remaining 6 days, simulate all 7 days economically."""
        assert self._current_obs is not None, "call reset() first"
        cfg = self.config

        # --- translate the pricing action ---
        action = int(np.clip(int(action), 0, len(cfg.pricing.multipliers) - 1))
        multiplier = cfg.pricing.multipliers[action]

        price_changes = 0
        weekly_reward = 0.0
        day_infos = []

        for _ in range(self.interval):
            if self.day > self.episode_length_days:
                break
            d = self.day

            # 2. pricing: new price at the week's first day, held after that.
            day_is_pricing_day = (d - cfg.timing.pricing_day_in_cycle) % self.interval == 0
            prev_price = self._current_price.copy()
            if day_is_pricing_day:
                self._current_price = self.base_price * multiplier
                price_changes += 1
            self._push_price_to_demand_source()

            # 3. replenishment hook may place an order (lead times sampled).
            qty = np.asarray(self.replenishment_hook(d, self._hook_view()), dtype=float)
            lead_times = self.pipeline.place_orders(qty, day=d)
            self._last_lead_times = np.where(
                lead_times > 0, lead_times, self._last_lead_times
            )

            # 4. demand + lost-sales transition.
            demand = np.asarray(self.demand_source(d), dtype=float)
            sales, lost = self.inventory.sell(demand)
            self._demand_history.append(demand)
            self.forecaster.update(d, demand)

            # 6. canonical shared reward, accumulated over the week.
            breakdown = compute_reward(
                prices=self._current_price,
                sales=sales,
                lost_sales=lost,
                inventory_end=self.inventory.on_hand,
                orders_placed=self._accepted_today,
                unit_costs=self.unit_cost,
                prev_prices=prev_price,
                reward_cfg=cfg.reward,
            )
            weekly_reward += breakdown.total
            day_infos.append({
                "day": d,
                "price": self._current_price.copy(),
                "demand": demand,
                "sales": sales,
                "lost_sales": lost,
                "reward_terms": {
                    "revenue": breakdown.revenue,
                    "ordering_cost": breakdown.ordering_cost,
                    "holding_cost": breakdown.holding_cost,
                    "stockout_cost": breakdown.stockout_cost,
                    "price_change_cost": breakdown.price_change_cost,
                },
            })

            # 7. advance the day; receive next day's arrivals.
            self.day += 1
            self._prev_price = prev_price
            if self.day <= self.episode_length_days:
                self._receive_arrivals()

        # episode end bookkeeping
        self.week += 1
        truncated = self.day > self.episode_length_days
        terminated = False
        self._current_obs = (
            self._build_obs() if not truncated else None
        )

        info = {
            "week": self.week,
            "action_multiplier": multiplier,
            "weekly_reward": weekly_reward,
            "days": day_infos,
            "n_price_changes": price_changes,
            "price": self._current_price.copy(),
            "total_on_hand": self.inventory.total_on_hand,
            "outstanding_orders": self.pipeline.outstanding(),
            "capacity_violation_events": self.inventory.capacity_violation_events,
            "capacity_clipped_units": self.inventory.capacity_clipped_units,
        }
        obs = (
            self._current_obs.copy() if self._current_obs is not None
            else np.zeros(self._obs_dim(), dtype=np.float32)
        )
        return obs, float(weekly_reward), terminated, truncated, info

    # ------------------------------------------------------------------ internals

    def _receive_arrivals(self) -> None:
        arrivals = self.pipeline.receive(self.day)
        accepted, clipped = self.inventory.receive(arrivals)
        self._arrivals_today = arrivals
        self._accepted_today = accepted
        self._clipped_today = clipped

    def _push_price_to_demand_source(self) -> None:
        if hasattr(self.demand_source, "set_current_price"):
            self.demand_source.set_current_price(self._current_price)

    def _hook_view(self) -> dict:
        """Everything a replenishment heuristic may see (Doc S2.3.2 data)."""
        return {
            "day": self.day,
            "on_hand": self.inventory.on_hand,
            "outstanding": self.pipeline.outstanding(),
            "forecast": self.forecaster.forecast(self.day),
            "lead_time_demand_buffer": float(
                self.config.lead_time.mean_days
                if self.config.lead_time.mean_days is not None
                else (self.config.lead_time.min_days + self.config.lead_time.max_days) / 2.0
            ),
            "price": self._current_price.copy(),
            "headroom": self.inventory.headroom,
        }

    def _build_obs(self) -> np.ndarray:
        n = self.n
        hist = np.stack(self._demand_history, axis=1)   # (n, k) oldest..newest
        if self.calendar_provider is not None:
            ctx = np.asarray(self.calendar_provider(self.day), dtype=float).ravel()
        else:
            weekday = (self.day - 1) % 7                # 0..6 stub context
            week_of_year = (self.day - 1) // 7 + 1
            ctx = np.array([float(weekday), float(week_of_year)])
        headroom = self.inventory.headroom
        blocks = [
            self.inventory.on_hand,                     # I_t
            hist.flatten(),                             # D_(t-k:t-1)
            self._current_price,                        # p (held since week start)
            ctx,                                        # x_t
            self.forecaster.forecast(self.day),         # D_hat_t
            np.full(n, headroom),                       # capacity signal
        ]
        return np.concatenate(blocks).astype(np.float32)

    def _empty_info(self) -> dict:
        return {
            "week": 0,
            "action_multiplier": None,
            "weekly_reward": 0.0,
            "days": [],
            "n_price_changes": 0,
            "price": self._current_price.copy(),
            "total_on_hand": self.inventory.total_on_hand,
            "outstanding_orders": self.pipeline.outstanding(),
            "capacity_violation_events": 0,
            "capacity_clipped_units": 0.0,
        }

    # ------------------------------------------------------------------ extras

    def observation_components(self) -> dict[str, np.ndarray]:
        """Named obs blocks; flattened obs = their concat, in this order."""
        n = self.n
        hist = np.stack(self._demand_history, axis=1)
        if self.calendar_provider is not None:
            ctx = np.asarray(self.calendar_provider(self.day), dtype=float).ravel()
        else:
            weekday = (self.day - 1) % 7
            week_of_year = (self.day - 1) // 7 + 1
            ctx = np.array([float(weekday), float(week_of_year)])
        return {
            "inventory": self.inventory.on_hand,
            "demand_history": hist,
            "price": self._current_price.copy(),
            "context": ctx,
            "forecast": self.forecaster.forecast(self.day),
            "headroom": np.full(n, self.inventory.headroom),
        }

    def set_current_price(self, prices: np.ndarray) -> None:
        """Override the env-held price with an absolute vector (demo/tests)."""
        self._current_price = np.asarray(prices, dtype=float).copy()
